"""Bridge to R: the R packages plm and splm fit the panel models of ``plm(backend="splm")``.

Non-spatial models (``pooled``, ``fe``, ``re``) are fitted with ``plm::plm``, the spatial lag and spatial error
models (``slm_fe``, ``slm_re``, ``sem_pooled``, ``sem_fe``, ``sem_re``) by maximum likelihood with
``splm::spml``; ``spdep`` turns the spatial weights into the weights object of R. R is started as a subprocess
(``Rscript``) and exchanges data with Python through temporary CSV files, as the lavaan engine of ``sem()`` does
(:mod:`apatools.r.lavaan`; the lookup of ``Rscript`` and the English messages are in :mod:`apatools.r.bridge`).

Private: the public function is :func:`apatools.plm.plm`.
"""

import json
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.stats

from .._deps import missing_dependency
from .bridge import r_literal, r_require, run_r

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    missing_dependency("plm", "plm", e)

# apatools model -> (R function, fixed arguments); ``effect`` of the fixed-effects models is set from ``effects``
SPLM_MODELS = {
    "pooled": ("plm", {"model": "pooling"}),
    "fe": ("plm", {"model": "within"}),
    "re": ("plm", {"model": "random"}),
    "slm_fe": ("spml", {"model": "within", "lag": True, "spatial.error": "none"}),
    "slm_re": ("spml", {"model": "random", "lag": True, "spatial.error": "none"}),
    "sem_pooled": ("spml", {"model": "pooling", "lag": False, "spatial.error": "b"}),
    "sem_fe": ("spml", {"model": "within", "lag": False, "spatial.error": "b"}),
    "sem_re": ("spml", {"model": "random", "lag": False, "spatial.error": "b"}),
}
_R_EFFECT = {"entity": "individual", "time": "time", "twoway": "twoways"}

_SCRIPT = """\
{require}
options(warn = 1)
dir <- {dir}
d <- read.csv(file.path(dir, "data.csv"), check.names = FALSE, na.strings = "NA")
d$e__ <- factor(d$e__, levels = sort(unique(d$e__))); d$t__ <- factor(d$t__, levels = sort(unique(d$t__)))
pd <- plm::pdata.frame(d, index = c("e__", "t__"))
args <- list(formula = {formula}, data = pd)
{weights}
base <- {base}
for (n in names(base)) args[[n]] <- base[[n]]
extra <- list()
{extra}
for (n in names(extra)) args[[n]] <- extra[[n]]
warns <- character()
fit <- withCallingHandlers(
  do.call({function}, args),
  warning = function(w) {{ warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }}
)
out <- function(x, name) write.csv(x, file.path(dir, name), row.names = FALSE, fileEncoding = "UTF-8")
spatial <- inherits(fit, "splm")
names_ <- character(); est <- numeric(); blocks <- list()
add <- function(values, vcov) {{
  if (is.null(values) || length(values) == 0) return(invisible())
  v <- as.numeric(values); nm <- names(values)
  vc <- if (is.null(vcov)) matrix(NA_real_, length(v), length(v)) else matrix(as.numeric(vcov), length(v), length(v))
  names_ <<- c(names_, nm); est <<- c(est, v); blocks[[length(blocks) + 1]] <<- vc
}}
if (spatial) {{
  add(fit$coefficients, fit$vcov); add(fit$arcoef, fit$vcov.arcoef); add(fit$errcomp, fit$vcov.errcomp)
}} else {{
  add(fit$coefficients, vcov(fit))
}}
k <- length(est); cv <- matrix(0, k, k); at <- 0
for (b in blocks) {{ m <- nrow(b); cv[at + seq_len(m), at + seq_len(m)] <- b; at <- at + m }}
dimnames(cv) <- list(names_, names_)
out(data.frame(term = names_, estimate = est, se = sqrt(diag(cv))), "terms.csv")
out(as.data.frame(cv, check.names = FALSE), "cov.csv")
ix <- plm::index(pd)
res <- as.numeric(residuals(fit)); fit_values <- tryCatch(as.numeric(fitted(fit)), error = function(e) rep(NA_real_, length(res)))
out(data.frame(e = as.integer(as.character(ix[[1]])), t = as.integer(as.character(ix[[2]])), resid = res, fitted = fit_values), "obs.csv")
info <- data.frame(plm = as.character(utils::packageVersion("plm")), splm = as.character(utils::packageVersion("splm")),
  loglik = tryCatch(as.numeric(unlist(fit$logLik))[1], error = function(e) NA_real_), df_resid = NA_real_,
  rsq = NA_real_, f = NA_real_, f_df1 = NA_real_, f_df2 = NA_real_)
if (!spatial) {{
  s <- summary(fit)
  info$df_resid <- fit$df.residual
  info$rsq <- as.numeric(s$r.squared[1])
  if (!is.null(s$fstatistic) && identical(names(s$fstatistic$parameter), c("df1", "df2"))) {{
    info$f <- as.numeric(s$fstatistic$statistic); info$f_df1 <- s$fstatistic$parameter[["df1"]]; info$f_df2 <- s$fstatistic$parameter[["df2"]]
  }}
}}
out(info, "info.csv")
out(data.frame(text = warns), "warnings.csv")
"""


@dataclass
class SplmFit:
    """What ``plm(backend="splm")`` returns for every model: the R fit converted to Python objects."""

    model: str  # apatools model name ("fe", "slm_re", ...)
    effects: str  # "entity", "time" or "twoway"
    table: "pd.DataFrame"  # coefficients: coef, se, stat (t or z), p-value, in the names of apatools
    cov: "pd.DataFrame"  # covariance matrix of the table rows (the spatial parameters are uncorrelated blocks)
    stat_name: str  # "t" (plm) or "z" (splm)
    y: np.ndarray  # dependent variable, in the layout of the panel (period blocks)
    fitted: np.ndarray
    resid: np.ndarray
    n_obs: int
    n_entities: int
    n_periods: int
    metrics: dict  # r_sq, r_sq_type, r_sq_adj, f_stat, f_pvalue, df_model, df_resid, aic, bic, llf (where available)
    ssr: float
    df_resid: float  # nan if not known
    design: "np.ndarray | None" = None  # pooled model: design matrix with the constant
    versions: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)  # warnings of R

    @property
    def summary(self):
        return self.table.round(4).to_string()


def _csv(path):
    return pd.read_csv(path, encoding="utf-8", na_values=["NA"], keep_default_na=True)


def fit_splm(name, panel, effects="entity", args=None, rscript=None, timeout=None):
    """
    Fit the model ``name`` of ``plm()`` with R and return a :class:`SplmFit`.

    ``args`` is a dict of further arguments of ``plm::plm`` / ``splm::spml`` (``{"zero.policy": True}``);
    they override the arguments set here. ``effects`` ("entity", "time", "twoway") applies to the
    fixed-effects models.
    """
    if name not in SPLM_MODELS:
        raise NotImplementedError(
            f"Model '{name}' is not available with backend='splm' (choose from {tuple(SPLM_MODELS)}); "
            "the GMM models need backend='spreg'."
        )
    function, base = SPLM_MODELS[name]
    base = dict(base)
    if name.endswith("_fe") or name == "fe":
        base["effect"] = _R_EFFECT[effects]
    spatial = function == "spml"
    if spatial and panel.w is None:
        raise ValueError(f"Model '{name}' needs spatial weights `w`.")
    x_names = list(panel.x.columns)
    renamed = {f"v{i + 1}": n for i, n in enumerate(x_names)}
    packages = ["plm"] + (["splm", "spdep"] if spatial else [])
    with tempfile.TemporaryDirectory(prefix="apatools_splm_") as tmp:
        tmp = Path(tmp)
        frame = pd.DataFrame(
            {f"v{i + 1}": panel.x.iloc[:, i].to_numpy() for i in range(len(x_names))}
        )
        frame.insert(0, "v0", panel.y.to_numpy())
        frame["e__"], frame["t__"] = panel.ent, panel.tim
        frame.to_csv(tmp / "data.csv", index=False, na_rep="NA", encoding="utf-8")
        weights = ""
        if spatial:
            pd.DataFrame(panel.w.sparse.toarray()).to_csv(
                tmp / "W.csv", index=False, header=False
            )
            weights = 'W <- as.matrix(read.csv(file.path(dir, "W.csv"), header = FALSE)); args$listw <- spdep::mat2listw(W)'
        formula = "v0 ~ " + " + ".join(renamed)
        script = _SCRIPT.format(
            require=r_require(*packages),
            dir=r_literal(tmp.as_posix()),
            formula=f"as.formula({json.dumps(formula)})",
            weights=weights,
            base=r_literal(base),
            extra="\n".join(
                f"extra[[{json.dumps(str(k))}]] <- {r_literal(v)}"
                for k, v in (args or {}).items()
            ),
            **{"function": "plm::plm" if function == "plm" else "splm::spml"},
        )
        run_r(script, tmp, rscript=rscript, timeout=timeout)
        terms, cov = _csv(tmp / "terms.csv"), _csv(tmp / "cov.csv")
        obs, info = _csv(tmp / "obs.csv"), _csv(tmp / "info.csv").iloc[0]
        messages = [
            " ".join(str(t).split())
            for t in _csv(tmp / "warnings.csv")["text"].dropna()
        ]
    for text in messages:
        warnings.warn(f"R ({function}): {text}", UserWarning, stacklevel=3)
    return _convert(
        name,
        panel,
        effects if "effect" in base else "entity",
        terms,
        cov,
        obs,
        info,
        renamed,
        messages,
    )


def _without_effects(y, panel, effects):
    """y minus its entity and / or period means (``effects``), as the within transformation of a balanced panel."""
    out = np.asarray(y, dtype=float).reshape(panel.T, panel.N)
    if effects == "entity":
        return (out - out.mean(axis=0)).ravel()
    if effects == "time":
        return (out - out.mean(axis=1, keepdims=True)).ravel()
    return (
        out - out.mean(axis=0) - out.mean(axis=1, keepdims=True) + out.mean()
    ).ravel()


def _convert(name, panel, effects, terms, cov, obs, info, renamed, messages):
    """The R tables as a :class:`SplmFit` in the conventions of apatools."""
    spatial = SPLM_MODELS[name][0] == "spml"
    lag = name.startswith("slm")
    rename = dict(renamed)
    rename["(Intercept)"] = "const"
    # splm calls the spatial lag coefficient "lambda" and the spatial error coefficient "rho", apatools (and
    # spreg) the other way round
    rename["lambda"] = "rho" if lag else "lambda"
    rename["rho"] = "lambda" if not lag else "rho"
    names = [rename.get(t, t) for t in terms["term"]]
    estimate, se = terms["estimate"].to_numpy(float), terms["se"].to_numpy(float)
    df_resid = float(info["df_resid"]) if pd.notna(info["df_resid"]) else np.nan
    stat = estimate / se
    p = 2 * (
        scipy.stats.t.sf(np.abs(stat), df_resid)
        if np.isfinite(df_resid)
        else scipy.stats.norm.sf(np.abs(stat))
    )
    table = pd.DataFrame(
        {"coef": estimate, "se": se, "stat": stat, "p-value": p}, index=names
    )
    covariance = pd.DataFrame(cov.to_numpy(float), index=names, columns=names)
    n = len(panel.y)
    position = obs["t"].to_numpy(int) * panel.N + obs["e"].to_numpy(
        int
    )  # period blocks of N entities
    order = np.empty(n, dtype=int)
    order[position] = np.arange(n)
    resid, fitted = (
        obs["resid"].to_numpy(float)[order],
        obs["fitted"].to_numpy(float)[order],
    )
    y = panel.y.to_numpy(float)
    if not np.isfinite(fitted).all():
        fitted = y - resid
    slopes = [c for c in names if c in panel.x.columns]
    k = len(slopes)
    metrics, design = {}, None
    if spatial:
        # spreg's pseudo R²: squared correlation of y (without the effects of a within model) and y - residual
        target = _without_effects(y, panel, effects) if name.endswith("_fe") else y
        metrics.update(
            r_sq=float(np.corrcoef(target, target - resid)[0, 1] ** 2),
            r_sq_type="pseudo",
        )
        loglik = float(info["loglik"])
        if np.isfinite(loglik):
            parameters = len(
                names
            )  # coefficients, spatial parameter and, for random effects, phi
            metrics.update(
                llf=loglik,
                aic=2 * parameters - 2 * loglik,
                bic=parameters * np.log(n) - 2 * loglik,
            )
    else:
        design = np.column_stack(
            ([np.ones(n)] if "const" in names else [])
            + [panel.x[c].to_numpy(float) for c in slopes]
        )
        if (
            name == "re"
        ):  # R's R² of the random-effects model refers to the transformed data: use the pseudo R²
            xb = (
                design
                @ table.loc[
                    (["const"] if "const" in names else []) + slopes, "coef"
                ].to_numpy()
            )
            metrics.update(
                r_sq=float(np.corrcoef(y, xb)[0, 1] ** 2), r_sq_type="pseudo"
            )
        else:
            r_sq = float(info["rsq"])
            absorbed = n - df_resid - k
            metrics.update(
                r_sq=r_sq,
                r_sq_type="within" if name == "fe" else "ols",
                r_sq_adj=1 - (1 - r_sq) * (n - absorbed) / df_resid,
            )
            if pd.notna(info["f"]):
                metrics.update(
                    f_stat=float(info["f"]),
                    f_pvalue=float(
                        scipy.stats.f.sf(info["f"], info["f_df1"], info["f_df2"])
                    ),
                    df_model=int(info["f_df1"]),
                    df_resid=int(info["f_df2"]),
                )
    return SplmFit(
        model=name,
        effects=effects,
        table=table,
        cov=covariance,
        stat_name="z" if spatial else "t",
        y=y,
        fitted=fitted,
        resid=resid,
        n_obs=n,
        n_entities=panel.N,
        n_periods=panel.T,
        metrics=metrics,
        ssr=float(np.sum(resid**2)),
        df_resid=df_resid,
        design=design,
        versions={"plm": str(info["plm"]), "splm": str(info["splm"])},
        warnings=messages,
    )
