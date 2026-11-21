"""Bridge to R: lavaan fits models (``sem(engine="lavaan")``) and semPlot draws them
(``sem_plot(engine="lavaan")``).

R is started as a subprocess (``Rscript``) and exchanges data with Python through temporary CSV
files, so no Python package beyond pandas is needed. Needs R with the R package ``lavaan`` (and
``semPlot`` for the diagrams); ``Rscript`` is looked up in this order: the ``rscript`` argument, the
environment variable ``APATOOLS_RSCRIPT``, the ``PATH``, ``R_HOME`` and the usual install locations.
The messages of R are in English (``LANGUAGE=en``) whatever the system language; set the environment
variable ``APATOOLS_R_LANGUAGE`` (e.g. ``de``) to get them translated.

Private: the public functions are :func:`apatools.sem.sem` and :func:`apatools.sem.sem_plot`.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._deps import missing_dependency

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    missing_dependency("sem", "sem", e)

ENV_RSCRIPT = "APATOOLS_RSCRIPT"
ENV_LANGUAGE = "APATOOLS_R_LANGUAGE"  # language of the messages of R; English by default

# semopy's names of the estimators (``sem(method=...)``) and the lavaan estimators that correspond to them
LAVAAN_ESTIMATORS = {
    "MLW": "ML",
    "ML": "ML",
    "FIML": "ML",  # together with missing="fiml"
    "ULS": "ULS",
    "GLS": "GLS",
    "DWLS": "DWLS",
    "WLS": "WLS",
}

_MISSING_PACKAGE = "MISSING_R_PACKAGE:"


def find_rscript(rscript=None):
    """
    Path of the ``Rscript`` executable.

    Looks at ``rscript`` (a path or a command name), the environment variable ``APATOOLS_RSCRIPT``,
    the ``PATH``, ``R_HOME`` and the usual install locations of Windows, macOS and Linux (the newest
    R version on Windows). Raises ``RuntimeError`` with installation hints if R is not found.
    """
    candidates = []
    for explicit in (rscript, os.environ.get(ENV_RSCRIPT)):
        if explicit:
            found = shutil.which(str(explicit))
            if found:
                return found
            raise RuntimeError(
                f"Rscript '{explicit}' was not found: check the path "
                f"(argument 'rscript' or environment variable {ENV_RSCRIPT})."
            )
    found = shutil.which("Rscript")
    if found:
        return found
    exe = "Rscript.exe" if os.name == "nt" else "Rscript"
    if os.environ.get("R_HOME"):
        candidates.append(os.path.join(os.environ["R_HOME"], "bin", exe))
    candidates += sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True)
    candidates += sorted(glob.glob(r"C:\Program Files\R\R-*\bin\x64\Rscript.exe"), reverse=True)
    candidates += [
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/bin/Rscript",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        "R was not found (needed for engine='lavaan'). Install R from https://cran.r-project.org and the R "
        "packages with install.packages(c('lavaan', 'semPlot')); if Rscript is not on the PATH, pass its "
        f"location as 'rscript' or set the environment variable {ENV_RSCRIPT}."
    )


def r_literal(value):
    """Python value as R code: None, bool, numbers, strings and (nested) lists / tuples of them."""
    if value is None:
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}L"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NA_real_"
        if np.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return repr(float(value))
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are valid R escapes
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [r_literal(v) for v in value]
        scalars = all(not isinstance(v, (list, tuple, dict, np.ndarray)) for v in value)
        return ("c(" if scalars else "list(") + ", ".join(items) + ")"
    if isinstance(value, dict):
        return "list(" + ", ".join(f"{json.dumps(str(k))} = {r_literal(v)}" for k, v in value.items()) + ")"
    raise TypeError(f"Cannot pass a value of type {type(value).__name__} to R.")


def run_r(script, workdir, rscript=None, timeout=None):
    """Run an R script (text) in ``workdir`` with Rscript; returns the stderr text (warnings and messages)."""
    exe = find_rscript(rscript)
    path = Path(workdir) / "script.R"
    path.write_text(script, encoding="utf-8")
    try:
        done = subprocess.run(
            [exe, str(path)],
            check=False,  # a failure is reported below, with the stderr of R
            cwd=str(workdir),
            env={**os.environ, "LANGUAGE": os.environ.get(ENV_LANGUAGE, "en")},  # English, whatever the system
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"R did not finish within {timeout} seconds.") from e
    if done.returncode != 0:
        err = done.stderr or ""
        for line in err.splitlines():
            if _MISSING_PACKAGE in line:
                package = line.split(_MISSING_PACKAGE, 1)[1].strip().strip("'\"")
                raise ImportError(
                    f"The R package '{package}' is not installed: run install.packages('{package}') in R."
                )
        raise RuntimeError("R failed:\n" + "\n".join(err.strip().splitlines()[-15:]))
    return done.stderr or ""


_REQUIRE = (
    'if (!requireNamespace("%s", quietly = TRUE)) stop("' + _MISSING_PACKAGE + '%s", call. = FALSE)\n'
)

_FIT_SCRIPT = """\
{require}
options(warn = 1)
dir <- {dir}
data <- read.csv(file.path(dir, "data.csv"), check.names = FALSE, stringsAsFactors = FALSE,
                 fileEncoding = "UTF-8", na.strings = "NA")
args <- list(model = {model}, data = data, estimator = {estimator})
extra <- list()
{extra}
for (name in names(extra)) args[[name]] <- extra[[name]]
{seed}
warns <- character()
fit <- withCallingHandlers(
  do.call(lavaan::sem, args),
  warning = function(w) {{ warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }}
)
out <- function(x, name) write.csv(x, file.path(dir, name), row.names = FALSE, fileEncoding = "UTF-8")
pe <- lavaan::parameterEstimates(fit, standardized = FALSE, ci = FALSE)
ss <- lavaan::standardizedSolution(fit, type = "std.all")
out(pe, "pe.csv")
out(ss, "ss.csv")
fm <- lavaan::fitMeasures(fit)
out(data.frame(name = names(fm), value = as.numeric(fm)), "fit.csv")
out(data.frame(version = as.character(utils::packageVersion("lavaan")),
               nobs = lavaan::lavInspect(fit, "nobs"),
               converged = lavaan::lavInspect(fit, "converged"),
               estimator = lavaan::lavInspect(fit, "options")$estimator,
               se = lavaan::lavInspect(fit, "options")$se), "info.csv")
out(data.frame(name = lavaan::lavNames(fit, "lv")), "latent.csv")
out(data.frame(text = warns), "warnings.csv")
"""


@dataclass
class LavaanFit:
    """What ``sem(engine="lavaan", return_model=True)`` returns instead of a ``semopy.Model``."""

    parameters: "pd.DataFrame"  # lavaan's parameterEstimates() and standardizedSolution() as returned by R
    fit_measures: "pd.Series"  # lavaan's fitMeasures()
    version: str = ""  # version of the R package lavaan
    converged: bool = True
    warnings: list = field(default_factory=list)  # warnings of lavaan


def _csv(path):
    return pd.read_csv(path, encoding="utf-8", na_values=["NA"], keep_default_na=True)


def lavaan_estimator(method, missing_lavaan_args):
    """lavaan estimator for ``method`` (semopy's or lavaan's name) and the extra arguments it implies."""
    key = str(method).upper()
    estimator = LAVAAN_ESTIMATORS.get(key, str(method))
    extra = {}
    if key == "FIML" and "missing" not in missing_lavaan_args:
        extra["missing"] = "fiml"
    return estimator, extra


def fit_lavaan(
    data,
    formula,
    estimator,
    se=None,
    bootstrap=0,
    lavaan_args=None,
    seed=13,
    rscript=None,
    timeout=None,
):
    """
    Fit a model with ``lavaan::sem`` in R and return ``(stats, metrics, fit)`` in the format of ``sem()``.

    ``estimator`` and ``se`` are lavaan's names, ``lavaan_args`` a dict of further arguments of
    ``lavaan::sem`` (``{"std.lv": True, "ordered": ["x1"]}``), ``bootstrap`` > 0 requests bootstrap
    standard errors with that many resamples.
    """
    lavaan_args = dict(lavaan_args or {})
    extra = dict(lavaan_args)
    if se is not None and "se" not in extra:
        extra["se"] = se
    if bootstrap:
        extra["se"] = "bootstrap"
        extra["bootstrap"] = int(bootstrap)
    with tempfile.TemporaryDirectory(prefix="apatools_lavaan_") as tmp:
        frame = data.copy()
        for column in frame.columns:  # R reads True / False as logical, but integers are safer
            if frame[column].dtype == bool:
                frame[column] = frame[column].astype(int)
        frame.to_csv(Path(tmp) / "data.csv", index=False, na_rep="NA", encoding="utf-8")
        script = _FIT_SCRIPT.format(
            require=_REQUIRE % ("lavaan", "lavaan"),
            dir=r_literal(Path(tmp).as_posix()),
            model=r_literal(str(formula)),
            estimator=r_literal(estimator),
            extra="\n".join(f"extra[[{json.dumps(str(k))}]] <- {r_literal(v)}" for k, v in extra.items()),
            seed=f"set.seed({int(seed)})" if seed is not None else "",
        )
        run_r(script, tmp, rscript=rscript, timeout=timeout)
        pe, ss = _csv(Path(tmp) / "pe.csv"), _csv(Path(tmp) / "ss.csv")
        measures = _csv(Path(tmp) / "fit.csv")
        info = _csv(Path(tmp) / "info.csv").iloc[0]
        latent = sorted(_csv(Path(tmp) / "latent.csv")["name"].dropna().astype(str))
        messages = [" ".join(str(t).split()) for t in _csv(Path(tmp) / "warnings.csv")["text"].dropna()]
    for text in messages:
        warnings.warn(f"lavaan: {text}", UserWarning, stacklevel=3)
    stats, metrics = convert_lavaan(pe, ss, measures, int(info["nobs"]), latent)
    fit = LavaanFit(
        parameters=pe.merge(ss, on=["lhs", "op", "rhs"], how="left", suffixes=("", "_std")),
        fit_measures=measures.set_index("name")["value"],
        version=str(info["version"]),
        converged=bool(info["converged"]),
        warnings=messages,
    )
    return stats, metrics, fit


# lavaan's fitMeasures() under the names of semopy's calc_stats(); SRMR is added because it is so common
_MEASURES = [
    ("DoF", "df"),
    ("DoF Baseline", "baseline.df"),
    ("chi2", "chisq"),
    ("chi2 p-value", "pvalue"),
    ("chi2 Baseline", "baseline.chisq"),
    ("CFI", "cfi"),
    ("GFI", "gfi"),
    ("AGFI", "agfi"),
    ("NFI", "nfi"),
    ("TLI", "tli"),
    ("RMSEA", "rmsea"),
    ("SRMR", "srmr"),
    ("AIC", "aic"),
    ("BIC", "bic"),
    ("LogLik", "logl"),
]
_STATS_COLUMNS = [
    "lval", "op", "rval", "estimate", "est_std", "std_err", "z-value", "p-value", "se_std", "z_std", "p_std",
]  # fmt: skip


def convert_lavaan(pe, ss, measures, nobs, latent):
    """
    lavaan's tables in the format of the ``sem()`` output.

    ``=~`` (loadings) become ``~`` rows from the indicator to the latent variable, as semopy
    reports them (``lval`` = indicator, ``rval`` = latent variable); regressions, (co)variances and
    intercepts (``~1``, with ``rval`` = ``"1"``) keep their meaning. Other rows (thresholds, defined
    parameters, constraints) are left out. Fixed parameters have no standard error, z-value or
    p-value (NaN), as in the semopy output.
    """
    if "group" in pe and pe["group"].nunique() > 1:
        raise NotImplementedError("Models with several groups are not supported by the lavaan engine.")
    keys = ["lhs", "op", "rhs"]
    table = pe.merge(ss[keys + ["est.std", "se", "z", "pvalue"]], on=keys, how="left", suffixes=("", "_std"))
    table = table[table["op"].isin(["=~", "~", "~~", "~1"])].reset_index(drop=True)
    if len(table) != len(pe[pe["op"].isin(["=~", "~", "~~", "~1"])]):
        raise RuntimeError("The parameter tables of lavaan could not be matched (duplicated parameters).")
    op, lhs, rhs = table["op"], table["lhs"], table["rhs"]
    loading, intercept = op == "=~", op == "~1"
    stats = pd.DataFrame(
        {
            "lval": np.where(loading, rhs, lhs),
            "op": np.where(op == "~~", "~~", "~"),
            "rval": np.where(loading, lhs, np.where(intercept, "1", rhs)),
            "estimate": table["est"],
            "est_std": table["est.std"],
            "std_err": table["se"],
            "z-value": table["z"],
            "p-value": table["pvalue"],
            "se_std": table["se_std"],
            "z_std": table["z_std"],
            "p_std": table["pvalue_std"],
        }
    )
    fixed = stats["std_err"].fillna(0) == 0  # lavaan reports se = 0 for fixed parameters
    stats.loc[fixed, ["std_err", "z-value", "p-value"]] = np.nan
    stats = stats[_STATS_COLUMNS].astype({"estimate": float})
    stats.attrs["latent"] = list(latent)

    values = measures.set_index("name")["value"]
    rows = {"N": float(nobs)}
    for name, key in _MEASURES:
        if key in values.index and np.isfinite(values[key]):
            rows[name] = float(values[key])
    metrics = pd.DataFrame({"value": pd.Series(rows)})
    return stats, metrics


_PLOT_SCRIPT = """\
{require}
suppressMessages(library(semPlot))
dir <- {dir}
e <- read.csv(file.path(dir, "edges.csv"), stringsAsFactors = FALSE, fileEncoding = "UTF-8")
v <- read.csv(file.path(dir, "vars.csv"), stringsAsFactors = FALSE, fileEncoding = "UTF-8")
pars <- data.frame(label = e$label, lhs = e$lhs, edge = e$edge, rhs = e$rhs, est = 0.5, std = 0.5,
                   group = "", fixed = FALSE, par = seq_len(nrow(e)), stringsAsFactors = FALSE)
vars <- data.frame(name = v$name, manifest = as.logical(v$manifest), exogenous = NA, stringsAsFactors = FALSE)
model <- new("semPlotModel", Pars = pars, Vars = vars, Thresholds = data.frame(), Computed = TRUE,
             ObsCovs = list(), ImpCovs = list(), Original = list())
file <- file.path(dir, {outfile})
device <- {device}
device(file, width = {width}, height = {height}{res})
semPlot::semPaths(model, whatLabels = "name", layout = {layout}, rotation = {rotation}, residuals = FALSE,
                  intercepts = FALSE, nCharNodes = 0, nCharEdges = 0, edge.label.cex = {label_cex},
                  edge.color = "black", mar = c(4, 4, 4, 4))
invisible(dev.off())
"""

_DEVICES = {
    "png": ("grDevices::png", True, "png"),
    "jpg": ("grDevices::jpeg", True, "jpg"),
    "jpeg": ("grDevices::jpeg", True, "jpg"),
    "tiff": ("grDevices::tiff", True, "tiff"),
    "bmp": ("grDevices::bmp", True, "bmp"),
    "pdf": ("grDevices::pdf", False, "pdf"),
    "svg": ("grDevices::svg", False, "svg"),
}


class RFigure:
    """A figure drawn by R: the file contents (``data``), the file type (``format``) and, if it was
    saved, its ``path``. Displays itself in Jupyter (PNG, SVG); ``save(path)`` writes a copy."""

    def __init__(self, data, format, path=None):
        self.data, self.format, self.path = data, format, path

    def save(self, path):
        Path(path).write_bytes(self.data)
        return path

    def _repr_png_(self):
        return self.data if self.format == "png" else None

    def _repr_svg_(self):
        return self.data.decode("utf-8") if self.format == "svg" else None

    def __repr__(self):
        return f"<RFigure {self.format}, {len(self.data)} bytes{', ' + str(self.path) if self.path else ''}>"


def semplot(
    order,
    latent,
    edges,
    covs,
    save_to_file=None,
    format_fig="png",
    dpi=300,
    layout="tree",
    rotation=2,
    width=None,
    height=None,
    fontsize=0.8,
    rscript=None,
    timeout=None,
):
    """
    Draw the path diagram with ``semPlot::semPaths`` in R and return it as :class:`RFigure`.

    ``order`` / ``latent`` / ``edges`` / ``covs`` come from ``apatools.sem._path_diagram``: the
    variables, the latent ones, and the labelled regression and covariance edges. The labels are
    drawn as given, so they are the ones of the other engines. ``layout`` is a semPlot layout
    (``"tree"``, ``"tree2"``, ``"circle"``, ``"circle2"``, ``"spring"``), ``rotation`` (1-4) turns the
    tree (2: from left to right), ``width`` / ``height`` are in inches (by default from the number of
    variables) and ``fontsize`` scales the edge labels.
    """
    suffix = os.path.splitext(str(save_to_file))[1].lstrip(".").lower() if save_to_file else ""
    kind = suffix or str(format_fig).lower()
    if kind not in _DEVICES:
        raise ValueError(f"Unsupported figure format '{kind}', use one of {sorted(_DEVICES)}.")
    device, uses_res, ext = _DEVICES[kind]
    n = max(len(order), 1)
    width = width if width is not None else max(7.0, 1.0 + 0.7 * n)
    height = height if height is not None else max(5.0, 0.6 * n)
    layouts = ("tree", "tree2", "circle", "circle2", "spring")
    if layout not in layouts:
        raise ValueError(f"Unknown layout '{layout}', use one of {layouts}.")
    with tempfile.TemporaryDirectory(prefix="apatools_semplot_") as tmp:
        rows = [(r, "->", rhs, lab) for r, rhs, lab in edges] + [(r, "<->", rhs, lab) for r, rhs, lab in covs]
        pd.DataFrame(rows, columns=["lhs", "edge", "rhs", "label"]).to_csv(
            Path(tmp) / "edges.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame({"name": list(order), "manifest": [n not in latent for n in order]}).to_csv(
            Path(tmp) / "vars.csv", index=False, encoding="utf-8"
        )
        script = _PLOT_SCRIPT.format(
            require=_REQUIRE % ("semPlot", "semPlot"),
            dir=r_literal(Path(tmp).as_posix()),
            outfile=r_literal(f"figure.{ext}"),
            device=device,
            width=r_literal(float(width)),
            height=r_literal(float(height)),
            res=f", units = \"in\", res = {int(dpi)}" if uses_res else "",
            layout=r_literal(layout),
            rotation=int(rotation),
            label_cex=r_literal(float(fontsize)),
        )
        run_r(script, tmp, rscript=rscript, timeout=timeout)
        data = (Path(tmp) / f"figure.{ext}").read_bytes()
    figure = RFigure(data, "jpg" if ext == "jpg" else ext)
    if save_to_file:
        figure.save(save_to_file)
        figure.path = str(save_to_file)
    return figure
