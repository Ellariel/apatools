"""
Panel regressions in the same format as :mod:`apatools.lm` and :mod:`apatools.slm`.

* ``backend="spreg"`` (default): pooled, fixed effects (FE) and random effects (RE) models,
  also with spatial lag / spatial error terms, estimated with *spreg*.
* ``backend="linearmodels"`` (optional dependency, ``pip install linearmodels``): non-spatial
  pooled / FE / RE models with *linearmodels*, which adds e.g. clustered standard errors
  and two-way fixed effects.

Models
------
===============  =========================================================  =========
name             model                                                      backends
===============  =========================================================  =========
``pooled``       pooled OLS                                                 both
``fe``           fixed effects (within estimator)                           both
``re``           random effects                                             both
``slm_fe``       spatial lag, fixed effects (ML)                            spreg
``slm_re``       spatial lag, random effects (ML)                           spreg
``sem_pooled``   spatial error, pooled (ML)                                 spreg
``sem_fe``       spatial error, fixed effects (ML)                          spreg
``sem_re``       spatial error, random effects (ML)                         spreg
``sem_pooled_gm``  spatial error, pooled (GMM)                              spreg
``sem_re_gm``    spatial error, random effects (GMM, Kapoor-Kelejian-Prucha)  spreg
===============  =========================================================  =========

Note that ``spreg`` 1.9.0 raises a ``TypeError`` (complex numbers) in ``ML_ErrorRE`` for the data
we tried; use ``sem_re_gm`` until this is fixed upstream.

Data layout
-----------
Pass an ordinary *long* DataFrame with an entity and a time column (or a two-level
``(entity, time)`` index). :func:`make_panel` checks that the panel is balanced, aligns the
entities with the rows of the spatial weights ``w`` (by ``w.id_order``, or by ``w_ids``)
and sorts the data in the layout spreg needs (period blocks, entities in the order of ``w``).
"""
import io
import inspect
import contextlib
import warnings
from dataclasses import dataclass, replace
from itertools import zip_longest

import numpy as np
import scipy.stats
import scipy.sparse as sps
import spreg as spr
from libpysal.weights import WSP
from pandas import DataFrame, Index, Series, MultiIndex, concat, to_numeric

from .citation import Citation
from .format import format_r, format_p, get_stars
from .slm import w_subset_rows

# https://pysal.org/spreg/index.html
CITATION = Citation(
    APA="Rey, S. J., & Anselin, L. (2007). \
PySAL: A python library of spatial analytical methods. \
The Review of Regional Studies, 37(1), 5-27."
)
# https://bashtage.github.io/linearmodels/
CITATION_LINEARMODELS = Citation(
    APA="Sheppard, K. (n.d.). linearmodels: Linear models for panel data, \
instrumental variables and system regression [Computer software]. \
https://github.com/bashtage/linearmodels"
)

# name -> (spreg class, effects, spatial interaction)
SPREG_MODELS = {
    "pooled": ("PooledOLS", "pooled", None),
    "fe": ("PanelFE", "fe", None),
    "re": ("PanelRE", "re", None),
    "slm_fe": ("ML_LagFE", "fe", "lag"),
    "slm_re": ("ML_LagRE", "re", "lag"),
    "sem_pooled": ("ML_ErrorPooled", "pooled", "error"),
    "sem_fe": ("ML_ErrorFE", "fe", "error"),
    "sem_re": ("ML_ErrorRE", "re", "error"),
    "sem_pooled_gm": ("GM_ErrorPooled", "pooled", "error"),
    "sem_re_gm": ("GM_ErrorRE", "re", "error"),
}
LINEARMODELS_MODELS = ("pooled", "fe", "re")

# rows of a spreg coefficient table that are not slope coefficients
_NON_SLOPE = {"CONSTANT", "const", "Intercept", "rho", "lambda", "phi", "sigma2_v", "sigma2_1"}
_CONSTANTS = ("CONSTANT", "const", "Intercept")


###################
#  panel layout   #
###################


@dataclass
class PanelData:
    """
    Balanced panel in the *long layout* used by spreg: ``T`` consecutive blocks (periods) of
    ``N`` rows (entities), the entities in the order of the rows of the weights ``w``.

    Attributes
    ----------
    y : pandas.Series
        Dependent variable, index ``(entity, time)``.
    x : pandas.DataFrame
        Predictors (without constant), same index.
    ent, tim : numpy.ndarray
        Entity / period position (0..N-1 / 0..T-1) of every row.
    w : libpysal.weights.W or None
        Weights aligned with the entities (``w.n == N``), or None.
    N, T : int
    entities, times : list
        Labels of the entities (in layout order) and of the periods.
    """

    y: Series
    x: DataFrame
    ent: np.ndarray
    tim: np.ndarray
    w: object
    N: int
    T: int
    entities: list
    times: list

    def subset(self, keep):
        """
        Panel restricted to the rows ``keep`` (boolean mask). Must remove whole entities
        and/or whole periods so that the panel stays balanced; ``w`` is restricted accordingly.
        """
        keep = np.asarray(keep, dtype=bool)
        ent_kept = np.unique(self.ent[keep])
        tim_kept = np.unique(self.tim[keep])
        if keep.sum() != len(ent_kept) * len(tim_kept):
            raise ValueError("Subsetting must keep whole entities or whole periods (balanced panel).")
        w = self.w
        if w is not None and len(ent_kept) != self.N:
            w = w_subset_rows(w, ent_kept)
        return PanelData(
            y=self.y[keep],
            x=self.x[keep],
            ent=np.searchsorted(ent_kept, self.ent[keep]),
            tim=np.searchsorted(tim_kept, self.tim[keep]),
            w=w,
            N=len(ent_kept),
            T=len(tim_kept),
            entities=[self.entities[i] for i in ent_kept],
            times=[self.times[i] for i in tim_kept],
        )


def _column(data, key):
    """Values of a column or of an index level."""
    if key in data.columns:
        return data[key].to_numpy()
    if key in (data.index.names or []):
        return data.index.get_level_values(key).to_numpy()
    raise KeyError(f"'{key}' is neither a column nor an index level of the data.")


def make_panel(data, y=None, x=None, entity=None, time=None, w=None, formula=None,
               dropna=True, balance="error", w_ids=None):
    """
    Convert a long DataFrame into a :class:`PanelData` in the layout needed by spreg.

    Parameters
    ----------
    data : pandas.DataFrame
        Long data. The index must be unique if ``formula`` is used.
    y : str
        Dependent variable (ignored if ``formula`` is given).
    x : list of str
        Predictors, without constant (ignored if ``formula`` is given).
    entity, time : str, optional
        Columns (or index levels) identifying entity and period. If both are None, the
        index must be a two-level ``(entity, time)`` MultiIndex.
    w : libpysal.weights.W, optional
        Spatial weights. The entity values must be the ids ``w.id_order`` (or ``w_ids``);
        entities missing in the data are removed from ``w``.
    formula : str, optional
        patsy formula (``"y ~ x1 + x2"``); the intercept is dropped (spreg and
        linearmodels handle constants themselves).
    dropna : bool
        Drop rows with missing values.
    balance : {"error", "drop"}
        What to do with entities that are not observed in every period: raise an error
        or drop these entities (with a warning).
    w_ids : sequence, optional
        Entity labels in the row order of ``w``. Default: ``w.id_order``.

    Returns
    -------
    PanelData
    """
    if balance not in ("error", "drop"):
        raise ValueError("balance must be 'error' or 'drop'.")
    if (entity is None) != (time is None):
        raise ValueError("Pass both `entity` and `time`, or none of them (MultiIndex).")
    if entity is None:
        if not isinstance(data.index, MultiIndex) or data.index.nlevels != 2:
            raise ValueError("Without `entity` and `time` the data need a two-level (entity, time) index.")
        ent_all = data.index.get_level_values(0).to_numpy()
        tim_all = data.index.get_level_values(1).to_numpy()
    else:
        ent_all, tim_all = _column(data, entity), _column(data, time)

    if formula is not None:
        from patsy import dmatrices

        if not data.index.is_unique:
            raise ValueError("A unique data index is required when a formula is used.")
        Y, Xd = dmatrices(formula, data, return_type="dataframe")  # drops rows with NAs
        y = Y.columns[0]
        x = [c for c in Xd.columns if c != "Intercept"]
        if not x:
            raise ValueError("The formula has no predictors.")
        values = concat([Y, Xd[x]], axis=1)
        pos = data.index.get_indexer(values.index)
    else:
        if y is None or x is None or len(x) == 0:
            raise ValueError("Either formula or x,y have to be explicitly defined.")
        values = data[[y] + list(x)]
        pos = np.arange(len(data))
        if dropna:
            ok = values.notna().all(axis=1).to_numpy()
            values, pos = values[ok], pos[ok]
    x = list(x)

    frame = values.reset_index(drop=True)
    frame["_e"], frame["_t"] = ent_all[pos], tim_all[pos]
    ok = DataFrame({"e": frame["_e"], "t": frame["_t"]}).notna().all(axis=1).to_numpy()
    frame = frame[ok].reset_index(drop=True)
    if frame.duplicated(["_e", "_t"]).any():
        raise ValueError("Some (entity, time) pairs occur more than once.")

    # balance
    n_periods = frame["_t"].nunique()
    counts = frame.groupby("_e")["_t"].nunique()
    incomplete = counts[counts < n_periods].index
    if len(incomplete):
        if balance == "error":
            raise ValueError(
                f"The panel is not balanced: {len(incomplete)} of {len(counts)} entities are not "
                f"observed in all {n_periods} periods (e.g. {list(incomplete[:3])}). "
                "Use balance='drop' to drop them."
            )
        warnings.warn(
            f"Dropped {len(incomplete)} entities that are not observed in all {n_periods} periods.",
            UserWarning,
        )
        frame = frame[~frame["_e"].isin(incomplete)].reset_index(drop=True)
        if frame.empty:
            raise ValueError("No entity is observed in all periods.")

    # entity order and weights
    present = list(frame["_e"].unique())
    if w is not None:
        ids = list(w.id_order) if w_ids is None else list(w_ids)
        if len(ids) != w.n:
            raise ValueError(f"`w_ids` has {len(ids)} labels but w has {w.n} observations.")
        id2pos = {k: i for i, k in enumerate(ids)}
        unknown = [e for e in present if e not in id2pos]
        if unknown:
            raise ValueError(
                f"{len(unknown)} entities are not ids of the weights (e.g. {unknown[:3]}). "
                "The entity values must match w.id_order; otherwise pass `w_ids` with the entity "
                "labels in the row order of w."
            )
        order = sorted(id2pos[e] for e in present)
        if len(order) < w.n:
            warnings.warn(
                f"The weights are restricted to the {len(order)} entities of the data (w has {w.n}).",
                UserWarning,
            )
            w = w_subset_rows(w, order)
        entities = [ids[i] for i in order]
    else:
        try:
            entities = sorted(present)
        except TypeError:
            entities = present
    times = sorted(frame["_t"].unique())
    ent_pos = frame["_e"].map({e: i for i, e in enumerate(entities)}).to_numpy()
    tim_pos = frame["_t"].map({t: i for i, t in enumerate(times)}).to_numpy()
    srt = np.lexsort((ent_pos, tim_pos))  # periods first, then entities
    frame, ent_pos, tim_pos = frame.iloc[srt], ent_pos[srt], tim_pos[srt]

    index = MultiIndex.from_arrays(
        [Index(entities).take(ent_pos), Index(times).take(tim_pos)],  # keeps the label dtypes
        names=["entity", "time"],
    )
    return PanelData(
        y=Series(frame[y].to_numpy(dtype=float), index=index, name=y),
        x=DataFrame(frame[x].to_numpy(dtype=float), index=index, columns=x),
        ent=ent_pos,
        tim=tim_pos,
        w=w,
        N=len(entities),
        T=len(times),
        entities=entities,
        times=times,
    )


#####################
#  model fitting    #
#####################


def _dummy_w(n):
    """Weights without any neighbours, for spreg models that need ``w`` only to get N."""
    return WSP(sps.csr_matrix((n, n))).to_W(silence_warnings=True)


def _model_kwargs(kwargs, name, stage="model"):
    prefix = f"{name}_{stage}_"
    return {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)}


def _fit_spreg(name, panel, kwargs):
    cls_name, effects, spatial = SPREG_MODELS[name]
    cls = getattr(spr, cls_name)
    w = panel.w
    if spatial and w is None:
        raise ValueError(f"Model '{name}' needs spatial weights `w`.")
    model_kwargs = _model_kwargs(kwargs, name)
    params = inspect.signature(cls.__init__).parameters
    slx_lags = kwargs.get("slx_lags", 0)
    if slx_lags:
        if "slx_lags" not in params:
            raise ValueError(f"Model '{name}' does not support slx_lags.")
        model_kwargs.setdefault("slx_lags", slx_lags)
    if w is None:
        w = _dummy_w(panel.N)
        if "spat_diag" in params:  # diagnostics are meaningless without weights
            model_kwargs.setdefault("spat_diag", False)
    with contextlib.redirect_stdout(io.StringIO()):  # spreg prints the model title
        return cls(panel.y.to_frame(), panel.x, w, **model_kwargs)


def _fit_linearmodels(name, panel, kwargs):
    try:
        from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "backend='linearmodels' needs the optional dependency: pip install linearmodels"
        ) from e
    if name not in LINEARMODELS_MODELS:
        raise NotImplementedError(
            f"Model '{name}' is not available with backend='linearmodels' "
            f"(choose from {LINEARMODELS_MODELS}); spatial models need backend='spreg'."
        )
    if kwargs.get("slx_lags", 0):
        raise NotImplementedError("slx_lags needs backend='spreg'.")
    model_kwargs = _model_kwargs(kwargs, name)
    fit_kwargs = _model_kwargs(kwargs, name, "fit")
    x = panel.x
    if name == "fe":
        model_kwargs.setdefault("entity_effects", True)
        model = PanelOLS(panel.y, x, **model_kwargs)
    else:
        x = x.copy()
        x.insert(0, "const", 1.0)  # like spreg, add the constant ourselves
        model = (PooledOLS if name == "pooled" else RandomEffects)(panel.y, x, **model_kwargs)
    return model.fit(**fit_kwargs)


def _fit_one(name, panel, backend, kwargs):
    if backend == "spreg":
        if name not in SPREG_MODELS:
            raise NotImplementedError(f"'{name}' is not implemented, try {tuple(SPREG_MODELS)}.")
        return _fit_spreg(name, panel, kwargs)
    if backend == "linearmodels":
        return _fit_linearmodels(name, panel, kwargs)
    raise ValueError("backend must be 'spreg' or 'linearmodels'.")


def _is_spreg(r):
    return type(r).__module__.startswith("spreg")


def _model_name(r):
    """Model name (key of :data:`SPREG_MODELS`) of a fitted result of either backend."""
    if _is_spreg(r):
        for name, (cls_name, _, _) in SPREG_MODELS.items():
            if type(r).__name__ == cls_name:
                return name
        return type(r).__name__
    cls = type(r.model).__name__
    if cls == "PanelOLS":
        return "fe"
    return {"RandomEffects": "re", "PooledOLS": "pooled"}.get(cls, cls)


def _clean_names(names):
    return [str(n).strip() for n in names]


def _param_table(r):
    """Coefficient table (coef, se, z/t, p-value, cil, cir) of a fitted result of either backend."""
    if _is_spreg(r):
        out = r.output.copy()
        out["var_names"] = _clean_names(out["var_names"])
        if type(r).__name__.startswith("ML_Lag"):
            # spreg names the lag coefficient "W_<name of y>"; call it rho
            name_y = r.name_y[0] if isinstance(r.name_y, (list, tuple)) else r.name_y
            out["var_names"] = ["rho" if n in ("W_dep_var", f"W_{name_y}") else n for n in out["var_names"]]
        t = out.set_index("var_names").apply(to_numeric, errors="coerce")
        stat = "t" if hasattr(r, "t_stat") else "z"  # pooled OLS reports t, the others z
        t = t.rename(columns={"coefficients": "coef", "std_err": "se", "zt_stat": stat, "prob": "p-value"})
        t.index.name = None
        zc = scipy.stats.norm.ppf(0.975)  # 95% CI, normal approximation
        t["cil"] = t["coef"] - zc * t["se"]
        t["cir"] = t["coef"] + zc * t["se"]
        return t[["coef", "se", stat, "p-value", "cil", "cir"]]
    ci = r.conf_int()
    return DataFrame(
        {
            "coef": r.params,
            "se": r.std_errors,
            "t": r.tstats,
            "p-value": r.pvalues,
            "cil": ci["lower"],
            "cir": ci["upper"],
        }
    )


def _cov_frame(r):
    """Covariance matrix of the coefficients as a DataFrame indexed by coefficient names."""
    if _is_spreg(r):
        vm = np.asarray(r.vm, dtype=float)
        names = _clean_names(r.output["var_names"])[: vm.shape[0]]
        return DataFrame(vm, index=names, columns=names)
    return r.cov


def _slopes(names):
    return [n for n in names if n not in _NON_SLOPE]


def _finite(v):
    try:
        v = float(np.ravel(v)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return v if np.isfinite(v) else None


#################
#    metrics    #
#################


def base_metrics(r, panel=None):
    """
    Fit metrics of a fitted panel model of either backend.

    Keys (where available): ``model``, ``backend``, ``r_sq`` and ``r_sq_type``
    (``"ols"``: pooled R², ``"within"``: R² of the within-transformed data of an FE model,
    ``"overall"``: R² of a RE model with linearmodels, ``"pseudo"``: spreg pseudo R²,
    the squared correlation of observed and predicted y), ``r_sq_adj``, ``f_stat``,
    ``f_pvalue``, ``df_model``, ``df_resid`` (FE: ``N·T - N - k``), ``aic``, ``bic``, ``llf``
    (spreg: they do not count the ``N`` fixed effects), ``n_obs`` (= N·T), ``n_entities``,
    ``n_periods``, ``n_params`` (slope coefficients), ``mae`` and ``mad`` (median absolute
    deviation of the residuals).

    Residuals (``mae``, ``mad``) follow the estimator: for non-spatial FE models (both backends) they
    are the residuals of the within model, i.e. they include the estimated entity effects; for
    pooled and RE models they are ``y - x·β`` (spreg) or the backend's residuals (linearmodels).
    spreg's own ``PanelFE`` reports ``y - x·β`` and a pseudo R² that ignore the effects, so the
    within R² and the residuals are recomputed here. spreg's FE standard errors use
    ``N·T - N`` instead of ``N·T - N - k`` degrees of freedom (slightly too small), and its
    confidence intervals / p-values are normal-based.
    """
    name = _model_name(r)
    table = _param_table(r)
    slopes = _slopes(table.index)
    k = len(slopes)
    outputs = {"model": name, "backend": "spreg" if _is_spreg(r) else "linearmodels"}

    if _is_spreg(r):
        _, effects, spatial = SPREG_MODELS[name]
        n_obs = int(r.n)  # N*T (spreg panel models report the stacked size)
        n_per = int(r.t)
        n_ent = n_obs // n_per
        resid = np.asarray(r.u, dtype=float).ravel()
        if effects == "pooled" and spatial is None:
            r_sq, r_type = float(r.r2), "ols"
            outputs["r_sq_adj"] = float(r.ar2)
            F, p = r.f_stat
            outputs.update(f_stat=float(F), f_pvalue=float(p), df_model=len(table) - 1,
                           df_resid=n_obs - len(table))
        else:
            r_sq = float(r.pr2)
            r_type = "within" if (effects == "fe" and spatial is None) else "pseudo"
        if effects == "fe" and spatial is None:
            resid, y_within = _within_resid(r)  # spreg's own u / pr2 leave out the fixed effects
            r_sq = 1 - np.sum(resid**2) / np.sum(y_within**2)
            df_resid = n_obs - n_ent - k
            outputs["r_sq_adj"] = 1 - (1 - r_sq) * (n_obs - n_ent) / df_resid
            f_stat = (r_sq / k) / ((1 - r_sq) / df_resid)
            outputs.update(f_stat=f_stat, f_pvalue=float(scipy.stats.f.sf(f_stat, k, df_resid)),
                           df_model=k, df_resid=df_resid)
        for key, attr in (("aic", "aic"), ("bic", "schwarz"), ("llf", "logll")):
            v = _finite(getattr(r, attr, None))
            if v is not None:
                outputs[key] = v
    else:
        n_obs = int(r.nobs)
        n_ent, n_per = int(r.entity_info["total"]), int(r.time_info["total"])
        resid = np.asarray(r.resids, dtype=float).ravel()
        has_const = any(c in table.index for c in _CONSTANTS)
        if name == "fe":
            r_sq, r_type = float(r.rsquared_within), "within"
            outputs["r_sq_adj"] = 1 - (1 - r_sq) * (n_obs - n_ent) / r.df_resid
        elif name == "re":
            r_sq, r_type = float(r.rsquared_overall), "overall"
        else:
            r_sq, r_type = float(r.rsquared), "ols"
            outputs["r_sq_adj"] = 1 - (1 - r_sq) * (n_obs - int(has_const)) / r.df_resid
        if name != "re":
            f = r.f_statistic if r._cov_type == "Unadjusted" else r.f_statistic_robust
            outputs.update(f_stat=float(f.stat), f_pvalue=float(f.pval), df_model=int(f.df),
                           df_resid=int(f.df_denom))
        v = _finite(getattr(r, "loglik", None))
        if v is not None:
            outputs["llf"] = v

    outputs.update(
        r_sq=r_sq,
        r_sq_type=r_type,
        n_obs=n_obs,
        n_entities=n_ent,
        n_periods=n_per,
        n_params=k,
        mae=float(np.mean(np.abs(resid))),
        mad=float(np.median(np.abs(resid - np.median(resid)))),
    )
    return outputs


def _within_resid(r):
    """
    Within residuals and within-demeaned y of a non-spatial spreg ``PanelFE`` result.

    spreg's ``PanelFE.u`` is ``y - x·β`` *without* the fixed effects, so the residuals of the
    within model are ``u`` minus its entity means.
    """
    n_per = int(r.t)
    n_ent = int(r.n) // n_per
    u = np.asarray(r.u, dtype=float).reshape(n_per, n_ent)
    y = np.asarray(r.y, dtype=float).reshape(n_per, n_ent)
    return (u - u.mean(axis=0)).ravel(), (y - y.mean(axis=0)).ravel()


def _coefs(r):
    return _param_table(r)["coef"]


def _predict_rows(coefs, name, panel_full, wy_full, train_mask, test_mask):
    """Predictions for the rows ``test_mask`` from coefficients fitted on the other rows."""
    spatial = SPREG_MODELS[name][2] if name in SPREG_MODELS else None
    effects = SPREG_MODELS[name][1] if name in SPREG_MODELS else {"fe": "fe", "re": "re"}.get(name, "pooled")
    slopes = list(panel_full.x.columns)
    xb = panel_full.x.to_numpy() @ coefs[slopes].to_numpy()
    for c in _CONSTANTS:
        if c in coefs.index:
            xb = xb + float(coefs[c])
    if spatial == "lag":
        xb = xb + float(coefs["rho"]) * wy_full
    if effects == "fe":
        # entity effect = mean over the training periods of (y - rho*Wy - x*beta)
        resid = panel_full.y.to_numpy() - xb
        mu = np.bincount(panel_full.ent[train_mask], weights=resid[train_mask],
                         minlength=panel_full.N) / np.bincount(panel_full.ent[train_mask], minlength=panel_full.N)
        xb = xb + mu[panel_full.ent]
    return xb[test_mask]


def pred_metrics(results, panel, backend="spreg", group="entity", exclude_neighbors=False, **kwargs):
    """
    Leave-one-group-out prediction metrics (R²pred, MAEpred, MADpred) for panel models.

    Leave-one-*observation*-out is not meaningful for panels: the other periods of the same
    entity leak into the training sample. Instead whole groups are held out in turn, the model
    (and the weights) are refitted without them and their rows are predicted:

    * ``group="entity"``: one entity at a time (all its periods). Predicts a *new entity*, so
      it is not possible for fixed-effects models (an entity effect cannot be estimated);
      their metrics are ``nan`` with a warning.
    * ``group="time"``: one period at a time (all entities). Works for every model; fixed
      effects are estimated from the remaining periods.

    Predictions are ``x·β`` plus the constant, plus ``ρ·(Wy)`` for spatial lag models (observed
    ``y`` of the neighbours in the same period, original weights) and plus the entity effect
    for FE models. Random-effects models use the population-average predictor (no
    entity effect). Models with ``slx_lags`` are not supported.

    .. warning::
       With spatial data the neighbours of a held-out entity stay in the training sample, which
       makes ``group="entity"`` optimistic (see :func:`apatools.slm.pred_metrics`).
       ``exclude_neighbors=True`` additionally removes the neighbours of the held-out entity
       from the training sample (only for ``group="entity"``; needs ``w``).

    Cost: one refit per group and model.

    Parameters
    ----------
    results : list
        Fitted models (from :func:`plm`), whose type determines the model names.
    panel : PanelData
    backend : {"spreg", "linearmodels"}
    group : {"entity", "time"}
    exclude_neighbors : bool
    **kwargs
        The model arguments passed to :func:`plm`.

    Returns
    -------
    list of dict
        ``pred_loo_r_sq`` (clipped to [-1, 1]), ``pred_loo_mae`` and ``pred_loo_mad`` per model.
    """
    if group not in ("entity", "time"):
        raise ValueError("group must be 'entity' or 'time'.")
    if exclude_neighbors and (group != "entity" or panel.w is None):
        raise ValueError("exclude_neighbors=True needs group='entity' and spatial weights `w`.")
    if kwargs.get("slx_lags", 0):
        raise NotImplementedError("pred_metrics is not implemented for slx_lags > 0.")
    names = [_model_name(r) for r in results]
    y = panel.y.to_numpy()
    wy = np.zeros_like(y)
    if panel.w is not None:  # (Wy) of every row, period by period, from the original weights
        wy = np.asarray(panel.w.sparse @ y.reshape(panel.T, panel.N).T).T.ravel()
    nan_out = {k: np.nan for k in ("pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad")}

    skip = set()
    if group == "entity":
        skip = {i for i, n in enumerate(names) if n in SPREG_MODELS and SPREG_MODELS[n][1] == "fe"}
        if skip:
            warnings.warn(
                "Fixed-effects models cannot predict new entities; their pred_* metrics are NaN. "
                "Use pred_group='time'.",
                UserWarning,
            )

    errors = {i: [] for i in range(len(results)) if i not in skip}
    fails, n_islands = [], 0
    n_groups = panel.N if group == "entity" else panel.T
    for g in range(n_groups):
        if group == "entity":
            drop = {g}
            if exclude_neighbors:
                drop |= {int(panel.w.id2i[j]) for j in panel.w.neighbors[panel.w.id_order[g]]}
            train_mask = ~np.isin(panel.ent, list(drop))
            test_mask = panel.ent == g
        else:
            train_mask = panel.tim != g
            test_mask = panel.tim == g
        train = panel.subset(train_mask)
        if train.w is not None and len(train.w.islands):
            n_islands += 1
        for i, name in enumerate(names):
            if i in skip:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = _fit_one(name, train, backend, kwargs)
                pred = _predict_rows(_coefs(r), name, panel, wy, train_mask, test_mask)
                errors[i].append((np.flatnonzero(test_mask), y[test_mask] - pred))
            except Exception as e:  # e.g. singular design, no convergence
                fails.append(str(e))
    if fails:
        warnings.warn(
            f"Some attempts to calculate leave-one-{group}-out metrics failed ({len(fails)}): {set(fails)}",
            UserWarning,
        )
    if n_islands:
        warnings.warn(
            f"In {n_islands} refits the training weights contained islands (entities without neighbours).",
            UserWarning,
        )

    ybar = y.mean()
    outputs = []
    for i in range(len(results)):
        if i in skip or not errors[i]:
            outputs.append(dict(nan_out))
            continue
        rows = np.concatenate([e[0] for e in errors[i]])
        err = np.concatenate([e[1] for e in errors[i]])
        sst = np.sum((y[rows] - ybar) ** 2)
        outputs.append(
            {
                "pred_loo_r_sq": float(np.clip(1 - np.sum(err**2) / sst, -1.0, 1.0)),
                "pred_loo_mae": float(np.mean(np.abs(err))),
                "pred_loo_mad": float(np.median(np.abs(err - np.median(err)))),
            }
        )
    return outputs


################
#     plm      #
################


def plm(data, y=None, x=None, w=None, entity=None, time=None, model="fe", formula=None,
        backend="spreg", **kwargs):
    """
    Fit panel regressions (pooled, fixed effects, random effects, optionally spatial).

    Parameters
    ----------
    data : pandas.DataFrame
        Long panel data with an entity and a time column (or an ``(entity, time)`` index).
    y : str
        Dependent variable; ignored if ``formula`` is given.
    x : list of str
        Predictors (no constant); ignored if ``formula`` is given.
    w : libpysal.weights.W, optional
        Spatial weights (required for the ``slm_*`` / ``sem_*`` models). The entity values must be
        its ids (``w.id_order``), or pass ``w_ids``. See :func:`make_panel`.
    entity, time : str, optional
        Entity and period columns (or index levels). Both None: use the two-level index.
    model : str or list of str
        Any of ``pooled, fe, re, slm_fe, slm_re, sem_pooled, sem_fe, sem_re, sem_pooled_gm,
        sem_re_gm`` (see the module docstring); linearmodels supports ``pooled, fe, re``.
    formula : str, optional
        patsy formula; overrides ``y``/``x``.
    backend : {"spreg", "linearmodels"}
        Estimation package. ``"linearmodels"`` is an optional dependency and allows e.g.
        ``fe_fit_cov_type="clustered"``, ``fe_fit_cluster_entity=True`` or
        ``fe_model_time_effects=True`` (two-way fixed effects).
    **kwargs
        Function options (below) and arguments for the backend named
        ``<model>_model_<argument>`` (passed to the model class, e.g.
        ``slm_fe_model_method="LU"``) or ``<model>_fit_<argument>`` (passed to
        ``.fit()``, linearmodels only), where ``<model>`` is one of the names above.

    Function options
    ----------------
    verbose : bool (default True)
        Print the panel dimensions and the model summaries.
    dropna : bool (default True)
        Drop rows with missing values.
    balance : {"error", "drop"} (default "error")
        Entities not observed in all periods: raise an error or drop them.
    w_ids : sequence, optional
        Entity labels in the row order of ``w`` (default ``w.id_order``).
    slx_lags : int (default 0)
        Number of spatial lags of the predictors (spreg models that support it).
    base_metrics : bool (default True)
        Compute :func:`base_metrics`.
    pred_metrics : bool (default False)
        Add leave-one-group-out prediction metrics, see :func:`pred_metrics`.
    pred_group : {"entity", "time"} (default "entity")
        Group that is held out.
    pred_exclude_neighbors : bool (default False)
        Also hold out the neighbours of the held-out entity (needs ``w``).

    Returns
    -------
    results : list
        One fitted model per entry of ``model`` (spreg or linearmodels objects). Each has an
        attribute ``plm_panel`` with the :class:`PanelData` used.
    metrics : list of dict
        See :func:`base_metrics` (plus ``pred_loo_*`` if requested).

    Example
    -------
    plm(df, "y", ["x1", "x2"], w=w, entity="county", time="year",
        model=["pooled", "fe", "re", "slm_fe"],
        verbose=True, balance="error", slx_lags=0,
        base_metrics=True, pred_metrics=True, pred_group="time",
        slm_fe_model_method="LU")
    """
    verbose = kwargs.get("verbose", True)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    pred_group = kwargs.pop("pred_group", "entity")
    pred_exclude_neighbors = kwargs.pop("pred_exclude_neighbors", False)
    panel = make_panel(
        data, y, x, entity=entity, time=time, w=w, formula=formula,
        dropna=kwargs.pop("dropna", True),
        balance=kwargs.pop("balance", "error"),
        w_ids=kwargs.pop("w_ids", None),
    )
    if verbose:
        print(f"N={panel.N} entities, T={panel.T} periods, N*T={len(panel.y)}")
        print(f"Formula: {panel.y.name} ~ " + " + ".join(panel.x.columns))

    models = list(model) if isinstance(model, (list, tuple)) else [model]
    results = []
    for m in models:
        r = _fit_one(m, panel, backend, kwargs)
        try:  # what plm_tests() and mundlak() need to refit / test the model
            r.plm_panel = panel
            r.plm_spec = {"backend": backend, "kwargs": dict(kwargs)}
        except AttributeError:  # pragma: no cover
            pass
        results.append(r)
        if verbose:
            print(f"model: {m} ({backend})")
            print(r.summary)

    metrics = []
    if add_base_metrics or add_pred_metrics:
        metrics = [base_metrics(r, panel) for r in results]
        if add_pred_metrics:
            pm = pred_metrics(results, panel, backend=backend, group=pred_group,
                              exclude_neighbors=pred_exclude_neighbors, **kwargs)
            metrics = [{**m_, **p_} for m_, p_ in zip(metrics, pm)]
    return results, metrics


_R_SQ_SUFFIX = {"ols": "", "within": "within", "overall": "overall", "pseudo": "pseudo"}


def plm_report(results, metrics={}, format_pval=True, add_stars=True, decimal=3,
               add_ftest=True, add_aic=True, add_bic=True, add_llf=True, add_mae=True, add_mad=True,
               add_pred_loo_mae=True, add_pred_loo_mad=True, add_n_obs=True, add_panel_dims=True,
               intercept_name="Intercept"):
    """
    Build one APA-style coefficient table per model from :func:`plm` output.

    Columns: coef, se, z (or t), p-value, cil, cir (95% CI), optional ``sig`` stars. The first
    row's ``model`` column holds a fit summary such as
    ``R²within = .36, R²adj = .34, F(2, 123) = 35.1, p < .001, MAE = ..., N = 150, n = 25, T = 6``;
    R²pred, MAEpred and MADpred appear if :func:`plm` was run with ``pred_metrics=True``.
    ``R²within`` is the R² of the within-transformed data (FE), ``R²overall`` the overall R² of a
    random-effects model (linearmodels), ``R²pseudo`` the squared correlation of observed and
    predicted y (spreg, spatial and RE models) and a plain ``R²`` is the pooled OLS R².
    The spatial parameters (``W_dep_var`` = ρ, ``lambda`` = λ) and variance parameters are rows of
    the table. ``format_pval`` / ``add_stars`` may be True or a callable.
    """
    output = []
    for r, i in zip_longest(results, metrics, fillvalue={}):
        s = []
        if "r_sq" in i:
            s.append(f"R²{_R_SQ_SUFFIX.get(i.get('r_sq_type', 'ols'), '')} {format_r(i['r_sq'], use_letter=False)}")
        if "r_sq_adj" in i:
            s.append(f"R²adj {format_r(i['r_sq_adj'], use_letter=False)}")
        if "pred_loo_r_sq" in i:
            s.append(f"R²pred {format_r(i['pred_loo_r_sq'], use_letter=False)}")
        if add_ftest and "f_stat" in i and "df_model" in i:
            s.append(f"F({i['df_model']}, {i['df_resid']}) = {i['f_stat']:.2f}, {format_p(i['f_pvalue'])}")
        if add_aic and "aic" in i:
            s.append(f"AIC = {i['aic']:.1f}")
        if add_bic and "bic" in i:
            s.append(f"BIC = {i['bic']:.1f}")
        if add_llf and "llf" in i:
            s.append(f"LL = {i['llf']:.1f}")
        if add_mae and "mae" in i:
            s.append(f"MAE = {i['mae']:.3f}")
        if add_mad and "mad" in i:
            s.append(f"MAD = {i['mad']:.3f}")
        if add_pred_loo_mae and "pred_loo_mae" in i:
            s.append(f"MAEpred = {i['pred_loo_mae']:.3f}")
        if add_pred_loo_mad and "pred_loo_mad" in i:
            s.append(f"MADpred = {i['pred_loo_mad']:.3f}")
        if add_n_obs and "n_obs" in i:
            s.append(f"N = {i['n_obs']}")
        if add_panel_dims and "n_entities" in i:
            s.append(f"n = {i['n_entities']}, T = {i['n_periods']}")
        s = ", ".join(s)

        params = _param_table(r)
        if decimal:
            for c in ["coef", "se", "cil", "cir"]:
                params[c] = params[c].round(decimal)
        if add_stars:
            add_stars = add_stars if callable(add_stars) else get_stars
            params["sig"] = [add_stars(c) for c in params["p-value"]]
        if format_pval:
            format_pval = (
                format_pval
                if callable(format_pval)
                else lambda x_: format_p(x_, use_letter=False, keep_spaces=False, no_equals=True)
            )
            params["p-value"] = [format_pval(c) for c in params["p-value"]]
        if len(i):
            params.loc[params.index[0], "model"] = s
        params.index = [intercept_name if n in _CONSTANTS else n for n in params.index]
        output.append(params)
    return output


#################
#     tests     #
#################


def _row(test, model, stat, df, p, label, df_denom=None):
    df_txt = f"{df}" if df_denom is None else f"{df}, {df_denom}"
    report = f"{label}({df_txt}) = {stat:.2f}, {format_p(p)}" if df is not None else f"{label} = {stat:.2f}, {format_p(p)}"
    return {"test": test, "model": model, "statistic": float(stat), "df": df, "df_denom": df_denom,
            "p-value": float(p), "report": report}


def hausman(r_fe, r_re):
    """
    Hausman test of a fixed- against a random-effects model with the same spatial structure.

    Compares the common slope coefficients: ``H = d' (V_fe - V_re)⁻¹ d`` ~ χ²(k).
    Works for both backends. Returns ``(statistic, df, p-value)``; statistic and p-value are
    ``nan`` (with a warning) if ``V_fe - V_re`` is not positive definite. This is common when the
    effects are correlated with the predictors (the random-effects variance is then too large),
    i.e. exactly when the test should reject; use :func:`mundlak` in that case.
    """
    t_fe, t_re = _param_table(r_fe), _param_table(r_re)
    common = [n for n in _slopes(t_fe.index) if n in t_re.index]
    if not common:
        raise ValueError("No common slope coefficients.")
    d = (t_fe.loc[common, "coef"] - t_re.loc[common, "coef"]).to_numpy()
    v = (_cov_frame(r_fe).loc[common, common] - _cov_frame(r_re).loc[common, common]).to_numpy()
    if np.any(np.linalg.eigvalsh((v + v.T) / 2) <= 0):
        warnings.warn("V_fe - V_re is not positive definite; the Hausman statistic is undefined.",
                      UserWarning)
        return np.nan, len(common), np.nan
    h = float(d @ np.linalg.solve(v, d))
    return h, len(common), float(scipy.stats.chi2.sf(h, len(common)))


def mundlak(r_re):
    """
    Mundlak test of correlated entity effects for a random-effects model.

    The model is refitted with the entity means of the (time-varying) predictors added; a
    Wald test checks that their coefficients are jointly zero. Under the null the random-effects
    estimator is consistent. The test is the regression-based equivalent of the Hausman test
    for the slope coefficients, but it stays defined when ``V_fe - V_re`` is not positive
    definite, which is typical when the effects *are* correlated with the predictors (the
    random-effects variance is then inflated), and it works for all random-effects models.

    Parameters
    ----------
    r_re : fitted random-effects model (``re``, ``slm_re``, ``sem_re_gm``, ...)
        Must come from :func:`plm`, which stores the panel and the fitting options on it.

    Returns
    -------
    (statistic, df, p-value) of the χ²(df) Wald test.
    """
    panel, spec = getattr(r_re, "plm_panel", None), getattr(r_re, "plm_spec", None)
    if panel is None or spec is None:
        raise ValueError("The model must come from plm() (panel and fit options are stored on it).")
    name = _model_name(r_re)
    x = panel.x
    means = x.groupby(panel.ent).transform("mean")
    within_sd = (x - means).std()
    varying = [c for c in x.columns if within_sd[c] > 1e-12]  # time-invariant means are collinear
    if not varying:
        raise ValueError("No predictor varies over time.")
    aug = concat([x, means[varying].add_suffix("_mean")], axis=1)
    r = _fit_one(name, replace(panel, x=aug), spec["backend"], spec["kwargs"])
    names = [f"{c}_mean" for c in varying]
    b = _param_table(r).loc[names, "coef"].to_numpy()
    v = _cov_frame(r).loc[names, names].to_numpy()
    stat = float(b @ np.linalg.solve(v, b))
    return stat, len(names), float(scipy.stats.chi2.sf(stat, len(names)))


def plm_tests(results, w=None, format_pval=True):
    """
    Specification tests for the models fitted with :func:`plm`, as a tidy DataFrame.

    Computed where the required models are among ``results``:

    * **Hausman** FE vs RE (``fe``/``re``, ``slm_fe``/``slm_re``, ``sem_fe``/``sem_re``) for
      the common slope coefficients. Null: random effects are consistent. Omitted (with a
      warning) when ``V_fe - V_re`` is not positive definite, see :func:`hausman`.
    * **Mundlak** test for every random-effects model, see :func:`mundlak`. Null: the effects
      are uncorrelated with the predictors (random effects are consistent).
    * **F test of entity effects** (``fe`` vs ``pooled``, non-spatial, same backend). Null: no
      entity effects.
    * **Baltagi-Song-Koh LM tests** (spreg, from ``pooled``; need weights): marginal random
      effects (LM1), marginal spatial error (LM2) and their joint test (LMJ), and the
      Anselin LM tests for a spatial lag / error (and their robust versions).

    Parameters
    ----------
    results : list
        Fitted models from :func:`plm`.
    w : libpysal.weights.W, optional
        Weights for the LM tests; by default those stored with the results.

    Returns
    -------
    pandas.DataFrame
        Columns ``test``, ``model``, ``statistic``, ``df``, ``df_denom``, ``p-value`` and an
        APA-style string ``report``.
    """
    by_name = {}
    for r in results:
        by_name.setdefault((_model_name(r), _is_spreg(r)), r)
    panel = next((getattr(r, "plm_panel", None) for r in results if hasattr(r, "plm_panel")), None)
    rows = []

    for fe_name, re_name in (("fe", "re"), ("slm_fe", "slm_re"), ("sem_fe", "sem_re"), ("sem_fe", "sem_re_gm")):
        for sp in (True, False):
            if (fe_name, sp) in by_name and (re_name, sp) in by_name:
                h, df, p = hausman(by_name[(fe_name, sp)], by_name[(re_name, sp)])
                if np.isfinite(h):
                    rows.append(_row("Hausman (FE vs RE)", f"{fe_name} vs {re_name}", h, df, p, "χ²"))

    for (name, sp), r in by_name.items():
        if name in SPREG_MODELS and SPREG_MODELS[name][1] == "re" and hasattr(r, "plm_spec"):
            try:
                stat, df, p = mundlak(r)
            except (ValueError, np.linalg.LinAlgError, KeyError):  # e.g. only time-invariant predictors
                continue
            rows.append(_row("Mundlak (correlated effects)", name, stat, df, p, "χ²"))

    for sp in (True, False):
        if ("fe", sp) in by_name and ("pooled", sp) in by_name:
            r_fe, r_p = by_name[("fe", sp)], by_name[("pooled", sp)]
            if sp:
                n_obs = int(r_fe.n)
                n_ent = n_obs // int(r_fe.t)
                k = len(_slopes(_param_table(r_fe).index))
                df1, df2 = n_ent - 1, n_obs - n_ent - k
                ssr_fe, ssr_p = float(np.sum(np.square(_within_resid(r_fe)[0]))), float(np.sum(np.square(r_p.u)))
                f = ((ssr_p - ssr_fe) / df1) / (ssr_fe / df2)
                p = float(scipy.stats.f.sf(f, df1, df2))
            else:
                fp = r_fe.f_pooled
                f, p, df1, df2 = float(fp.stat), float(fp.pval), int(fp.df), int(fp.df_denom)
            rows.append(_row("F test of entity effects", "fe vs pooled", f, df1, p, "F", df2))

    r_p = by_name.get(("pooled", True))
    if r_p is not None:
        w = w if w is not None else (panel.w if panel is not None else None)
        if w is not None:
            from spreg import diagnostics_panel as dp

            bsk = dp.BSK_tests(r_p, w, which=["LM1", "LM2", "LMJ"])
            for _, b in bsk.iterrows():
                lab = "χ²" if int(b["df"]) > 1 else "LM"
                rows.append(_row(b["Test"], "pooled", b["Statistic"], int(b["df"]) if lab == "χ²" else None,
                                 b["p-value"], lab))
            y, xx = r_p.y, r_p.x[:, 1:]  # without the constant
            for label, fn in (("LM lag", dp.panel_LMlag), ("LM error", dp.panel_LMerror),
                              ("Robust LM lag", dp.panel_rLMlag), ("Robust LM error", dp.panel_rLMerror)):
                stat, p = fn(y, xx, w)
                rows.append(_row(label, "pooled", stat, 1, p, "χ²"))

    cols = ["test", "model", "statistic", "df", "df_denom", "p-value", "report"]
    return DataFrame(rows, columns=cols)
