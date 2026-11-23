"""
Panel regressions in the same format as :mod:`apatools.lm` and :mod:`apatools.slm`.

* ``backend="spreg"`` (default): pooled, fixed effects (FE) and random effects (RE) models,
  also with spatial lag / spatial error terms, estimated with *spreg*.
* ``backend="linearmodels"`` (optional dependency, ``pip install linearmodels``): non-spatial
  pooled / FE / RE models with *linearmodels*, which adds e.g. clustered standard errors
  and two-way fixed effects.
* ``backend="splm"`` (needs R with the packages plm, splm and spdep, see :mod:`apatools.r.bridge`): the
  non-spatial models with R's *plm*, the spatial ML models with R's *splm*, including period effects of the
  spatial lag model. Every model starts R (a few seconds).

Models
------
================  =========================================================  ==================
name              model                                                      backends
================  =========================================================  ==================
``pooled``        pooled OLS                                                 all
``fe``            fixed effects (within estimator)                           all
``re``            random effects                                             all
``slm_fe``        spatial lag, fixed effects (ML)                            spreg, splm
``slm_re``        spatial lag, random effects (ML)                           spreg, splm
``sem_pooled``    spatial error, pooled (ML)                                 spreg, splm
``sem_fe``        spatial error, fixed effects (ML)                          spreg, splm
``sem_re``        spatial error, random effects (ML)                         spreg, splm
``sem_pooled_gm``  spatial error, pooled (GMM)                               spreg
``sem_re_gm``     spatial error, random effects (GMM, Kapoor-Kelejian-Prucha)  spreg
================  =========================================================  ==================

Note that ``sem_re`` (spreg's ``ML_ErrorRE``, version 1.9.0) raised a ``TypeError`` (complex numbers) with
the older numpy/scipy versions we tested (numpy 2.0-2.2, scipy 1.13-1.15) and worked with recent ones
(numpy 2.5, scipy 1.18); ``sem_re_gm`` worked everywhere.

Data layout
-----------
Pass an ordinary *long* DataFrame with an entity and a time column (or a two-level
``(entity, time)`` index). :func:`make_panel` checks that the panel is balanced, aligns the
entities with the rows of the spatial weights ``w`` (by ``w.id_order``, or by ``w_ids``)
and sorts the data in the layout spreg needs (period blocks, entities in the order of ``w``).
"""

import contextlib
import inspect
import io
import warnings
from dataclasses import dataclass, replace
from itertools import zip_longest

import numpy as np
import scipy.sparse as sps
import scipy.stats

from ._deps import missing_dependency

try:
    import patsy  # noqa: F401  (used for formulas)
    import spreg as spr
    from libpysal.weights import WSP
    from pandas import DataFrame, Index, MultiIndex, Series, concat, to_numeric
except ImportError as e:  # pragma: no cover
    missing_dependency("plm", "plm", e)

from ._common import (
    INTERCEPT_NAMES,
    format_significance,
    rename_intercept,
    round_columns,
    w_subset_rows,
)
from ._impacts import effects_table
from .format import format_p, format_r
from .r.splm import SplmFit, fit_splm

# https://pysal.org/spreg/index.html
# https://bashtage.github.io/linearmodels/
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
_NON_SLOPE = {
    *INTERCEPT_NAMES,
    "rho",
    "lambda",
    "phi",
    "sigma2_v",
    "sigma2_1",
}


###################
#  panel layout   #
###################


@dataclass
class PanelData:
    """
    Balanced panel in the *long layout* used by spreg: ``T`` consecutive blocks (periods) of
    ``N`` rows (entities), the entities in the order of the rows of the weights ``w``.

    Usually created by :func:`make_panel` (called by :func:`plm`, which stores the instance on
    every fitted model as ``plm_panel``).

    Attributes
    ----------
    y : pandas.Series
        Dependent variable (float), ``MultiIndex`` ``(entity, time)`` with level names
        ``"entity"`` and ``"time"``, sorted by period first and then by entity.
    x : pandas.DataFrame
        Predictors (float, without constant), same index.
    ent, tim : numpy.ndarray
        Entity / period position (0..N-1 / 0..T-1) of every row.
    w : libpysal.weights.W or None
        Weights aligned with the entities (``w.n == N``), or None.
    N, T : int
        Number of entities and of periods; ``len(y) == N * T``.
    entities, times : list
        Labels of the entities (in layout order) and of the periods (sorted).
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

        Parameters
        ----------
        keep : array-like of bool
            Mask with one element per row of the panel.

        Returns
        -------
        PanelData
            New panel (the positions ``ent`` / ``tim`` are renumbered from 0; ``w`` is
            restricted with the original weight values kept, and the original transformation
            re-applied).

        Raises
        ------
        ValueError
            If the kept rows do not form a balanced rectangle (entities x periods).
        """
        keep = np.asarray(keep, dtype=bool)
        ent_kept = np.unique(self.ent[keep])
        tim_kept = np.unique(self.tim[keep])
        if keep.sum() != len(ent_kept) * len(tim_kept):
            raise ValueError(
                "Subsetting must keep whole entities or whole periods (balanced panel)."
            )
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


def make_panel(
    data,
    y=None,
    x=None,
    entity=None,
    time=None,
    w=None,
    formula=None,
    dropna=True,
    balance="error",
    w_ids=None,
):
    """
    Convert a long DataFrame into a :class:`PanelData` in the layout needed by spreg.

    The rows are sorted by period first and then by entity; the entities are ordered like
    the rows of ``w`` (with ``w``) or sorted (without ``w``; in order of appearance if they
    cannot be sorted). Rows with a missing entity or period label are dropped.

    Parameters
    ----------
    data : pandas.DataFrame
        Long data. The index must be unique if ``formula`` is used.
    y : str
        Dependent variable (ignored if ``formula`` is given).
    x : list of str
        Predictors, without constant (ignored if ``formula`` is given).
    entity, time : str, optional
        Columns (or index levels) identifying entity and period. Pass both or none; if both
        are None, the index must be a two-level ``(entity, time)`` MultiIndex.
    w : libpysal.weights.W, optional
        Spatial weights. The entity values must be the ids ``w.id_order`` (or ``w_ids``);
        entities missing in the data are removed from ``w`` (with a ``UserWarning``).
    formula : str, optional
        patsy formula (``"y ~ x1 + x2"``); the intercept is dropped (spreg and
        linearmodels handle constants themselves). Rows with NAs are always dropped by patsy.
    dropna : bool, default True
        Drop rows with missing values in ``y``/``x`` (not used with ``formula``).
    balance : {"error", "drop"}, default "error"
        What to do with entities that are not observed in every period: raise a
        ``ValueError`` or drop these entities (with a ``UserWarning``).
    w_ids : sequence, optional
        Entity labels in the row order of ``w``. Default: ``w.id_order``.

    Returns
    -------
    PanelData
        The balanced panel (``N`` entities, ``T`` periods, ``N * T`` rows).

    Raises
    ------
    ValueError
        For an invalid ``balance`` / ``entity`` / ``time`` combination, a MultiIndex of the wrong
        shape, duplicated ``(entity, time)`` pairs, an unbalanced panel with
        ``balance="error"``, entities that are not ids of ``w``, or missing ``y``/``x``.
    KeyError
        If ``entity`` / ``time`` is neither a column nor an index level.
    """
    if balance not in ("error", "drop"):
        raise ValueError("balance must be 'error' or 'drop'.")
    if (entity is None) != (time is None):
        raise ValueError("Pass both `entity` and `time`, or none of them (MultiIndex).")
    if entity is None:
        if not isinstance(data.index, MultiIndex) or data.index.nlevels != 2:
            raise ValueError(
                "Without `entity` and `time` the data need a two-level (entity, time) index."
            )
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
            raise ValueError(
                f"`w_ids` has {len(ids)} labels but w has {w.n} observations."
            )
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
        [
            Index(entities).take(ent_pos),
            Index(times).take(tim_pos),
        ],  # keeps the label dtypes
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
    return {k[len(prefix) :]: v for k, v in kwargs.items() if k.startswith(prefix)}


def _effects_kwargs(effects, backend, kwargs):
    """Translate ``effects`` ("entity", "time", "twoway") into the linearmodels arguments of the FE model."""
    if effects not in ("entity", "time", "twoway"):
        raise ValueError("effects must be 'entity', 'time' or 'twoway'.")
    if effects == "entity":
        return
    if backend in (
        "spreg",
        "splm",
    ):  # spreg: period dummies, see _time_dummies; splm: the argument of R
        kwargs["effects"] = effects
        return
    kwargs.setdefault("fe_model_entity_effects", effects == "twoway")
    kwargs.setdefault("fe_model_time_effects", True)


def _time_dummies(panel):
    """Period dummies (all but the first period) as a DataFrame in the row order of the panel."""
    names = [f"time[{t}]" for t in panel.times[1:]]
    values = (panel.tim[:, None] == np.arange(1, panel.T)[None, :]).astype(float)
    return DataFrame(values, index=panel.x.index, columns=names)


def _fit_spreg(name, panel, kwargs):
    cls_name, kind, spatial = SPREG_MODELS[name]
    effects = kwargs.get("effects", "entity") if kind == "fe" else "entity"
    x = panel.x
    if effects != "entity":
        # Period effects of spreg's FE models: period dummies as extra regressors (the entity effects
        # are removed by spreg), which is the model with entity + period effects for any weights.
        # Without entity effects the pooled model with the dummies is the period-effects model.
        if kwargs.get("slx_lags", 0):
            raise NotImplementedError("slx_lags is not available with period effects.")
        if effects == "time":
            if name == "slm_fe":
                raise NotImplementedError(
                    "effects='time' is not available for slm_fe (spreg has no pooled spatial lag model); "
                    "use effects='twoway'."
                )
            cls_name = {"fe": "PooledOLS", "sem_fe": "ML_ErrorPooled"}[name]
        x = concat([x, _time_dummies(panel)], axis=1)
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
        r = cls(panel.y.to_frame(), x, w, **model_kwargs)
    r.plm_model = name
    r.plm_effects = effects
    r.plm_time = list(x.columns[panel.x.shape[1] :])  # names of the period dummies
    return r


def _fit_linearmodels(name, panel, kwargs):
    # check the request first: the message about spatial models is useful whether or not
    # linearmodels is installed
    if name not in LINEARMODELS_MODELS:
        raise NotImplementedError(
            f"Model '{name}' is not available with backend='linearmodels' "
            f"(choose from {LINEARMODELS_MODELS}); spatial models need backend='spreg'."
        )
    if kwargs.get("slx_lags", 0):
        raise NotImplementedError("slx_lags needs backend='spreg'.")
    try:
        from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "backend='linearmodels' needs the optional dependency linearmodels: "
            'pip install "apatools[linearmodels]"'
        ) from e
    model_kwargs = _model_kwargs(kwargs, name)
    fit_kwargs = _model_kwargs(kwargs, name, "fit")
    x = panel.x
    if name == "fe":
        model_kwargs.setdefault("entity_effects", True)
        model = PanelOLS(panel.y, x, **model_kwargs)
    else:
        x = x.copy()
        x.insert(0, "const", 1.0)  # like spreg, add the constant ourselves
        model = (PooledOLS if name == "pooled" else RandomEffects)(
            panel.y, x, **model_kwargs
        )
    return model.fit(**fit_kwargs)


def _fit_one(name, panel, backend, kwargs):
    if backend == "spreg":
        if name not in SPREG_MODELS:
            raise NotImplementedError(
                f"'{name}' is not implemented, try {tuple(SPREG_MODELS)}."
            )
        return _fit_spreg(name, panel, kwargs)
    if backend == "linearmodels":
        return _fit_linearmodels(name, panel, kwargs)
    if backend == "splm":
        return _fit_splm(name, panel, kwargs)
    raise ValueError("backend must be 'spreg', 'linearmodels' or 'splm'.")


def _fit_splm(name, panel, kwargs):
    if kwargs.get("slx_lags", 0):
        raise NotImplementedError("slx_lags needs backend='spreg'.")
    return fit_splm(
        name,
        panel,
        effects=kwargs.get("effects", "entity"),
        args=_model_kwargs(kwargs, name),
        rscript=kwargs.get("rscript"),
        timeout=kwargs.get("r_timeout"),
    )


def _is_spreg(r):
    return type(r).__module__.startswith("spreg")


def _is_splm(r):
    return isinstance(r, SplmFit)


def _backend_of(r):
    """``"spreg"``, ``"splm"`` or ``"linearmodels"``: the backend that fitted the result."""
    return "spreg" if _is_spreg(r) else "splm" if _is_splm(r) else "linearmodels"


def _model_name(r):
    """Model name (key of :data:`SPREG_MODELS`) of a fitted result of any backend."""
    if _is_splm(r):
        return r.model
    if _is_spreg(r):
        if getattr(
            r, "plm_model", None
        ):  # period-effects models are fitted with pooled classes
            return r.plm_model
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


def _rename_lag(r, names):
    """spreg calls the lag coefficient ``W_<name of y>``; call it ``rho``, like the other backends."""
    if not type(r).__name__.startswith("ML_Lag"):
        return list(names)
    name_y = r.name_y[0] if isinstance(r.name_y, (list, tuple)) else r.name_y
    return ["rho" if n in ("W_dep_var", f"W_{name_y}") else n for n in names]


def _param_table(r):
    """Coefficient table (coef, se, z/t, p-value, cil, cir) of a fitted result of any backend."""
    if _is_splm(r):
        table = r.table.rename(columns={"stat": r.stat_name}).copy()
        critical = (
            scipy.stats.t.ppf(0.975, r.df_resid)
            if np.isfinite(r.df_resid)
            else scipy.stats.norm.ppf(0.975)
        )
        table["cil"] = table["coef"] - critical * table["se"]
        table["cir"] = table["coef"] + critical * table["se"]
        return table[["coef", "se", r.stat_name, "p-value", "cil", "cir"]]
    if _is_spreg(r):
        out = r.output.copy()
        out["var_names"] = _clean_names(out["var_names"])
        out["var_names"] = _rename_lag(r, out["var_names"])
        t = out.set_index("var_names").apply(to_numeric, errors="coerce")
        stat = (
            "t" if hasattr(r, "t_stat") else "z"
        )  # pooled OLS reports t, the others z
        t = t.rename(
            columns={
                "coefficients": "coef",
                "std_err": "se",
                "zt_stat": stat,
                "prob": "p-value",
            }
        )
        t.index.name = None
        zc = scipy.stats.norm.ppf(0.975)  # 95% CI, normal approximation
        t["cil"] = t["coef"] - zc * t["se"]
        t["cir"] = t["coef"] + zc * t["se"]
        t = t.drop(
            index=getattr(r, "plm_time", [])
        )  # the period effects are not reported
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
    if _is_splm(r):
        return r.cov
    if _is_spreg(r):
        vm = np.asarray(r.vm, dtype=float)
        names = _rename_lag(r, _clean_names(r.output["var_names"]))[: vm.shape[0]]
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

    Parameters
    ----------
    r : fitted model
        A result of :func:`plm` (spreg panel model or linearmodels ``PanelEffectsResults`` /
        ``PanelResults``).
    panel : PanelData, optional
        Not used (kept so that ``base_metrics(r, panel)`` has the same call as the other
        helpers).

    Returns
    -------
    dict
        See above. ``f_stat`` / ``f_pvalue`` / ``df_model`` / ``df_resid`` are missing for
        spatial and random-effects models (and for the linearmodels random-effects model);
        ``r_sq_adj`` exists for pooled OLS and non-spatial FE models only; ``aic`` / ``bic``
        exist only for spreg models that report them, ``llf`` if the backend provides a finite
        log-likelihood.

    Notes
    -----
    The F test of an FE model compares the within R² with ``df_model = k`` and
    ``df_resid = N·T - N - k``. With period effects (``effects="time"`` or ``"twoway"`` of :func:`plm`, both
    backends) R² is the within R² after removing all effects and ``df_resid = N·T - A - k`` with ``A`` the
    number of effects estimated away (``N + T - 1`` for two-way, ``T`` for time effects with the constant). The
    linearmodels F statistic follows the covariance type of the fit (robust version if the covariance is not
    ``"Unadjusted"``).
    """
    name = _model_name(r)
    table = _param_table(r)
    slopes = _slopes(table.index)
    k = len(slopes)
    outputs = {"model": name, "backend": _backend_of(r)}

    if _is_splm(
        r
    ):  # the fit statistics were computed with the R fit (see apatools.r.splm)
        resid = np.asarray(r.resid, dtype=float)
        outputs.update(r.metrics)
        outputs.update(
            n_obs=r.n_obs,
            n_entities=r.n_entities,
            n_periods=r.n_periods,
            n_params=k,
            mae=float(np.mean(np.abs(resid))),
            mad=float(np.median(np.abs(resid - np.median(resid)))),
        )
        return outputs

    if _is_spreg(r):
        _, effects, spatial = SPREG_MODELS[name]
        n_obs = int(r.n)  # N*T (spreg panel models report the stacked size)
        n_per = int(r.t)
        n_ent = n_obs // n_per
        resid = np.asarray(r.u, dtype=float).ravel()
        time_effects = getattr(r, "plm_effects", "entity") != "entity"
        if effects == "pooled" and spatial is None:
            r_sq, r_type = float(r.r2), "ols"
            outputs["r_sq_adj"] = float(r.ar2)
            F, p = r.f_stat
            outputs.update(
                f_stat=float(F),
                f_pvalue=float(p),
                df_model=len(table) - 1,
                df_resid=n_obs - len(table),
            )
        elif not (time_effects and spatial is None):
            r_sq = float(r.pr2)
            r_type = "within" if (effects == "fe" and spatial is None) else "pseudo"
        if effects == "fe" and spatial is None:
            resid, y_within = _within_resid(
                r
            )  # spreg's own u / pr2 leave out the fixed effects
            r_sq, r_type = 1 - np.sum(resid**2) / np.sum(y_within**2), "within"
            absorbed = _absorbed(r, n_ent, n_per)
            df_resid = n_obs - absorbed - k
            outputs["r_sq_adj"] = 1 - (1 - r_sq) * (n_obs - absorbed) / df_resid
            f_stat = (r_sq / k) / ((1 - r_sq) / df_resid)
            outputs.update(
                f_stat=f_stat,
                f_pvalue=float(scipy.stats.f.sf(f_stat, k, df_resid)),
                df_model=k,
                df_resid=df_resid,
            )
        for key, attr in (("aic", "aic"), ("bic", "schwarz"), ("llf", "logll")):
            v = _finite(getattr(r, attr, None))
            if v is not None:
                outputs[key] = v
    else:
        n_obs = int(r.nobs)
        n_ent, n_per = int(r.entity_info["total"]), int(r.time_info["total"])
        resid = np.asarray(r.resids, dtype=float).ravel()
        has_const = any(c in table.index for c in INTERCEPT_NAMES)
        if name == "fe":
            # R² after removing *all* effects (entity only: the within R²; with time effects: two-way within R²)
            r_sq, r_type = float(r.rsquared), "within"
            absorbed = (
                n_obs - int(r.df_resid) - k
            )  # entity (and time) effects estimated away
            outputs["r_sq_adj"] = 1 - (1 - r_sq) * (n_obs - absorbed) / r.df_resid
        elif name == "re":
            r_sq, r_type = float(r.rsquared_overall), "overall"
        else:
            r_sq, r_type = float(r.rsquared), "ols"
            outputs["r_sq_adj"] = 1 - (1 - r_sq) * (n_obs - int(has_const)) / r.df_resid
        if name != "re":
            f = r.f_statistic if r._cov_type == "Unadjusted" else r.f_statistic_robust
            outputs.update(
                f_stat=float(f.stat),
                f_pvalue=float(f.pval),
                df_model=int(f.df),
                df_resid=int(f.df_denom),
            )
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


def _absorbed(r, n_ent, n_per):
    """Number of estimated effects (entity, period, constant) of a non-spatial spreg FE-type result."""
    effects = getattr(r, "plm_effects", "entity")
    if effects == "entity":
        return n_ent
    if effects == "twoway":
        return n_ent + n_per - 1
    return n_per  # period dummies and the constant


def _within_resid(r):
    """
    Residuals and y without the effects of a non-spatial spreg fixed-effects result.

    spreg's ``PanelFE.u`` is ``y - x·β`` *without* the fixed effects, so the residuals of the
    within model are ``u`` minus its entity means. With period effects (period dummies among the
    regressors) y is also demeaned by period; without entity effects (pooled model with period
    dummies) ``u`` already is the residual.
    """
    n_per = int(r.t)
    n_ent = int(r.n) // n_per
    effects = getattr(r, "plm_effects", "entity")
    u = np.asarray(r.u, dtype=float).reshape(n_per, n_ent)
    y = np.asarray(r.y, dtype=float).reshape(n_per, n_ent)
    if effects == "time":
        return u.ravel(), (y - y.mean(axis=1, keepdims=True)).ravel()
    y = y - y.mean(axis=0)  # entity means
    if effects == "twoway":
        y = y - y.mean(axis=1, keepdims=True)  # period means of the entity-demeaned y
    return (u - u.mean(axis=0)).ravel(), y.ravel()


def _coefs(r):
    return _param_table(r)["coef"]


def _predict_rows(coefs, name, panel_full, wy_full, train_mask, test_mask):
    """Predictions for the rows ``test_mask`` from coefficients fitted on the other rows."""
    spatial = SPREG_MODELS[name][2] if name in SPREG_MODELS else None
    effects = (
        SPREG_MODELS[name][1]
        if name in SPREG_MODELS
        else {"fe": "fe", "re": "re"}.get(name, "pooled")
    )
    slopes = list(panel_full.x.columns)
    xb = panel_full.x.to_numpy() @ coefs[slopes].to_numpy()
    for c in INTERCEPT_NAMES:
        if c in coefs.index:
            xb = xb + float(coefs[c])
    if spatial == "lag":
        xb = xb + float(coefs["rho"]) * wy_full
    if effects == "fe":
        # entity effect = mean over the training periods of (y - rho*Wy - x*beta)
        resid = panel_full.y.to_numpy() - xb
        mu = np.bincount(
            panel_full.ent[train_mask],
            weights=resid[train_mask],
            minlength=panel_full.N,
        ) / np.bincount(panel_full.ent[train_mask], minlength=panel_full.N)
        xb = xb + mu[panel_full.ent]
    return xb[test_mask]


def pred_metrics(
    results, panel, backend="spreg", group="entity", exclude_neighbors=False, **kwargs
):
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
        The panel the models were fitted on (``results[i].plm_panel``).
    backend : {"spreg", "linearmodels"}, default "spreg"
        Estimation package used to refit the models.
    group : {"entity", "time"}, default "entity"
        Which groups are held out in turn.
    exclude_neighbors : bool, default False
        Also remove the neighbours (according to ``panel.w``) of the held-out entity from the
        training sample; needs ``group="entity"`` and weights (``ValueError`` otherwise).
    **kwargs
        The model arguments passed to :func:`plm` (``<model>_model_<argument>`` etc.).
        ``slx_lags > 0`` raises ``NotImplementedError``.

    Returns
    -------
    list of dict
        One dict per model with ``pred_loo_r_sq`` (clipped to [-1, 1]), ``pred_loo_mae`` and
        ``pred_loo_mad`` (median absolute deviation of the prediction errors); all ``nan`` for
        models that could not be refitted (FE models with ``group="entity"``, or every
        refit failed). Refits that fail are skipped and reported in a ``UserWarning``;
        another warning reports training weights with islands.

    Notes
    -----
    A cross-validation-style metric (leave-one-group-out). With ``e`` the prediction errors of
    all held-out rows, ``pred_loo_r_sq = 1 - Σ e² / Σ (y - ȳ)²`` (``ȳ``: mean of all ``y``,
    the sums run over the predicted rows), ``pred_loo_mae = mean(|e|)`` and
    ``pred_loo_mad = median(|e - median(e)|)``.

    References
    ----------
    Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal,
    spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913-929.
    """
    if group not in ("entity", "time"):
        raise ValueError("group must be 'entity' or 'time'.")
    if exclude_neighbors and (group != "entity" or panel.w is None):
        raise ValueError(
            "exclude_neighbors=True needs group='entity' and spatial weights `w`."
        )
    if kwargs.get("slx_lags", 0):
        raise NotImplementedError("pred_metrics is not implemented for slx_lags > 0.")
    if backend == "splm":
        warnings.warn(
            "pred_* metrics are not available with backend='splm' (every refit would start R); they are NaN.",
            UserWarning,
        )
        return [
            dict.fromkeys(("pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad"), np.nan)
            for _ in results
        ]
    names = [_model_name(r) for r in results]
    y = panel.y.to_numpy()
    wy = np.zeros_like(y)
    if (
        panel.w is not None
    ):  # (Wy) of every row, period by period, from the original weights
        wy = np.asarray(panel.w.sparse @ y.reshape(panel.T, panel.N).T).T.ravel()
    nan_out = {k: np.nan for k in ("pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad")}

    skip = {
        i
        for i, r in enumerate(results)
        if getattr(r, "plm_effects", "entity") != "entity"
    }
    if skip:
        warnings.warn(
            "pred_* metrics are not available for models with period effects (the effect of a held-out "
            "period cannot be estimated); they are NaN.",
            UserWarning,
        )
    if group == "entity":
        skip |= {
            i
            for i, n in enumerate(names)
            if n in SPREG_MODELS and SPREG_MODELS[n][1] == "fe"
        }
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
                drop |= {
                    int(panel.w.id2i[j]) for j in panel.w.neighbors[panel.w.id_order[g]]
                }
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
            except Exception as e:  # noqa: BLE001  (e.g. singular design, no convergence: skipped and reported)
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


def plm(
    data,
    y=None,
    x=None,
    w=None,
    entity=None,
    time=None,
    model="fe",
    formula=None,
    backend="spreg",
    **kwargs,
):
    """
    Fit panel regressions (pooled, fixed effects, random effects, optionally spatial).

    The data are converted with :func:`make_panel` (balanced panel in the layout of spreg)
    and every requested model is fitted on it. Spatial models: spatial lag with fixed or random
    effects (Elhorst, 2003, 2014), spatial error with pooled, fixed or random effects by
    maximum likelihood (Anselin, 1988) or the generalized-moments estimator of Kapoor,
    Kelejian and Prucha (2007).

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
    backend : {"spreg", "linearmodels", "splm"}
        Estimation package. ``"linearmodels"`` is an optional dependency and allows e.g.
        ``fe_fit_cov_type="clustered"`` or ``fe_fit_cluster_entity=True``, and time or two-way
        fixed effects (``effects``). ``"splm"`` uses R (packages plm, splm, spdep) through ``Rscript``; the
        results are :class:`apatools.r.splm.SplmFit` objects instead of spreg or linearmodels results.
    **kwargs
        Function options (below) and arguments for the backend named
        ``<model>_model_<argument>`` (passed to the model class, e.g.
        ``slm_fe_model_method="LU"``; with ``"splm"`` to ``plm::plm`` / ``splm::spml``, e.g.
        ``slm_fe_model_zero.policy=True``) or ``<model>_fit_<argument>`` (passed to
        ``.fit()``, linearmodels only), where ``<model>`` is one of the names above.

    Other Parameters
    ----------------
    effects : {"entity", "time", "twoway"}, default "entity"
        Effects of the fixed-effects models (``fe``, ``slm_fe``, ``sem_fe``): entity (the usual
        within model), time (period) or both (two-way fixed effects); the other models ignore
        it. Where each is available:

        ================  ====================  ====================  ====================
        model             spreg                 linearmodels          splm (R)
        ================  ====================  ====================  ====================
        ``fe``            entity, time, twoway  entity, time, twoway  entity, time, twoway
        ``slm_fe``        entity, twoway        not available         entity, time, twoway
        ``sem_fe``        entity, time, twoway  not available         entity, time, twoway
        all other models  ignored               ignored               ignored
        ================  ====================  ====================  ====================

        linearmodels: sets its arguments ``entity_effects`` and
        ``time_effects`` (``fe_model_entity_effects`` / ``fe_model_time_effects`` still work and
        take precedence). splm: sets the argument ``effect`` of R's ``plm`` / ``spml``
        (``fe_model_effect`` takes precedence). spreg has no such option, so the period effects are period dummies
        added to the regressors (``time[<period>]`` in ``r.summary``; hidden by
        :func:`plm_report`), valid for any spatial weights; ``"time"`` is then the pooled model
        with the dummies (``fe`` and ``sem_fe``; there is no pooled spatial lag model, so not
        ``slm_fe``), and ``slx_lags`` is not available. :func:`plm_tests` compares models
        with period effects by F / likelihood-ratio tests (no Hausman test), and they have no
        ``pred_*`` metrics. The standard errors
        of spreg's entity-effects models use ``N·T - N`` degrees of freedom (see
        :func:`base_metrics`), so with ``"twoway"`` they are slightly too small. The standard errors
        of ``rho`` / ``lambda`` come from spreg's analytic information matrix and understate the
        sampling spread somewhat in small panels (the within ML estimator is biased towards zero).
    rscript : str, optional
        ``backend="splm"``: path of ``Rscript`` (default: the environment variable ``APATOOLS_RSCRIPT``, the
        ``PATH`` and the usual install locations). ``r_timeout`` limits the time of every R run (seconds).
    verbose : bool, default True
        Print the panel dimensions and the model summaries.
    dropna : bool, default True
        Drop rows with missing values.
    balance : {"error", "drop"}, default "error"
        Entities not observed in all periods: raise an error or drop them.
    w_ids : sequence, optional
        Entity labels in the row order of ``w`` (default ``w.id_order``).
    slx_lags : int, default 0
        Number of spatial lags of the predictors (spreg models that support it; not with
        linearmodels).
    base_metrics : bool, default True
        Compute :func:`base_metrics`.
    pred_metrics : bool, default False
        Add leave-one-group-out prediction metrics, see :func:`pred_metrics`.
    pred_group : {"entity", "time"}, default "entity"
        Group that is held out.
    pred_exclude_neighbors : bool, default False
        Also hold out the neighbours of the held-out entity (needs ``w``).

    Returns
    -------
    results : list
        One fitted model per entry of ``model`` (spreg or linearmodels objects, or ``SplmFit`` for R). Each has the
        attributes ``plm_panel`` (the :class:`PanelData` used) and ``plm_spec`` (backend and
        fitting options), which :func:`mundlak` and :func:`plm_tests` need.
    metrics : list of dict
        See :func:`base_metrics` (plus ``pred_loo_*`` if requested). Empty if neither
        ``base_metrics`` nor ``pred_metrics`` is requested.

    Raises
    ------
    ValueError
        For an unknown ``backend`` or spatial model without ``w`` (and the errors of
        :func:`make_panel`).
    NotImplementedError
        For an unknown model name, a spatial model / ``slx_lags`` with
        ``backend="linearmodels"``, ``slx_lags`` or a GMM model with ``backend="splm"``.
    ImportError
        If ``backend="linearmodels"`` and linearmodels is not installed, or an R package is missing with
        ``backend="splm"``.
    RuntimeError
        If R is not found or fails (``backend="splm"``).

    References
    ----------
    Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer Academic.

    Elhorst, J. P. (2003). Specification and estimation of spatial panel data models.
    *International Regional Science Review, 26*(3), 244-268.

    Elhorst, J. P. (2014). *Spatial econometrics: From cross-sectional data to spatial panels*.
    Springer.

    Kapoor, M., Kelejian, H. H., & Prucha, I. R. (2007). Panel data models with spatially
    correlated error components. *Journal of Econometrics, 140*(1), 97-130.

    Baltagi, B. H. *Econometric analysis of panel data*. Wiley.

    Examples
    --------
    >>> results, metrics = plm(
    ...     df, "y", ["x1", "x2"], w=w, entity="county", time="year",
    ...     model=["pooled", "fe", "re", "slm_fe"],
    ...     verbose=True, balance="error", slx_lags=0,
    ...     base_metrics=True, pred_metrics=True, pred_group="time",
    ...     slm_fe_model_method="LU",
    ... )  # doctest: +SKIP
    """
    verbose = kwargs.get("verbose", True)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    pred_group = kwargs.pop("pred_group", "entity")
    pred_exclude_neighbors = kwargs.pop("pred_exclude_neighbors", False)
    _effects_kwargs(kwargs.pop("effects", "entity"), backend, kwargs)
    panel = make_panel(
        data,
        y,
        x,
        entity=entity,
        time=time,
        w=w,
        formula=formula,
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
            pm = pred_metrics(
                results,
                panel,
                backend=backend,
                group=pred_group,
                exclude_neighbors=pred_exclude_neighbors,
                **kwargs,
            )
            metrics = [{**m_, **p_} for m_, p_ in zip(metrics, pm)]
    return results, metrics


_R_SQ_SUFFIX = {"ols": "", "within": "within", "overall": "overall", "pseudo": "pseudo"}


def plm_report(
    results,
    metrics=None,
    format_pval=True,
    add_stars=True,
    decimal=3,
    add_ftest=True,
    add_aic=True,
    add_bic=True,
    add_llf=True,
    add_mae=True,
    add_mad=True,
    add_pred_loo_mae=True,
    add_pred_loo_mad=True,
    add_n_obs=True,
    add_panel_dims=True,
    intercept_name="Intercept",
):
    """
    Build one APA-style coefficient table per model from :func:`plm` output.

    Columns: coef, se, z (or t), p-value, cil, cir (95% CI), optional ``sig`` stars. The first
    row's ``model`` column holds a fit summary such as
    ``R²within = .36, R²adj = .34, F(2, 123) = 35.1, p < .001, MAE = ..., N = 150, n = 25, T = 6``;
    R²pred, MAEpred and MADpred appear if :func:`plm` was run with ``pred_metrics=True``.
    ``R²within`` is the R² of the within-transformed data (FE), ``R²overall`` the overall R² of a
    random-effects model (linearmodels), ``R²pseudo`` the squared correlation of observed and
    predicted y (spreg, spatial and RE models) and a plain ``R²`` is the pooled OLS R².
    The spatial parameters (``rho`` = ρ of the lag models, ``lambda`` = λ of the error models)
    and the variance parameters of spreg (e.g. ``phi``, ``sigma2_v``) are rows of the table.

    Parameters
    ----------
    results : list
        Fitted models (first value returned by :func:`plm`).
    metrics : list of dict, optional
        Metrics (second value returned by :func:`plm`), matched with ``results`` by position;
        without them the tables have no ``model`` column.
    format_pval : bool or callable, default True
        ``False`` keeps the numeric p-values. ``True`` replaces them by APA strings without the
        ``p =`` prefix (``".009"``, ``"<.001"``; :func:`apatools.format.format_p`). A callable
        is applied to every p-value and its return value is used. The stars are always
        computed from the numeric p-values.
    add_stars : bool or callable, default True
        ``False``: no ``sig`` column. ``True``: add a ``sig`` column with significance stars
        (``***`` p < .001, ``**`` p < .01, ``*`` p < .05, ``⁺`` p < .10, otherwise an empty
        string; :func:`apatools.format.get_stars`). A callable maps one p-value to the marker.
    decimal : int or None, default 3
        Round the columns ``coef``, ``se``, ``cil`` and ``cir`` to this many decimals (the
        statistic and the p-values are not rounded here). ``None`` or 0: no rounding.
    add_ftest, add_aic, add_bic, add_llf, add_mae, add_mad, add_pred_loo_mae, add_pred_loo_mad, add_n_obs : bool, default True
        Include the corresponding part (F test, AIC, BIC, ``LL``, ``MAE``, ``MAD``,
        ``MAEpred``, ``MADpred``, ``N``) in the fit summary string if the metric is available.
        R² (with its type, see above), R²adj and R²pred are always shown when available.
    add_panel_dims : bool, default True
        Append the number of entities and periods, ``n = ..., T = ...``.
    intercept_name : str, default "Intercept"
        Row label of the constant (``CONSTANT`` of spreg, ``const`` of linearmodels); other
        labels are unchanged.

    Returns
    -------
    list of pandas.DataFrame
        One table per model, indexed by the coefficient names, with the columns ``coef``,
        ``se``, ``z`` (spreg models other than pooled OLS) or ``t`` (pooled OLS and linearmodels),
        ``p-value``, ``cil`` / ``cir`` (limits of the 95% confidence interval, confidence level
        0.95: normal approximation ``coef ∓ 1.96 · se`` for spreg, the interval of
        linearmodels otherwise), ``sig`` (if ``add_stars``) and ``model`` (fit summary, first
        row only, if metrics are given). Fixed effects are not listed as coefficients.
    """
    if metrics is None:  # reporting without metrics is allowed
        metrics = []
    output = []
    for r, i in zip_longest(results, metrics, fillvalue={}):
        s = []
        if "r_sq" in i:
            s.append(
                f"R²{_R_SQ_SUFFIX.get(i.get('r_sq_type', 'ols'), '')} {format_r(i['r_sq'], use_letter=False)}"
            )
        if "r_sq_adj" in i:
            s.append(f"R²adj {format_r(i['r_sq_adj'], use_letter=False)}")
        if "pred_loo_r_sq" in i:
            s.append(f"R²pred {format_r(i['pred_loo_r_sq'], use_letter=False)}")
        if add_ftest and "f_stat" in i and "df_model" in i:
            s.append(
                f"F({i['df_model']}, {i['df_resid']}) = {i['f_stat']:.2f}, {format_p(i['f_pvalue'])}"
            )
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
        round_columns(params, decimal)
        format_significance(params, add_stars, format_pval)
        if len(i):
            params.loc[params.index[0], "model"] = s
        params.index = rename_intercept(params.index, intercept_name)
        output.append(params)
    return output


def plm_effects(
    results,
    w=None,
    inference="simulation",
    n_draws=1000,
    seed=13,
    conf_level=0.95,
    traces="auto",
    n_probes=50,
    format_pval=False,
    add_stars=False,
    decimal=None,
):
    """
    Direct, indirect and total effects of the regressors of the spatial lag panel models fitted with :func:`plm`.

    In a spatial lag model (``slm_fe``, ``slm_re``; every backend that fits them: spreg and splm; with spreg
    also the Durbin model of ``slx_lags``) the coefficient of a regressor is not its effect: a change in one entity
    also changes its neighbours through ``ρWy``, and their neighbours in turn. LeSage & Pace (2009) summarize the
    effect matrix by three averages, which Elhorst (2014) uses for panels as well (the weights ``W`` of the entities
    apply in every period): the **direct** effect (the change of an entity caused by a change of its own regressor,
    including the feedback through its neighbours), the **total** effect (the change of an entity caused by a unit
    change of the regressor in *all* entities) and the **indirect** effect, their difference (the spillover). The
    averages are computed from the traces of the powers of ``W``, without inverting ``I - ρW``; for the Durbin model
    the effect matrix is ``(I - ρW)⁻¹ (β_k I + θ_k W)``.

    Parameters
    ----------
    results : list
        Fitted models from :func:`plm`. Only the spatial lag models have these effects; the others are skipped
        with a warning (the coefficients of the spatial error models already are the direct and total effects).
    w : libpysal.weights.W, optional
        The weights of the entities; by default those stored with the results (the panel's).
    inference : {"simulation", "analytic"}, default "simulation"
        How the standard errors, z values, p-values and confidence intervals are obtained. ``"simulation"``:
        LeSage & Pace's simulation, ``n_draws`` draws of the coefficients and ``ρ`` from their estimated joint
        normal distribution; the standard error is the standard deviation of the effects of the draws and the
        confidence interval is made of their percentiles (so it can be asymmetric). ``"analytic"``: the delta method
        with the gradient of the trace series (the interval is estimate ± z · se).
    n_draws, seed, conf_level, traces, n_probes, format_pval, add_stars, decimal
        As in :func:`apatools.slm.slm_effects`.

    Returns
    -------
    pandas.DataFrame
        One row per model, regressor and effect: the columns ``model`` (the model name; a repeated one gets the
        suffix ``_2``, ...), ``variable``, ``effect`` (``"Direct"``, ``"Indirect"``, ``"Total"``), ``estimate``,
        ``se``, ``z``, ``p-value``, ``cil``, ``cir`` and optionally ``sig``.

    Notes
    -----
    The constant has no effect and is left out, as are the period effects of ``effects="time"`` / ``"twoway"``
    (they are not regressors). The standard errors use the covariance matrix of the coefficients and ``ρ`` of the
    backend: spreg's covers the coefficients and ``ρ`` jointly, that of splm treats ``ρ`` as uncorrelated with the
    coefficients, so the standard errors of the indirect and total effects differ somewhat between the backends.
    spreg's estimates of an entity-effects model are those of the within estimator.

    The delta method (``inference="analytic"``) treats the effects as linear in the estimates. That holds when
    ``ρ`` is estimated precisely; with few units and a large ``ρ`` the standard errors of the indirect and total
    effects can be far too small (in a 64-unit example with ρ = 0.6 the analytic standard error of an indirect effect
    was 0.44, against 1.48 by simulation with 100,000 draws and 1.43 in R's ``impacts``), so prefer the simulation
    then. The direct effect is much less sensitive. The simulated standard errors of the indirect and total effects
    are noisy in the same situation, because a few draws of ``ρ`` near its bound give huge effects: in that example
    the standard error of the indirect effect was 1.03 with 5000 draws and 1.48 with 100,000 (R's own simulation
    behaves the same way). Raise ``n_draws`` and check that the result is stable with another ``seed``.

    References
    ----------
    LeSage, J., & Pace, R. K. (2009). *Introduction to spatial econometrics*. CRC Press.
    https://doi.org/10.1201/9781420064254

    Elhorst, J. P. (2014). *Spatial econometrics: From cross-sectional data to spatial panels*. Springer.
    https://doi.org/10.1007/978-3-642-40340-8

    Examples
    --------
    >>> results, metrics = plm(df, "y", ["x1", "x2"], w=w, entity="county", time="year", model="slm_fe")  # doctest: +SKIP
    >>> plm_effects(results, inference="analytic", decimal=3)  # doctest: +SKIP
    """
    entries, seen = [], {}
    for r in results:
        name = _model_name(r)
        table = _param_table(r)
        if "rho" not in table.index:
            reason = (
                "the coefficients of a spatial error model already are the direct and total effects"
                if "lambda" in table.index
                else "it has no spatial lag parameter"
            )
            warnings.warn(
                f"Model '{name}' has no direct and indirect effects ({reason}); skipped.",
                UserWarning,
            )
            continue
        if w is None:
            w = getattr(getattr(r, "plm_panel", None), "w", None)
        cov = _cov_frame(r)
        names = [n for n in table.index if n in cov.index]
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}_{seen[name]}"
        entries.append(
            (
                label,
                names,
                table.loc[names, "coef"].to_numpy(),
                cov.loc[names, names].to_numpy(),
            )
        )
    if entries and w is None:
        raise ValueError(
            "The spatial weights `w` are needed (the results do not carry them)."
        )
    return effects_table(
        entries,
        w,
        inference,
        n_draws,
        seed,
        conf_level,
        traces,
        n_probes,
        format_pval,
        add_stars,
        decimal,
    )


#################
#     tests     #
#################


def _row(test, model, stat, df, p, label, df_denom=None):
    df_txt = f"{df}" if df_denom is None else f"{df}, {df_denom}"
    report = (
        f"{label}({df_txt}) = {stat:.2f}, {format_p(p)}"
        if df is not None
        else f"{label} = {stat:.2f}, {format_p(p)}"
    )
    return {
        "test": test,
        "model": model,
        "statistic": float(stat),
        "df": df,
        "df_denom": df_denom,
        "p-value": float(p),
        "report": report,
    }


def hausman(r_fe, r_re):
    """
    Hausman test of a fixed- against a random-effects model with the same spatial structure.

    Compares the common slope coefficients: ``H = d' (V_fe - V_re)⁻¹ d`` ~ χ²(k).
    Works for both backends. Returns ``(statistic, df, p-value)``; statistic and p-value are
    ``nan`` (with a warning) if ``V_fe - V_re`` is not positive definite. This is common when the
    effects are correlated with the predictors (the random-effects variance is then too large),
    i.e. exactly when the test should reject; use :func:`mundlak` in that case.

    Parameters
    ----------
    r_fe, r_re : fitted models
        Fixed- and random-effects results from :func:`plm` of the same backend and spatial
        structure (e.g. ``fe`` and ``re``, or ``slm_fe`` and ``slm_re``). The statistic uses the
        slope coefficients (not the constant, ``rho``, ``lambda`` or variance parameters)
        present in both.

    Returns
    -------
    statistic : float
        The Hausman statistic ``H``.
    df : int
        Degrees of freedom, the number of compared slope coefficients.
    p_value : float
        ``P(χ²(df) > H)``.

    Raises
    ------
    ValueError
        If there is no common slope coefficient.

    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica, 46*(6),
    1251-1271.
    """
    t_fe, t_re = _param_table(r_fe), _param_table(r_re)
    common = [n for n in _slopes(t_fe.index) if n in t_re.index]
    if not common:
        raise ValueError("No common slope coefficients.")
    d = (t_fe.loc[common, "coef"] - t_re.loc[common, "coef"]).to_numpy()
    v = (
        _cov_frame(r_fe).loc[common, common] - _cov_frame(r_re).loc[common, common]
    ).to_numpy()
    if np.any(np.linalg.eigvalsh((v + v.T) / 2) <= 0):
        warnings.warn(
            "V_fe - V_re is not positive definite; the Hausman statistic is undefined.",
            UserWarning,
        )
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
    statistic : float
        Wald statistic ``b' V⁻¹ b`` of the coefficients ``b`` (covariance ``V``) of the added
        ``<x>_mean`` regressors; only predictors that vary within entities get a mean.
    df : int
        Number of added regressors.
    p_value : float
        ``P(χ²(df) > statistic)``.

    Raises
    ------
    ValueError
        If the model does not come from :func:`plm` or no predictor varies over time.

    References
    ----------
    Mundlak, Y. (1978). On the pooling of time series and cross section data.
    *Econometrica, 46*(1), 69-85.
    """
    panel, spec = getattr(r_re, "plm_panel", None), getattr(r_re, "plm_spec", None)
    if panel is None or spec is None:
        raise ValueError(
            "The model must come from plm() (panel and fit options are stored on it)."
        )
    name = _model_name(r_re)
    x = panel.x
    means = x.groupby(panel.ent).transform("mean")
    within_sd = (x - means).std()
    varying = [
        c for c in x.columns if within_sd[c] > 1e-12
    ]  # time-invariant means are collinear
    if not varying:
        raise ValueError("No predictor varies over time.")
    aug = concat([x, means[varying].add_suffix("_mean")], axis=1)
    r = _fit_one(name, replace(panel, x=aug), spec["backend"], spec["kwargs"])
    names = [f"{c}_mean" for c in varying]
    b = _param_table(r).loc[names, "coef"].to_numpy()
    v = _cov_frame(r).loc[names, names].to_numpy()
    stat = float(b @ np.linalg.solve(v, b))
    return stat, len(names), float(scipy.stats.chi2.sf(stat, len(names)))


_BACKENDS = ("spreg", "linearmodels", "splm")


def _loglik(r):
    """Log-likelihood of a spatial ML result (spreg or splm)."""
    return float(r.metrics["llf"]) if _is_splm(r) else float(np.ravel(r.logll)[0])


def _pooled_arrays(r):
    """Design matrix (with the constant), y, residuals and coefficients of a pooled OLS result (spreg or splm)."""
    if _is_splm(r):
        columns = [
            c for c in r.table.index if c not in _NON_SLOPE or c in INTERCEPT_NAMES
        ]
        return (
            r.design,
            np.asarray(r.y, float),
            np.asarray(r.resid, float),
            r.table.loc[columns, "coef"].to_numpy(),
        )
    return (
        np.asarray(r.x, dtype=float),
        np.asarray(r.y, dtype=float).ravel(),
        np.asarray(r.u, dtype=float).ravel(),
        np.asarray(r.betas, dtype=float).ravel(),
    )


def _bsk_lm_tests(e, w):
    """
    Baltagi, Song & Koh (2003) LM tests on the residuals ``e`` of the pooled OLS of a balanced panel (stacked
    in period blocks): the marginal test of random effects (LM1, Honda's one-sided normal form), the marginal
    test of spatial error dependence (LM2, the one-sided form of Anselin's LM error test) and their joint test
    (LMJ). The statistics agree with ``spreg.diagnostics_panel.BSK_tests`` and with R's ``plm::plmtest`` /
    ``splm::slmtest``. Returns ``[(label, statistic, df or None, p-value)]``.
    """
    n_ent = w.n
    n_per = len(e) // n_ent
    n = len(e)
    sigma2 = e @ e / n
    per_entity = e.reshape(n_per, n_ent).sum(axis=0)
    lm1 = np.sqrt(n / (2 * (n_per - 1))) * (per_entity @ per_entity / (e @ e) - 1)
    trace = (w.sparse.T @ w.sparse + w.sparse @ w.sparse).diagonal().sum() * n_per
    big_w = sps.kron(sps.identity(n_per), w.sparse, format="csr")
    lm2 = (e @ (big_w @ e) / sigma2) / np.sqrt(trace)
    joint = (
        lm1**2 + lm2**2
    )  # the information matrix is block-diagonal because the trace of W is zero
    return [
        ("LM Marginal RE (LM1)", float(lm1), None, float(scipy.stats.norm.sf(lm1))),
        (
            "LM Marginal Spatial (LM2)",
            float(lm2),
            None,
            float(scipy.stats.norm.sf(lm2)),
        ),
        ("LM Joint (LMJ)", float(joint), 2, float(scipy.stats.chi2.sf(joint, 2))),
    ]


def _anselin_lm_tests(r_pooled, w):
    """
    Anselin's LM tests for a spatial lag / error in the pooled OLS of a balanced panel (stacked data, weights
    ``I_T ⊗ W``): the statistics of Anselin (1988) with the ML variance ``e'e / (N·T)`` and the trace term
    ``T·tr(W'W + W²)`` (Anselin, Le Gallo & Jayet, 2008). spreg's ``diagnostics_panel.panel_LMlag`` divides
    by ``tr(W'W + W²)²`` instead and its panel tests use ``e'e / (N·T - k)``; they differ from the
    textbook statistics (and from R's ``splm::slmtest``), so they are not used.

    Returns ``{label: (statistic, p-value)}`` for ``LM lag``, ``LM error`` and the robust versions.
    """
    x, y, e, betas = _pooled_arrays(r_pooled)
    n = len(y)
    big_w = sps.kron(sps.identity(n // w.n), w.sparse, format="csr")
    sigma2 = e @ e / n
    wy, we = big_w @ y, big_w @ e
    wxb = big_w @ (x @ betas)
    projected = wxb - x @ np.linalg.lstsq(x, wxb, rcond=None)[0]  # M (W X b)
    trace = (w.sparse.T @ w.sparse + w.sparse @ w.sparse).diagonal().sum() * (n // w.n)
    denominator = (
        projected @ projected / sigma2 + trace
    )  # information of the lag parameter
    error, lag = e @ we / sigma2, e @ wy / sigma2  # e'We / s2 and e'Wy / s2
    out = {
        "LM lag": lag**2 / denominator,
        "LM error": error**2 / trace,
        "Robust LM lag": (lag - error) ** 2 / (denominator - trace),
        "Robust LM error": (error - trace / denominator * lag) ** 2
        / (trace * (1 - trace / denominator)),
    }
    return {
        label: (float(stat), float(scipy.stats.chi2.sf(stat, 1)))
        for label, stat in out.items()
    }


def _effects_of(r):
    """Effects of a fitted result: ``"entity"``, ``"time"`` or ``"twoway"`` (``"other"``: none, linearmodels)."""
    if _is_splm(r):
        return r.effects
    if _is_spreg(r):
        return getattr(r, "plm_effects", "entity")
    model = r.model
    if type(model).__name__ != "PanelOLS":
        return "entity"  # pooled / random effects: not an effects variant
    flags = (bool(model.entity_effects), bool(model.time_effects))
    return {(True, False): "entity", (False, True): "time", (True, True): "twoway"}.get(
        flags, "other"
    )


def _ssr_df(r):
    """Residual sum of squares and residual degrees of freedom of a non-spatial pooled / FE result."""
    if _is_splm(r):
        return r.ssr, r.df_resid
    if not _is_spreg(r):
        return float(r.resid_ss), float(r.df_resid)
    n, t = int(r.n), int(r.t)
    k = len(_slopes(_param_table(r).index))
    if _model_name(r) == "pooled":
        return float(np.sum(np.square(r.u))), n - k - 1
    return float(np.sum(np.square(_within_resid(r)[0]))), n - _absorbed(
        r, n // t, t
    ) - k


def _label(r):
    effects = _effects_of(r)
    return _model_name(r) if effects == "entity" else f"{_model_name(r)} ({effects})"


def plm_tests(results, w=None, format_pval=False, add_stars=False, decimal=None):
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
    * Models with period effects (``effects="time"`` / ``"twoway"``; Hausman and Mundlak need random-effects
      models with period effects, which are not available): **F tests** of period effects
      (``twoway`` vs ``fe``, ``time`` vs ``pooled``), of entity effects given period effects (``twoway`` vs
      ``time``) and of both (``twoway`` vs ``pooled``), for the non-spatial models of the same backend that
      are among ``results``; **likelihood-ratio test** of period effects (``slm_fe`` / ``sem_fe`` with
      ``twoway`` vs without, ``T - 1`` degrees of freedom). The Baltagi-Song-Koh and Anselin LM tests
      are available for the pooled model only. The likelihood-ratio statistic is multiplied by the residual
      degrees of freedom over ``N·T`` (the within likelihood counts ``N·T`` observations; unscaled it rejects
      12-16 % of the time at the 5 % level in simulations with ``N`` = 36-100 and ``T`` = 6, scaled 6-7 %). Note: the F test of time effects against ``pooled`` is conservative when the data have
      strong entity effects, because they inflate the pooled residuals (in simulations with ``N`` = 36, ``T`` = 6:
      0 % rejections of true nulls at the 5 % level and 87 % power for time effects with an SD of half the
      error SD, against 7 % and 97 % for ``twoway`` vs ``fe``); with entity effects in the data prefer that
      entity-conditional test.
    * **Baltagi-Song-Koh LM tests** (spreg, from ``pooled``; need weights): marginal random
      effects (LM1), marginal spatial error (LM2) and their joint test (LMJ), and the
      Anselin LM tests for a spatial lag / error (and their robust versions).

    Parameters
    ----------
    results : list
        Fitted models from :func:`plm`.
    w : libpysal.weights.W, optional
        Weights for the LM tests; by default those stored with the results.
    format_pval : bool or callable, optional
        ``True`` turns the p-values into APA strings (``.633``, ``<.001``); a callable
        is used to format each p-value; ``False`` (default) keeps the numeric p-values.
    add_stars : bool or callable, optional
        ``True`` adds a ``sig`` column with significance stars; a callable maps a
        p-value to the marker; ``False`` (default) adds no column. Computed from the numeric p-values.
    decimal : int, optional
        Rounds the ``statistic`` column to this many decimals; ``None`` or 0 (default) keeps
        the full precision. The ``report`` strings are not affected.

    Returns
    -------
    pandas.DataFrame
        Columns ``test``, ``model``, ``statistic``, ``df``, ``df_denom``, ``p-value`` (strings
        unless ``format_pval=False``), optionally ``sig``, and an APA-style string ``report``.
        The table is empty (with these columns) if none of the tests applies to ``results``.
        Use ``format_pval=False, add_stars=False, decimal=None`` for the raw numbers.

    Notes
    -----
    The ``model`` column holds the compared models (``"fe vs re"``) or the model name. ``df``
    is None for the LM tests with one degree of freedom reported as ``LM``; ``df_denom`` is
    set only for the F test. The Baltagi-Song-Koh and Anselin LM tests need spreg and
    ``w``. The Baltagi-Song-Koh tests come from spreg (``spreg.diagnostics_panel``); the Anselin tests are
    computed here for the stacked pooled model (spreg's panel versions differ from the textbook statistics).
    Both agree with R (``plmtest``, ``splm::slmtest``).

    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica, 46*(6),
    1251-1271.

    Mundlak, Y. (1978). On the pooling of time series and cross section data.
    *Econometrica, 46*(1), 69-85.

    Baltagi, B. H., Song, S. H., & Koh, W. (2003). Testing panel data regression models with
    spatial error correlation. *Journal of Econometrics, 117*(1), 123-150.

    Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications
    to model specification in econometrics. *Review of Economic Studies, 47*(1), 239-253.

    Anselin, L., Le Gallo, J., & Jayet, H. (2008). Spatial panel econometrics. In L. Mátyás &
    P. Sevestre (Eds.), *The econometrics of panel data* (pp. 625-660). Springer.
    """
    by_name, variants = {}, {}
    for r in results:
        variants.setdefault((_model_name(r), _backend_of(r), _effects_of(r)), r)
        if _effects_of(r) != "entity":
            continue  # models with period effects are not comparable with the RE / pooled models
        by_name.setdefault((_model_name(r), _backend_of(r)), r)
    panel = next(
        (getattr(r, "plm_panel", None) for r in results if hasattr(r, "plm_panel")),
        None,
    )
    rows = []

    for fe_name, re_name in (
        ("fe", "re"),
        ("slm_fe", "slm_re"),
        ("sem_fe", "sem_re"),
        ("sem_fe", "sem_re_gm"),
    ):
        for sp in _BACKENDS:
            if (fe_name, sp) in by_name and (re_name, sp) in by_name:
                h, df, p = hausman(by_name[(fe_name, sp)], by_name[(re_name, sp)])
                if np.isfinite(h):
                    rows.append(
                        _row(
                            "Hausman (FE vs RE)",
                            f"{fe_name} vs {re_name}",
                            h,
                            df,
                            p,
                            "χ²",
                        )
                    )

    for (name, sp), r in by_name.items():
        if (
            name in SPREG_MODELS
            and SPREG_MODELS[name][1] == "re"
            and hasattr(r, "plm_spec")
        ):
            try:
                stat, df, p = mundlak(r)
            except (
                ValueError,
                np.linalg.LinAlgError,
                KeyError,
            ):  # e.g. only time-invariant predictors
                continue
            rows.append(_row("Mundlak (correlated effects)", name, stat, df, p, "χ²"))

    for sp in _BACKENDS:
        if ("fe", sp) in by_name and ("pooled", sp) in by_name:
            r_fe, r_p = by_name[("fe", sp)], by_name[("pooled", sp)]
            if sp == "splm":
                (ssr_fe, df2), (ssr_p, df_p) = _ssr_df(r_fe), _ssr_df(r_p)
                df1 = int(round(df_p - df2))
                f = ((ssr_p - ssr_fe) / df1) / (ssr_fe / df2)
                p, df2 = float(scipy.stats.f.sf(f, df1, df2)), int(round(df2))
            elif sp == "spreg":
                n_obs = int(r_fe.n)
                n_ent = n_obs // int(r_fe.t)
                k = len(_slopes(_param_table(r_fe).index))
                df1, df2 = n_ent - 1, n_obs - n_ent - k
                ssr_fe, ssr_p = (
                    float(np.sum(np.square(_within_resid(r_fe)[0]))),
                    float(np.sum(np.square(r_p.u))),
                )
                f = ((ssr_p - ssr_fe) / df1) / (ssr_fe / df2)
                p = float(scipy.stats.f.sf(f, df1, df2))
            else:
                fp = r_fe.f_pooled
                f, p, df1, df2 = (
                    float(fp.stat),
                    float(fp.pval),
                    int(fp.df),
                    int(fp.df_denom),
                )
            rows.append(
                _row("F test of entity effects", "fe vs pooled", f, df1, p, "F", df2)
            )

    for sp in _BACKENDS:  # F tests of nested non-spatial models
        get = lambda name, eff: variants.get((name, sp, eff))  # noqa: E731
        pooled, fe, fe_time, fe_two = (
            get("pooled", "entity"),
            get("fe", "entity"),
            get("fe", "time"),
            get("fe", "twoway"),
        )
        for label, full, restricted in (
            ("F test of time effects", fe_two, fe),
            ("F test of time effects", fe_time, pooled),
            ("F test of entity effects (with time effects)", fe_two, fe_time),
            ("F test of entity and time effects", fe_two, pooled),
        ):
            if full is None or restricted is None:
                continue
            ssr_u, df_u = _ssr_df(full)
            ssr_r, df_r = _ssr_df(restricted)
            df1 = int(round(df_r - df_u))
            if df1 <= 0:
                continue
            f = ((ssr_r - ssr_u) / df1) / (ssr_u / df_u)
            rows.append(
                _row(
                    label,
                    f"{_label(full)} vs {_label(restricted)}",
                    f,
                    df1,
                    float(scipy.stats.f.sf(f, df1, df_u)),
                    "F",
                    int(round(df_u)),
                )
            )
    for name, sp in (
        (n, b) for n in ("slm_fe", "sem_fe") for b in ("spreg", "splm")
    ):  # LR test of period effects
        full, restricted = (
            variants.get((name, sp, "twoway")),
            variants.get((name, sp, "entity")),
        )
        if full is None or restricted is None:
            continue
        n_obs, n_per = (
            (full.n_obs, full.n_periods) if sp == "splm" else (int(full.n), int(full.t))
        )
        k = (
            len(_slopes(_param_table(full).index)) + 1
        )  # slopes and the spatial parameter
        # The within likelihood counts N·T observations although the effects use up N + T - 1 of
        # them: the plain LR statistic is too large (12-16 % rejections at the 5 % level in simulations).
        # It is multiplied by the residual degrees of freedom over N·T (about 6-7 % rejections).
        scale = (n_obs - n_obs // n_per - (n_per - 1) - k) / n_obs
        lr = scale * 2 * (_loglik(full) - _loglik(restricted))
        df = n_per - 1
        rows.append(
            _row(
                "LR test of time effects",
                f"{_label(full)} vs {_label(restricted)}",
                lr,
                df,
                float(scipy.stats.chi2.sf(lr, df)),
                "χ²",
            )
        )

    r_p = by_name.get(("pooled", "spreg")) or by_name.get(("pooled", "splm"))
    if r_p is not None:
        w = w if w is not None else (panel.w if panel is not None else None)
        if w is not None:
            _, _, residuals, _ = _pooled_arrays(r_p)
            for label, stat, df, p in _bsk_lm_tests(residuals, w):
                rows.append(_row(label, "pooled", stat, df, p, "χ²" if df else "LM"))
            anselin = _anselin_lm_tests(r_p, w)
            for label in ("LM lag", "LM error", "Robust LM lag", "Robust LM error"):
                stat, p = anselin[label]
                rows.append(_row(label, "pooled", stat, 1, p, "χ²"))

    res = DataFrame(
        rows,
        columns=["test", "model", "statistic", "df", "df_denom", "p-value", "report"],
    )
    # an empty table has object columns: make them numeric so that rounding and stars also work then
    res["statistic"] = res["statistic"].astype(float)
    res["p-value"] = res["p-value"].astype(float)
    if decimal:
        res["statistic"] = res["statistic"].round(decimal)
    format_significance(
        res, add_stars, format_pval
    )  # stars come from the numeric p-values

    return res
