import contextlib
import io
import warnings
from itertools import zip_longest

import numpy as np
import scipy

from ._deps import missing_dependency

try:
    import spreg as spr
    from pandas import to_numeric
    from statsmodels.api import add_constant
    from statsmodels.formula.formulatools import handle_formula_data
except ImportError as e:  # pragma: no cover
    missing_dependency("slm", "slm", e)

from ._common import (
    df_standardize,
    format_significance,
    rename_intercept,
    round_columns,
    w_subset_rows,
)
from .citation import Citation
from .format import format_p, format_r

# https://pysal.org/spreg/index.html
CITATION = Citation(
    APA="Rey, S. J., & Anselin, L. (2007). \
PySAL: A python library of spatial analytical methods. \
The Review of Regional Studies, 37(1), 5-27."
)


def fit_model(model, Y, X, w, **kwargs):  # model.fit() generator function
    """
    Generator that fits one or several spreg models and yields the fitted results.

    Parameters
    ----------
    model : str, spreg model class, or list/tuple of them
        ``"ols"`` (``spreg.OLS``), ``"slm"`` (spatial lag, maximum likelihood, ``spreg.ML_Lag``)
        or ``"sem"`` (spatial error, maximum likelihood, ``spreg.ML_Error``), the classes
        themselves, or a list/tuple (one result per entry, in order). Other strings raise
        ``NotImplementedError``. An unknown class is called as ``model(Y, X, w, **kwargs)``
        with the ``model_``-prefixed keywords.
    Y : array-like
        Dependent variable (``n`` or ``n x 1``).
    X : array-like or pandas.DataFrame
        Predictors without constant; spreg adds the constant itself.
    w : libpysal.weights.W or None
        Spatial weights aligned with the rows (``None`` only for ``"ols"`` without spatial
        diagnostics).
    **kwargs
        ``verbose`` (bool, default False) prints the model name and the arguments passed on.
        Arguments for spreg are named ``<model>_model_<argument>`` with ``<model>`` in
        ``ols``/``slm``/``sem`` (the first prefix is dropped, then ``model_``), e.g.
        ``ols_model_spat_diag=True`` or ``slm_model_method="LU"``. Other keywords are ignored.

    Yields
    ------
    spreg model object
        ``model(Y, X, w, **model_kwargs)`` (spreg models are fitted on construction).
    """
    if isinstance(model, (list, tuple)):
        for m in model:
            yield next(fit_model(m, Y, X, w, **kwargs))
    elif isinstance(model, str):
        if model == "ols":
            yield next(fit_model(spr.OLS, Y, X, w, **kwargs))
        elif model == "slm":
            yield next(fit_model(spr.ML_Lag, Y, X, w, **kwargs))
        elif model == "sem":
            yield next(fit_model(spr.ML_Error, Y, X, w, **kwargs))
        else:
            raise NotImplementedError(
                f"'{model}' is not implemented, try 'ols', 'slm' or 'sem'."
            )
    else:
        verbose = kwargs.get("verbose", False)
        # `model` is a class here (not an instance), so compare with `is`
        if model is spr.OLS:
            if verbose:
                print("model: OLS")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ols_")}
        elif model is spr.ML_Lag:
            if verbose:
                print("model: SLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("slm_")}
        elif model is spr.ML_Error:
            if verbose:
                print("model: SEM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("sem_")}
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:]: v for k, v in _kwargs.items() if k.startswith("model_")}

        if verbose:
            print("model_kwargs:", model_kwargs)

        yield model(Y, X, w, **model_kwargs)


def base_metrics(results):
    """
    Collect fit metrics (R²/pseudo R², adjusted R², F-test, AIC/BIC/LL, MAE, MAD) from a
    spreg results object.

    * ``spreg.OLS``: ``r2``, ``ar2`` and the F test (``f_stat``) of spreg.
    * ``spreg.ML_Lag`` / ``spreg.ML_Error``: ``r_sq`` is spreg's pseudo R² (``pr2``, the
      squared correlation between observed ``y`` and the predicted ``predy``);
      ``r_sq_adj = 1 - (1 - R²)(n - 1) / df_resid`` and the F statistic
      ``(R² / df_model) / ((1 - R²) / df_resid)`` with its F-distribution p-value are computed
      here (a rough, descriptive analogue; there is no exact F test for ML models).
      The parameter count ``k`` of spreg includes ``rho`` for the lag model but not ``lambda``
      of the error model, which is added here.

    Parameters
    ----------
    results : spreg model object
        Fitted ``OLS``, ``ML_Lag`` or ``ML_Error``.

    Returns
    -------
    dict
        ``r_sq``, ``r_sq_adj``, ``df_model`` (number of parameters minus the constant, so
        including ``rho``/``lambda``), ``df_resid`` (``n - k``), ``f_stat``, ``f_pvalue``,
        ``n_obs``, ``n_params`` (including the constant and ``rho``/``lambda``), ``mae`` (mean
        absolute residual ``y - predy``), ``mad`` (median absolute deviation of the residuals,
        ``median(|e - median(e)|)``, not rescaled), ``aic``, ``bic`` (spreg's Schwarz criterion)
        and ``llf`` (log-likelihood; spreg's ``logll``).

    References
    ----------
    Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer Academic.
    """

    outputs = {}
    resid = results.y.flatten() - results.predy.flatten()
    # spreg's `k` counts the constant and the predictors; it also counts rho for
    # ML_Lag but does NOT count lambda for ML_Error (added below).
    n_params = int(results.k)
    n_obs = int(results.n)

    if isinstance(results, spr.ML_Error):
        n_params += 1  # lambda counts
    df_model = n_params - 1  # predictors (+ rho / lambda), without the constant
    df_resid = n_obs - n_params  # n - k, as in spreg's OLS F-test

    if isinstance(results, (spr.ML_Lag, spr.ML_Error)):
        r_sq = results.pr2
        r_sq_adj = 1 - (1 - r_sq) * (n_obs - 1) / df_resid
        f_stat = (r_sq / df_model) / ((1 - r_sq) / df_resid)
        f_pvalue = scipy.stats.f.sf(f_stat, df_model, df_resid)
    else:
        r_sq = results.r2
        r_sq_adj = results.ar2
        f_stat, f_pvalue = results.f_stat

    outputs.update(
        {
            "r_sq": r_sq,
            "r_sq_adj": r_sq_adj,
            "df_model": df_model,
            "df_resid": df_resid,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "n_obs": n_obs,
            "n_params": n_params,
            "mae": np.mean(np.abs(resid)),
            "mad": np.median(np.abs(resid - np.median(resid))),
            "aic": results.aic,
            "bic": results.schwarz,
            "llf": results.logll,
        }
    )
    return outputs


def pred_metrics(results, model, Y, X, w, exclude_neighbors=False, **kwargs):
    """
    Leave-one-out prediction metrics (R²pred, MAEpred, MADpred) for spreg models.

    Every observation ``i`` is held out in turn, the model is refitted on the
    remaining observations (with the spatial weights restricted to them, see
    :func:`w_subset_rows`) and ``y_i`` is predicted. The prediction depends on the model:

    * ``ols`` and ``sem`` (spatial error): ``x_i·β`` (the trend; the error process
      is not used for prediction, which is also what spreg's ``predy`` is);
    * ``slm`` (spatial lag): ``ρ·(Wy)_i + x_i·β``, i.e. the observed ``y`` values
      of the neighbours of ``i`` are used through the *original* weights row of ``i``.
      This predictor uses more information than the one for ols/sem, so the
      R²pred values of the models are not strictly comparable.

    .. warning::
       Spatial data are autocorrelated. In plain leave-one-out the neighbours of the
       held-out observation stay in the training sample, and their (correlated)
       values leak into the fit, so R²pred is *optimistic* with respect to predicting
       a new, unseen region. Use ``exclude_neighbors=True`` for a more honest
       estimate: the neighbours of ``i`` (according to ``w``) are removed from
       the training sample as well (``i`` is still the only observation that is
       predicted). Because the refit has fewer rows this variant is slightly
       pessimistic for small samples and may create islands in the training weights.

    Cost: one model fit per observation and model (``n`` fits). For ML lag/error
    models each fit computes the log-determinant of ``W``; pass e.g.
    ``slm_model_method="LU"`` / ``sem_model_method="LU"`` or ``"ord"`` for large ``n``.

    Parameters
    ----------
    results : list
        Fitted spreg models returned by the full-sample fit, one per entry of ``model``
        (used for the design matrix, so that a constant added by spreg is handled).
    model : str or list of str
        The models that were fitted (``"ols"``, ``"slm"``, ``"sem"``).
    Y, X : array-like
        Data used for the full-sample fit; ``w`` must be aligned with the rows.
    w : libpysal.weights.W or None
        Spatial weights. ``None`` is only possible for plain ``ols``.
    exclude_neighbors : bool
        Also remove the neighbours of the held-out observation from the training sample.
    **kwargs
        The same model arguments as passed to :func:`slm`. Diagnostics of OLS are
        switched off inside the loop for speed.

    Returns
    -------
    list of dict
        One dict per model with ``pred_loo_r_sq`` (clipped to [-1, 1], as in
        :func:`apatools.lm.pred_metrics`), ``pred_loo_mae`` and ``pred_loo_mad``
        (median absolute deviation of the prediction errors). ``nan`` if every
        refit failed. Failed refits are skipped with a warning.

    Notes
    -----
    This is a cross-validation-style metric. With ``e_i = y_i - ŷ_i`` the prediction
    errors of the successfully refitted observations,
    ``pred_loo_r_sq = 1 - Σ e_i² / Σ (y_i - ȳ)²`` (``ȳ`` is the mean of all observations,
    the sums run over the predicted observations), clipped to [-1, 1];
    ``pred_loo_mae = mean(|e_i|)``; ``pred_loo_mad = median(|e_i - median(e)|)``.
    A ``UserWarning`` is issued if refits failed or if the training weights of some refits
    contain islands (observations without neighbours). The ``w`` must have as many
    observations as the data (``ValueError`` otherwise).

    References
    ----------
    Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal,
    spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913-929.

    Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer Academic.
    """
    y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    x = np.asarray(X, dtype=float)
    n = y.shape[0]
    if w is None and exclude_neighbors:
        raise ValueError("exclude_neighbors=True requires spatial weights `w`.")
    if w is not None and w.n != n:
        raise ValueError(f"Weights have {w.n} observations but the data {n}.")

    _kwargs = kwargs.copy()
    _kwargs.update(
        verbose=False, ols_model_spat_diag=False, ols_model_nonspat_diag=False
    )
    designs = [r.x for r in results]  # n x k design (incl. constant added by spreg)
    wy = None if w is None else w.sparse @ y  # (Wy)_i from the original weights

    errors = {}
    fit_fails = []
    n_islands = 0
    all_pos = np.arange(n)
    for i in range(n):
        drop = {i}
        if exclude_neighbors:
            drop |= {int(w.id2i[j]) for j in w.neighbors[w.id_order[i]]}
        train = np.array([k for k in all_pos if k not in drop])
        w_train = None if w is None else w_subset_rows(w, train)
        if w_train is not None and len(w_train.islands):
            n_islands += 1
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")  # spreg is chatty on every refit
                fitted = list(fit_model(model, y[train], x[train], w_train, **_kwargs))
        except Exception as e:  # noqa: BLE001  (e.g. singular design, no convergence: skipped and reported)
            fit_fails.append(str(e))
            continue
        for idx, r in enumerate(fitted):
            d = designs[idx]
            beta = np.asarray(r.betas, dtype=float).ravel()[: d.shape[1]]
            pred = float(d[i] @ beta)
            if isinstance(r, spr.ML_Lag):
                pred += float(r.rho) * float(wy[i, 0])
            errors.setdefault(idx, []).append((i, y[i, 0] - pred))

    if fit_fails:
        warnings.warn(
            f"Some attempts to calculate LeaveOneOut metrics failed ({len(fit_fails)}): {set(fit_fails)}",
            UserWarning,
        )
    if n_islands:
        warnings.warn(
            f"In {n_islands} leave-one-out refits the training weights contained islands "
            "(observations without neighbours).",
            UserWarning,
        )

    ybar = y.mean()
    outputs = []
    for idx in range(len(results)):
        if idx not in errors:
            outputs.append(
                {k: np.nan for k in ("pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad")}
            )
            continue
        pos, err = zip(*errors[idx])
        err = np.asarray(err)
        sst = np.sum((y[list(pos), 0] - ybar) ** 2)
        outputs.append(
            {
                "pred_loo_r_sq": float(np.clip(1 - np.sum(err**2) / sst, -1.0, 1.0)),
                "pred_loo_mae": float(np.mean(np.abs(err))),
                "pred_loo_mad": float(np.median(np.abs(err - np.median(err)))),
            }
        )
    return outputs


def slm(data, y=None, x=None, w=None, model="ols", formula=None, **kwargs):
    """
    Fit OLS, spatial lag (SLM, ``ML_Lag``) and spatial error (SEM, ``ML_Error``) models with spreg.

    The spatial lag model is ``y = ρWy + Xβ + ε`` and the spatial error model is
    ``y = Xβ + u``, ``u = λWu + ε`` (Anselin, 1988), both estimated by maximum likelihood.
    Requires the optional dependencies ``spreg`` and ``libpysal`` (for ``w``).

    Parameters
    ----------
    data : pandas.DataFrame
        The data; its rows must be in the order of ``w.id_order``.
    y : str
        Dependent variable; ignored if ``formula`` is given.
    x : list of str
        Predictors; ignored if ``formula`` is given.
    w : libpysal.weights.W
        Spatial weights (required for SLM/SEM and for OLS spatial diagnostics).
    model : str or list of str
        ``"ols"``, ``"slm"``, ``"sem"`` or a list of them.
    formula : str, optional
        patsy formula; overrides ``y``/``x``.
    **kwargs
        Function options (below) and spreg arguments named ``<model>_model_<argument>``
        with ``<model>`` in ols/slm/sem, e.g. ``ols_model_spat_diag=True``.

    Other Parameters
    ----------------
    verbose : bool, default True
        Print the sample size, the formula and the model summaries.
    dropna : bool, default True
        Drop rows with NAs (not with ``formula``: patsy always drops them), with a
        ``UserWarning``. ``w`` is restricted to the remaining rows automatically
        (``formula``: needs a unique ``data`` index).
    intercept : bool, default False
        Has no effect on the estimates: spreg always includes a constant (a constant column that
        is already present is not duplicated, so True and False give the same model). It only
        changes the printed formula. A model without intercept is not possible with spreg.
    standardize : False, True/``"z"`` or callable, default False
        z-transform (or apply a callable to) the numeric/boolean columns; other columns are
        dropped. A ``UserWarning`` is issued for z-transformation, because it sets the intercept
        estimate to zero.
    base_metrics : bool, default True
        Compute :func:`base_metrics` for every model.
    pred_metrics : bool, default False
        Add leave-one-out prediction metrics, which needs one refit per observation.
        Leave-one-out is optimistic for spatially autocorrelated data, see
        :func:`pred_metrics`.
    pred_exclude_neighbors : bool, default False
        With ``pred_metrics``: also remove the neighbours of the held-out observation
        from the training sample (less optimistic; needs ``w``).

    Returns
    -------
    results : list of spreg model objects
        One fitted ``spreg.OLS`` / ``spreg.ML_Lag`` / ``spreg.ML_Error`` per model.
    metrics : list of dict
        One dict per model (see :func:`base_metrics`): ``r_sq`` (pseudo R² for SLM/SEM),
        ``r_sq_adj``, ``f_stat``, ``f_pvalue``, ``df_model``, ``df_resid`` (n - k),
        ``n_params``, ``aic``, ``bic``, ``llf``, ``mae``, ``mad`` (median absolute deviation
        of residuals), ``n_obs``. With ``pred_metrics=True`` also ``pred_loo_r_sq``,
        ``pred_loo_mae``, ``pred_loo_mad`` (see :func:`pred_metrics` for how the predictions
        are made and for the caveat on spatial autocorrelation). Empty if neither
        ``base_metrics`` nor ``pred_metrics`` is requested. There is no VIF for spreg models.

    Raises
    ------
    ValueError
        If neither ``formula`` nor non-empty ``x`` together with ``y`` is given.

    References
    ----------
    Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer Academic.

    Examples
    --------
    >>> results, metrics = slm(
    ...     test_data, Y, X, w, model=["ols", "slm"],
    ...     verbose=True,
    ...     dropna=True,
    ...     intercept=False,  # keep False, spreg adds the constant by default
    ...     standardize="z",  # keeps np.number or bool columns only
    ...     base_metrics=True,
    ...     pred_metrics=True,  # leave-one-out R²pred, MAEpred, MADpred
    ...     pred_exclude_neighbors=False,
    ...     ols_model_spat_diag=True,
    ...     ols_model_nonspat_diag=True,
    ... )  # doctest: +SKIP
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    constant = kwargs.pop("intercept", False)
    standardize = kwargs.pop("standardize", False)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    pred_exclude_neighbors = kwargs.pop("pred_exclude_neighbors", False)

    if formula is None:
        if x is None or y is None or len(x) == 0:
            raise ValueError("Either formula or x,y have to be explicitly defined.")

        df = data[[y] + x]
        if dropna:
            keep = df.notna().all(axis=1).to_numpy()
            df = df[keep]
            if w is not None and not keep.all():
                # keep the weights aligned with the rows that survived NA-dropping
                w = w_subset_rows(w, np.flatnonzero(keep))
        if len(df) != len(data):
            warnings.warn(
                f"Rows with NAs were dropped. Ntotal={len(data)}",
                UserWarning,
            )

        if standardize:
            df = df_standardize(df, func=standardize)

        X, Y = df[x], df[y]

        if verbose:
            print(f"N={len(Y)}")
            print(f"Formula: {y} ~ {'1 + ' if constant else ''}" + " + ".join(x))

        if constant:
            X = add_constant(X)  # no effect: spreg adds the constant itself and does not duplicate it
    else:
        (Y, X), _, _ = handle_formula_data(data, X=None, formula=formula)
        if w is not None and len(Y) != len(data):
            # patsy dropped rows with NAs: restrict the weights to the remaining rows
            if not data.index.is_unique:
                raise ValueError(
                    "A unique data index is required to align the weights after NA-dropping."
                )
            w = w_subset_rows(w, data.index.get_indexer(Y.index))
            warnings.warn(
                f"Rows with NAs were dropped. Ntotal={len(data)}",
                UserWarning,
            )
        if verbose:
            print(f"N={len(Y)}")
            print(f"Specified formula: {formula}")

        if standardize:
            X = df_standardize(X, func=standardize)
            Y = df_standardize(Y, func=standardize)

    if standardize == "z" or (isinstance(standardize, bool) and standardize):
        warnings.warn(
            "Using z-transformation sets the intercept estimate to zero. This may lead to wrong results.",
            UserWarning,
        )

    results = []
    for r in fit_model(model, Y, X, w, **kwargs):
        results.append(r)
        if verbose:
            print(r.summary)

    metrics = []
    if add_base_metrics or add_pred_metrics:
        metrics = [base_metrics(r) for r in results]
        if add_pred_metrics:
            metrics = [
                {**m, **p}
                for m, p in zip(
                    metrics,
                    pred_metrics(
                        results,
                        model,
                        Y,
                        X,
                        w,
                        exclude_neighbors=pred_exclude_neighbors,
                        **kwargs,
                    ),
                )
            ]

    return results, metrics


def slm_report(
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
    add_pred_loo_mad=True,  # shown only if pred_metrics=True was used; no vif for spreg
    add_n_obs=True,
    add_n_groups=False,
    add_vif=False,
    intercept_name="Intercept",
):
    """
    Build one APA-style coefficient table per model from :func:`slm` output.

    Parameters
    ----------
    results : list
        Fitted spreg models (first value returned by :func:`slm`).
    metrics : list of dict, optional
        Metrics (second value returned by :func:`slm`), matched with ``results`` by position;
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
        ``MAEpred``, ``MADpred``, ``N``) in the fit summary string if the metric is available
        (the pred_* metrics only exist if :func:`slm` was run with ``pred_metrics=True``).
        R², R²adj and R²pred are always shown when available.
    add_n_groups : bool, default False
        Include ``G = ...`` if the metrics have ``n_groups`` (never the case for spreg
        models; kept for symmetry with :func:`apatools.lm.lm_report`).
    add_vif : bool, default False
        Join a ``vif`` column if the metrics contain one (spreg models do not).
    intercept_name : str, default "Intercept"
        Row label of the constant (spreg's ``CONSTANT``); other labels are unchanged.

    Returns
    -------
    list of pandas.DataFrame
        One table per model, indexed by the variable names of spreg (the spatial parameter
        ``W_<y>`` (rho) or ``lambda`` appear as rows), with the columns

        * ``coef``: estimate; ``se``: standard error;
        * ``z``: the test statistic (spreg's ``zt_stat``; it is a t statistic for OLS but
          is labelled ``z`` for all models);
        * ``p-value``: p-value (formatted, see ``format_pval``);
        * ``cil``, ``cir``: limits of the 95% confidence interval, ``coef ∓ z_0.975 · se``
          with the normal quantile 1.96 (confidence level 0.95; also for OLS);
        * ``sig``: significance stars (if ``add_stars``);
        * ``model``: filled in the first row only (if metrics are given): the fit summary, e.g.
          ``R² = .35, R²adj = .31, R²pred = .22, F(3, 45) = 8.06, p < .001, AIC = ..., N = 49``.
          R²pred, MAEpred and MADpred appear only if :func:`slm` was run with
          ``pred_metrics=True`` (leave-one-out, see :func:`pred_metrics`).
    """

    if metrics is None:  # reporting without metrics is allowed
        metrics = []
    output = []
    for r, i in zip_longest(results, metrics, fillvalue={}):
        s = []
        if "r_sq" in i:
            s.append(f"R² {format_r(i['r_sq'], use_letter=False)}")
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
        if add_n_groups and "n_groups" in i:
            s.append(f"G = {i['n_groups']}")

        s = ", ".join(s)
        params = r.output.set_index("var_names")
        params = params.apply(to_numeric, errors="coerce").rename(
            columns={
                "coefficients": "coef",
                "prob": "p-value",
                "std_err": "se",
                "zt_stat": "z",
            },
        )
        alpha_z = scipy.stats.norm.ppf(0.975)  # 95% CI
        params["cil"] = params["coef"] - alpha_z * params["se"]
        params["cir"] = params["coef"] + alpha_z * params["se"]
        round_columns(params, decimal)
        format_significance(params, add_stars, format_pval)
        if add_vif and "vif" in i and i["vif"] is not None:
            params = params.join(i["vif"])
        if len(i):
            params.loc[params.index[0], "model"] = s
        params.index = rename_intercept(params.index, intercept_name)
        output.append(params)

    return output
