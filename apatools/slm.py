import contextlib
import io
import warnings
from itertools import zip_longest

import numpy as np
import scipy
from scipy.stats import chi2, jarque_bera

from ._deps import missing_dependency

try:
    import spreg as spr
    from pandas import DataFrame, to_numeric
    from spreg import diagnostics as spr_diag
    from spreg import diagnostics_sp as spr_sp
    from statsmodels.api import add_constant
    from statsmodels.stats.diagnostic import het_breuschpagan
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
from .format import format_p, format_r


# https://pysal.org/spreg/index.html
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
        ``median(|e - median(e)|)``, not rescaled), ``aic``, ``bic`` (Schwarz criterion;
        like spreg's, but counting ``lambda`` of the error model as a parameter) and ``llf`` (log-likelihood; spreg's ``logll``).

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

    aic, bic = results.aic, results.schwarz
    if isinstance(results, spr.ML_Error):
        n_params += 1  # lambda counts
        # spreg's criteria of the error model leave lambda out of k (those of the lag model count rho):
        # count it, so that the criteria of the three models are comparable
        aic += 2.0
        bic += np.log(n_obs)
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
            "aic": aic,
            "bic": bic,
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
            X = add_constant(
                X
            )  # no effect: spreg adds the constant itself and does not duplicate it
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


def _breusch_pagan(reg, robust):
    """Breusch-Pagan (``robust=False``) or Koenker-Bassett (``robust=True``) test of an OLS fit of spreg."""
    stat, p, _, _ = het_breuschpagan(
        np.asarray(reg.u, dtype=float).ravel(),
        np.asarray(reg.x, dtype=float),
        robust=robust,
    )
    return {"bp": stat, "df": reg.x.shape[1] - 1, "pvalue": p}


def _moran_permutation(e, w, n_perm, seed):
    """Moran's I of ``e`` and its two-sided permutation p-value (``n_perm`` random reorderings of ``e``)."""
    z = np.asarray(e, dtype=float).ravel()
    z = z - z.mean()
    denominator = float(z @ z)
    scale = len(z) / float(w.s0)

    def moran(values):
        return scale * float(values @ (w.sparse @ values)) / denominator

    observed = moran(z)
    rng = np.random.default_rng(seed)
    extreme = sum(
        abs(moran(rng.permutation(z))) >= abs(observed) for _ in range(n_perm)
    )
    return observed, (extreme + 1) / (n_perm + 1)


def slm_tests(
    results,
    w=None,
    format_pval=False,
    add_stars=False,
    decimal=None,
    n_perm=999,
    seed=13,
):
    """
    Diagnostic tests of the models fitted with :func:`slm`, as a tidy DataFrame.

    They check the assumptions of the models and the need for a spatial term; they are not
    measures of the quality of the fit (see ``base_metrics`` / ``pred_metrics`` for those).
    Computed per model of ``results`` (a list as returned by :func:`slm`):

    * **OLS** (``"ols"``): Moran's I of the residuals (``z`` test; Cliff & Ord, 1972), the
      Lagrange multiplier tests for a missing spatial lag (``LM-lag``) and for spatial error
      dependence (``LM-error``), their robust versions (robust to the other alternative) and
      ``LM-SARMA`` for both together (Anselin et al., 1996; Burridge, 1980; Bera & Yoon, 1993),
      and the tests of normality (Jarque-Bera) and heteroskedasticity (Breusch-Pagan,
      Koenker-Bassett with the regressors of the model as in R's ``bptest``, White) of the residuals. The spatial tests need ``w``. Null hypotheses: no
      spatial autocorrelation of the residuals / no spatial lag / no spatial error dependence /
      neither; normally distributed residuals; homoskedastic residuals.
    * **Spatial lag** (``"slm"``) and **spatial error** (``"sem"``) models: the z test of the
      spatial parameter (``rho`` / ``lambda``, null: 0), the likelihood-ratio test of the same
      null against the OLS model (needs an OLS model fitted to the same data among ``results``;
      1 df), Jarque-Bera test of the residuals and Moran's I of the residuals with a permutation
      p-value (``n_perm`` reorderings; needs ``w``). The residuals are those of the model: for the
      lag model ``y - rho Wy - Xb``, for the error model the filtered ones, ``(I - lambda W)(y -
      Xb)``. A significant Moran's I means that spatial dependence is left in the model.

    A common way to use the OLS tests (Anselin & Rey, 2014): if LM-lag and LM-error are both
    insignificant, OLS is fine; otherwise fit the model of the more significant one, and if both
    are significant, decide by the robust versions. This function does not make that decision.

    Parameters
    ----------
    results : list of spreg models
        As returned by :func:`slm` (``spreg.OLS``, ``spreg.ML_Lag`` and ``spreg.ML_Error``
        objects). Other models are skipped with a warning.
    w : libpysal.weights.W, optional
        The spatial weights the models were fitted with (the results do not keep them). Without
        it only the tests that do not need weights are computed (with a warning).
    format_pval : bool or callable, default False
        Format the p-values as strings (True: APA style without the ``p``, ``"<.001"``;
        a callable receives one p-value); False keeps the numbers.
    add_stars : bool or callable, default False
        Add a ``sig`` column with significance stars (True: :func:`apatools.format.get_stars`;
        or a callable). They are computed from the unformatted p-values.
    decimal : int, optional
        Round ``statistic`` to this many decimals (None: no rounding).
    n_perm : int, default 999
        Number of permutations for the p-value of Moran's I of the residuals of the ML models
        (0 leaves that test out).
    seed : int, default 13
        Seed of the random generator for the permutations (a local generator).

    Returns
    -------
    pandas.DataFrame
        One row per test with the columns ``test``, ``model`` (``ols``, ``slm``, ``sem``; a
        repeated model gets the suffix ``_2``, ...), ``statistic`` (chi-square, ``z``, Moran's I
        for the permutation test), ``df`` (NaN where not applicable), ``p-value``, optionally
        ``sig``, and an APA-style string ``report``. Use ``format_pval=False, add_stars=False,
        decimal=None`` for the raw numbers.

    Notes
    -----
    With many observations nearly every test becomes significant, and among many tests some are
    by chance; judge the size of the departure (e.g. Moran's I, the spatial parameter), not only
    the p-value. The permutation test of Moran's I treats the residuals as exchangeable, an
    approximation for model residuals.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
    https://doi.org/10.1007/978-94-015-7799-1

    Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996). Simple diagnostic tests for spatial
    dependence. *Regional Science and Urban Economics, 26*(1), 77-104.
    https://doi.org/10.1016/0166-0462(95)02111-6

    Anselin, L., & Rey, S. J. (2014). *Modern Spatial Econometrics in Practice: A Guide to GeoDa,
    GeoDaSpace and PySAL*. GeoDa Press.

    Bera, A. K., & Yoon, M. J. (1993). Specification testing with locally misspecified
    alternatives. *Econometric Theory, 9*(4), 649-658. https://doi.org/10.1017/S0266466600008021

    Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random
    coefficient variation. *Econometrica, 47*(5), 1287-1294. https://doi.org/10.2307/1911963

    Burridge, P. (1980). On the Cliff-Ord test for spatial correlation. *Journal of the Royal
    Statistical Society: Series B, 42*(1), 107-108.
    https://doi.org/10.1111/j.2517-6161.1980.tb01108.x

    Cliff, A., & Ord, K. (1972). Testing for spatial autocorrelation among regression residuals.
    *Geographical Analysis, 4*(3), 267-284. https://doi.org/10.1111/j.1538-4632.1972.tb00475.x

    Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial
    independence of regression residuals. *Economics Letters, 6*(3), 255-259.
    https://doi.org/10.1016/0165-1765(80)90024-5

    Koenker, R., & Bassett, G. (1982). Robust tests for heteroscedasticity based on regression
    quantiles. *Econometrica, 50*(1), 43-61. https://doi.org/10.2307/1912528

    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct
    test for heteroskedasticity. *Econometrica, 48*(4), 817-838. https://doi.org/10.2307/1912934

    Examples
    --------
    >>> results, metrics = slm(df, "y", ["x1", "x2"], w, model=["ols", "slm"])  # doctest: +SKIP
    >>> slm_tests(results, w, format_pval=True, add_stars=True, decimal=3)  # doctest: +SKIP
    """
    rows = []

    def add(model, label, statistic, df, p, text):
        rows.append((label, model, statistic, df, p, f"{text}, {format_p(p)}"))

    def classical(
        r, model, label, compute, key
    ):  # a test of spreg.diagnostics: {key: statistic, df, pvalue}
        d = compute(r)
        add(
            model,
            label,
            float(d[key]),
            d["df"],
            float(d["pvalue"]),
            f"{label}({d['df']}) = {d[key]:.2f}",
        )

    def attempt(model, label, *args):
        try:
            classical(*args)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"{label} could not be computed for '{model}': {e}",
                UserWarning,
                stacklevel=3,
            )

    kinds = {spr.OLS: "ols", spr.ML_Lag: "slm", spr.ML_Error: "sem"}
    supported = [r for r in results if type(r) in kinds]
    for r in results:
        if type(r) not in kinds:
            warnings.warn(
                f"{type(r).__name__} is not supported by slm_tests and was skipped.",
                UserWarning,
            )
    ols_fits = [r for r in supported if type(r) is spr.OLS]
    if w is None and supported:
        warnings.warn(
            "Without w only the tests that do not need spatial weights are computed.",
            UserWarning,
        )

    seen = {}
    for r in supported:
        kind = kinds[type(r)]
        seen[kind] = seen.get(kind, 0) + 1
        model = kind if seen[kind] == 1 else f"{kind}_{seen[kind]}"
        if kind == "ols":
            if w is not None:
                mi = spr_sp.MoranRes(r, w, z=True)
                add(
                    model,
                    "Moran's I (residuals)",
                    float(mi.zI),
                    np.nan,
                    float(mi.p_norm),
                    f"Moran's I = {mi.I:.3f}, z = {mi.zI:.2f}",
                )
                lm = spr_sp.LMtests(r, w)
                for label, (stat, p), df in (
                    ("LM-lag", lm.lml, 1),
                    ("LM-error", lm.lme, 1),
                    ("Robust LM-lag", lm.rlml, 1),
                    ("Robust LM-error", lm.rlme, 1),
                    ("LM-SARMA", lm.sarma, 2),
                ):
                    add(
                        model,
                        label,
                        float(stat),
                        df,
                        float(p),
                        f"{label}({df}) = {stat:.2f}",
                    )
            for label, compute, key in (
                ("Jarque-Bera", spr_diag.jarque_bera, "jb"),
                # the regressors of the model explain the variance, as in R's bptest and statsmodels (spreg's
                # breusch_pagan / koenker_bassett use their squares by default)
                ("Breusch-Pagan", lambda reg: _breusch_pagan(reg, robust=False), "bp"),
                ("Koenker-Bassett", lambda reg: _breusch_pagan(reg, robust=True), "bp"),
                ("White", spr_diag.white, "wh"),
            ):
                attempt(model, label, r, model, label, compute, key)
            continue

        name, parameter = ("rho", r.rho) if kind == "slm" else ("lambda", r.lam)
        z, p = r.z_stat[-1]
        add(
            model,
            f"z test of {name}",
            float(z),
            np.nan,
            float(p),
            f"{name} = {float(parameter):.3f}, z = {z:.2f}",
        )
        base = next((o for o in ols_fits if o.n == r.n), None)
        if base is not None:
            lr = max(2 * (float(r.logll) - float(base.logll)), 0.0)
            add(
                model,
                f"LR test of {name}",
                lr,
                1,
                float(chi2.sf(lr, 1)),
                f"LR test of {name}(1) = {lr:.2f}",
            )
        residuals = np.asarray(r.u, dtype=float).ravel()
        if kind == "sem" and w is not None:
            residuals = residuals - float(r.lam) * (
                w.sparse @ residuals
            )  # filtered: (I - lambda W) u
        jb = jarque_bera(residuals)
        add(
            model,
            "Jarque-Bera",
            float(jb.statistic),
            2,
            float(jb.pvalue),
            f"Jarque-Bera(2) = {jb.statistic:.2f}",
        )
        if w is not None and n_perm:
            moran, p_perm = _moran_permutation(residuals, w, n_perm, seed)
            add(
                model,
                "Moran's I (residuals)",
                moran,
                np.nan,
                p_perm,
                f"Moran's I = {moran:.3f} ({n_perm} permutations)",
            )

    res = DataFrame(
        rows, columns=["test", "model", "statistic", "df", "p-value", "report"]
    )
    res["statistic"] = res["statistic"].astype(
        float
    )  # also for an empty table (object columns)
    res["df"] = res["df"].astype(float)
    res["p-value"] = res["p-value"].astype(float)
    if decimal:
        res["statistic"] = res["statistic"].round(decimal)
    format_significance(res, add_stars, format_pval)
    return res
