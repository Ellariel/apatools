import warnings
from io import StringIO
from itertools import zip_longest

import numpy as np
import scipy
from scipy.stats import shapiro

from ._deps import missing_dependency

try:
    import lxml  # noqa: F401  (pandas.read_html in lm_report needs it)
    import statsmodels.api as sm
    from pandas import DataFrame, Series, read_html, to_numeric
    from statsmodels.api import add_constant
    from statsmodels.formula.formulatools import handle_formula_data
    from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
    from statsmodels.stats import diagnostic as sm_diag
    from statsmodels.stats.outliers_influence import (
        OLSInfluence,
        variance_inflation_factor,
    )
    from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
except ImportError as e:  # pragma: no cover
    missing_dependency("lm", "lm", e)

from ._common import (
    INTERCEPT_NAMES,
    df_check_intercept,
    df_standardize,
    format_significance,
    rename_intercept,
    round_columns,
)
from .format import format_p, format_r


# https://www.statsmodels.org/stable/index.html
def vif(results, sort=False, decimal=2):
    """
    Variance inflation factors (VIF), a measure of multicollinearity.

    The VIF of a predictor is ``1 / (1 - R²_j)``, where ``R²_j`` is the R² of the regression
    of that column of the design matrix on all the other columns (see
    :func:`statsmodels.stats.outliers_influence.variance_inflation_factor`). As a rule of
    thumb, VIF > 5 (some authors use 10) indicates that the variable is highly collinear
    with the other input variables. The VIF is computed for every column of
    ``results.model.exog``, **including the constant**; the VIF of the constant is
    not interpretable as a collinearity diagnostic (and is large when the predictors are
    not centered), ignore it.

    Parameters
    ----------
    results : statsmodels results object
        Fitted model (OLS, RLM, GLM, QuantReg, MixedLM) with ``results.model.exog`` and
        ``results.model.exog_names``.
    sort : bool, default False
        If True, sort the rows by VIF in ascending order.
    decimal : int or None, default 2
        Round the VIFs to this number of decimals; ``None`` or 0 keeps the full precision.

    Returns
    -------
    pandas.DataFrame or None
        A single column ``vif``, indexed by the names of the design matrix columns
        (``results.model.exog_names``). If the calculation fails for any reason, a
        ``UserWarning`` is issued and ``None`` is returned.

    References
    ----------
    Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression diagnostics:
    Identifying influential data and sources of collinearity*. Wiley.
    """
    try:
        vif_df = DataFrame()
        vif_df["vif"] = [
            variance_inflation_factor(results.model.exog, i)
            for i in range(results.model.exog.shape[1])
        ]
        vif_df.index = results.model.exog_names
        vif_df = vif_df if not sort else vif_df.sort_values("vif")
        if decimal:
            vif_df = vif_df.round(decimal)
        return vif_df
    except Exception as e:  # noqa: BLE001  (any failure of statsmodels' VIF only becomes a warning)
        warnings.warn(
            f"VIF calculation failed ({e!s}).",
            UserWarning,
        )
        return None


def mlm_wrapper(Y, X, **kwargs):
    """
    Constructor of :class:`statsmodels.regression.mixed_linear_model.MixedLM` with ``groups`` as a keyword.

    ``MixedLM(endog, exog, groups, ...)`` takes the grouping variable as the third *positional*
    argument, whereas :func:`fit_model` calls every model class as ``model(Y, X, **model_kwargs)``.
    This wrapper makes ``groups`` an ordinary keyword argument (``mlm_model_groups`` in
    :func:`lm`).

    Parameters
    ----------
    Y : array-like or pandas.Series
        Dependent variable.
    X : array-like or pandas.DataFrame
        Design matrix of the fixed effects (include the constant yourself if wanted).
    **kwargs
        ``groups`` (required, array-like of group labels, one per row; ``KeyError`` if missing),
        and any other keyword argument of ``MixedLM`` (e.g. ``exog_re``, ``re_formula``,
        ``use_sqrt``).

    Returns
    -------
    statsmodels.regression.mixed_linear_model.MixedLM
        The unfitted model.
    """
    groups = kwargs.pop("groups")
    return sm.MixedLM(Y, X, groups, **kwargs)


def fit_model(model, Y, X, **kwargs):  # model.fit() generator function
    """
    Generator that fits one or several statsmodels models and yields the fitted results.

    Parameters
    ----------
    model : str, statsmodels model class, or list/tuple of them
        A string is one of ``"ols"`` (``sm.OLS``), ``"rlm"`` (``sm.RLM``), ``"glm"``
        (``sm.GLM``), ``"qlm"`` (``sm.QuantReg``) or ``"mlm"`` (``sm.MixedLM``, built with
        :func:`mlm_wrapper`); anything else raises ``NotImplementedError``. A class (e.g. a
        custom model with the signature ``model(Y, X, **kw).fit(**kw)``) is used as is. A list or
        tuple yields one result per entry, in order.
    Y : array-like or pandas.Series
        Dependent variable.
    X : array-like or pandas.DataFrame
        Design matrix (the constant is not added here).
    **kwargs
        ``verbose`` (bool, default False) prints the model name and the arguments passed on.
        Arguments for statsmodels are selected by prefix, ``<model>_<stage>_<argument>``:
        ``<model>`` is ``ols``, ``rlm``, ``glm``, ``qlm`` or ``mlm`` (the prefix is dropped
        first; for a custom class the keywords are used without a model prefix), ``<stage>`` is
        ``model`` (passed to the constructor, e.g. ``glm_model_family`` or ``mlm_model_groups``) or
        ``fit`` (passed to ``.fit()``, e.g. ``ols_fit_cov_type="HC1"``, ``qlm_fit_q=0.5``).
        Keywords that match none of the prefixes are ignored.

    Yields
    ------
    statsmodels results object
        The result of ``model(Y, X, **model_kwargs).fit(**fit_kwargs)``.

    Raises
    ------
    NotImplementedError
        For an unknown model name.
    """
    if isinstance(model, (list, tuple)):
        for m in model:
            yield next(fit_model(m, Y, X, **kwargs))
    elif isinstance(model, str):
        if model == "ols":
            yield next(fit_model(sm.OLS, Y, X, **kwargs))
        elif model == "rlm":
            yield next(fit_model(sm.RLM, Y, X, **kwargs))
        elif model == "glm":
            yield next(fit_model(sm.GLM, Y, X, **kwargs))
        elif model == "qlm":
            yield next(fit_model(sm.QuantReg, Y, X, **kwargs))
        elif model == "mlm":
            yield next(fit_model(sm.MixedLM, Y, X, **kwargs))
        else:
            raise NotImplementedError(
                f"'{model}' is not implemented, try 'ols', 'rlm', 'glm', 'qlm' or 'mlm'."
            )
    else:
        verbose = kwargs.get("verbose", False)
        if model == sm.OLS:
            if verbose:
                print("model: OLS")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ols_")}
        elif model == sm.RLM:
            if verbose:
                print("model: RLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("rlm_")}
        elif model == sm.GLM:
            if verbose:
                print("model: GLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("glm_")}
        elif model == sm.QuantReg:
            if verbose:
                print("model: QLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("qlm_")}
        elif model == sm.MixedLM:
            model = mlm_wrapper  # using wrapper because of one positional argument
            if verbose:
                print("model: MLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("mlm_")}
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:]: v for k, v in _kwargs.items() if k.startswith("model_")}
        fit_kwargs = {k[4:]: v for k, v in _kwargs.items() if k.startswith("fit_")}

        if verbose:
            print("model_kwargs:", model_kwargs)
            print("fit_kwargs:", fit_kwargs)
        yield model(Y, X, **model_kwargs).fit(**fit_kwargs)


def pred_metrics(model, Y, X, **kwargs):
    """
    Leave-one-out prediction metrics (R²pred, MAEpred, MADpred) of statsmodels models.

    A cross-validation-style metric: every observation ``i`` is held out in turn, the model
    is refitted on the remaining ``n - 1`` rows and ``y_i`` is predicted from ``x_i``
    (the prediction error is the PRESS residual; for OLS see Allen, 1974). It needs one refit
    per observation and model (``n`` fits). Refits that fail are skipped and reported in a
    single ``UserWarning``.

    The metrics are, with ``e_i = y_i - ŷ_(i)`` the leave-one-out errors:

    * ``pred_loo_r_sq`` = ``1 - Σ e_i² / (n · var(y))``, i.e. ``1 - PRESS / SST`` with the
      total sum of squares around the mean of all ``y``, clipped to [-1, 1]
      (also known as predicted R² or Q²);
    * ``pred_loo_mae`` = mean of ``|e_i|``;
    * ``pred_loo_mad`` = median of ``|e_i - median(e)|`` (median absolute deviation, not
      rescaled to the normal SD).

    .. warning::
       Plain leave-one-out treats rows as independent. For grouped (``mlm``), spatial or
       temporal data the rows that are held out stay correlated with the training sample, which
       makes the metrics optimistic; see Roberts et al. (2017) and, for spatial data,
       :func:`apatools.slm.pred_metrics`. For mixed models the held-out observation is
       predicted from the fixed effects only (``results.predict``) while its group stays in
       the training sample.

    Parameters
    ----------
    model : str, class, or list of them
        The model(s) as in :func:`fit_model`.
    Y : pandas.Series
        Dependent variable (positional indexing with ``.iloc`` is used).
    X : pandas.DataFrame
        Design matrix, including the constant if the model has one.
    **kwargs
        The model arguments as passed to :func:`lm`, see :func:`fit_model`. ``verbose`` is
        switched off. ``mlm_model_groups`` must be a ``pandas.Series`` aligned with ``X`` (it
        is subset together with the training rows).

    Returns
    -------
    list of dict
        One dict per model with the keys ``pred_loo_r_sq``, ``pred_loo_mae`` and
        ``pred_loo_mad``.

    References
    ----------
    Allen, D. M. (1974). The relationship between variable selection and data augmentation
    and a method for prediction. *Technometrics, 16*(1), 125-127.

    Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal,
    spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913-929.
    """
    errors = {}
    fit_fails = []
    _kwargs = kwargs.copy()
    _kwargs["verbose"] = False
    groups = kwargs.get("mlm_model_groups", None)  # mlm

    all_index = np.arange(len(X))
    for i in all_index:  # leave one out: train on all rows but i, test on row i
        train_index, test_index = np.delete(all_index, i), all_index[i : i + 1]
        if groups is not None:  # mlm
            x_train, x_test, _groups = (
                X.iloc[train_index],
                X.iloc[test_index],
                groups.iloc[train_index],
            )
            _kwargs["mlm_model_groups"] = _groups
        else:
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        try:
            for idx, r in enumerate(fit_model(model, y_train, x_train, **_kwargs)):
                errors.setdefault(idx, [])
                errors[idx].append(y_test.iloc[0] - r.predict(x_test).iloc[0])
        except Exception as e:  # noqa: BLE001  (a failed refit is skipped and reported in a warning)
            fit_fails.append(str(e))
    if len(fit_fails):
        warnings.warn(
            f"Some attempts to calculate LeaveOneOut metrics failed ({len(fit_fails)}): {set(fit_fails)}",
            UserWarning,
        )
    return [
        {
            "pred_loo_r_sq": np.clip(
                1
                - np.sum(np.square(err))
                / (np.var(np.asarray(Y, dtype=float)) * Y.size),
                -1.0,
                1.0,
            ),
            "pred_loo_mae": np.mean(np.abs(err)),
            "pred_loo_mad": np.median(np.abs(np.asarray(err) - np.median(err))),
        }
        for err in errors.values()
    ]


def _random_effects_variance(results):
    """Variance of the random effects: diagonal of ``cov_re`` plus the ``vc_formula`` components."""
    variance = 0.0
    if results.cov_re.shape[0] > 0:
        variance += np.sum(np.diag(results.cov_re))
    vcomp = getattr(results, "vcomp", None)
    if vcomp is not None:
        variance += float(np.sum(vcomp))
    return variance


def mlm_icc(results):
    """
    Intraclass correlation coefficient (ICC) of a fitted mixed model (approximate).

    ``ICC = τ² / (τ² + σ²)``, the share of the (non-fixed-effect) variance that lies between
    groups. ``τ²`` is the random-effects variance, the sum of the diagonal of the
    random-effects covariance matrix ``results.cov_re`` plus the variance components declared
    with ``vc_formula`` (``results.vcomp``, if any; the same ``τ²`` as in :func:`mlm_r_sq`), and
    ``σ²`` is the residual variance ``results.scale``. For a random-intercept model ``cov_re``
    is 1x1 and this is the usual ICC. For random-slope models the sum of the variances of
    intercepts and slopes is used, which is not a proper ICC (it also depends on the values of
    the predictors), so a ``UserWarning`` is issued whenever ``cov_re`` is not 1x1.

    Parameters
    ----------
    results : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted mixed model (``model="mlm"`` of :func:`lm`).

    Returns
    -------
    float
        ICC in [0, 1].

    References
    ----------
    Nakagawa, S., & Schielzeth, H. (2010). Repeatability for Gaussian and non-Gaussian data: A
    practical guide for biologists. *Biological Reviews, 85*(4), 935-956.
    """
    var_random = _random_effects_variance(results)
    var_resid = results.scale  # Residual variance
    if results.cov_re.shape != (1, 1):
        warnings.warn(
            "ICC is only well-defined for random-intercept models.",
            UserWarning,
        )
    return var_random / (var_random + var_resid)


def mlm_r_sq(results):
    """
    Marginal and conditional R² of a fitted linear mixed model (Nakagawa & Schielzeth).

    With ``σ²_f`` the variance of the fixed-effects linear predictor
    (``np.var(X @ fe_params)``), ``σ²_r`` the random-effects variance (sum of the diagonal of
    ``results.cov_re`` plus the variance components ``results.vcomp``, if any) and ``σ²_e`` the
    residual variance (``results.scale``)::

        R²_m = σ²_f / (σ²_f + σ²_r + σ²_e)
        R²_c = (σ²_f + σ²_r) / (σ²_f + σ²_r + σ²_e)

    ``R²_m`` is the variance explained by the fixed effects alone, ``R²_c`` the variance
    explained by fixed and random effects together. For random-slope models summing the
    diagonal of ``cov_re`` is only an approximation of the random-effects variance (the
    exact value depends on the predictors); Nakagawa & Schielzeth (2013) and Johnson (2014)
    give the full formulas.

    Parameters
    ----------
    results : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted mixed model.

    Returns
    -------
    r2_m, r2_c : float
        Marginal and conditional R².

    References
    ----------
    Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for obtaining R²
    from generalized linear mixed-effects models. *Methods in Ecology and Evolution, 4*(2),
    133-142.

    Johnson, P. C. D. (2014). Extension of Nakagawa & Schielzeth's R²GLMM to random slopes
    models. *Methods in Ecology and Evolution, 5*(9), 944-946.
    """
    # Fixed effects variance
    var_fixed = np.var(np.dot(results.model.exog, results.fe_params.values))
    var_random = _random_effects_variance(results)
    var_resid = results.scale  # Residual variance
    # Marginal R² (r2_m) → variance explained by fixed effects only
    r2_m = var_fixed / (var_fixed + var_random + var_resid)
    # Conditional R² (r2_c) → variance explained by fixed + random effects
    r2_c = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    return r2_m, r2_c


def base_metrics(results):
    """
    Calculating/retrieving R²/R²pseudo, R²adj and other fit metrics
    for OLS, RLM, GLM, QuantReg and MLM from statsmodels fitting results.

    Every value is taken from the results object when statsmodels provides a finite one
    and is computed here otherwise (so the function also works for results objects that lack
    some attributes). Non-finite values from statsmodels are replaced by the computed ones.

    * ``r_sq``: for mixed models the conditional R² (``r_sq_c``, see :func:`mlm_r_sq`); for
      other models ``pseudo_rsquared`` (QuantReg, GLM where available) or ``prsquared``, then
      ``rsquared`` (OLS, RLM), else ``1 - SSe / SSt`` with the (weighted) sums of squares
      of the residuals and of the observed values around their mean.
    * ``r_sq_adj``: ``rsquared_adj`` if available, else
      ``1 - (1 - R²)(n - c) / df_resid`` with ``c = 1`` if the model has a constant, else 0.
    * ``f_stat`` / ``f_pvalue``: ``fvalue`` / ``f_pvalue`` of the results, else
      ``(R² / df_model) / ((1 - R²) / df_resid)`` and its F-distribution p-value.
    * ``df_model`` and ``df_resid`` are corrected to be at least the number of predictors
      (without constant) and ``n - df_model``, respectively.

    Parameters
    ----------
    results : statsmodels results object
        Fitted OLS, RLM, GLM, QuantReg or MixedLM results.

    Returns
    -------
    dict
        Always present: ``r_sq``, ``r_sq_adj``, ``df_model``, ``df_resid``, ``f_stat``,
        ``f_pvalue``, ``n_obs``, ``n_params`` (predictors without the constant),
        ``mae`` (mean absolute residual) and ``mad`` (median absolute deviation of the
        residuals: ``median(|e - median(e)|)``, not rescaled). Present when finite/available:
        ``aic``, ``bic`` (``bic_llf`` if the results have it, else ``bic``), ``llf``.
        Mixed models add ``r_sq_c`` (= ``r_sq``), ``r_sq_m`` (marginal R²) and, if the model
        has that attribute, ``n_groups``.

    Warns
    -----
    UserWarning
        If the model has no predictors or zero model degrees of freedom.

    Notes
    -----
    For robust (RLM) and weighted models there is no single accepted R²; the value
    reported is the one returned by statsmodels (see Rousseeuw & Leroy, 1987, p. 42, and
    https://stats.stackexchange.com/questions/83826). The F statistic of a mixed model is
    only a rough heuristic computed from the conditional R².

    References
    ----------
    Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust regression and outlier detection*. Wiley.
    """

    outputs = {}
    n_obs = results.nobs
    params = set(results.params.index)
    n_params = getattr(
        results, "k_fe", len(params)
    )  # k_fe counts only fixed effects in MLM
    # if "const" in params or "Intercept" in params:
    if df_check_intercept(columns=params):
        n_params -= 1  # number of predictors, without intercept/const
    n_params = max(n_params, 0)
    df_model = max(getattr(results, "df_model", 0), n_params)  # technical correction
    if df_model == 0 or n_params == 0:
        warnings.warn(
            "A model has no parameters or zero degrees or freedom.",
            UserWarning,
        )

    df_resid = max(
        getattr(results, "df_resid", 0), n_obs - df_model
    )  # technical correction

    resid = getattr(results, "resid", getattr(results, "resid_working", None))
    fitted = results.fittedvalues
    observed = resid + fitted

    # There is a discussion on proper R2 metric for (weighted) robust regresson
    # see also Rousseeuw, P. J., & Leroy, A. M. (1987) p. 42
    # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/metrics/regression.py#L370
    # SSe = np.sum(weights * resid ** 2)
    # SSt = np.sum(weights * (observed - np.sum(weights * observed) / np.sum(weights)) ** 2)

    # https://stats.stackexchange.com/a/375752
    # https://web.maths.unsw.edu.au/~adelle/Garvan/Assays/GoodnessOfFit.html
    # SSe = np.sum(weights * resid ** 2)
    # SSt = np.sum(weights * (observed - np.mean(observed)) ** 2)

    # M = getattr(results.model, "M", False)  # r_sq_pseudo
    # if M and callable(M.rho):
    #    SSe = np.sum(M.rho(resid))
    #    SSt = np.sum(M.rho(observed - np.mean(observed)))
    #    outputs.update({"r_sq_pseudo": 1 - SSe / SSt})

    weights = getattr(results, "weights", 1)
    SSe = np.sum(weights * resid**2)
    SSt = np.sum(weights * (observed - np.mean(observed)) ** 2)

    r_sq_def = 1 - SSe / SSt
    if isinstance(results, MixedLMResultsWrapper):
        if hasattr(results.model, "n_groups"):
            outputs.update(
                {
                    "n_groups": int(results.model.n_groups),
                }
            )
        r_sq_m, r_sq_c = mlm_r_sq(results)
        outputs.update(
            {
                "r_sq_c": r_sq_c,
                "r_sq_m": r_sq_m,
            }
        )
        r_sq = r_sq_c
    else:
        r_sq = getattr(
            results,
            "pseudo_rsquared",
            getattr(results, "prsquared", getattr(results, "rsquared", r_sq_def)),
        )
    r_sq = r_sq() if callable(r_sq) else r_sq
    # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.rsquared_adj.html
    if not np.isfinite(r_sq):
        r_sq = r_sq_def

    # centered models: (n - 1) / df_resid; models without intercept: n / df_resid
    # has_const = "const" in params or "Intercept" in params
    has_const = bool(df_check_intercept(columns=params))

    r_sq_adj_def = 1 - (1 - r_sq) * (n_obs - int(has_const)) / df_resid
    r_sq_adj = getattr(results, "rsquared_adj", r_sq_adj_def)
    if not np.isfinite(r_sq_adj):
        r_sq_adj = r_sq_adj_def

    # https://www.slideshare.net/slideshow/multiple-regressionppt-252604177/252604177#8
    f_stat_def = (
        (r_sq / df_model) / ((1 - r_sq) / df_resid) if df_model and df_resid else np.nan
    )  # (SSt / df_model) / (SSe / df_resid)
    f_stat = getattr(results, "fvalue", f_stat_def)
    if not np.isfinite(f_stat):
        f_stat = f_stat_def

    f_pvalue_def = scipy.stats.f.sf(f_stat, df_model, df_resid)
    f_pvalue = getattr(results, "f_pvalue", f_pvalue_def)
    if not np.isfinite(f_pvalue):
        f_pvalue = f_pvalue_def
    # LL, AIC & BIC and others
    try:
        if hasattr(results, "aic") and np.isfinite(results.aic):
            outputs.update(
                {
                    "aic": results.aic,
                }
            )
        if hasattr(results, "bic_llf") and np.isfinite(results.bic_llf):
            outputs.update(
                {
                    "bic": results.bic_llf,
                }
            )
        elif hasattr(results, "bic") and np.isfinite(results.bic):
            outputs.update(
                {
                    "bic": results.bic,
                }
            )
    except NotImplementedError:
        pass
    try:
        if hasattr(results, "llf") and np.isfinite(results.llf):
            outputs.update(
                {
                    "llf": results.llf,
                }
            )
    except NotImplementedError:
        pass

    outputs.update(
        {
            "r_sq": r_sq,
            "r_sq_adj": r_sq_adj,
            "df_model": int(df_model),
            "df_resid": int(df_resid),
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "n_obs": int(n_obs),
            "n_params": int(n_params),
            "mae": np.mean(np.abs(resid)),
            "mad": np.median(np.abs(resid - np.median(resid))),
        }
    )
    return outputs


def lm(data, y=None, x=None, model="ols", formula=None, **kwargs):
    """
    Fit OLS, RLM, GLM, QuantReg or MixedLM models with statsmodels.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    y : str
        Dependent variable (column name). Not used if ``formula`` is given.
    x : list of str
        Predictor column names (without the constant). Not used if ``formula`` is given.
    model : str or list of str
        ``"ols"``, ``"rlm"`` (robust linear model), ``"glm"`` (generalized linear model),
        ``"qlm"`` (quantile regression) or ``"mlm"`` (linear mixed model), or a list of them.
        Every entry is fitted on the same data.
    formula : str, optional
        statsmodels/patsy formula; overrides ``y``/``x``, ``intercept`` and ``dropna`` (patsy
        always drops rows with NAs and adds an intercept unless ``- 1`` is used).
    **kwargs
        Options of this function (below) and arguments for statsmodels, named
        ``<model>_<stage>_<argument>``: ``<model>`` is ``ols``/``rlm``/``glm``/``qlm``/``mlm``;
        ``<stage>`` is ``model`` (passed to the constructor) or ``fit`` (passed to ``.fit()``),
        e.g. ``ols_fit_cov_type`` or ``glm_model_family``. ``mlm_model_groups`` may be a
        column name of ``data`` (required for ``"mlm"``).

    Other Parameters
    ----------------
    verbose : bool, default True
        Print the sample size, the formula and the statsmodels summary of every model.
    dropna : bool, default True
        Drop rows with NAs in ``y``/``x`` (a ``UserWarning`` reports it). Ignored with ``formula``.
    intercept : bool, default True
        Add a constant to ``x``. Ignored with ``formula``.
    standardize : False, True/``"z"`` or callable, default False
        z-transform (or apply the callable to) every numeric/boolean column; other columns
        are dropped. The constant is left untouched. Combined with an intercept, z-scored
        data give an intercept estimate of zero and a ``UserWarning`` is issued.
    base_metrics : bool, default True
        Compute :func:`base_metrics` for every model.
    pred_metrics : bool, default False
        Add leave-one-out metrics (:func:`pred_metrics`); one refit per observation and model.
    vif : bool, default False
        Add the :func:`vif` table of every model to the metrics (key ``"vif"``).

    Returns
    -------
    results : list
        One fitted statsmodels results object per model.
    metrics : list of dict
        One dict per model, see :func:`base_metrics`: ``r_sq``, ``r_sq_adj``, ``f_stat``,
        ``f_pvalue``, ``df_model``, ``df_resid``, ``aic``/``bic``/``llf`` (if available), ``mae``,
        ``mad`` (median absolute deviation of the residuals), ``n_obs``, ``n_params``;
        mixed models also ``r_sq_m``, ``r_sq_c`` and ``n_groups``. With ``pred_metrics=True``
        also ``pred_loo_r_sq``, ``pred_loo_mae``, ``pred_loo_mad``; with ``vif=True`` also
        ``vif`` (a DataFrame, or None if it failed). The list is empty if neither
        ``base_metrics`` nor ``pred_metrics`` nor ``vif`` is requested.

    Raises
    ------
    ValueError
        If neither ``formula`` nor non-empty ``x`` together with ``y`` is given.

    Notes
    -----
    Requires statsmodels (and lxml/pandas for :func:`lm_report`).

    Examples
    --------
    >>> results, metrics = lm(
    ...     test_data, Y, X, model=["ols", "rlm"],
    ...     verbose=True,
    ...     intercept=True,  # ignored when formula is defined
    ...     dropna=True,  # ignored when formula is defined
    ...     standardize="z",  # keeps np.number or bool columns only
    ...     base_metrics=True,
    ...     pred_metrics=False,
    ...     vif=False,
    ...     qlm_fit_q=0.5,
    ...     qlm_fit_cov_type="boot",
    ...     qlm_fit_cov_kwds={"n_boot": 100},
    ...     ols_fit_cov_type="HC1",
    ...     rlm_model_M=sm.robust.norms.RamsayE(),
    ... )  # doctest: +SKIP
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    constant = kwargs.pop("intercept", True)
    standardize = kwargs.pop("standardize", False)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    calc_vif = kwargs.pop("vif", False)

    groups = kwargs.get("mlm_model_groups", False)  # mlm
    if isinstance(groups, str):
        kwargs["mlm_model_groups"] = data[groups]

    if formula is None:
        if x is None or y is None or len(x) == 0:
            raise ValueError("Either formula or x,y have to be explicitly defined.")

        df = data[[y] + x]
        if dropna:
            df = df.dropna()
        if len(df) != len(data):
            warnings.warn(
                f"Rows with NAs were dropped. Ntotal={len(data)}",
                UserWarning,
            )

        if standardize:
            df = df_standardize(df, func=standardize)

        X, Y = df[x], df[y]
        if isinstance(kwargs.get("mlm_model_groups"), Series):
            # keep the grouping variable aligned with the rows that survived NA-dropping
            kwargs["mlm_model_groups"] = kwargs["mlm_model_groups"].loc[df.index]

        if verbose:
            print(f"N={len(Y)}")
            print(f"Formula: {y} ~ {'1 + ' if constant else ''}" + " + ".join(x))

        if constant:
            X = add_constant(X)
    else:
        (Y, X), _, _ = handle_formula_data(data, X=None, formula=formula)
        if isinstance(kwargs.get("mlm_model_groups"), Series):
            # patsy dropped the rows with NAs: keep the grouping variable aligned with the rest
            kwargs["mlm_model_groups"] = kwargs["mlm_model_groups"].loc[Y.index]
        if verbose:
            print(f"N={len(Y)}")
            print(f"Specified formula: {formula}")

        if standardize:
            X = df_standardize(X, func=standardize)
            Y = df_standardize(Y, func=standardize)

    if (constant or df_check_intercept(X)) and (
        standardize == "z" or (isinstance(standardize, bool) and standardize)
    ):
        warnings.warn(
            "Having intercept=True and using z-transformation sets the intercept estimate to zero. This may lead to wrong results.",
            UserWarning,
        )

    results = []
    for r in fit_model(model, Y, X, **kwargs):
        results.append(r)
        if verbose:
            print(r.summary())

    metrics = []
    if add_base_metrics or add_pred_metrics:
        metrics = [base_metrics(r) for r in results]
        if add_pred_metrics:
            metrics = [
                {
                    **r,
                    **m,
                }
                for r, m in zip(metrics, pred_metrics(model, Y, X, **kwargs))
            ]
    if calc_vif:
        metrics = metrics or [{} for _ in results]  # also without base / pred metrics
        metrics = [{**i, "vif": vif(r)} for i, r in zip(metrics, results)]

    return results, metrics


def lm_report(
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
    add_n_groups=True,
    add_vif=True,
    intercept_name="Intercept",
):
    """
    Build one APA-style coefficient table per model from :func:`lm` output.

    Parameters
    ----------
    results : list
        Fitted statsmodels results (first value returned by :func:`lm`).
    metrics : list of dict, optional
        Metrics (second value returned by :func:`lm`), matched with ``results`` by position.
        If None or shorter than ``results``, the tables of the models without metrics have no
        ``model`` column.
    format_pval : bool or callable, default True
        ``False`` keeps the numeric p-values. ``True`` replaces them by APA strings without the
        ``p =`` prefix (``".009"``, ``"<.001"``; :func:`apatools.format.format_p`). A callable
        is applied to every p-value and its return value is used. The stars are always
        computed from the numeric p-values, whatever the value of this argument.
    add_stars : bool or callable, default True
        ``False``: no ``sig`` column. ``True``: add a ``sig`` column with significance stars
        (``***`` p < .001, ``**`` p < .01, ``*`` p < .05, ``⁺`` p < .10, otherwise an empty
        string; :func:`apatools.format.get_stars`). A callable maps one p-value to the
        marker.
    decimal : int or None, default 3
        Round the columns ``coef``, ``se``, ``cil`` and ``cir`` to this many decimals; the
        test statistic, the p-values and the VIFs are not rounded here. ``None`` or 0
        switches the rounding off.
    add_ftest, add_aic, add_bic, add_llf, add_mae, add_mad, add_pred_loo_mae, add_pred_loo_mad, add_n_obs, add_n_groups : bool, default True
        Include the corresponding part (F test, AIC, BIC, log-likelihood ``LL``, ``MAE``,
        ``MAD``, ``MAEpred``, ``MADpred``, number of observations ``N``, number of groups
        ``G``) in the fit summary string, if the metric is available. (R², R²adj and
        R²pred are always shown when available.)
    add_vif : bool, default True
        Join the ``vif`` column to the table if ``lm(..., vif=True)`` was used.
    intercept_name : str, default "Intercept"
        Row label of the constant (``const``, ``Intercept`` or ``CONSTANT``); other labels are
        unchanged.

    Returns
    -------
    list of pandas.DataFrame
        One table per model (as many as the longer of ``results`` and ``metrics``), indexed
        by the coefficient names, with the columns

        * ``coef``: estimate; ``se``: standard error;
        * the test statistic, named by statsmodels: ``t`` (OLS, QuantReg) or ``z`` (RLM, GLM,
          MixedLM);
        * ``p-value``: p-value (formatted, see ``format_pval``);
        * ``cil``, ``cir``: lower and upper limits of the 95% confidence interval
          (confidence level 0.95, as in the statsmodels summary);
        * ``sig``: significance stars (if ``add_stars``);
        * ``vif``: variance inflation factor (if available and ``add_vif``);
        * ``model``: filled in the first row only (if metrics are given), the fit summary
          such as ``R² = .34, R²adj = .34, R²pred = .34, F(1, 416) = 6.71, p = .009, N = 418``.

        For mixed models the table also contains the variance parameters (e.g. ``Group Var``)
        as rows.

    Notes
    -----
    The table is built from the summary table of the statsmodels results
    (``results.summary().tables[1]``), so the confidence level is always 95%. Requires
    ``lxml`` for non-mixed models (``pandas.read_html``).
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
        if isinstance(r, MixedLMResultsWrapper):
            params = r.summary().tables[1]
        else:
            params = read_html(
                StringIO(r.summary().tables[1].as_html()), header=0, index_col=0
            )[0]
        params = params.apply(to_numeric, errors="coerce").rename(
            columns={
                "Coef.": "coef",
                "P>|z|": "p-value",
                "P>|t|": "p-value",
                "std err": "se",
                "Std.Err.": "se",
                "[0.025": "cil",
                "0.975]": "cir",
            },
        )
        round_columns(params, decimal)
        format_significance(params, add_stars, format_pval)
        if add_vif and "vif" in i and i["vif"] is not None:
            params = params.join(i["vif"])
        if len(i):
            params.loc[params.index[0], "model"] = s
        params.index = rename_intercept(params.index, intercept_name)
        output.append(params)

    return output


def _white_df(exog):
    """Degrees of freedom of statsmodels' White test: the rank of the auxiliary regression without constant."""
    first, second = np.triu_indices(exog.shape[1])
    auxiliary = exog[:, first] * exog[:, second]
    return int(np.linalg.matrix_rank(auxiliary)) - 1


def lm_tests(
    results, format_pval=False, add_stars=False, decimal=None, lags=None, reset_power=3
):
    """
    Diagnostic tests of the OLS models fitted with :func:`lm`, as a tidy DataFrame.

    The tests check the assumptions of the linear model, the specification and the influence of
    single observations; they are not measures of the quality of the fit (see ``base_metrics`` and
    ``pred_metrics`` for those). Computed for every OLS model of ``results`` (a list as returned by
    :func:`lm`); other models (RLM, GLM, QuantReg, MixedLM) are skipped with a warning.

    * **Normality of the residuals** (null: normally distributed errors): Jarque-Bera (skewness and
      kurtosis; Jarque & Bera, 1980), the omnibus test of D'Agostino & Pearson (1973; the same
      ingredients, different combination) and Shapiro-Wilk (1965; the statistic is W, no df).
    * **Homoskedasticity** (null: constant error variance): Breusch-Pagan in the studentized version of
      Koenker (1981), which does not assume normal errors, and White (1980), which also detects
      misspecification of the variance by squares and cross products of the predictors. Breusch-Pagan
      has power against a variance that changes linearly with the predictors, White also against
      curved changes (e.g. a variance that grows with the square of a predictor). Both need a
      constant in the model; df is the number of auxiliary regressors.
    * **Functional form** (null: no omitted nonlinearity): Ramsey's RESET (1969), an F test of the
      powers 2 to ``reset_power`` of the fitted values added to the model.
    * **Autocorrelation** (only with ``lags``; null: no autocorrelation of the errors up to ``lags``):
      Breusch-Godfrey LM test (Breusch, 1978; Godfrey, 1978) and the Durbin-Watson statistic (1950,
      1951; about 2 without first-order autocorrelation, below 2 for positive autocorrelation; no
      p-value). They are only meaningful if the rows are ordered in time.
    * **Outliers and influence**: the Bonferroni outlier test (statistic: the largest absolute
      studentized residual; null: no outlier), and the largest Cook's distance (1977; no p-value;
      the report says how many observations exceed the rule of thumb ``4 / n``).
    * **Multicollinearity** (no p-values): the largest variance inflation factor of the predictors
      (rule of thumb: values above 10 are a concern) and the condition number of the design matrix
      (Belsley et al., 1980: above 30 is a concern).

    Parameters
    ----------
    results : list of statsmodels results
        As returned by :func:`lm`.
    format_pval : bool or callable, default False
        Format the p-values as strings (True: APA style without the ``p``, ``"<.001"``;
        a callable receives one p-value); False keeps the numbers.
    add_stars : bool or callable, default False
        Add a ``sig`` column with significance stars (True: :func:`apatools.format.get_stars`;
        or a callable). They are computed from the unformatted p-values.
    decimal : int, optional
        Round ``statistic`` to this many decimals (None: no rounding).
    lags : int, optional
        Number of lags of the Breusch-Godfrey test; None leaves the autocorrelation tests out (they
        are meaningless for cross-sectional data).
    reset_power : int, default 3
        Highest power of the fitted values in RESET (3: squares and cubes, as ``resettest`` in R).

    Returns
    -------
    pandas.DataFrame
        One row per test with the columns ``test``, ``model`` (``ols``; a repeated model gets the suffix
        ``_2``, ...), ``statistic`` (chi-square, F, W, the Durbin-Watson statistic, the largest studentized
        residual, ...), ``df`` and ``df_denom`` (NaN where not applicable), ``p-value`` (NaN for the
        statistics without a test), optionally ``sig``, and an APA-style string ``report``. Use
        ``format_pval=False, add_stars=False, decimal=None`` for the raw numbers.

    Notes
    -----
    With many observations nearly every test becomes significant, and among many tests some are by
    chance; look at the size of the departure (plots of the residuals, the influence of single
    observations), not only at the p-value. The tests use the residuals, so they do not depend on
    the covariance estimator chosen for the standard errors (``ols_fit_cov_type``).

    References
    ----------
    Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential
    Data and Sources of Collinearity*. Wiley. https://doi.org/10.1002/0471725153

    Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. *Australian Economic
    Papers, 17*(31), 334-355. https://doi.org/10.1111/j.1467-8454.1978.tb00635.x

    Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics,
    19*(1), 15-18. https://doi.org/10.1080/00401706.1977.10489493

    D'Agostino, R., & Pearson, E. S. (1973). Tests for departure from normality. Empirical results for
    the distributions of b2 and sqrt(b1). *Biometrika, 60*(3), 613-622.
    https://doi.org/10.1093/biomet/60.3.613

    Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression. I.
    *Biometrika, 37*(3-4), 409-428. https://doi.org/10.1093/biomet/37.3-4.409

    Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression. II.
    *Biometrika, 38*(1-2), 159-178. https://doi.org/10.1093/biomet/38.1-2.159

    Godfrey, L. G. (1978). Testing against general autoregressive and moving average error models when
    the regressors include lagged dependent variables. *Econometrica, 46*(6), 1293-1301.
    https://doi.org/10.2307/1913829

    Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial
    independence of regression residuals. *Economics Letters, 6*(3), 255-259.
    https://doi.org/10.1016/0165-1765(80)90024-5

    Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. *Journal of Econometrics,
    17*(1), 107-112. https://doi.org/10.1016/0304-4076(81)90062-2

    Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression
    analysis. *Journal of the Royal Statistical Society: Series B, 31*(2), 350-371.
    https://doi.org/10.1111/j.2517-6161.1969.tb00796.x

    Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete
    samples). *Biometrika, 52*(3-4), 591-611. https://doi.org/10.1093/biomet/52.3-4.591

    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for
    heteroskedasticity. *Econometrica, 48*(4), 817-838. https://doi.org/10.2307/1912934

    Examples
    --------
    >>> results, metrics = lm(df, "y", ["x1", "x2"])  # doctest: +SKIP
    >>> lm_tests(results, format_pval=True, add_stars=True, decimal=3)  # doctest: +SKIP
    """
    rows = []

    def add(model, label, statistic, df, df_denom, p, text):
        report = f"{text}, {format_p(p)}" if not np.isnan(p) else text
        rows.append((label, model, statistic, df, df_denom, p, report))

    def attempt(model, label, compute, *args):
        try:
            compute(model, *args)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            warnings.warn(f"{label} could not be computed for '{model}': {e}", UserWarning, stacklevel=3)

    nan = np.nan
    supported = []
    for r in results:
        if isinstance(getattr(r, "model", None), sm.OLS):
            supported.append(r)
        else:
            name = type(getattr(r, "model", r)).__name__
            warnings.warn(f"{name} models are not supported by lm_tests (only OLS) and were skipped.", UserWarning, stacklevel=2)

    def normality(model, r):
        resid = np.asarray(r.resid, dtype=float)
        jb, p = jarque_bera(resid)[:2]
        add(model, "Jarque-Bera", float(jb), 2, nan, float(p), f"Jarque-Bera(2) = {jb:.2f}")
        k2, p = omni_normtest(resid)
        add(model, "Omnibus", float(k2), 2, nan, float(p), f"Omnibus(2) = {k2:.2f}")
        w_stat, p = shapiro(resid)
        add(model, "Shapiro-Wilk", float(w_stat), nan, nan, float(p), f"Shapiro-Wilk W = {w_stat:.3f}")

    def variance(model, r):
        resid, exog = np.asarray(r.resid, dtype=float), r.model.exog
        lm_stat, p = sm_diag.het_breuschpagan(resid, exog)[:2]
        df = exog.shape[1] - 1
        add(model, "Breusch-Pagan", float(lm_stat), df, nan, float(p), f"Breusch-Pagan({df}) = {lm_stat:.2f}")
        lm_stat, p = sm_diag.het_white(resid, exog)[:2]
        df = _white_df(exog)
        add(model, "White", float(lm_stat), df, nan, float(p), f"White({df}) = {lm_stat:.2f}")

    def specification(model, r):
        reset = sm_diag.linear_reset(r, power=reset_power, test_type="fitted", use_f=True)
        df, df_denom = int(reset.df_num), int(reset.df_denom)
        add(model, "RESET", float(reset.fvalue), df, df_denom, float(reset.pvalue), f"RESET F({df}, {df_denom}) = {float(reset.fvalue):.2f}")

    def autocorrelation(model, r):
        bg, p = sm_diag.acorr_breusch_godfrey(r, nlags=lags)[:2]
        add(model, "Breusch-Godfrey", float(bg), lags, nan, float(p), f"Breusch-Godfrey({lags}) = {bg:.2f}")
        dw = float(durbin_watson(np.asarray(r.resid, dtype=float)))
        add(model, "Durbin-Watson", dw, nan, nan, nan, f"Durbin-Watson = {dw:.2f}")

    def influence(model, r):
        influence = OLSInfluence(r)
        resid = np.asarray(r.resid, dtype=float)
        leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
        df_denom = int(r.df_resid) - 1
        # externally studentized residuals in closed form (statsmodels refits the model n times for them)
        variance_without = (r.df_resid * r.mse_resid - resid**2 / (1 - leverage)) / df_denom
        with np.errstate(divide="ignore", invalid="ignore"):
            student = resid / np.sqrt(variance_without * (1 - leverage))
        position = int(np.nanargmax(np.abs(student)))
        largest = float(abs(student[position]))
        p = float(min(1.0, len(student) * 2 * scipy.stats.t.sf(largest, df_denom)))  # Bonferroni
        label = r.resid.index[position]
        add(model, "Outlier test", largest, nan, df_denom, p, f"max |studentized residual| = {largest:.2f} (observation {label}), Bonferroni")
        cook = np.asarray(influence.cooks_distance[0], dtype=float)
        count = int((cook > 4 / len(cook)).sum())
        add(model, "Cook's distance", float(cook.max()), nan, nan, nan, f"max Cook's D = {cook.max():.3f} ({count} above 4/n)")

    def collinearity(model, r):
        exog, names = r.model.exog, list(r.model.exog_names)
        columns = [i for i, name in enumerate(names) if name not in INTERCEPT_NAMES]
        vifs = [variance_inflation_factor(exog, i) for i in columns] if len(columns) > 1 else [1.0]
        add(model, "Max VIF", float(max(vifs)), nan, nan, nan, f"max VIF = {max(vifs):.2f}")
        condition = float(r.condition_number)
        add(model, "Condition number", condition, nan, nan, nan, f"condition number = {condition:.1f}")

    seen = {}
    for r in supported:
        seen["ols"] = seen.get("ols", 0) + 1
        model = "ols" if seen["ols"] == 1 else f"ols_{seen['ols']}"
        for label, compute in (
            ("Normality tests", normality),
            ("Heteroskedasticity tests", variance),
            ("RESET", specification),
            ("Influence diagnostics", influence),
            ("Collinearity diagnostics", collinearity),
        ):
            attempt(model, label, compute, r)
        if lags:
            attempt(model, "Autocorrelation tests", autocorrelation, r)

    res = DataFrame(rows, columns=["test", "model", "statistic", "df", "df_denom", "p-value", "report"])
    for column in ("statistic", "df", "df_denom", "p-value"):
        res[column] = res[column].astype(float)  # also for an empty table (object columns)
    if decimal:
        res["statistic"] = res["statistic"].round(decimal)
    format_significance(res, add_stars, format_pval, na_rep="")  # statistics without a test: empty cells
    return res
