import warnings
import scipy
import numpy as np
from io import StringIO
import statsmodels.api as sm
from itertools import zip_longest
from statsmodels.api import add_constant
from sklearn.model_selection import LeaveOneOut
from pandas import read_html, to_numeric, DataFrame
from statsmodels.formula.formulatools import handle_formula_data
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper


from .citation import Citation
from .format import format_r, format_p, get_stars
from .utils import df_standardize, df_check_intercept


# https://www.statsmodels.org/stable/index.html
CITATION = Citation(
    APA="Seabold, S., & Perktold, J. (2010). \
statsmodels: Econometric and statistical modeling with python. \
9th Python in Science Conference (pp. 57-61), \
Austin, Texas, United States. \
https://doi.org/10.25080/Majora-92bf1922-011"
)

  

def vif(results, sort=False, decimal=2):
    """
    VIF, the variance inflation factor, is a measure of multicollinearity.
    VIF > 5 for a variable indicates that it is highly collinear with the
    other input variables.
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
    except Exception as e:
        warnings.warn(
            f"VIF calculation failed ({str(e)}).",
            UserWarning,
        )
        return None   


def mlm_wrapper(Y, X, **kwargs):
    groups = kwargs.pop("groups")
    return sm.MixedLM(Y, X, groups, **kwargs)


def fit_model(model, Y, X, **kwargs):  # model.fit() generator function
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
            model = mlm_wrapper # using wrapper because of one positional argument
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
    Calculating R²pred for statsmodels

    https://stats.stackexchange.com/questions/592653/how-to-get-predicted-r-square-from-statmodels
    """
    errors = {}
    fit_fails = []
    _kwargs = kwargs.copy()
    _kwargs["verbose"] = False
    groups = kwargs.get("mlm_model_groups", None) # mlm
    
    for train_index, test_index in LeaveOneOut().split(X):
        if groups is not None: # mlm
            x_train, x_test, _groups = X.iloc[train_index], X.iloc[test_index], groups.iloc[train_index]
            _kwargs["mlm_model_groups"] = _groups
        else:
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        try:
            for idx, r in enumerate(fit_model(model, y_train, x_train, **_kwargs)):
                errors.setdefault(idx, [])
                errors[idx].append(y_test.iloc[0] - r.predict(x_test).iloc[0])
        except Exception as e:
            fit_fails.append(str(e))
    if len(fit_fails):
        warnings.warn(
            f"Some attempts to calculate LeaveOneOut metrics failed ({len(fit_fails)}): {set(fit_fails)}",
            UserWarning,
        )
    return [{
            "pred_loo_r_sq": np.clip(
                1 - np.sum(np.square(err)) / (np.var(Y) * Y.size),
                -1.0,
                1.0,
            ),
            "pred_loo_mae": np.mean(np.abs(err)),
            "pred_loo_mad": np.median(np.abs(np.asarray(err) - np.median(err))),
        }
        for err in errors.values()
    ]


def mlm_icc(results):
    """
    Intraclass Correlation Coefficient (ICC) (approx.)
    """
    var_random = 0.0  # Random effects variance
    if results.cov_re.shape[0] > 0:
        var_random += np.sum(np.diag(results.cov_re))
    var_resid = results.scale  # Residual variance
    if results.cov_re.shape != (1, 1):
        warnings.warn(
            "ICC is only well-defined for random-intercept models.",
            UserWarning,
        )
    return var_random / (var_random + var_resid)


def mlm_r_sq(results):
    # Fixed effects variance
    var_fixed = np.var(np.dot(results.model.exog, results.fe_params.values))
    var_random = 0.0  # Random effects variance
    if results.cov_re.shape[0] > 0:
        var_random += np.sum(np.diag(results.cov_re))
    # Add variance components
    if hasattr(results, "vcomp") and results.vcomp is not None:
        var_random += sum(results.vcomp)
    var_resid = results.scale  # Residual variance
    # Marginal R² (r2_m) → variance explained by fixed effects only
    r2_m = var_fixed / (var_fixed + var_random + var_resid)
    # Conditional R² (r2_c) → variance explained by fixed + random effects
    r2_c = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    return r2_m, r2_c


def base_metrics(results):
    """
    Calculating/retrieving R²/R²pseudo, R²adj
    for OLS, RLM, GLM, MLM from statsmodels fitting results

    https://stats.stackexchange.com/questions/83826/is-a-weighted-r2-in-robust-linear-model-meaningful-for-goodness-of-fit-analys
    https://stats.stackexchange.com/questions/55236/prove-f-test-is-equal-to-t-test-squared
    """

    outputs = {}
    n_obs = results.nobs
    params = set(results.params.index)
    n_params = getattr(
        results, "k_fe", len(params)
    )  # k_fe counts only fixed effects in MLM
    if "const" in params or "Intercept" in params:
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

    M = getattr(results.model, "M", False)  # r_sq_pseudo
    if M and callable(M.rho):
        SSe = np.sum(M.rho(resid))
        SSt = np.sum(M.rho(observed - np.mean(observed)))
        outputs.update({"r_sq_pseudo": 1 - SSe / SSt})

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

    r_sq_adj_def = 1 - (1 - r_sq) * n_obs / df_resid
    r_sq_adj = getattr(results, "rsquared_adj", r_sq_adj_def)
    if not np.isfinite(r_sq_adj):
        r_sq_adj = r_sq_adj_def

    # https://www.slideshare.net/slideshow/multiple-regressionppt-252604177/252604177#8
    f_stat_def = (r_sq / df_model) / (
        (1 - r_sq) / df_resid
    ) if df_model and df_resid else np.nan # (SSt / df_model) / (SSe / df_resid)
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
            "mad": np.median(np.abs(resid) - np.median(resid)),
        }
    )
    return outputs


def lm(data, y=None, x=None, model="ols", formula=None, **kwargs):
    """
    Fitting OLS, RLM, GLM from statsmodels

    lm(test_data, Y, X, model=['ols', 'rlm'],
                    verbose=True,
                    intercept=True, # ignored when formula is defined
                    dropna=True, # ignored when formula is defined
                    standardize='z' # keeps np.number or bool columns only
                    base_metrics=True,
                    pred_metrics=False,
                    vif=False,
                    
                    qlm_fit_q=0.5,
                    qlm_fit_cov_type='boot',
                    qlm_fit_cov_kwds={'n_boot': 100},
                    ols_fit_cov_type='HC1',
                    rlm_model_M=sm.robust.norms.RamsayE())
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    constant = kwargs.pop("intercept", True)
    standardize = kwargs.pop("standardize", False)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    calc_vif = kwargs.pop("vif", False)
    
    groups = kwargs.get("mlm_model_groups", False) #mlm
    if isinstance(groups, str):
        kwargs["mlm_model_groups"] = data[groups]

    if formula is None:
        if x is None or y is None or len(x) == 0:
            raise ValueError(
                "Either formula or x,y have to be explicitely defined."
            )
        
        df = data[[y] + x]
        if dropna:
            df.dropna(inplace=True)
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
            X = add_constant(X)
    else:
        if verbose:
           (Y, X), _, _ = handle_formula_data(data, X=None, formula=formula)
           print(f"N={len(Y)}")
           print(f"Specified formula: {formula}")
           
        if standardize: # need a test
            X = df_standardize(X, func=standardize)
            Y = df_standardize(Y, func=standardize)

    if (constant or df_check_intercept(X)) and\
       (standardize == 'z' or (isinstance(standardize, bool) and standardize)):
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
        metrics = [{**i, "vif": vif(r)} for i, r in zip(metrics, results)]

    return results, metrics


def lm_report(results, metrics={}, format_pval=True, add_stars=True, decimal=None, 
              add_ftest=True, add_aic=True, add_bic=True, add_llf=True, add_mae=True, add_mad=True,
              add_pred_loo_mae=True, add_pred_loo_mad=True, 
              add_n_obs=True, add_n_groups=True, add_vif=True):
    # R² = .34, R²adj = .34, R²pred = .34, F(1, 416) = 6.71, p = .009

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
                else lambda x: format_p(
                    x, use_letter=False, keep_spaces=False, no_equals=True
                )
            )
            params["p-value"] = [format_pval(c) for c in params["p-value"]]
        if add_vif and "vif" in i and i["vif"] is not None:
            params = params.join(i["vif"])
        if len(i):
            params.loc[params.index[0], "model"] = s
        output.append(params)

    return output
