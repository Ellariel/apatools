import warnings
import scipy
import numpy as np
import spreg as spr
from pandas import to_numeric
from itertools import zip_longest
from statsmodels.api import add_constant
# from sklearn.model_selection import LeaveOneOut
from statsmodels.formula.formulatools import handle_formula_data


from .citation import Citation
from .format import format_r, format_p, get_stars
from .utils import df_standardize, df_check_intercept


# https://pysal.org/spreg/index.html
CITATION = Citation(
    APA="Rey, S. J., & Anselin, L. (2007). \
PySAL: A python library of spatial analytical methods. \
The Review of Regional Studies, 37(1), 5-27."
)



def fit_model(model, Y, X, w, **kwargs):  # model.fit() generator function
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
        if isinstance(model, spr.OLS):
            if verbose:
                print("model: OLS")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ols_")}
        elif isinstance(model, spr.ML_Lag):
            if verbose:
                print("model: SLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("slm_")}
        elif isinstance(model, spr.ML_Error):
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
    Calculating/retrieving R²/R²pseudo etc.
    """

    outputs = {}
    resid = results.y.flatten() - results.predy.flatten()
    # k - variables for which coefficients are estimated 
    # (including the constant, excluding rho / lambda) <-- wrong description, rho is excluded, but not lambda
    n_params = int(results.k)
    n_obs = int(results.n)
    
    if isinstance(results, spr.ML_Error):
        n_params += 1 # lambda counts
    df_model = n_params - 1
    df_resid = n_obs - df_model
    
    if isinstance(results, (spr.ML_Lag, spr.ML_Error)):
        r_sq = results.pr2
        r_sq_adj = 1 - (1 - r_sq) * n_obs / df_resid
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
            "mad": np.median(np.abs(resid) - np.median(resid)),
            "aic": results.aic,
            "bic": results.schwarz,
            "llf": results.logll,
        }
    )
    return outputs


def slm(data, y=None, x=None, w=None, model="ols", formula=None, **kwargs):
    """
    Fitting OLS, SLM, SEM from spreg

    slm(test_data, Y, X, model=['ols', 'slm'],
                    verbose=True,
                    dropna=True, 
                    intercept=False, # ignored, spreg adds it by default 
                    standardized='z' # keeps np.number or bool columns only
                    base_metrics=True,
                    pred_metrics=False, # no this, and no vif for spreg
                    
                    ols_model_spat_diag=True,
                    ols_model_nonspat_diag=True,
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    constant = kwargs.pop("intercept", False)
    standardize = kwargs.pop("standardize", False)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    
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
            X = add_constant(X) # ignoredin spreg
    else:
        if verbose:
           (Y, X), _, _ = handle_formula_data(data, X=None, formula=formula)
           print(f"N={len(Y)}")
           print(f"Specified formula: {formula}")
           
        if standardize: # need a test
            X = df_standardize(X, func=standardize)
            Y = df_standardize(Y, func=standardize)

    if standardize == 'z' or (isinstance(standardize, bool) and standardize):
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
        #if add_pred_metrics:
        #    metrics = [
        #        {
        #            **r,
        #            **m,
        #        }
        #        for r, m in zip(metrics, pred_metrics(model, Y, X, **kwargs))
        #    ]
    
    return results, metrics


def slm_report(results, metrics={}, format_pval=True, add_stars=True, decimal=3, 
              add_ftest=True, add_aic=True, add_bic=True, add_llf=True, add_mae=True, add_mad=True,
              add_pred_loo_mae=False, add_pred_loo_mad=False, # no this, and no vif for spreg
              add_n_obs=True, add_n_groups=False, add_vif=False,
              intercept_name='Intercept'):
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
        params = r.output.set_index('var_names')
        params = params.apply(to_numeric, errors="coerce").rename(
            columns={
                "coefficients": "coef",
                "prob": "p-value",
                "std_err": "se",
                "zt_stat": "z",
            },
        )
        alpha_z = scipy.stats.norm.ppf(0.975) # 95% CI
        params["cil"] = params["coef"] - alpha_z * params["se"]
        params["cir"] = params["coef"] + alpha_z * params["se"]
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
        const_name = df_check_intercept(params, params.index)  
        if const_name != intercept_name:
            params.index = params.index.str.replace(const_name, intercept_name)
        output.append(params)

    return output