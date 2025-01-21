import scipy
import numpy as np
from io import StringIO
import statsmodels.api as sm
from itertools import zip_longest
from pandas import read_html, DataFrame
from sklearn.model_selection import LeaveOneOut
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .format import format_r, format_p, get_stars


def vif(results, sort=False, decimal=2):
    """
    VIF, the variance inflation factor, is a measure of multicollinearity.
    VIF > 5 for a variable indicates that it is highly collinear with the
    other input variables.
    """
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
        else:
            raise NotImplementedError(
                f"{model} is not implemented, try 'ols', 'rlm' or 'glm'."
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
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:]: v for k, v in _kwargs.items() if k.startswith("model_")}
        fit_kwargs = {k[4:]: v for k, v in _kwargs.items() if k.startswith("fit_")}

        if verbose:
            print("model_kwargs:", model_kwargs)
            print("fit_kwargs:", fit_kwargs)

        yield model(Y, X, **model_kwargs).fit(**fit_kwargs)


def pred_r_sq(model, Y, X, **kwargs):
    """
    Calculating R²pred for statsmodels

    https://stats.stackexchange.com/questions/592653/how-to-get-predicted-r-square-from-statmodels
    """
    res = {}
    errors = []
    kwargs["verbose"] = False
    for train_index, test_index in LeaveOneOut().split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        try:
            for idx, r in enumerate(fit_model(model, y_train, x_train, **kwargs)):
                res.setdefault(idx, [])
                res[idx].append(*(y_test - r.predict(x_test)))
        except Exception as e:
            errors.append(str(e))
    if len(errors):
        print(
            f"Some attempts to calculate R²pred have been unsuccessful ({len(errors)}): {set(errors)}"
        )
    return np.clip(
        [1 - np.sum(np.square(i)) / np.var(Y) / Y.size for _, i in res.items()],
        -1.0,
        1.0,
    )


def r_sq(results):
    """
    Calculating/retrieving R²/R²pseudo, R²adj, R²pred
    for OLS, RLM, GLM from statsmodels fitting results

    https://stats.stackexchange.com/questions/83826/is-a-weighted-r2-in-robust-linear-model-meaningful-for-goodness-of-fit-analys
    https://stats.stackexchange.com/questions/55236/prove-f-test-is-equal-to-t-test-squared
    """

    weights = getattr(results, "weights", 1)
    resid = getattr(results, "resid", getattr(results, "resid_working", None))
    nobs = int(results.nobs)
    n_pred = len(
        set(results.params.index) - set(["const"])
    )  # number of predictors, without intercept
    df_model = max(int(results.df_model), n_pred)  # technical correction
    df_resid = max(int(results.df_resid), nobs - df_model)  # technical correction
    fitted = results.fittedvalues
    observed = resid + fitted
    SSe = np.sum((weights * resid) ** 2)
    SSt = np.sum((weights * (observed - np.mean(weights * observed))) ** 2)
    r_sq = getattr(
        results, "rsquared", getattr(results, "pseudo_rsquared", 1 - SSe / SSt)
    )
    r_sq = r_sq() if callable(r_sq) else r_sq

    # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.rsquared_adj.html
    r_sq_adj = getattr(results, "rsquared_adj", 1 - (1 - r_sq) * nobs / df_resid)

    # https://www.slideshare.net/slideshow/multiple-regressionppt-252604177/252604177#8
    f_stat = getattr(
        results, "fvalue", (r_sq / df_model) / ((1 - r_sq) / df_resid)
    )  # (SSt / df_model) / (SSe / df_resid)
    f_pvalue = getattr(
        results, "f_pvalue", scipy.stats.f.sf(f_stat, df_model, df_resid)
    )

    return {
        "r_sq": r_sq,
        "r_sq_adj": r_sq_adj,
        "df_model": df_model,
        "df_resid": df_resid,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
    }


def lm(data, y, x, model="ols", **kwargs):
    """
    Fitting OLS, RLM, GLM from statsmodels

    lm(test_data, Y, X, model=['ols', 'rlm'],
                    verbose=True,
                    constant=True,
                    standardized=False, # keeps np.number columns only
                    r_sq = True,
                    vif = False,
                    pred_r_sq = True,
                    ols_fit_cov_type='HC1',
                    rlm_model_M=sm.robust.norms.RamsayE())
    """

    verbose = kwargs.get("verbose", True)
    constant = kwargs.pop("constant", True)
    standardized = kwargs.pop("standardized", False)
    add_r_sq = kwargs.pop("r_sq", False)
    add_pred_r_sq = kwargs.pop("pred_r_sq", False)
    calc_vif = kwargs.pop("vif", False)

    if verbose and constant and standardized:
        print(
            "Having constant=True and standardized=True at the same time does not make sense and can lead to errors."
        )

    df = data[[y] + x].dropna()
    if verbose and len(df) != len(data):
        print(f"N={len(data)}")
        print("Rows with NAs were dropped!")

    if standardized:
        df = df.select_dtypes(include=[np.number, "bool"]).apply(scipy.stats.zscore)
    X, Y = df[x], df[y]

    if verbose:
        print(f"N={len(Y)}")
        print(f"formula: {y} ~ {'1 + ' if constant else ''}" + " + ".join(x))

    if constant:
        X = sm.add_constant(X)

    results = []
    for r in fit_model(model, Y, X, **kwargs):
        results.append(r)
        if verbose:
            print(r.summary())

    metrics = []
    if add_r_sq or add_pred_r_sq:
        metrics = [r_sq(r) for r in results]
        if add_pred_r_sq:
            metrics = [
                {**r, **{"pred_r_sq": rr}}
                for r, rr in zip(metrics, pred_r_sq(model, Y, X, **kwargs))
            ]
    if calc_vif:
        metrics = [{**i, **{"vif": vif(r)}} for i, r in zip(metrics, results)]

    return results, metrics


def lm_report(results, metrics={}, format_pval=True, add_stars=True, decimal=None):
    # R² = .34, R²adj = .34, R²pred = .34, F(1, 416) = 6.71, p = .009

    output = []
    for r, i in zip_longest(results, metrics, fillvalue={}):
        s = []
        if "r_sq" in i:
            s.append(f"R² {format_r(i['r_sq'], use_letter=False)}")
        if "r_sq_adj" in i:
            s.append(f"R²adj {format_r(i['r_sq_adj'], use_letter=False)}")
        if "pred_r_sq" in i:
            s.append(f"R²pred {format_r(i['pred_r_sq'], use_letter=False)}")
        s.append(
            f"F({i['df_model']}, {i['df_resid']}) = {i['f_stat']:.2f}, {format_p(i['f_pvalue'])}"
        )
        s = ", ".join(s)
        params = read_html(
            StringIO(r.summary().tables[1].as_html()), header=0, index_col=0
        )[0].rename(
            columns={
                "P>|z|": "p-value",
                "std err": "se",
                "[0.025": "cil",
                "0.975]": "cir",
            }
        )
        if decimal:
            for c in ["coef", "se", "cil", "cir"]:
                params[c] = params[c].round(decimal)

        if add_stars:
            add_stars = add_stars if callable(add_stars) else get_stars
            params["sig"] = [get_stars(c) for c in params["p-value"]]

        if format_pval:
            format_pval = (
                format_pval
                if callable(format_pval)
                else lambda x: format_p(
                    x, use_letter=False, keep_spaces=False, no_equals=True
                )
            )
            params["p-value"] = [format_pval(c) for c in params["p-value"]]
        if "vif" in i:
            params = params.join(i["vif"])
        if len(i):
            params.loc[params.index[0], "model"] = s
        output.append(params)

    return output
