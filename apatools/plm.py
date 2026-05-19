import warnings
import numpy as np
import spreg as sr
import scipy
import statsmodels.api as sm
from itertools import zip_longest
from sklearn.model_selection import LeaveOneOut
from spreg import OLS, ML_Lag, ML_LagRE, ML_ErrorRE, ML_ErrorFE, PooledOLS, PanelFE, PanelRE#,# OLS


from .citation import Citation
from .format import format_r, format_p, get_stars
from .utils import df_standardize, df_check_intercept


# https://www.statsmodels.org/stable/index.html
CITATION = Citation(
    APA="Seabold, S., & Perktold, J. (2010). \
statsmodels: Econometric and Statistical Modeling with Python. \
9th Python in Science Conference (pp. 57-61), \
Austin, Texas, United States. \
https://doi.org/10.25080/Majora-92bf1922-011"
)



     



def fit_model(model, Y, X, w, **kwargs):  # model.fit() generator function
    if isinstance(model, (list, tuple)):
        for m in model:
            yield next(fit_model(m, Y, X, w, **kwargs))
    elif isinstance(model, str):
        if model == "ols":
            yield next(fit_model(sr.OLS, Y, X, w, **kwargs))
        #elif model == "rlm":
        #    yield next(fit_model(sm.RLM, Y, X, **kwargs))
        #elif model == "glm":
        #    yield next(fit_model(sm.GLM, Y, X, **kwargs))
        #elif model == "qlm":
        #    yield next(fit_model(sm.QuantReg, Y, X, **kwargs))
        #elif model == "mlm":
        #    yield next(fit_model(sm.MixedLM, Y, X, **kwargs))
        else:
            raise NotImplementedError(
                f"'{model}' is not implemented, try 'ols', 'rlm', 'glm', 'qlm' or 'mlm'."
            )
    else:
        verbose = kwargs.get("verbose", False)
        if model == sr.OLS:
            if verbose:
                print("model: OLS")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ols_")}
        #elif model == sm.RLM:
        #    if verbose:
        #        print("model: RLM")
        #    _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("rlm_")}
        #elif model == sm.GLM:
        #    if verbose:
        #        print("model: GLM")
        #    _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("glm_")}
        #elif model == sm.QuantReg:
        #    if verbose:
        #        print("model: QLM")
        #    _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("qlm_")}
        #elif model == sm.MixedLM:
        #    model = mlm_wrapper # using wrapper because of one positional argument
        #    if verbose:
        #        print("model: MLM")
        #    _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("mlm_")}
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:]: v for k, v in _kwargs.items() if k.startswith("model_")}

        if verbose:
            print("model_kwargs:", model_kwargs)

        yield model(Y, X, w, **model_kwargs)


def plm(data, y=None, x=None, w=None, model="ols", **kwargs):
    """
    Fitting OLS, RLM, GLM from statsmodels

    lm(test_data, Y, X, model=['ols', 'rlm'],
                    verbose=True,
                    dropna=True, 
                    standardized='z' # keeps np.number or bool columns only
                    base_metrics=True,
                    pred_metrics=False,
                    
                    qlm_fit_q=0.5,
                    qlm_fit_cov_type='boot',
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    standardize = kwargs.pop("standardize", False)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    
    if x is None or y is None or len(x) == 0:
        raise ValueError(
            "x,y have to be explicitely defined."
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
        if standardize == 'z' or (isinstance(standardize, bool) and standardize):
            warnings.warn(
                "Having standardize=True and using z-transformation sets the intercept estimate (which is enabled by default) to zero, this may lead to wrong results.",
                UserWarning,
            )

    X, Y = df[x], df[y]

    if verbose:
        print(f"N={len(Y)}")
        print(f"formula: {y} ~ 1 + " + " + ".join(x))

    results = []
    for r in fit_model(model, Y, X, w, **kwargs):
        results.append(r)
        if verbose:
            print(r.summary)
    
    return results


