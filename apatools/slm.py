import warnings
import scipy
import numpy as np
import spreg as spr
from itertools import zip_longest
from statsmodels.api import add_constant
from sklearn.model_selection import LeaveOneOut
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
        if model == spr.OLS:
            if verbose:
                print("model: OLS")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ols_")}
        elif model == spr.ML_Lag:
            if verbose:
                print("model: SLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("slm_")}
        elif model == spr.ML_Error:
            if verbose:
                print("model: SEM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("sem_")}
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:]: v for k, v in _kwargs.items() if k.startswith("model_")}

        if verbose:
            print("model_kwargs:", model_kwargs)

        yield model(Y, X, w, **model_kwargs)


def slm(data, y=None, x=None, w=None, model="ols", formula=None, **kwargs):
    """
    Fitting OLS, SLM, SEM from spreg

    slm(test_data, Y, X, model=['ols', 'slm'],
                    verbose=True,
                    dropna=True, 
                    intercept=False, # ignored, in spreg it is added by default 
                    standardized='z' # keeps np.number or bool columns only
                    base_metrics=True,
                    pred_metrics=False,
                    
                    ols_model_spat_diag=True,
                    ols_model_nonspat_diag=True,
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    constant = kwargs.pop("intercept", False)
    standardize = kwargs.pop("standardize", False)
    #add_base_metrics = kwargs.pop("base_metrics", True)
    #add_pred_metrics = kwargs.pop("pred_metrics", False)
    
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
    
    return results


