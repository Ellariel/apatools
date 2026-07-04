import numpy as np
import pandas as pd
import libpysal

from apatools.slm import slm, slm_report


def s(x):
    return f"{x:.2f}"


def test_slm():
    db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    cols = ["HOVAL", "INC", "CRIME"]
    df = pd.DataFrame([db.by_col(c) for c in cols], index=cols).T
    assert len(df) == 49
    y, x = cols[0], cols[1:]
    f = "HOVAL ~ 1 + INC + CRIME"

    #OLS
    results, metrics = slm(df, y, x, model='ols', 
                  standardize=False,
                  ols_model_spat_diag=False,
                  ols_model_nonspat_diag=True)
    print(metrics)
    assert s(results[0].betas[0][0]) == "46.43"

    results, metrics = slm(df, y, x, w, model='ols', # spatial diag
                  standardize=False,
                  ols_model_spat_diag=True,
                  ols_model_nonspat_diag=True)
    print(metrics)
    assert s(results[0].betas[0][0]) == "46.43"
    
    results, metrics = slm(df, model='ols', 
                  formula=f, # formula test
                  standardize=False,
                  ols_model_spat_diag=False,
                  ols_model_nonspat_diag=True)
    print(metrics)
    assert s(results[0].betas[0][0]) == "46.43" 
    
    #SLM
    results, metrics = slm(df, y, x, w, model='slm', 
                  standardize=False,
                  ols_model_spat_diag=True,
                  ols_model_nonspat_diag=True)
    print(metrics)
    assert s(results[0].betas[0][0]) == "37.27"

    #SEM
    results, metrics = slm(df, y, x, w, model='sem', 
                  standardize=False,
                  ols_model_spat_diag=True,
                  ols_model_nonspat_diag=True)
    print(metrics)
    assert s(results[0].betas[0][0]) == "48.01"
    
    results, metrics = slm(df, w=w, model='sem',
                  formula=f, # formula test
                  standardize=False,
                  ols_model_spat_diag=True,
                  ols_model_nonspat_diag=True)
    print(metrics)
    assert s(results[0].betas[0][0]) == "48.01"

    results_rep = slm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)
    assert s(metrics[0]["r_sq"]) == "0.35"
    
    results, metrics = slm(df, w=w, model=['sem', 'ols','slm'],
                  formula=f, # formula test
                  standardize=False,
                  ols_model_spat_diag=True,
                  ols_model_nonspat_diag=True)
    
    results_rep = slm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)
    
    assert (
        results_rep[0].iloc[0]["model"]
        == "R² = .35, R²adj = .31, F(3, 46) = 8.24, p < .001, AIC = 404.5, BIC = 410.1, LL = -199.2, MAE = 11.440, MAD = 14.073, N = 49"
    )
    
    assert (
        results_rep[1].iloc[0]["model"]
        == "R² = .35, R²adj = .32, F(2, 47) = 12.36, p < .0001, AIC = 408.7, BIC = 414.4, LL = -201.4, MAE = 11.268, MAD = 14.578, N = 49"
    )
    
    assert (
        results_rep[2].iloc[0]["model"]
        == "R² = .38, R²adj = .34, F(3, 46) = 9.50, p < .0001, AIC = 408.9, BIC = 416.4, LL = -200.4, MAE = 10.943, MAD = 14.385, N = 49"
    )
    
    #5/0



if __name__ == "__main__":
    test_slm()
    print("Tests for apatools.slm are PASSED!")
