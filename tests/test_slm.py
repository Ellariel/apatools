import numpy as np
import pandas as pd
import libpysal

from apatools.slm import slm#, slm_report


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


if __name__ == "__main__":
    test_slm()
    print("Tests for apatools.slm are PASSED!")
