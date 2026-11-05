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
        == "R² = .35, R²adj = .31, F(3, 45) = 8.06, p < .001, AIC = 404.5, BIC = 410.1, LL = -199.2, MAE = 11.440, MAD = 8.518, N = 49"
    )
    
    assert (
        results_rep[1].iloc[0]["model"]
        == "R² = .35, R²adj = .32, F(2, 46) = 12.36, p < .001, AIC = 408.7, BIC = 414.4, LL = -201.4, MAE = 11.268, MAD = 6.869, N = 49"
    )
    
    assert (
        results_rep[2].iloc[0]["model"]
        == "R² = .38, R²adj = .34, F(3, 45) = 9.29, p < .001, AIC = 408.9, BIC = 416.4, LL = -200.4, MAE = 10.943, MAD = 7.149, N = 49"
    )
    
    #5/0



def test_slm_kwargs_routing_and_df():
    import pytest

    db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'), 'r')
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    cols = ["HOVAL", "INC", "CRIME"]
    df = pd.DataFrame([db.by_col(c) for c in cols], index=cols).T

    # ols_model_* arguments must reach spreg: spatial diagnostics are only
    # computed when requested
    r_no, _ = slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False,
                  ols_model_spat_diag=False)
    r_yes, _ = slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False,
                   ols_model_spat_diag=True)
    assert getattr(r_no[0], "lm_error", None) is None
    assert r_yes[0].lm_error is not None

    # OLS: F(k - 1, n - k) as reported by spreg itself
    results, metrics = slm(df, "HOVAL", ["INC", "CRIME"], model="ols", verbose=False)
    n, k = results[0].n, results[0].k
    assert (metrics[0]["df_model"], metrics[0]["df_resid"]) == (k - 1, n - k)
    assert metrics[0]["f_stat"] == pytest.approx(results[0].f_stat[0])
    assert metrics[0]["r_sq_adj"] == pytest.approx(results[0].ar2)
    # MAD is the median absolute deviation of the residuals
    resid = results[0].y.flatten() - results[0].predy.flatten()
    assert metrics[0]["mad"] == pytest.approx(np.median(np.abs(resid - np.median(resid))))

    # formula path works without verbose output
    r_f, m_f = slm(df, model="ols", formula="HOVAL ~ 1 + INC + CRIME", verbose=False)
    assert m_f[0]["r_sq"] == pytest.approx(metrics[0]["r_sq"])

    # unknown model name
    with pytest.raises(NotImplementedError):
        slm(df, "HOVAL", ["INC"], w, model="xyz", verbose=False)
    # neither formula nor x/y
    with pytest.raises(ValueError):
        slm(df, w=w, verbose=False)


if __name__ == "__main__":
    test_slm()
    print("Tests for apatools.slm are PASSED!")
