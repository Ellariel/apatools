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


import warnings

import pytest
import spreg

import apatools.slm as slm_module
from apatools.slm import pred_metrics, w_subset_rows


@pytest.fixture(scope="module")
def columbus():
    db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'), 'r')
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    cols = ["HOVAL", "INC", "CRIME"]
    df = pd.DataFrame([db.by_col(c) for c in cols], index=cols).T
    return df, w


def _loo_summary(err, y):
    err = np.asarray(err)
    return {
        "pred_loo_r_sq": 1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2),
        "pred_loo_mae": np.mean(np.abs(err)),
        "pred_loo_mad": np.median(np.abs(err - np.median(err))),
    }


def test_w_subset_rows_keeps_weights_and_transform():
    # non-binary weights must survive the subsetting
    neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
    weights = {0: [2.0, 4.0], 1: [2.0, 1.0], 2: [4.0, 1.0, 3.0], 3: [3.0]}
    w = libpysal.weights.W(neighbors, weights)
    sub = w_subset_rows(w, [0, 2, 3])
    assert sub.n == 3 and sub.transform == "O"
    assert sub.id_order == [0, 1, 2]  # ids are the new positions
    assert sub.weights[0] == [4.0]  # (old 0 -> 2) weight 4.0 kept
    assert sub.weights[1] == [4.0, 3.0]  # (old 2 -> 0, 3)
    # the transformation of w is re-applied to the subset
    w.transform = "r"
    sub = w_subset_rows(w, [0, 2, 3])
    assert sub.transform == "R"
    assert np.allclose(sub.sparse.sum(axis=1).A.ravel(), 1.0)
    # relative weights are those of the original matrix, renormalized
    assert sub.weights[1] == pytest.approx([4 / 7, 3 / 7])
    # observations that lose all their neighbours become islands
    assert w_subset_rows(w, [0, 3]).islands == [0, 1]


def test_pred_metrics_ols_equals_closed_form(columbus):
    df, w = columbus
    results, metrics = slm(df, "HOVAL", ["INC", "CRIME"], model="ols", verbose=False,
                           pred_metrics=True)
    X = results[0].x
    y = results[0].y.ravel()
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    loo_err = (y - hat @ y) / (1 - np.diag(hat))  # PRESS residuals
    expected = _loo_summary(loo_err, y)
    for k, v in expected.items():
        assert metrics[0][k] == pytest.approx(v), k
    # base metrics are still there and unchanged
    assert metrics[0]["r_sq"] == pytest.approx(results[0].r2)
    assert metrics[0]["pred_loo_r_sq"] < metrics[0]["r_sq"]


def test_pred_metrics_ols_exclude_neighbors_matches_manual(columbus):
    df, w = columbus
    results, metrics = slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False,
                           pred_metrics=True, pred_exclude_neighbors=True)
    X, y = results[0].x, results[0].y.ravel()
    err = []
    for i in range(len(y)):
        drop = {i} | set(w.neighbors[i])
        keep = [k for k in range(len(y)) if k not in drop]
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        err.append(y[i] - X[i] @ beta)
    for k, v in _loo_summary(err, y).items():
        assert metrics[0][k] == pytest.approx(v), k
    # plain leave-one-out is more optimistic than leaving the neighbourhood out
    _, plain = slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False, pred_metrics=True)
    assert plain[0]["pred_loo_r_sq"] > metrics[0]["pred_loo_r_sq"]
    assert plain[0]["pred_loo_mae"] < metrics[0]["pred_loo_mae"]


@pytest.mark.parametrize("exclude", [False, True])
def test_pred_metrics_lag_and_error_match_manual_refits(columbus, exclude):
    df, w = columbus
    results, metrics = slm(df, "HOVAL", ["INC", "CRIME"], w, model=["slm", "sem"], verbose=False,
                           pred_metrics=True, pred_exclude_neighbors=exclude)
    y = results[0].y
    x = df[["INC", "CRIME"]].to_numpy()
    wy = w.sparse @ y
    errors = {"slm": [], "sem": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(y)):
            drop = {i} | (set(w.neighbors[i]) if exclude else set())
            train = np.array([k for k in range(len(y)) if k not in drop])
            wt = w_subset_rows(w, train)
            design = np.hstack([np.ones((len(y), 1)), x])
            lag = spreg.ML_Lag(y[train], x[train], wt)
            err_ = spreg.ML_Error(y[train], x[train], wt)
            p_lag = design[i] @ lag.betas[:3, 0] + lag.rho * wy[i, 0]  # rho * (Wy)_i + x_i b
            p_err = design[i] @ err_.betas[:3, 0]  # trend only
            errors["slm"].append(y[i, 0] - p_lag)
            errors["sem"].append(y[i, 0] - p_err)
    for idx, name in enumerate(["slm", "sem"]):
        for k, v in _loo_summary(errors[name], y.ravel()).items():
            assert metrics[idx][k] == pytest.approx(v, rel=1e-6), (name, k)


def test_pred_metrics_output_and_report(columbus, capsys):
    df, w = columbus
    results, metrics = slm(df, "HOVAL", ["INC", "CRIME"], w, model=["ols", "slm", "sem"],
                           verbose=False, pred_metrics=True)
    assert len(metrics) == 3
    for m in metrics:
        assert {"pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad", "r_sq", "mae"} <= set(m)
        assert -1.0 <= m["pred_loo_r_sq"] <= 1.0
        assert m["pred_loo_mae"] > 0 and m["pred_loo_mad"] >= 0
        assert isinstance(m["pred_loo_r_sq"], float)
    # reported in the model summary, and absent when not requested
    rep = slm_report(results, metrics)
    for t in rep:
        s_ = t.iloc[0]["model"]
        assert "R²pred = " in s_ and "MAEpred = " in s_ and "MADpred = " in s_
    _, plain = slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False)
    assert "pred_loo_r_sq" not in plain[0]
    assert "R²pred" not in slm_report(results[:1], plain)[0].iloc[0]["model"]
    # the refits are silent
    capsys.readouterr()
    slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False, pred_metrics=True)
    assert capsys.readouterr().out == ""


def test_pred_metrics_islands_warning(columbus):
    df, w = columbus
    with pytest.warns(UserWarning, match="islands"):
        slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False,
            pred_metrics=True, pred_exclude_neighbors=True)


def test_pred_metrics_errors_and_failures(columbus, monkeypatch):
    df, w = columbus
    with pytest.raises(ValueError, match="requires spatial weights"):
        slm(df, "HOVAL", ["INC", "CRIME"], model="ols", verbose=False,
            pred_metrics=True, pred_exclude_neighbors=True)
    results, _ = slm(df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False)
    with pytest.raises(ValueError, match="Weights have"):
        pred_metrics(results, "ols", df["HOVAL"][:-1], df[["INC", "CRIME"]][:-1], w)

    # when every refit fails: NaN metrics and a warning, not an exception
    def broken(*args, **kwargs):
        raise RuntimeError("no convergence")
        yield  # pragma: no cover

    monkeypatch.setattr(slm_module, "fit_model", broken)
    with pytest.warns(UserWarning, match="no convergence"):
        out = pred_metrics(results, "ols", df["HOVAL"], df[["INC", "CRIME"]], w)
    assert all(np.isnan(v) for v in out[0].values())


def test_slm_na_rows_keep_weights_aligned(columbus):
    df, w = columbus
    df2 = df.copy()
    df2.loc[[3, 17], "INC"] = np.nan
    with pytest.warns(UserWarning, match="NAs"):
        results, metrics = slm(df2, "HOVAL", ["INC", "CRIME"], w, model=["ols", "slm"],
                               verbose=False, pred_metrics=True)
    assert metrics[0]["n_obs"] == 47
    # identical to fitting on the complete rows with the correspondingly subset weights
    keep = np.flatnonzero(df2.notna().all(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref, ref_m = slm(df.iloc[keep], "HOVAL", ["INC", "CRIME"], w_subset_rows(w, keep),
                         model=["ols", "slm"], verbose=False, pred_metrics=True)
    assert results[1].rho == pytest.approx(ref[1].rho)
    for k in ("r_sq", "pred_loo_r_sq", "pred_loo_mae"):
        assert metrics[1][k] == pytest.approx(ref_m[1][k]), k
    # the same holds for the formula interface (patsy drops the NA rows)
    with pytest.warns(UserWarning, match="NAs"):
        res_f, _ = slm(df2, w=w, model="slm", formula="HOVAL ~ 1 + INC + CRIME", verbose=False)
    assert res_f[0].rho == pytest.approx(ref[1].rho)


if __name__ == "__main__":
    test_slm()
    print("Tests for apatools.slm are PASSED!")
