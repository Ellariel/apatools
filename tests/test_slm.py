import warnings

import libpysal
import numpy as np
import pandas as pd
import pytest
import spreg
from statsmodels.stats.diagnostic import het_breuschpagan

import apatools.slm as slm_module
from apatools._common import w_subset_rows
from apatools.slm import (
    _moran_permutation,
    pred_metrics,
    slm,
    slm_report,
    slm_tests,
)


def s(x):
    return f"{x:.2f}"


def test_slm():
    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    cols = ["HOVAL", "INC", "CRIME"]
    df = pd.DataFrame([db.by_col(c) for c in cols], index=cols).T
    assert len(df) == 49
    y, x = cols[0], cols[1:]
    f = "HOVAL ~ 1 + INC + CRIME"

    # OLS
    results, metrics = slm(
        df,
        y,
        x,
        model="ols",
        standardize=False,
        ols_model_spat_diag=False,
        ols_model_nonspat_diag=True,
    )
    print(metrics)
    assert s(results[0].betas[0][0]) == "46.43"

    results, metrics = slm(
        df,
        y,
        x,
        w,
        model="ols",  # spatial diag
        standardize=False,
        ols_model_spat_diag=True,
        ols_model_nonspat_diag=True,
    )
    print(metrics)
    assert s(results[0].betas[0][0]) == "46.43"

    results, metrics = slm(
        df,
        model="ols",
        formula=f,  # formula test
        standardize=False,
        ols_model_spat_diag=False,
        ols_model_nonspat_diag=True,
    )
    print(metrics)
    assert s(results[0].betas[0][0]) == "46.43"

    # SLM
    results, metrics = slm(
        df,
        y,
        x,
        w,
        model="slm",
        standardize=False,
        ols_model_spat_diag=True,
        ols_model_nonspat_diag=True,
    )
    print(metrics)
    assert s(results[0].betas[0][0]) == "37.27"

    # SEM
    results, metrics = slm(
        df,
        y,
        x,
        w,
        model="sem",
        standardize=False,
        ols_model_spat_diag=True,
        ols_model_nonspat_diag=True,
    )
    print(metrics)
    assert s(results[0].betas[0][0]) == "48.01"

    results, metrics = slm(
        df,
        w=w,
        model="sem",
        formula=f,  # formula test
        standardize=False,
        ols_model_spat_diag=True,
        ols_model_nonspat_diag=True,
    )
    print(metrics)
    assert s(results[0].betas[0][0]) == "48.01"

    results_rep = slm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)
    assert s(metrics[0]["r_sq"]) == "0.35"

    results, metrics = slm(
        df,
        w=w,
        model=["sem", "ols", "slm"],
        formula=f,  # formula test
        standardize=False,
        ols_model_spat_diag=True,
        ols_model_nonspat_diag=True,
    )

    results_rep = slm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)

    assert (
        results_rep[0].iloc[0]["model"]
        == "R² = .35, R²adj = .31, F(3, 45) = 8.06, p < .001, AIC = 406.5, BIC = 414.0, LL = -199.2, MAE = 11.440, MAD = 8.518, N = 49"
    )

    assert (
        results_rep[1].iloc[0]["model"]
        == "R² = .35, R²adj = .32, F(2, 46) = 12.36, p < .001, AIC = 408.7, BIC = 414.4, LL = -201.4, MAE = 11.268, MAD = 6.869, N = 49"
    )

    assert (
        results_rep[2].iloc[0]["model"]
        == "R² = .38, R²adj = .34, F(3, 45) = 9.29, p < .001, AIC = 408.9, BIC = 416.4, LL = -200.4, MAE = 10.943, MAD = 7.149, N = 49"
    )

    # 5/0


def test_slm_kwargs_routing_and_df():
    import pytest

    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    cols = ["HOVAL", "INC", "CRIME"]
    df = pd.DataFrame([db.by_col(c) for c in cols], index=cols).T

    # ols_model_* arguments must reach spreg: spatial diagnostics are only
    # computed when requested
    r_no, _ = slm(
        df,
        "HOVAL",
        ["INC", "CRIME"],
        w,
        model="ols",
        verbose=False,
        ols_model_spat_diag=False,
    )
    r_yes, _ = slm(
        df,
        "HOVAL",
        ["INC", "CRIME"],
        w,
        model="ols",
        verbose=False,
        ols_model_spat_diag=True,
    )
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
    assert metrics[0]["mad"] == pytest.approx(
        np.median(np.abs(resid - np.median(resid)))
    )

    # formula path works without verbose output
    r_f, m_f = slm(df, model="ols", formula="HOVAL ~ 1 + INC + CRIME", verbose=False)
    assert m_f[0]["r_sq"] == pytest.approx(metrics[0]["r_sq"])

    # unknown model name
    with pytest.raises(NotImplementedError):
        slm(df, "HOVAL", ["INC"], w, model="xyz", verbose=False)
    # neither formula nor x/y
    with pytest.raises(ValueError):
        slm(df, w=w, verbose=False)


@pytest.fixture(scope="module")
def columbus():
    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
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
    results, metrics = slm(
        df, "HOVAL", ["INC", "CRIME"], model="ols", verbose=False, pred_metrics=True
    )
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
    results, metrics = slm(
        df,
        "HOVAL",
        ["INC", "CRIME"],
        w,
        model="ols",
        verbose=False,
        pred_metrics=True,
        pred_exclude_neighbors=True,
    )
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
    _, plain = slm(
        df, "HOVAL", ["INC", "CRIME"], w, model="ols", verbose=False, pred_metrics=True
    )
    assert plain[0]["pred_loo_r_sq"] > metrics[0]["pred_loo_r_sq"]
    assert plain[0]["pred_loo_mae"] < metrics[0]["pred_loo_mae"]


@pytest.mark.parametrize("exclude", [False, True])
def test_pred_metrics_lag_and_error_match_manual_refits(columbus, exclude):
    df, w = columbus
    results, metrics = slm(
        df,
        "HOVAL",
        ["INC", "CRIME"],
        w,
        model=["slm", "sem"],
        verbose=False,
        pred_metrics=True,
        pred_exclude_neighbors=exclude,
    )
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
            p_lag = (
                design[i] @ lag.betas[:3, 0] + lag.rho * wy[i, 0]
            )  # rho * (Wy)_i + x_i b
            p_err = design[i] @ err_.betas[:3, 0]  # trend only
            errors["slm"].append(y[i, 0] - p_lag)
            errors["sem"].append(y[i, 0] - p_err)
    for idx, name in enumerate(["slm", "sem"]):
        for k, v in _loo_summary(errors[name], y.ravel()).items():
            assert metrics[idx][k] == pytest.approx(v, rel=1e-6), (name, k)


def test_pred_metrics_output_and_report(columbus, capsys):
    df, w = columbus
    results, metrics = slm(
        df,
        "HOVAL",
        ["INC", "CRIME"],
        w,
        model=["ols", "slm", "sem"],
        verbose=False,
        pred_metrics=True,
    )
    assert len(metrics) == 3
    for m in metrics:
        assert {"pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad", "r_sq", "mae"} <= set(
            m
        )
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
        slm(
            df,
            "HOVAL",
            ["INC", "CRIME"],
            w,
            model="ols",
            verbose=False,
            pred_metrics=True,
            pred_exclude_neighbors=True,
        )


def test_pred_metrics_errors_and_failures(columbus, monkeypatch):
    df, w = columbus
    with pytest.raises(ValueError, match="requires spatial weights"):
        slm(
            df,
            "HOVAL",
            ["INC", "CRIME"],
            model="ols",
            verbose=False,
            pred_metrics=True,
            pred_exclude_neighbors=True,
        )
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
        results, metrics = slm(
            df2,
            "HOVAL",
            ["INC", "CRIME"],
            w,
            model=["ols", "slm"],
            verbose=False,
            pred_metrics=True,
        )
    assert metrics[0]["n_obs"] == 47
    # identical to fitting on the complete rows with the correspondingly subset weights
    keep = np.flatnonzero(df2.notna().all(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref, ref_m = slm(
            df.iloc[keep],
            "HOVAL",
            ["INC", "CRIME"],
            w_subset_rows(w, keep),
            model=["ols", "slm"],
            verbose=False,
            pred_metrics=True,
        )
    assert results[1].rho == pytest.approx(ref[1].rho)
    for k in ("r_sq", "pred_loo_r_sq", "pred_loo_mae"):
        assert metrics[1][k] == pytest.approx(ref_m[1][k]), k
    # the same holds for the formula interface (patsy drops the NA rows)
    with pytest.warns(UserWarning, match="NAs"):
        res_f, _ = slm(
            df2, w=w, model="slm", formula="HOVAL ~ 1 + INC + CRIME", verbose=False
        )
    assert res_f[0].rho == pytest.approx(ref[1].rho)


def test_slm_report_renames_only_the_intercept(columbus):
    df, w = columbus
    d = df.rename(columns={"INC": "const_income"})
    results, metrics = slm(
        d, "HOVAL", ["const_income", "CRIME"], w, model="ols", verbose=False
    )
    assert list(slm_report(results, metrics)[0].index) == [
        "Intercept",
        "const_income",
        "CRIME",
    ]


if __name__ == "__main__":
    test_slm()
    print("Tests for apatools.slm are PASSED!")


def test_slm_intercept_option_does_not_change_the_model(columbus):
    df, w = columbus
    fits = [
        slm(
            df, "HOVAL", ["INC", "CRIME"], w, model="ols", intercept=flag, verbose=False
        )[0][0]
        for flag in (False, True)
    ]
    assert (
        fits[0].name_x == fits[1].name_x == ["CONSTANT", "INC", "CRIME"]
    )  # no duplicated constant
    assert fits[0].betas.ravel().tolist() == pytest.approx(
        fits[1].betas.ravel().tolist()
    )


def test_slm_formula_with_standardize(columbus):
    df, w = columbus
    with pytest.warns(UserWarning, match="z-transformation"):
        results, _ = slm(
            df,
            formula="HOVAL ~ INC + CRIME",
            w=w,
            model="ols",
            standardize="z",
            verbose=False,
        )
    z = (df[["HOVAL", "INC", "CRIME"]] - df[["HOVAL", "INC", "CRIME"]].mean()) / df[
        ["HOVAL", "INC", "CRIME"]
    ].std(ddof=0)
    expected = np.linalg.lstsq(
        np.column_stack([np.ones(len(z)), z[["INC", "CRIME"]]]), z["HOVAL"], rcond=None
    )[0]
    assert results[0].betas.ravel().tolist() == pytest.approx(
        expected.tolist(), abs=1e-8
    )


###############
# slm_tests   #
###############


@pytest.fixture(scope="module")
def columbus_fits(columbus):
    df, w = columbus
    results, _ = slm(
        df, "HOVAL", ["INC", "CRIME"], w, model=["ols", "slm", "sem"], verbose=False
    )
    return df, w, results


def test_slm_tests_ols_diagnostics_are_those_of_spreg(columbus_fits):
    df, w, results = columbus_fits
    tests = slm_tests(results, w)
    assert list(tests.columns) == [
        "test",
        "model",
        "statistic",
        "df",
        "p-value",
        "report",
    ]
    ols = tests[tests["model"] == "ols"].set_index("test")
    # the tests that spreg computes together with the OLS fit when asked for the spatial diagnostics
    ref = spreg.OLS(
        df[["HOVAL"]].to_numpy(),
        df[["INC", "CRIME"]].to_numpy(),
        w=w,
        spat_diag=True,
        moran=True,
        white_test=True,
    )
    expected = {
        "Moran's I (residuals)": (ref.moran_res[1], ref.moran_res[2], np.nan),
        "LM-lag": (*ref.lm_lag, 1),
        "LM-error": (*ref.lm_error, 1),
        "Robust LM-lag": (*ref.rlm_lag, 1),
        "Robust LM-error": (*ref.rlm_error, 1),
        "LM-SARMA": (*ref.lm_sarma, 2),
        "Jarque-Bera": (ref.jarque_bera["jb"], ref.jarque_bera["pvalue"], 2),
        # the regressors themselves explain the variance (R's bptest, statsmodels); spreg's default are their squares
        "Breusch-Pagan": (*het_breuschpagan(ref.u.ravel(), ref.x, robust=False)[:2], 2),
        "Koenker-Bassett": (
            *het_breuschpagan(ref.u.ravel(), ref.x, robust=True)[:2],
            2,
        ),
        "White": (ref.white["wh"], ref.white["pvalue"], 5),
    }
    assert list(ols.index) == list(expected)
    for name, (stat, p, df_) in expected.items():
        assert ols.loc[name, "statistic"] == pytest.approx(stat) and ols.loc[
            name, "p-value"
        ] == pytest.approx(p), name
        assert (np.isnan(ols.loc[name, "df"]) and np.isnan(df_)) or ols.loc[
            name, "df"
        ] == df_, name
    # the report strings are APA-style
    assert (
        ols.loc["LM-error", "report"]
        == f"LM-error(1) = {ref.lm_error[0]:.2f}, p = .046"
    )
    assert ols.loc["Jarque-Bera", "report"].endswith("p < .001")
    assert ols.loc["Moran's I (residuals)", "report"].startswith(
        "Moran's I = 0.207, z = 2.50, p = .012"
    )


def test_slm_tests_likelihood_ratio_and_wald_tests_of_the_spatial_parameters(
    columbus_fits,
):
    from scipy.stats import chi2, norm

    _, w, results = columbus_fits
    ols, lag, err = results
    tests = slm_tests(results, w).set_index(["model", "test"])
    for model, fit, name in (("slm", lag, "rho"), ("sem", err, "lambda")):
        lr = 2 * (fit.logll - ols.logll)
        assert tests.loc[(model, f"LR test of {name}"), "statistic"] == pytest.approx(
            lr
        )
        assert tests.loc[(model, f"LR test of {name}"), "p-value"] == pytest.approx(
            chi2.sf(lr, 1)
        )
        assert tests.loc[(model, f"LR test of {name}"), "df"] == 1
        z = tests.loc[(model, f"z test of {name}"), "statistic"]
        assert z == pytest.approx(fit.z_stat[-1][0])
        assert tests.loc[(model, f"z test of {name}"), "p-value"] == pytest.approx(
            2 * norm.sf(abs(z))
        )
    assert (
        tests.loc[("slm", "LR test of rho"), "report"]
        == "LR test of rho(1) = 1.86, p = .173"
    )
    # the lag / error model are the ones that remove the spatial dependence of the residuals
    ols_moran = tests.loc[("ols", "Moran's I (residuals)"), "statistic"]
    assert (
        tests.loc[("sem", "Moran's I (residuals)"), "p-value"]
        > 0.5
        > 0.05
        > tests.loc[("ols", "Moran's I (residuals)"), "p-value"]
    )
    assert (
        abs(tests.loc[("sem", "Moran's I (residuals)"), "statistic"]) < 0.05
        and ols_moran > 2
    )  # I of the OLS residuals is z
    # Jarque-Bera of the residuals of the ML models (df 2), from the residuals of the model
    from scipy.stats import jarque_bera

    assert tests.loc[("slm", "Jarque-Bera"), "statistic"] == pytest.approx(
        jarque_bera(np.asarray(lag.u).ravel()).statistic
    )


def test_slm_tests_options_and_edge_cases(columbus_fits):
    _, w, results = columbus_fits
    raw = slm_tests(results, w)
    pd.testing.assert_frame_equal(
        raw, slm_tests(results, w)
    )  # reproducible (seeded permutations)
    assert not raw["p-value"].isna().any() and raw["statistic"].dtype == float
    # decimals, stars and formatted p-values as in plm_tests
    formatted = slm_tests(results, w, format_pval=True, add_stars=True, decimal=3)
    assert formatted["statistic"].tolist() == raw["statistic"].round(3).tolist()
    assert formatted["p-value"].map(
        lambda v: isinstance(v, str)
    ).all() and "<.001" in set(formatted["p-value"])
    assert formatted.loc[formatted["test"] == "Jarque-Bera", "sig"].iloc[0] != ""
    # another seed changes only the permutation p-values
    other = slm_tests(results, w, seed=1)
    different = (other["p-value"] != raw["p-value"]).to_numpy()
    assert (
        set(raw.loc[different, "test"]) == {"Moran's I (residuals)"}
        and (raw.loc[different, "model"] != "ols").all()
    )
    # no permutations: no Moran's I for the ML models
    assert "Moran's I (residuals)" not in set(
        slm_tests(results, w, n_perm=0).query("model != 'ols'")["test"]
    )
    # without weights only the tests that need none, with a warning
    with pytest.warns(UserWarning, match="Without w"):
        nospatial = slm_tests(results)
    assert set(nospatial["test"]) == {
        "Jarque-Bera",
        "Breusch-Pagan",
        "Koenker-Bassett",
        "White",
        "z test of rho",
        "LR test of rho",
        "z test of lambda",
        "LR test of lambda",
    }
    # no OLS fit among the results: no likelihood-ratio tests
    assert not any(t.startswith("LR") for t in slm_tests(results[1:], w)["test"])
    # a repeated model gets a suffix; unsupported models are skipped with a warning
    twice = slm_tests([results[0], results[0]], w)
    assert set(twice["model"]) == {"ols", "ols_2"}
    with pytest.warns(UserWarning, match="not supported"):
        assert set(slm_tests([results[0], "not a model"], w)["model"]) == {"ols"}
    empty = slm_tests([], w, format_pval=True, add_stars=True, decimal=2)
    assert (
        empty.empty and list(empty.columns)[:2] == ["test", "model"] and "sig" in empty
    )


def test_moran_permutation_p_values_are_calibrated(columbus):
    _, w = columbus
    rng = np.random.default_rng(0)
    n = w.n
    # independent values: not significant in most cases; the observed I is the usual statistic
    p_values = [
        _moran_permutation(rng.normal(size=n), w, 199, seed=b)[1] for b in range(60)
    ]
    assert np.mean(np.array(p_values) < 0.05) < 0.15 and 0.3 < np.mean(p_values) < 0.7
    # values smoothed over the neighbours: strongly positive and significant
    smooth = rng.normal(size=n)
    for _ in range(3):
        smooth = w.sparse @ smooth
    moran, p = _moran_permutation(smooth, w, 499, seed=1)
    assert moran > 0.5 and p < 0.01
    # the statistic is Moran's I of the centred values
    centred = smooth - smooth.mean()
    assert moran == pytest.approx(
        (n / w.s0) * (centred @ (w.sparse @ centred)) / (centred @ centred)
    )
    assert _moran_permutation(smooth, w, 99, seed=3) == _moran_permutation(
        smooth, w, 99, seed=3
    )


def test_slm_tests_find_the_spatial_dependence_of_simulated_data():
    w = libpysal.weights.lat2W(14, 14)
    w.transform = "r"
    rng = np.random.default_rng(2)
    n = w.n
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    inverse = np.linalg.inv(np.eye(n) - 0.6 * w.full()[0])
    lag_data = pd.DataFrame(
        {"x": x, "y": inverse @ (1.0 + 0.5 * x + noise)}
    )  # spatial lag process
    err_data = pd.DataFrame(
        {"x": x, "y": 1.0 + 0.5 * x + inverse @ noise}
    )  # spatial error process
    for data, process in ((lag_data, "slm"), (err_data, "sem")):
        results, _ = slm(
            data, "y", ["x"], w, model=["ols", "slm", "sem"], verbose=False
        )
        tests = slm_tests(results, w, n_perm=199).set_index(["model", "test"])[
            "p-value"
        ]
        assert (
            tests[("ols", "Moran's I (residuals)")] < 1e-4
            and tests[("ols", "LM-lag")] < 1e-3
            and tests[("ols", "LM-error")] < 1e-3
        )
        assert (
            tests[(process, f"LR test of {'rho' if process == 'slm' else 'lambda'}")]
            < 1e-6
        )
        # the model of the true process leaves no spatial dependence in the residuals
        assert tests[(process, "Moran's I (residuals)")] > 0.01
    # the robust tests point to the true process (Anselin et al., 1996)
    lag_tests = slm_tests(
        slm(lag_data, "y", ["x"], w, model="ols", verbose=False)[0], w
    ).set_index("test")["p-value"]
    err_tests = slm_tests(
        slm(err_data, "y", ["x"], w, model="ols", verbose=False)[0], w
    ).set_index("test")["p-value"]
    assert (
        lag_tests["Robust LM-lag"] < lag_tests["Robust LM-error"]
        and err_tests["Robust LM-error"] < err_tests["Robust LM-lag"]
    )
