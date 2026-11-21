import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from apatools.lm import lm, lm_report


def s(x):
    return f"{x:.2f}"


def test_lm():
    test_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    assert len(test_data) == 32

    y = "mpg"
    x = ["wt", "vs"]
    results, metrics = lm(
        test_data,
        y,
        x,
        model=["ols", "rlm", "glm"],
        verbose=True,
        intercept=True,
        standardize=False,
        vif=True,
        base_metrics=True,
        pred_metrics=True,
        ols_fit_cov_type="HC1",
        rlm_model_M=sm.robust.norms.RamsayE(),
        glm_fit_cov_type="HC1",
        glm_model_family=sm.families.Gaussian(),
    )
    # OLS
    print(metrics[1])
    assert s(results[0].fvalue) == "40.58"
    assert s(results[0].params.iloc[1]) == "-4.44"
    assert s(metrics[0]["r_sq"]) == "0.80"
    assert s(metrics[0]["pred_loo_r_sq"]) == "0.75"
    # RLM
    assert s(results[1].params.iloc[1]) == "-4.44"
    assert s(metrics[1]["r_sq"]) == "0.82"
    assert s(metrics[1]["pred_loo_r_sq"]) == "0.75"
    # GLM
    assert s(results[2].pvalues.iloc[1]) == "0.00"
    assert s(metrics[2]["r_sq"]) == "0.97"
    assert s(metrics[2]["pred_loo_r_sq"]) == "0.75"

    results_rep = lm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)

    # lm_report OLS
    assert results_rep[0].loc["vs"]["p-value"] == ".002"
    assert s(results_rep[0].loc["vs"]["vif"]) == "1.44"
    assert (
        results_rep[0].iloc[0]["model"]
        == "R² = .80, R²adj = .79, R²pred = .75, F(2, 30) = 40.58, p < .001, AIC = 159.1, BIC = 163.5, LL = -76.5, MAE = 2.179, MAD = 1.818, MAEpred = 2.413, MADpred = 2.090, N = 32"
    )
    # lm_report RLM
    assert results_rep[1].loc["vs"]["p-value"] == ".017"
    assert s(results_rep[1].loc["vs"]["vif"]) == "1.44"
    assert (
        results_rep[1].iloc[0]["model"]
        == "R² = .82, R²adj = .82, R²pred = .75, F(2, 30) = 70.07, p < .001, MAE = 2.165, MAD = 1.865, MAEpred = 2.422, MADpred = 2.167, N = 32"
    )
    # lm_report GLM
    assert results_rep[2].loc["vs"]["p-value"] == ".001"
    assert s(results_rep[2].loc["vs"]["vif"]) == "1.44"
    assert (
        results_rep[2].iloc[0]["model"]
        == "R² = .97, R²adj = .97, R²pred = .75, F(2, 30) = 563.42, p < .001, AIC = 159.1, BIC = 163.5, LL = -76.5, MAE = 2.179, MAD = 1.818, MAEpred = 2.413, MADpred = 2.090, N = 32"
    )

    # test the formula param
    results, metrics = lm(
        test_data,
        formula="mpg ~ 1",  # UserWarning: A model has no parameters or zero degrees or freedom.
        model=["ols"],
        verbose=True,
        intercept=True,
        standardize=False,
        vif=False,
        base_metrics=True,
        pred_metrics=True,
        ols_fit_cov_type="HC1",
        rlm_model_M=sm.robust.norms.RamsayE(),
        glm_fit_cov_type="HC1",
        glm_model_family=sm.families.Gaussian(),
    )

    # OLS
    print(metrics[0])
    assert s(results[0].aic) == "206.76"

    results, metrics = lm(
        test_data,
        formula="mpg ~ 1 + wt + vs",
        model=["ols"],
        verbose=True,
        intercept=True,
        standardize=False,
        vif=True,
        base_metrics=True,
        pred_metrics=True,
        ols_fit_cov_type="HC1",
        rlm_model_M=sm.robust.norms.RamsayE(),
        glm_fit_cov_type="HC1",
        glm_model_family=sm.families.Gaussian(),
    )

    # OLS
    print(metrics[0])
    assert s(results[0].aic) == "159.09"
    assert s(results[0].fvalue) == "40.58"
    assert s(results[0].params.iloc[1]) == "-4.44"
    assert s(metrics[0]["r_sq"]) == "0.80"
    assert s(metrics[0]["pred_loo_r_sq"]) == "0.75"

    # test for standardazed and const options
    results, metrics = lm(
        test_data,
        formula="mpg ~ 1 + wt",
        model=["ols"],
        verbose=True,
        intercept=False,
        standardize=True,
        vif=False,
        base_metrics=True,
        pred_metrics=True,
        ols_fit_cov_type="HC1",
        rlm_model_M=sm.robust.norms.RamsayE(),
        glm_fit_cov_type="HC1",
        glm_model_family=sm.families.Gaussian(),
    )

    # OLS
    print(metrics[0])
    assert s(results[0].aic) == "50.09"
    assert results[0].params.index[0] == "Intercept"
    assert s(results[0].params.iloc[0]) == "-0.00"

    # MLM
    results, metrics = lm(
        test_data,
        formula="mpg ~ 1 + vs",
        model="mlm",
        verbose=True,
        intercept=False,
        standardize=False,
        vif=False,
        base_metrics=True,
        pred_metrics=True,
        mlm_model_groups="am",  # test_data["am"] tested below
    )
    results_rep = lm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)
    assert s(metrics[0]["r_sq"]) == "0.71"
    assert s(metrics[0]["pred_loo_r_sq"]) == "0.38"

    results, metrics = lm(
        test_data,
        formula="mpg ~ 1 + vs",
        model="mlm",
        verbose=True,
        intercept=False,
        standardize=False,
        vif=False,
        base_metrics=True,
        pred_metrics=True,
        mlm_model_groups=test_data["am"],
    )
    results_rep = lm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)
    assert s(metrics[0]["r_sq"]) == "0.71"
    assert s(metrics[0]["pred_loo_r_sq"]) == "0.38"
    # print(results_rep[0].to_string(), metrics)
    # 5/0


def test_lm_regressions():
    d = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    # formula path must work without verbose output
    results, metrics = lm(d, formula="mpg ~ wt + vs", model="ols", verbose=False)
    assert s(metrics[0]["r_sq"]) == "0.80"
    # MAD is the median absolute deviation of the residuals
    res = results[0].resid
    assert abs(metrics[0]["mad"] - np.median(np.abs(res - np.median(res)))) < 1e-9
    # leave-one-out R2 is a plain float
    _, m = lm(
        d,
        "mpg",
        ["wt", "vs"],
        model="ols",
        verbose=False,
        base_metrics=True,
        pred_metrics=True,
    )
    assert isinstance(m[0]["pred_loo_r_sq"], float)
    # NA rows with a grouping variable (MixedLM) stay aligned
    d2 = d.copy()
    d2.loc[d2.index[0], "wt"] = np.nan
    results, _ = lm(
        d2, "mpg", ["wt"], model="mlm", verbose=False, mlm_model_groups="cyl"
    )
    assert results[0].nobs == 31


def test_lm_report_names_the_coefficients_of_models_without_intercept():
    """Regression: the coefficient names used to become NaN when there was no intercept to rename."""
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x1": rng.normal(size=80), "x2": rng.normal(size=80)})
    d["y"] = 1.5 * d.x1 - d.x2 + rng.normal(size=80)
    for results, metrics in (
        lm(d, "y", ["x1", "x2"], intercept=False, verbose=False),
        lm(d, formula="y ~ x1 + x2 - 1", verbose=False),
    ):
        assert list(lm_report(results, metrics)[0].index) == ["x1", "x2"]
    # with an intercept it is renamed, to the name that is asked for
    results, metrics = lm(d, "y", ["x1", "x2"], verbose=False)
    assert list(lm_report(results, metrics)[0].index) == ["Intercept", "x1", "x2"]
    renamed = lm_report(results, metrics, intercept_name="(Intercept)")
    assert list(renamed[0].index) == ["(Intercept)", "x1", "x2"]
    # only the intercept is renamed, not variables whose names contain "const"
    d2 = d.rename(columns={"x1": "const_income", "x2": "myconst"})
    results, metrics = lm(d2, "y", ["const_income", "myconst"], verbose=False)
    assert list(lm_report(results, metrics)[0].index) == ["Intercept", "const_income", "myconst"]


if __name__ == "__main__":
    test_lm()
    print("Tests for apatools.lm are PASSED!")


def test_lm_formula_with_nas_keeps_the_groups_of_a_mixed_model_aligned():
    rng = np.random.default_rng(1)
    d = pd.DataFrame(rng.normal(size=(60, 2)), columns=["x1", "x2"])
    d["y"] = d.x1 + rng.normal(size=60)
    d["g"] = np.repeat(range(6), 10)
    d.loc[[3, 59], "x1"] = np.nan  # patsy drops these rows; the last one is the highest group index
    results, _ = lm(d, formula="y ~ x1 + x2", model="mlm", mlm_model_groups="g", verbose=False)
    clean = d.dropna()
    reference, _ = lm(clean, "y", ["x1", "x2"], model="mlm", mlm_model_groups="g", verbose=False)
    assert results[0].nobs == len(clean)
    assert results[0].params.tolist() == pytest.approx(reference[0].params.tolist(), abs=1e-6)
    # a Series of groups is aligned in the same way
    as_series, _ = lm(d, formula="y ~ x1 + x2", model="mlm", mlm_model_groups=d["g"], verbose=False)
    assert as_series[0].params.tolist() == pytest.approx(results[0].params.tolist())


def test_lm_vif_without_the_other_metrics():
    rng = np.random.default_rng(2)
    d = pd.DataFrame(rng.normal(size=(50, 2)), columns=["x1", "x2"])
    d["y"] = d.x1 + rng.normal(size=50)
    results, metrics = lm(d, "y", ["x1", "x2"], base_metrics=False, vif=True, verbose=False)
    assert len(metrics) == len(results) == 1 and list(metrics[0]) == ["vif"]
    full = lm(d, "y", ["x1", "x2"], vif=True, verbose=False)[1][0]["vif"]
    pd.testing.assert_frame_equal(metrics[0]["vif"], full)
    # without any metrics the list stays empty
    assert lm(d, "y", ["x1", "x2"], base_metrics=False, verbose=False)[1] == []


def test_lm_formula_with_standardize_gives_standardized_coefficients():
    rng = np.random.default_rng(3)
    d = pd.DataFrame(rng.normal(size=(80, 2)), columns=["x1", "x2"])
    d["y"] = 2 * d.x1 - d.x2 + rng.normal(size=80)
    with pytest.warns(UserWarning, match="intercept"):
        results, _ = lm(d, formula="y ~ x1 + x2", standardize="z", verbose=False)
    z = (d - d.mean()) / d.std(ddof=0)
    manual = sm.OLS(z["y"], sm.add_constant(z[["x1", "x2"]])).fit()
    assert results[0].params.tolist() == pytest.approx(manual.params.tolist(), abs=1e-8)
    assert abs(results[0].params.iloc[0]) < 1e-10


def test_mlm_icc_and_r_sq_use_the_same_random_effects_variance():
    import statsmodels.formula.api as smf

    from apatools.lm import mlm_icc, mlm_r_sq

    rng = np.random.default_rng(4)
    n_groups, per_group = 25, 12
    d = pd.DataFrame({"g": np.repeat(range(n_groups), per_group), "x": rng.normal(size=n_groups * per_group)})
    d["h"] = np.tile(np.repeat(range(3), 4), n_groups)  # a second, crossed grouping used as variance component
    d["y"] = (
        d.x + rng.normal(size=n_groups)[d.g] + rng.normal(size=3)[d.h] * 0.8 + rng.normal(size=len(d))
    )
    plain = smf.mixedlm("y ~ x", d, groups="g").fit()
    with_vc = smf.mixedlm("y ~ x", d, groups="g", re_formula="1", vc_formula={"h": "0 + C(h)"}).fit()

    # random-intercept model: the usual ICC
    tau2, sigma2 = float(plain.cov_re.iloc[0, 0]), plain.scale
    assert mlm_icc(plain) == pytest.approx(tau2 / (tau2 + sigma2))
    # variance components are part of the random-effects variance in both functions
    tau2_vc = float(with_vc.cov_re.iloc[0, 0]) + float(np.sum(with_vc.vcomp))
    assert float(np.sum(with_vc.vcomp)) > 0
    assert mlm_icc(with_vc) == pytest.approx(tau2_vc / (tau2_vc + with_vc.scale))
    r2_m, r2_c = mlm_r_sq(with_vc)
    var_fixed = np.var(with_vc.model.exog @ with_vc.fe_params.values)
    assert r2_c - r2_m == pytest.approx(tau2_vc / (var_fixed + tau2_vc + with_vc.scale))
