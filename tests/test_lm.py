import numpy as np
import pandas as pd
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
