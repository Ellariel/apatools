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
        == "R² = .80, R²adj = .79, R²pred = .75, F(2, 30) = 40.58, p < .0001, AIC = 159.1, BIC = 163.5, LL = -76.5, MAE = 2.179, MAD = 2.317, MAEpred = 2.413, MADpred = 2.090, N = 32"
    )
    # lm_report RLM
    assert results_rep[1].loc["vs"]["p-value"] == ".017"
    assert s(results_rep[1].loc["vs"]["vif"]) == "1.44"
    assert (
        results_rep[1].iloc[0]["model"]
        == "R² = .82, R²adj = .81, R²pred = .75, F(2, 30) = 70.07, p < .0001, MAE = 2.165, MAD = 2.070, MAEpred = 2.422, MADpred = 2.167, N = 32"
    )
    # lm_report GLM
    assert results_rep[2].loc["vs"]["p-value"] == ".001"
    assert s(results_rep[2].loc["vs"]["vif"]) == "1.44"
    assert (
        results_rep[2].iloc[0]["model"]
        == "R² = .97, R²adj = .97, R²pred = .75, F(2, 30) = 563.42, p < .0001, AIC = 159.1, BIC = 163.5, LL = -76.5, MAE = 2.179, MAD = 2.317, MAEpred = 2.413, MADpred = 2.090, N = 32"
    )
    
    # test the formula param
    results, metrics = lm(
        test_data,
        formula = "mpg ~ 1", # UserWarning: A model has no parameters or zero degrees or freedom.
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
        formula = "mpg ~ 1 + wt + vs",
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
        formula = "mpg ~ 1 + wt",
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
        formula = "mpg ~ 1 + vs",
        model="mlm",
        verbose=True,
        intercept=False,
        standardize=False,
        vif=False,
        base_metrics=True,
        pred_metrics=True,
        mlm_model_groups="am", # test_data["am"] tested below
        
    )
    results_rep = lm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)
    assert s(metrics[0]["r_sq"]) == "0.71"
    assert s(metrics[0]["pred_loo_r_sq"]) == "0.38"
    
    results, metrics = lm(
        test_data,
        formula = "mpg ~ 1 + vs",
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
    

if __name__ == "__main__":
    test_lm()
    print("Tests for metatools.lm are PASSED!")
