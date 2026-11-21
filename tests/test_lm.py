import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from apatools.format import format_p
from apatools.lm import lm, lm_report, lm_tests


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


###############
# lm_tests    #
###############


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x1", "x2", "x3"])
    d["y"] = 1 + d.x1 - 0.5 * d.x2 + rng.normal(size=n)
    return d


def _ols(d, y="y"):
    return sm.OLS(d[y], sm.add_constant(d.drop(columns=y))).fit()


def test_lm_tests_agree_with_the_textbook_formulas():
    from scipy.stats import chi2, f, jarque_bera, shapiro, t

    d = _data(120, seed=3)
    res = _ols(d)
    tests = lm_tests([res], lags=1).set_index("test")
    assert list(lm_tests([res]).columns) == ["test", "model", "statistic", "df", "df_denom", "p-value", "report"]
    e, X, n, k = res.resid.to_numpy(), res.model.exog, 120, 4
    # Jarque-Bera: n / 6 * (S^2 + (K - 3)^2 / 4)
    skew, kurt = ((e - e.mean()) ** 3).mean() / e.std() ** 3, ((e - e.mean()) ** 4).mean() / e.std() ** 4
    jb = n / 6 * (skew**2 + (kurt - 3) ** 2 / 4)
    assert tests.loc["Jarque-Bera", "statistic"] == pytest.approx(jb) and tests.loc["Jarque-Bera", "p-value"] == pytest.approx(chi2.sf(jb, 2))
    assert tests.loc["Shapiro-Wilk", "statistic"] == pytest.approx(shapiro(e).statistic)
    assert tests.loc["Omnibus", "df"] == 2 and 0 < tests.loc["Omnibus", "p-value"] < 1

    def n_r2(y, design):  # n * R^2 of the auxiliary regression
        return len(y) * sm.OLS(y, design).fit().rsquared

    bp = n_r2(e**2, X)  # Koenker's studentized Breusch-Pagan: n R^2 of e^2 on the regressors
    assert tests.loc["Breusch-Pagan", "statistic"] == pytest.approx(bp) and tests.loc["Breusch-Pagan", "df"] == k - 1
    assert tests.loc["Breusch-Pagan", "p-value"] == pytest.approx(chi2.sf(bp, k - 1))
    first, second = np.triu_indices(k)
    aux = X[:, first] * X[:, second]  # the regressors, their squares and cross products
    white = n_r2(e**2, aux)
    assert tests.loc["White", "statistic"] == pytest.approx(white) and tests.loc["White", "df"] == k * (k + 1) // 2 - 1
    assert tests.loc["White", "p-value"] == pytest.approx(chi2.sf(white, k * (k + 1) // 2 - 1))
    # RESET: F test of the squared and cubed fitted values
    fitted = res.fittedvalues.to_numpy()
    unrestricted = sm.OLS(d["y"], np.column_stack([X, fitted**2, fitted**3])).fit()
    f_stat = ((res.ssr - unrestricted.ssr) / 2) / (unrestricted.ssr / (n - k - 2))
    assert tests.loc["RESET", "statistic"] == pytest.approx(f_stat) and tests.loc["RESET", "df"] == 2
    assert tests.loc["RESET", "df_denom"] == n - k - 2 and tests.loc["RESET", "p-value"] == pytest.approx(f.sf(f_stat, 2, n - k - 2))
    # Durbin-Watson and Breusch-Godfrey (one lag, the first lagged residual set to 0)
    assert tests.loc["Durbin-Watson", "statistic"] == pytest.approx(np.sum(np.diff(e) ** 2) / np.sum(e**2))
    assert np.isnan(tests.loc["Durbin-Watson", "p-value"])
    lagged = np.r_[0.0, e[:-1]]
    bg = n_r2(e, np.column_stack([X, lagged]))
    assert tests.loc["Breusch-Godfrey", "statistic"] == pytest.approx(bg) and tests.loc["Breusch-Godfrey", "df"] == 1
    # Cook's distance and the Bonferroni outlier test
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    s2 = e @ e / (n - k)
    cook = e**2 * h / (k * s2 * (1 - h) ** 2)
    assert tests.loc["Cook's distance", "statistic"] == pytest.approx(cook.max())
    assert f"({int((cook > 4 / n).sum())} above 4/n)" in tests.loc["Cook's distance", "report"]
    s2_i = ((n - k) * s2 - e**2 / (1 - h)) / (n - k - 1)  # variance without the observation
    student = e / np.sqrt(s2_i * (1 - h))
    assert tests.loc["Outlier test", "statistic"] == pytest.approx(np.abs(student).max())
    assert tests.loc["Outlier test", "p-value"] == pytest.approx(min(1.0, n * 2 * t.sf(np.abs(student).max(), n - k - 1)))
    assert tests.loc["Outlier test", "df_denom"] == n - k - 1
    reference = res.outlier_test(method="bonf")  # statsmodels (refits the model for every observation)
    assert np.allclose(student, reference["student_resid"]) and tests.loc["Outlier test", "p-value"] == pytest.approx(reference["bonf(p)"].min())
    # multicollinearity: VIF = 1 / (1 - R^2) of a predictor on the others; condition number of the design matrix
    vifs = [1 / (1 - sm.OLS(d[c], sm.add_constant(d[["x1", "x2", "x3"]].drop(columns=c))).fit().rsquared) for c in ("x1", "x2", "x3")]
    assert tests.loc["Max VIF", "statistic"] == pytest.approx(max(vifs))
    assert tests.loc["Condition number", "statistic"] == pytest.approx(np.linalg.cond(X))
    assert tests.loc[["Max VIF", "Condition number", "Durbin-Watson", "Cook's distance"], "p-value"].isna().all()  # no tests
    # the report strings are APA-style
    assert tests.loc["Breusch-Pagan", "report"] == f"Breusch-Pagan(3) = {bp:.2f}, {format_p(tests.loc['Breusch-Pagan', 'p-value'])}"
    assert tests.loc["RESET", "report"].startswith(f"RESET F(2, {n - k - 2}) = ")
    assert tests.loc["Max VIF", "report"] == f"max VIF = {max(vifs):.2f}"


def test_lm_tests_have_the_right_size_and_find_the_departures():
    rng = np.random.default_rng(11)
    n, sims = 150, 250
    rejections = {}

    def run(make, tests, level=0.05, count=60):  # count: number of simulated samples (many for the size)
        counts = dict.fromkeys(tests, 0)
        for _ in range(count):
            table = lm_tests([_ols(make())], lags=1).set_index("test")["p-value"]
            for name in tests:
                counts[name] += table[name] < level
        return {name: c / count for name, c in counts.items()}

    def frame(y_of, x=None):
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y_of(x1, x2)})

    names = ["Jarque-Bera", "Shapiro-Wilk", "Breusch-Pagan", "White", "RESET", "Breusch-Godfrey"]
    size = run(lambda: frame(lambda a, b: 1 + a + b + rng.normal(size=n)), names, count=sims)
    assert all(0.015 <= rate <= 0.10 for rate in size.values()), size  # about 5 % false positives under H0
    # a variance that changes with the predictor (sd = exp(x1 / 2)): BP and White reject
    hetero = run(lambda: frame(lambda a, b: 1 + a + b + np.exp(a / 2) * rng.normal(size=n)), ["Breusch-Pagan", "White"])
    assert min(hetero.values()) > 0.7, hetero
    # sd proportional to |x1|: the variance is not linear in x1, so only White (squares) has power
    curved_variance = run(lambda: frame(lambda a, b: 1 + a + b + np.abs(a) * 2 * rng.normal(size=n)), ["Breusch-Pagan", "White"])
    assert curved_variance["White"] > 0.9 and curved_variance["Breusch-Pagan"] < curved_variance["White"] - 0.3, curved_variance
    # skewed / heavy tailed errors: the normality tests reject
    heavy = run(lambda: frame(lambda a, b: 1 + a + b + rng.standard_t(3, size=n)), ["Jarque-Bera", "Shapiro-Wilk"])
    assert min(heavy.values()) > 0.7, heavy
    # an omitted quadratic term: RESET rejects
    curved = run(lambda: frame(lambda a, b: 1 + a + b + 0.8 * a**2 + rng.normal(size=n)), ["RESET"])
    assert curved["RESET"] > 0.8, curved
    # AR(1) errors: Breusch-Godfrey rejects, Durbin-Watson is well below 2
    def ar1(a, b):
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.7 * e[i - 1] + rng.normal()
        return 1 + a + b + e

    auto = run(lambda: frame(ar1), ["Breusch-Godfrey"])
    assert auto["Breusch-Godfrey"] > 0.9, auto
    dw = lm_tests([_ols(frame(ar1))], lags=1).set_index("test").loc["Durbin-Watson", "statistic"]
    assert dw < 1.2


def test_lm_tests_outliers_and_collinearity():
    d = _data(100, seed=5)
    clean = lm_tests([_ols(d)]).set_index("test")
    assert clean.loc["Outlier test", "p-value"] > 0.05 and clean.loc["Max VIF", "statistic"] < 2
    d.loc[17, "y"] += 12  # a gross outlier
    dirty = lm_tests([_ols(d)]).set_index("test")
    assert dirty.loc["Outlier test", "p-value"] < 1e-5 and "observation 17" in dirty.loc["Outlier test", "report"]
    assert dirty.loc["Cook's distance", "statistic"] > 5 * clean.loc["Cook's distance", "statistic"]
    # nearly collinear predictors
    d["x3"] = d["x1"] + 0.02 * np.random.default_rng(0).normal(size=100)
    collinear = lm_tests([_ols(d)]).set_index("test")
    assert collinear.loc["Max VIF", "statistic"] > 100 and collinear.loc["Condition number", "statistic"] > 30
    # a single predictor has a VIF of 1
    assert lm_tests([sm.OLS(d["y"], sm.add_constant(d[["x1"]])).fit()]).set_index("test").loc["Max VIF", "statistic"] == 1.0


def test_lm_tests_options_and_edge_cases():
    d = _data(100, seed=8)
    results, _ = lm(d, "y", ["x1", "x2", "x3"], model=["ols", "qlm"], qlm_fit_q=0.5, verbose=False)
    with pytest.warns(UserWarning, match="QuantReg models are not supported"):
        raw = lm_tests(results)
    assert set(raw["model"]) == {"ols"}  # only the OLS model
    assert not {"Breusch-Godfrey", "Durbin-Watson"} & set(raw["test"])  # autocorrelation only with lags
    assert {"Breusch-Godfrey", "Durbin-Watson"} <= set(lm_tests(results[:1], lags=2)["test"])
    assert lm_tests(results[:1], lags=2).set_index("test").loc["Breusch-Godfrey", "df"] == 2
    pd.testing.assert_frame_equal(raw, lm_tests(results[:1]))  # deterministic
    assert raw["statistic"].dtype == float and raw["df"].dtype == float
    # formatted p-values, stars, decimals as in plm_tests
    formatted = lm_tests(results[:1], format_pval=True, add_stars=True, decimal=2)
    assert formatted["statistic"].tolist() == raw["statistic"].round(2).tolist()
    assert formatted["p-value"].map(lambda v: isinstance(v, str)).all() and "sig" in formatted
    assert formatted.loc[formatted["test"] == "Max VIF", "p-value"].iloc[0] == ""  # no p-value, no text
    # reset_power: squares only
    assert lm_tests(results[:1], reset_power=2).set_index("test").loc["RESET", "df"] == 1
    # a repeated model gets a suffix, an empty list gives an empty table with the columns
    assert set(lm_tests([results[0], results[0]])["model"]) == {"ols", "ols_2"}
    empty = lm_tests([], format_pval=True, add_stars=True, decimal=2)
    assert empty.empty and list(empty.columns)[:2] == ["test", "model"]
    # a model without constant: the tests that need none are computed, the others are skipped with a warning
    no_constant, _ = lm(d, "y", ["x1", "x2"], intercept=False, verbose=False)
    with pytest.warns(UserWarning):
        table = lm_tests(no_constant)
    assert {"Jarque-Bera", "RESET", "Outlier test"} <= set(table["test"])
    # results of the formula interface
    by_formula, _ = lm(d, formula="y ~ x1 + x2", verbose=False)
    assert len(lm_tests(by_formula)) >= 10


def test_lm_tests_work_with_arrays_and_label_the_observations_by_position():
    rng = np.random.default_rng(4)
    x = sm.add_constant(rng.normal(size=(60, 2)))
    y = x @ [1.0, 1.0, -1.0] + rng.normal(size=60)
    y[11] += 10
    table = lm_tests([sm.OLS(y, x).fit()]).set_index("test")
    assert "(observation 11)" in table.loc["Outlier test", "report"]
    # with pandas data the label is the index
    labelled = pd.Series(y, index=[f"row{i}" for i in range(60)])
    assert "(observation row11)" in lm_tests([sm.OLS(labelled, x).fit()]).set_index("test").loc["Outlier test", "report"]


###################################
# lm_tests: GLM, RLM and MixedLM  #
###################################


def _glm(y, x, family, **kwargs):
    return sm.GLM(y, sm.add_constant(pd.DataFrame({"x": x})), family=family, **kwargs).fit()


def _poisson_fit(rng, n=200, over=0.0, curve=0.0):
    x = rng.normal(size=n)
    mu = np.exp(0.3 + 0.4 * x + curve * x**2)
    y = rng.poisson(mu * rng.gamma(1 / over, over, size=n)) if over else rng.poisson(mu)
    return _glm(y, x, sm.families.Poisson())


def _logit_fit(rng, n=200, curve=0.0):
    x = rng.normal(size=n)
    eta = -0.2 + 0.8 * x + curve * x**2
    return _glm(rng.binomial(1, 1 / (1 + np.exp(-eta))), x, sm.families.Binomial())


def test_glm_diagnostics_agree_with_their_formulas():
    from scipy.stats import chi2, norm

    from apatools.lm import _glm_link_test, _hosmer_lemeshow, _poisson_overdispersion, _randomized_quantile_residuals

    rng = np.random.default_rng(2)
    res = _poisson_fit(rng, n=200)
    table = lm_tests([res]).set_index("test")
    y, mu = np.asarray(res.model.endog, float), np.asarray(res.mu, float)
    n = len(y)
    assert list(lm_tests([res])["model"].unique()) == ["glm"]
    # dispersion, Pearson and deviance goodness of fit
    pearson = np.sum((y - mu) ** 2 / mu)
    deviance = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
    assert table.loc["Dispersion", "statistic"] == pytest.approx(pearson / (n - 2))
    assert table.loc["Pearson GOF", "statistic"] == pytest.approx(pearson) and table.loc["Pearson GOF", "df"] == n - 2
    assert table.loc["Pearson GOF", "p-value"] == pytest.approx(chi2.sf(pearson, n - 2))
    assert table.loc["Deviance GOF", "statistic"] == pytest.approx(deviance)
    # Cameron & Trivedi: t statistic of mu in the regression of ((y - mu)^2 - y) / mu on mu
    z = ((y - mu) ** 2 - y) / mu
    slope = (mu @ z) / (mu @ mu)
    se = np.sqrt(((z - slope * mu) ** 2).sum() / (n - 1) / (mu @ mu))
    assert table.loc["Overdispersion", "statistic"] == pytest.approx(slope / se) and table.loc["Overdispersion", "df_denom"] == n - 1
    assert _poisson_overdispersion(res)[0] == pytest.approx(slope / se)
    # Pregibon's link test: the squared linear predictor added to the model
    eta = res.model.exog @ res.params.to_numpy()
    refit = sm.GLM(y, np.column_stack([res.model.exog, eta**2]), family=sm.families.Poisson()).fit()
    assert table.loc["Link test", "statistic"] == pytest.approx(refit.tvalues[-1])
    assert table.loc["Link test", "p-value"] == pytest.approx(2 * norm.sf(abs(refit.tvalues[-1])))
    # leverage = diagonal of the weighted hat matrix; Cook's distance from statsmodels' GLMInfluence
    w = mu
    h = np.einsum("ij,jk,ik->i", res.model.exog * np.sqrt(w)[:, None], np.linalg.inv((res.model.exog * w[:, None]).T @ res.model.exog), res.model.exog * np.sqrt(w)[:, None])
    assert table.loc["Max leverage", "statistic"] == pytest.approx(h.max())
    assert table.loc["Cook's distance", "statistic"] == pytest.approx(res.get_influence().cooks_distance[0].max())
    # Hosmer-Lemeshow for binary responses, against a rank-based grouping (n divisible by 10)
    logit = _logit_fit(rng, n=200)
    expected, observed = np.asarray(logit.mu), np.asarray(logit.model.endog)
    ranks = np.argsort(np.argsort(expected, kind="stable"), kind="stable")
    group = ranks * 10 // 200
    stat = sum((observed[group == g].sum() - expected[group == g].sum()) ** 2 / (expected[group == g].sum() * (1 - expected[group == g].mean())) for g in range(10))
    hl = lm_tests([logit]).set_index("test").loc["Hosmer-Lemeshow"]
    assert hl["statistic"] == pytest.approx(stat) and hl["df"] == 8 and hl["p-value"] == pytest.approx(chi2.sf(stat, 8))
    assert _hosmer_lemeshow(observed, expected)[0] == pytest.approx(stat)
    assert not {"Pearson GOF", "Deviance GOF", "Overdispersion"} & set(lm_tests([logit])["test"])  # not for binary data
    # randomized quantile residuals: Gaussian ones are the standardized residuals, discrete ones lie in the cdf jump
    gauss = _glm(rng.normal(size=100) + 2, rng.normal(size=100), sm.families.Gaussian())
    assert _randomized_quantile_residuals(gauss, 1) == pytest.approx(np.asarray(gauss.resid_response) / np.sqrt(gauss.scale), abs=1e-6)
    from scipy.stats import poisson

    quantile = _randomized_quantile_residuals(res, 5)
    assert np.all(norm.cdf(quantile) <= poisson.cdf(y, mu) + 1e-9) and np.all(norm.cdf(quantile) >= poisson.cdf(y - 1, mu) - 1e-9)
    assert _randomized_quantile_residuals(res, 5).tolist() == quantile.tolist() != _randomized_quantile_residuals(res, 6).tolist()
    assert _glm_link_test(res)[0] == pytest.approx(refit.tvalues[-1])


def test_glm_tests_have_the_right_size_and_find_misspecification():
    rng = np.random.default_rng(7)

    def rate(make, tests, level=0.05, count=150):
        hits = dict.fromkeys(tests, 0)
        for seed in range(count):
            table = lm_tests([make()], seed=seed).set_index("test")["p-value"]  # a new draw for the quantile residuals
            for name in tests:
                hits[name] += table[name] < level
        return {name: v / count for name, v in hits.items()}

    poisson_tests = ["Overdispersion", "Pearson GOF", "Link test"]
    assert all(0.015 <= v <= 0.11 for v in rate(lambda: _poisson_fit(rng), poisson_tests).values())
    assert rate(lambda: _poisson_fit(rng, over=0.5), ["Overdispersion"], count=60)["Overdispersion"] > 0.9
    assert rate(lambda: _poisson_fit(rng, curve=0.3), ["Link test"], count=60)["Link test"] > 0.9
    logit_tests = ["Hosmer-Lemeshow", "Link test"]
    assert all(0.015 <= v <= 0.11 for v in rate(lambda: _logit_fit(rng), logit_tests).values())
    assert min(rate(lambda: _logit_fit(rng, n=400, curve=0.8), logit_tests, count=60).values()) > 0.8
    # normality of the quantile residuals: right size (conservative for discrete data), power against a wrong family
    quantile = "Shapiro-Wilk (quantile residuals)"
    assert 0.01 <= rate(lambda: _poisson_fit(rng), [quantile])[quantile] <= 0.10
    assert 0.01 <= rate(lambda: _logit_fit(rng), [quantile])[quantile] <= 0.10
    assert rate(lambda: _poisson_fit(rng, n=300, over=0.5), [quantile], count=60)[quantile] > 0.8

    def gamma_fit(shape_error=False):
        x = rng.normal(size=200)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.gamma(2.0, mu / 2.0) if not shape_error else rng.lognormal(np.log(mu), 1.2)  # not a gamma distribution
        return _glm(y, x, sm.families.Gamma(sm.families.links.Log()))

    def gaussian_fit():
        x = rng.normal(size=200)
        return _glm(1 + x + rng.normal(size=200), x, sm.families.Gaussian())

    def nbinom_fit():
        x = rng.normal(size=200)
        mu = np.exp(0.3 + 0.4 * x)
        return _glm(rng.negative_binomial(2.0, 2.0 / (2.0 + mu)), x, sm.families.NegativeBinomial(alpha=0.5))

    def inverse_gaussian_fit():
        x = rng.normal(size=200)
        mu = np.exp(0.5 + 0.3 * x)
        return _glm(rng.wald(mu, 4.0), x, sm.families.InverseGaussian(sm.families.links.Log()))

    for name, make in (("Gamma", gamma_fit), ("Gaussian", gaussian_fit), ("NegativeBinomial", nbinom_fit), ("InverseGaussian", inverse_gaussian_fit)):
        assert rate(make, [quantile], count=100)[quantile] <= 0.13, name
    assert rate(lambda: gamma_fit(True), [quantile], count=60)[quantile] > 0.8  # log-normal data are not gamma


def test_glm_tests_cover_families_offsets_weights_and_grouped_data():
    rng = np.random.default_rng(9)
    n = 120
    x = rng.normal(size=n)
    exposure = rng.uniform(1, 3, size=n)
    y = rng.poisson(exposure * np.exp(0.2 + 0.4 * x))
    with_offset = lm_tests([_glm(y, x, sm.families.Poisson(), offset=np.log(exposure))]).set_index("test")
    assert np.isfinite(with_offset["statistic"]).all() and 0 < with_offset.loc["Link test", "p-value"] < 1
    # the Gaussian family with identity link: no goodness of fit / overdispersion tests, but a link test and normality
    gaussian = lm_tests([_glm(1 + x + rng.normal(size=n), x, sm.families.Gaussian())])
    assert {"Dispersion", "Link test", "Shapiro-Wilk (quantile residuals)", "Cook's distance"} <= set(gaussian["test"])
    assert not {"Pearson GOF", "Overdispersion", "Hosmer-Lemeshow"} & set(gaussian["test"])
    # grouped binomial data (successes, failures): goodness of fit tests, but no Hosmer-Lemeshow and no quantile residuals
    trials = rng.integers(5, 30, size=n)
    successes = rng.binomial(trials, 1 / (1 + np.exp(-(0.2 + 0.5 * x))))
    grouped = sm.GLM(np.column_stack([successes, trials - successes]), sm.add_constant(x), family=sm.families.Binomial()).fit()
    with pytest.warns(UserWarning, match="Normality of the quantile residuals could not be computed.*binary"):
        table = lm_tests([grouped])
    assert {"Pearson GOF", "Deviance GOF", "Link test"} <= set(table["test"]) and "Hosmer-Lemeshow" not in set(table["test"])
    # weights and unsupported families: the other diagnostics are still computed, with a warning for the residuals
    weighted = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson(), freq_weights=np.full(n, 2.0)).fit()
    with pytest.warns(UserWarning, match="freq_weights"):
        assert "Link test" in set(lm_tests([weighted])["test"])
    tweedie = sm.GLM(rng.gamma(2.0, 1.0, size=n) + 0.1, sm.add_constant(x), family=sm.families.Tweedie(var_power=1.5)).fit()
    with pytest.warns(UserWarning, match="Tweedie family is not supported"):
        assert {"Dispersion", "Link test"} <= set(lm_tests([tweedie])["test"])


def _rlm_data(n=150, seed=0, outliers=0, errors="normal"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    e = rng.normal(size=n) if errors == "normal" else rng.standard_t(3, size=n)
    y = 1 + x @ [1.0, -0.5] + e
    y[:outliers] += 10
    return y, sm.add_constant(x)


def test_rlm_diagnostics_are_descriptive_and_agree_with_their_definitions():
    y, X = _rlm_data(150, seed=1, outliers=6)
    rlm = sm.RLM(y, X).fit()
    ols = sm.OLS(y, X).fit()
    table = lm_tests([rlm]).set_index("test")
    assert list(table.index) == ["Min robust weight", "Scale ratio", "Coefficient change", "Max leverage", "Max VIF", "Condition number"]
    assert table["p-value"].isna().all() and (table["model"] == "rlm").all()  # descriptive: no p-values without approximate=True
    assert table.loc["Min robust weight", "statistic"] == pytest.approx(rlm.weights.min())
    assert f"{int((rlm.weights < 0.5).sum())} below .5)" in table.loc["Min robust weight", "report"] and (rlm.weights < 0.5).sum() >= 4
    assert table.loc["Scale ratio", "statistic"] == pytest.approx(rlm.scale / np.sqrt(ols.mse_resid)) and table.loc["Scale ratio", "statistic"] < 0.6
    change = np.abs(rlm.params - ols.params) / ols.bse
    assert table.loc["Coefficient change", "statistic"] == pytest.approx(change.max())
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    assert table.loc["Max leverage", "statistic"] == pytest.approx(np.diag(hat).max())
    # clean data: nothing is downweighted much and the scales agree
    y, X = _rlm_data(150, seed=1)
    clean = lm_tests([sm.RLM(y, X).fit()]).set_index("test")
    assert clean.loc["Scale ratio", "statistic"] > 0.85 and clean.loc["Min robust weight", "statistic"] > 0.4
    assert clean.loc["Coefficient change", "statistic"] < table.loc["Coefficient change", "statistic"]


def test_rlm_approximate_tests_are_only_given_on_request_and_are_labelled():
    y, X = _rlm_data(150, seed=2)
    rlm = sm.RLM(y, X).fit()
    default = set(lm_tests([rlm])["test"])
    approx = lm_tests([rlm], approximate=True, lags=1).set_index("test")
    extra = set(approx.index) - default
    assert extra == {"Breusch-Pagan (approx.)", "White (approx.)", "RESET (approx.)", "Breusch-Godfrey (approx.)", "Durbin-Watson"}
    assert all("approx." in r for t, r in approx["report"].items() if t.endswith("(approx.)"))
    assert approx.loc["Breusch-Pagan (approx.)", "df"] == 2 and approx.loc["White (approx.)", "df"] == 5
    assert approx.loc["RESET (approx.)", "df"] == 2  # powers 2 and 3 of the fitted values
    assert "Breusch-Godfrey (approx.)" not in set(lm_tests([rlm], approximate=True)["test"])  # only with lags
    assert lm_tests([rlm], approximate=True, reset_power=2).set_index("test").loc["RESET (approx.)", "df"] == 1
    # Breusch-Pagan on the robust residuals is n R^2 of the squared residuals on the regressors
    e = np.asarray(rlm.resid)
    manual = len(e) * sm.OLS(e**2, X).fit().rsquared
    assert approx.loc["Breusch-Pagan (approx.)", "statistic"] == pytest.approx(manual)


def test_rlm_approximate_tests_have_about_the_right_size():
    rng_seed = iter(range(1000, 2000))

    def rate(count=120, **kwargs):
        hits = {"Breusch-Pagan (approx.)": 0, "White (approx.)": 0, "RESET (approx.)": 0}
        for _ in range(count):
            y, X = _rlm_data(200, seed=next(rng_seed), **kwargs)
            table = lm_tests([sm.RLM(y, X).fit()], approximate=True).set_index("test")["p-value"]
            for name in hits:
                hits[name] += table[name] < 0.05
        return {name: v / count for name, v in hits.items()}

    for errors in ("normal", "t"):
        assert all(0.01 <= v <= 0.14 for v in rate(errors=errors).values()), errors
    # heteroskedastic errors and an omitted quadratic term are found
    def heteroskedastic(seed):
        y, X = _rlm_data(200, seed=seed)
        return y + (np.exp(X[:, 1] / 2) - 1) * np.random.default_rng(seed + 1).normal(size=200) * 2, X

    found = [lm_tests([sm.RLM(*heteroskedastic(s), ).fit()], approximate=True).set_index("test").loc["Breusch-Pagan (approx.)", "p-value"] < 0.05 for s in range(40)]
    assert np.mean(found) > 0.7


def _mlm_fit(seed=0, tau=0.8, groups=15, size=10, slope=False):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), size)
    x = rng.normal(size=len(g))
    y = 1 + 0.5 * x + (rng.normal(size=groups) * tau)[g] + rng.normal(size=len(g))
    if slope:
        y = y + (rng.normal(size=groups) * 0.5)[g] * x
    data = pd.DataFrame({"y": y, "x": x, "g": g})
    return sm.MixedLM(data["y"], sm.add_constant(data[["x"]]), groups=data["g"], exog_re=sm.add_constant(data[["x"]]) if slope else None).fit()


def test_mlm_diagnostics_agree_with_their_definitions():
    from scipy.stats import chi2, shapiro

    from apatools.lm import mlm_icc

    res = _mlm_fit(seed=3)
    table = lm_tests([res]).set_index("test")
    assert (table["model"] == "mlm").all()
    assert list(table.index) == [
        "ICC", "LR test of random effect", "Shapiro-Wilk (random effects, approx.)",
        "Shapiro-Wilk (conditional residuals, approx.)", "Cook's distance (groups)", "Max VIF", "Condition number",
    ]  # fmt: skip
    assert table.loc["ICC", "statistic"] == pytest.approx(mlm_icc(res))
    # likelihood-ratio test of the random intercept: ML fits, 50:50 mixture of chi-squares with 0 and 1 df
    ml = res.model.fit(reml=False)
    lr = 2 * (ml.llf - sm.OLS(res.model.endog, res.model.exog).fit().llf)
    assert table.loc["LR test of random effect", "statistic"] == pytest.approx(lr)
    assert table.loc["LR test of random effect", "p-value"] == pytest.approx(0.5 * chi2.sf(lr, 1))
    effects = [float(v.iloc[0]) for v in res.random_effects.values()]
    assert table.loc["Shapiro-Wilk (random effects, approx.)", "statistic"] == pytest.approx(shapiro(effects).statistic)
    assert table.loc["Shapiro-Wilk (conditional residuals, approx.)", "statistic"] == pytest.approx(shapiro(res.resid).statistic)
    # Cook's distance of the groups: refit without every group, (b - b_(g))' V^-1 (b - b_(g)) / k
    k = res.k_fe
    inverse = np.linalg.inv(np.asarray(res.cov_params())[:k, :k])
    data = pd.DataFrame({"y": res.model.endog, "g": np.asarray(res.model.groups)}).join(pd.DataFrame(res.model.exog, columns=["const", "x"]))
    distances = {}
    for g in data["g"].unique():
        part = data[data["g"] != g]
        fit = sm.MixedLM(part["y"], part[["const", "x"]], groups=part["g"]).fit()
        change = res.fe_params.to_numpy() - fit.fe_params.to_numpy()
        distances[g] = change @ inverse @ change / k
    assert table.loc["Cook's distance (groups)", "statistic"] == pytest.approx(max(distances.values()), rel=1e-4)
    worst = max(distances, key=distances.get)
    assert f"(group {worst};" in table.loc["Cook's distance (groups)", "report"]


def test_mlm_likelihood_ratio_test_has_the_right_size_and_power():
    def rate(tau, count):
        hits = 0
        for seed in range(count):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # max_refits=0: the group influence (a refit per group) is not needed here
                table = lm_tests([_mlm_fit(seed=500 + seed, tau=tau, groups=20, size=6)], max_refits=0)
            hits += table.set_index("test").loc["LR test of random effect", "p-value"] < 0.05
        return hits / count

    assert rate(0.0, 100) <= 0.12  # about 5 % false positives with the boundary correction
    assert rate(0.8, 40) > 0.75


def test_mlm_diagnostics_options_and_limits():
    res = _mlm_fit(seed=4)
    # a group influence cap
    with pytest.warns(UserWarning, match="skipped for 15 groups"):
        capped = lm_tests([res], max_refits=10)
    assert "Cook's distance (groups)" not in set(capped["test"]) and "LR test of random effect" in set(capped["test"])
    # random slopes: no likelihood-ratio test of a single variance, no ICC, but the other diagnostics
    with pytest.warns(UserWarning, match="exactly one random effect"):
        slopes = lm_tests([_mlm_fit(seed=5, slope=True)])
    assert {"ICC", "LR test of random effect"}.isdisjoint(slopes["test"])
    assert {"Shapiro-Wilk (conditional residuals, approx.)", "Cook's distance (groups)", "Max VIF"} <= set(slopes["test"])
    # variance components: no group influence
    rng = np.random.default_rng(6)
    data = pd.DataFrame({"y": rng.normal(size=120), "x": rng.normal(size=120), "g": np.repeat(range(12), 10), "h": np.tile(np.repeat(range(3), 4), 10)})
    vc = sm.MixedLM.from_formula("y ~ x", data, groups="g", re_formula="1", vc_formula={"h": "0 + C(h)"}).fit()
    with pytest.warns(UserWarning):
        table = lm_tests([vc])
    assert "Cook's distance (groups)" not in set(table["test"]) and "Max VIF" in set(table["test"])
    # the formatted table, and a mixed list of models
    ols, _ = lm(data, "y", ["x"], model=["ols", "rlm", "mlm"], mlm_model_groups="g", verbose=False)
    mixed = lm_tests(ols, format_pval=True, add_stars=True, decimal=2)
    assert set(mixed["model"]) == {"ols", "rlm", "mlm"} and mixed["p-value"].map(lambda v: isinstance(v, str)).all()
