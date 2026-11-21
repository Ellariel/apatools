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
    results, _ = lm(d, "y", ["x1", "x2", "x3"], model=["ols", "rlm", "glm"], verbose=False)
    with pytest.warns(UserWarning, match="RLM models are not supported"), pytest.warns(UserWarning, match="GLM models"):
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
