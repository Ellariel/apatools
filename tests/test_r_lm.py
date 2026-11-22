"""lm, lm_report and lm_tests against R: base lm / glm, lmtest, MASS (rlm) and lme4 (lmer).

The tests are skipped without R and these packages. Conventions that differ from R are pinned where they occur.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from apatools.lm import lm, lm_report, lm_tests

from .r_reference import require_r, run_r

require_r("lmtest", "MASS", "lme4")

N, GROUPS = 240, 24


@pytest.fixture(scope="module", params=[3, 8], ids=lambda s: f"seed{s}")
def data(request):
    rng = np.random.default_rng(request.param)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=N),
            "x2": rng.normal(size=N) * 2,
            "g": np.repeat(np.arange(GROUPS), N // GROUPS),
        }
    )
    group_effect = rng.normal(size=GROUPS)[df["g"]]
    df["y"] = (
        1
        + 0.8 * df["x1"]
        + 0.4 * df["x2"]
        + (1 + 0.4 * np.abs(df["x1"])) * rng.normal(size=N)
        + group_effect
    )
    df["y"] += 4 * (rng.uniform(size=N) < 0.04)  # a few outliers for the robust model
    df["b"] = (df["y"] + rng.logistic(size=N) > df["y"].median()).astype(int)
    df["cnt"] = rng.poisson(np.exp(0.3 * df["x1"] + 0.5))
    return df


def fitted(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, metrics = lm(df, verbose=False, **kwargs)
    return results, metrics


def stat(tests, name, column="statistic"):
    return tests.set_index("test").loc[name, column]


@pytest.fixture(scope="module")
def ols_reference(data):
    script = """
f <- y ~ x1 + x2
o <- lm(f, d)
bpk <- bptest(o, studentize = TRUE); bp <- bptest(o, studentize = FALSE)
wh <- bptest(o, ~ x1 + x2 + I(x1^2) + I(x2^2) + x1:x2, data = d, studentize = TRUE)
rs <- resettest(o, power = 2:3, type = "fitted")
sw <- shapiro.test(resid(o))
e <- resid(o); m2 <- mean(e^2); jb <- length(e) / 6 * ((mean(e^3) / m2^1.5)^2 + (mean(e^4) / m2^2 - 3)^2 / 4)
bg <- bgtest(o, order = 2, type = "Chisq"); dw <- dwtest(o)
st <- rstudent(o); tmax <- max(abs(st)); pb <- min(1, 2 * length(e) * pt(-tmax, o$df.residual - 1))
X <- model.matrix(o); vifs <- sapply(1:2, function(j) 1 / (1 - summary(lm(X[, j + 1] ~ X[, -(j + 1)]))$r.squared))
res <- data.frame(
  test = c("bp", "bpk", "white", "reset", "shapiro", "jb", "bg", "dw", "outlier", "cook", "vif", "cond"),
  stat = c(bp$statistic, bpk$statistic, wh$statistic, rs$statistic, sw$statistic, jb, bg$statistic, dw$statistic, tmax, max(cooks.distance(o)), max(vifs), kappa(X, exact = TRUE)),
  p = c(bp$p.value, bpk$p.value, wh$p.value, rs$p.value, sw$p.value, pchisq(jb, 2, lower.tail = FALSE), bg$p.value, dw$p.value, pb, NA, NA, NA),
  df = c(bp$parameter, bpk$parameter, wh$parameter, rs$parameter[1], NA, 2, bg$parameter, NA, NA, NA, NA, NA))
"""
    return run_r(script, frames={"d": data}, outputs=["res"], packages=["lmtest"])[
        "res"
    ].set_index("test")


def test_ols_diagnostics_match_lmtest_and_base_r(data, ols_reference):
    results, _ = fitted(data, y="y", x=["x1", "x2"], model="ols")
    tests = lm_tests(results, decimal=None, lags=2).set_index("test")
    ref = ols_reference
    pairs = [
        ("Breusch-Pagan", "bpk"),  # Koenker's studentized version (R's default)
        ("White", "white"),
        ("RESET", "reset"),
        ("Shapiro-Wilk", "shapiro"),
        ("Jarque-Bera", "jb"),
        ("Outlier test", "outlier"),
        ("Cook's distance", "cook"),
        ("Max VIF", "vif"),
        ("Condition number", "cond"),
    ]
    for ours, theirs in pairs:
        assert tests.loc[ours, "statistic"] == pytest.approx(
            ref.loc[theirs, "stat"], rel=1e-6
        ), ours
        if np.isfinite(ref.loc[theirs, "df"]):
            assert tests.loc[ours, "df"] == ref.loc[theirs, "df"], ours
        if np.isfinite(ref.loc[theirs, "p"]) and np.isfinite(
            tests.loc[ours, "p-value"]
        ):
            assert tests.loc[ours, "p-value"] == pytest.approx(
                ref.loc[theirs, "p"], rel=1e-5
            ), ours


def test_ols_autocorrelation_tests_match_lmtest(data, ols_reference):
    results, _ = fitted(data, y="y", x=["x1", "x2"], model="ols")
    tests = lm_tests(results, decimal=None, lags=2).set_index("test")
    names = [t for t in tests.index if "Godfrey" in t or "Durbin" in t]
    assert len(names) == 2, list(tests.index)
    godfrey = next(t for t in names if "Godfrey" in t)
    durbin = next(t for t in names if "Durbin" in t)
    assert tests.loc[godfrey, "statistic"] == pytest.approx(
        ols_reference.loc["bg", "stat"], rel=1e-6
    )
    assert tests.loc[godfrey, "p-value"] == pytest.approx(
        ols_reference.loc["bg", "p"], rel=1e-5
    )
    assert tests.loc[durbin, "statistic"] == pytest.approx(
        ols_reference.loc["dw", "stat"], rel=1e-6
    )


@pytest.fixture(scope="module")
def glm_reference(data):
    script = """
fb <- glm(b ~ x1 + x2, binomial, d); fp <- glm(cnt ~ x1 + x2, poisson, d)
one <- function(g, name, fam) {
  d$eta <- predict(g, type = "link")
  link <- summary(glm(update(formula(g), . ~ . + I(eta^2)), family = fam, data = d))$coefficients
  data.frame(model = name, dispersion = sum(residuals(g, "pearson")^2) / g$df.residual,
    pearson = sum(residuals(g, "pearson")^2), deviance = deviance(g), df = g$df.residual,
    leverage = max(hatvalues(g)), cook = max(cooks.distance(g)), link_z = link["I(eta^2)", "z value"])
}
res <- rbind(one(fb, "binomial", binomial), one(fp, "poisson", poisson))
"""
    return run_r(script, frames={"d": data}, outputs=["res"])["res"].set_index("model")


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("binomial", dict(y="b", glm_model_family=sm.families.Binomial())),
        ("poisson", dict(y="cnt", glm_model_family=sm.families.Poisson())),
    ],
)
def test_glm_diagnostics_match_r(data, glm_reference, name, kwargs):
    results, _ = fitted(data, x=["x1", "x2"], model="glm", **kwargs)
    tests = lm_tests(results, decimal=None).set_index("test")
    ref = glm_reference.loc[name]
    assert tests.loc["Dispersion", "statistic"] == pytest.approx(
        ref["dispersion"], rel=1e-8
    )
    # (leverage and Cook's distance of a GLM: the IRLS of statsmodels stops at a looser tolerance than R's)
    assert tests.loc["Max leverage", "statistic"] == pytest.approx(
        ref["leverage"], rel=1e-4
    )
    assert tests.loc["Cook's distance", "statistic"] == pytest.approx(
        ref["cook"], rel=1e-4
    )
    assert tests.loc["Link test", "statistic"] == pytest.approx(
        ref["link_z"], rel=1e-3
    )  # (a refit: optimizer tolerance)
    if name == "poisson":
        assert tests.loc["Pearson GOF", "statistic"] == pytest.approx(
            ref["pearson"], rel=1e-8
        )
        assert tests.loc["Deviance GOF", "statistic"] == pytest.approx(
            ref["deviance"], rel=1e-8
        )
        assert tests.loc["Pearson GOF", "df"] == ref["df"]


@pytest.fixture(scope="module")
def rlm_reference(data):
    script = """
r <- MASS::rlm(y ~ x1 + x2, d)
cf <- data.frame(term = names(coef(r)), estimate = coef(r), se = summary(r)$coefficients[, "Std. Error"])
info <- data.frame(min_weight = min(r$w), scale = r$s)
"""
    return run_r(script, frames={"d": data}, outputs=["cf", "info"], packages=["MASS"])


def test_rlm_matches_mass_rlm(data, rlm_reference):
    results, _ = fitted(data, y="y", x=["x1", "x2"], model="rlm")
    table = lm_report(results, decimal=None, add_stars=False)[0]
    ref = rlm_reference["cf"].set_index("term")
    np.testing.assert_allclose(
        table["coef"].to_numpy(), ref["estimate"].to_numpy(), rtol=2e-3, atol=2e-4
    )
    np.testing.assert_allclose(table["se"].to_numpy(), ref["se"].to_numpy(), rtol=2e-2)
    tests = lm_tests(results, decimal=None)
    assert stat(tests, "Min robust weight") == pytest.approx(
        rlm_reference["info"].loc[0, "min_weight"], rel=2e-2
    )


@pytest.fixture(scope="module")
def mlm_reference(data):
    script = """
m <- lmer(y ~ x1 + x2 + (1 | g), d, REML = TRUE)
vc <- as.data.frame(VarCorr(m))
fixed <- data.frame(term = rownames(coef(summary(m))), estimate = coef(summary(m))[, "Estimate"], se = coef(summary(m))[, "Std. Error"])
ml <- lmer(y ~ x1 + x2 + (1 | g), d, REML = FALSE); ols <- lm(y ~ x1 + x2, d)
lrt <- 2 * (as.numeric(logLik(ml)) - as.numeric(logLik(ols)))
var <- data.frame(group = vc$vcov[1], resid = vc$vcov[2], icc = vc$vcov[1] / sum(vc$vcov), lr = lrt, p = 0.5 * pchisq(lrt, 1, lower.tail = FALSE))
"""
    return run_r(
        script, frames={"d": data}, outputs=["fixed", "var"], packages=["lme4"]
    )


def test_mixed_model_matches_lme4(data, mlm_reference):
    results, _ = fitted(
        data, formula="y ~ x1 + x2", model="mlm", mlm_model_groups=data["g"]
    )
    table = lm_report(results, decimal=None, add_stars=False)[0]
    ref = mlm_reference["fixed"].set_index("term")
    # statsmodels maximizes the REML criterion with a looser tolerance than lme4: the fixed effects agree to about
    # 0.1 %, the standard errors and the variance components to about 1 %
    np.testing.assert_allclose(
        table.iloc[:3]["coef"].to_numpy(), ref["estimate"].to_numpy(), rtol=5e-3
    )
    np.testing.assert_allclose(
        table.iloc[:3]["se"].to_numpy(), ref["se"].to_numpy(), rtol=2e-2
    )
    tests = lm_tests(results, decimal=None)
    var = mlm_reference["var"].iloc[0]
    assert stat(tests, "ICC") == pytest.approx(var["icc"], rel=2e-2)
    assert stat(tests, "LR test of random effect") == pytest.approx(var["lr"], rel=1e-2)
    assert stat(tests, "LR test of random effect", "p-value") == pytest.approx(
        var["p"], rel=5e-2
    )
