"""slm, slm_report, base_metrics and slm_tests against the R packages spatialreg, spdep and lmtest.

``spatialreg`` is the reference implementation of the spatial regression models, ``spdep`` of the LM tests and
Moran's I, and ``lmtest`` of the heteroskedasticity tests. The tests are skipped without R and these packages.

Conventions that differ from R (pinned below):

* R's AIC and BIC count the error variance as a parameter, ours (like spreg's) do not: a constant difference of
  2 (AIC) or log(n) (BIC) for every model;
* the p-value of Moran's I of the residuals is two-sided here, one-sided (greater) in ``lm.morantest``.
"""

import warnings

import libpysal
import numpy as np
import pandas as pd
import pytest

from apatools.slm import slm, slm_report, slm_tests

from .r_reference import require_r, run_r

require_r("spdep", "spatialreg", "lmtest")

DATASETS = [
    (5, 10, False, 0.45),
    (11, 8, True, 0.6),
]  # (seed, side of the lattice, queen contiguity, rho)


@pytest.fixture(scope="module", params=DATASETS, ids=lambda p: f"seed{p[0]}-side{p[1]}")
def data(request):
    seed, side, queen, rho = request.param
    rng = np.random.default_rng(seed)
    w = libpysal.weights.lat2W(side, side, rook=not queen)
    w.transform = "r"
    n, matrix = w.n, w.sparse.toarray()
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    noise = rng.normal(size=n) * (
        1 + 0.5 * np.abs(x1)
    )  # heteroskedastic: the tests have something to find
    y = np.linalg.solve(np.eye(n) - rho * matrix, 1 + x1 - 0.5 * x2 + noise)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, metrics = slm(
            df, "y", ["x1", "x2"], w=w, model=["ols", "slm", "sem"], verbose=False
        )
    return df, w, matrix, results, metrics


@pytest.fixture(scope="module")
def reference(data):
    df, _, matrix, *_ = data
    script = """
lw <- mat2listw(W, style = "W"); f <- y ~ x1 + x2
o <- lm(f, d); lag <- lagsarlm(f, d, lw); err <- errorsarlm(f, d, lw)
tab <- function(m, kind) {
  cf <- as.data.frame(summary(m)$Coef); names(cf)[1:2] <- c("estimate", "se"); cf$term <- rownames(cf)
  sp <- if (kind == "lag") data.frame(estimate = m$rho, se = m$rho.se, term = "rho") else data.frame(estimate = m$lambda, se = m$lambda.se, term = "lambda")
  out <- rbind(cf[, c("estimate", "se", "term")], sp); out$model <- kind; out }
coefs <- rbind(tab(lag, "lag"), tab(err, "err"))
fit <- data.frame(model = c("ols", "lag", "err"), ll = c(logLik(o), logLik(lag), logLik(err)), aic = c(AIC(o), AIC(lag), AIC(err)), bic = c(BIC(o), BIC(lag), BIC(err)))
rs <- lm.RStests(o, lw, test = "all")
rst <- data.frame(test = names(rs), stat = sapply(rs, function(x) unname(x$statistic)), p = sapply(rs, function(x) x$p.value))
mt <- lm.morantest(o, lw); mor <- data.frame(z = unname(mt$statistic), p = mt$p.value)
bp <- bptest(o, studentize = FALSE); kb <- bptest(o, studentize = TRUE)
wh <- bptest(o, ~ x1 + x2 + I(x1^2) + I(x2^2) + x1:x2, data = d, studentize = TRUE)
het <- data.frame(test = c("bp", "kb", "white"), stat = c(bp$statistic, kb$statistic, wh$statistic), df = c(bp$parameter, kb$parameter, wh$parameter), p = c(bp$p.value, kb$p.value, wh$p.value))
"""
    return run_r(
        script,
        frames={"d": df},
        matrices={"W": matrix},
        outputs=["coefs", "fit", "rst", "mor", "het"],
        packages=["spdep", "spatialreg", "lmtest"],
    )


@pytest.mark.parametrize(
    "position, kind, spatial", [(1, "lag", "rho"), (2, "err", "lambda")]
)
def test_ml_models_match_spatialreg(data, reference, position, kind, spatial):
    *_, results, metrics = data
    table = slm_report([results[position]], decimal=None, add_stars=False)[0]
    table.index = ["Intercept", "x1", "x2", spatial]
    ref = reference["coefs"].query("model == @kind").set_index("term")
    ref.index = ["Intercept", "x1", "x2", spatial]
    np.testing.assert_allclose(table["coef"], ref["estimate"], rtol=1e-4)
    np.testing.assert_allclose(table["se"], ref["se"], rtol=1e-4)


@pytest.mark.parametrize("position, name", [(0, "ols"), (1, "lag"), (2, "err")])
def test_log_likelihood_and_information_criteria_match_r(
    data, reference, position, name
):
    *_, results, metrics = data
    m = metrics[position]
    ref = reference["fit"].set_index("model").loc[name]
    n = m["n_obs"]
    assert m["llf"] == pytest.approx(ref["ll"], rel=1e-8)
    # R counts the error variance as a parameter: +2 (AIC) and +log(n) (BIC) for every model, and the
    # error model counts lambda (spreg's own criteria do not: the metrics add it)
    assert m["aic"] == pytest.approx(ref["aic"] - 2, abs=1e-6)
    assert m["bic"] == pytest.approx(ref["bic"] - np.log(n), abs=1e-6)


def test_ols_spatial_diagnostics_match_spdep(data, reference):
    _, w, _, results, _ = data
    tests = slm_tests(results[:1], w=w, decimal=None).set_index("test")
    ref = reference["rst"].set_index("test")
    for label, name in (
        ("LM-error", "RSerr"),
        ("LM-lag", "RSlag"),
        ("Robust LM-error", "adjRSerr"),
        ("Robust LM-lag", "adjRSlag"),
        ("LM-SARMA", "SARMA"),
    ):
        assert tests.loc[label, "statistic"] == pytest.approx(
            ref.loc[name, "stat"], rel=1e-7
        ), label
        assert tests.loc[label, "p-value"] == pytest.approx(
            ref.loc[name, "p"], rel=1e-6
        ), label
    moran = reference["mor"].iloc[0]
    assert tests.loc["Moran's I (residuals)", "statistic"] == pytest.approx(
        moran["z"], rel=1e-7
    )
    assert tests.loc["Moran's I (residuals)", "p-value"] == pytest.approx(
        2 * moran["p"], rel=1e-6
    )  # two-sided


def test_heteroskedasticity_tests_match_lmtest(data, reference):
    _, w, _, results, _ = data
    tests = slm_tests(results[:1], w=w, decimal=None).set_index("test")
    ref = reference["het"].set_index("test")
    for label, name in (
        ("Breusch-Pagan", "bp"),
        ("Koenker-Bassett", "kb"),
        ("White", "white"),
    ):
        assert tests.loc[label, "statistic"] == pytest.approx(
            ref.loc[name, "stat"], rel=1e-7
        ), label
        assert tests.loc[label, "df"] == ref.loc[name, "df"], label
        assert tests.loc[label, "p-value"] == pytest.approx(
            ref.loc[name, "p"], rel=1e-6
        ), label


@pytest.mark.parametrize("model, name", [("slm", "lag"), ("sem", "err")])
def test_lr_test_of_the_spatial_parameter_matches_loglik_difference(
    data, reference, model, name
):
    _, w, _, results, _ = data
    tests = slm_tests(results, w=w, decimal=None, n_perm=0).set_index(["model", "test"])
    ll = reference["fit"].set_index("model")["ll"]
    spatial = "rho" if model == "slm" else "lambda"
    assert tests.loc[(model, f"LR test of {spatial}"), "statistic"] == pytest.approx(
        2 * (ll[name] - ll["ols"]), rel=1e-7
    )


@pytest.fixture(scope="module")
def impacts_reference(data):
    """spatialreg::impacts of the spatial lag model: point estimates and standard errors of its simulation."""
    df, _, matrix, *_ = data
    script = """
lw <- mat2listw(W, style = "W"); lag <- lagsarlm(y ~ x1 + x2, d, lw)
set.seed(1); imp <- impacts(lag, listw = lw, R = 5000)
s <- summary(imp, zstats = TRUE, short = TRUE)
ref <- data.frame(variable = c("x1", "x2"), d = imp$res$direct, i = imp$res$indirect, t = imp$res$total,
  se_d = s$semat[, "Direct"], se_i = s$semat[, "Indirect"], se_t = s$semat[, "Total"])
"""
    out = run_r(
        script,
        frames={"d": df},
        matrices={"W": matrix},
        outputs=["ref"],
        packages=["spdep", "spatialreg"],
    )
    return out["ref"].set_index("variable")


def test_effects_of_the_spatial_lag_model_match_spatialreg(data, impacts_reference):
    from apatools.slm import slm_effects

    _, w, _, results, _ = data
    ref = impacts_reference
    simulated = slm_effects(
        results[1:2], w, inference="simulation", n_draws=5000
    ).pivot(index="variable", columns="effect", values="se")
    analytic = slm_effects(results[1:2], w, inference="analytic").pivot(
        index="variable", columns="effect", values="estimate"
    )
    # the point estimates: the same numbers as R's (exact inverse of the weights)
    for effect, column in (("Direct", "d"), ("Indirect", "i"), ("Total", "t")):
        np.testing.assert_allclose(analytic[effect], ref[column], atol=1e-6)
    # the standard errors of the simulation, which has noise (5000 draws each) and uses spreg's covariance matrix
    for effect, column in (("Direct", "se_d"), ("Indirect", "se_i"), ("Total", "se_t")):
        np.testing.assert_allclose(simulated[effect], ref[column], rtol=0.15)
    # the delta method for the direct effect (the indirect and total effects are less linear in rho: see the docstring)
    se = slm_effects(results[1:2], w, inference="analytic").pivot(
        index="variable", columns="effect", values="se"
    )
    np.testing.assert_allclose(se["Direct"], ref["se_d"], rtol=0.12)
