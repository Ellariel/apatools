"""plm, plm_report, base_metrics and plm_tests against the R packages plm, splm and spdep.

R is the reference implementation of panel econometrics, so these comparisons catch mistakes that checks against
other Python code (or our own formulas) cannot. The tests are skipped without R and the three packages.

Differences from R that are conventions, not errors (each is pinned below so that a change is noticed):

* spreg's fixed-effects standard errors use N·T - N degrees of freedom (R and linearmodels: N·T - N - k);
* the adjusted R² of plm uses (N·T - 1) in the numerator, ours (N·T - effects), like linearmodels;
* the spatial-lag coefficient of splm is called ``lambda`` and the spatial-error coefficient ``rho`` (spreg and
  apatools: the other way round).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from apatools.plm import make_panel, plm_report, plm_tests

from .r_reference import require_r, run_r
from .test_plm import HAS_LM, N, T, fit, simulate

require_r("plm", "splm", "spdep")

BACKENDS = [
    "spreg",
    pytest.param(
        "linearmodels", marks=pytest.mark.skipif(not HAS_LM, reason="no linearmodels")
    ),
]
DATASETS = [(0.4, 7), (0.6, 3)]  # (spatial lag of the simulation, seed)


@pytest.fixture(scope="module", params=DATASETS, ids=lambda p: f"rho{p[0]}-seed{p[1]}")
def data(request):
    rho, seed = request.param
    df, w = simulate(rho=rho, seed=seed)
    panel = make_panel(df, "y", ["x1", "x2"], entity="county", time="year", w=w)
    frame = pd.DataFrame(
        {
            "y": panel.y.to_numpy(),
            "x1": panel.x["x1"].to_numpy(),
            "x2": panel.x["x2"].to_numpy(),
            "id": panel.ent,
            "t": panel.tim,
        }
    )
    return df, w, frame, w.sparse.toarray()


@pytest.fixture(scope="module")
def reference(data):
    """Everything R computes for the non-spatial panel models and tests (one R run per data set)."""
    _, _, frame, w = data
    script = """
d$id <- factor(d$id); d$t <- factor(d$t)
pd <- pdata.frame(d, index = c("id", "t"))
f <- y ~ x1 + x2
pool <- plm(f, pd, model = "pooling"); re <- plm(f, pd, model = "random")
fits <- list(individual = plm(f, pd, model = "within"), time = plm(f, pd, model = "within", effect = "time"),
             twoways = plm(f, pd, model = "within", effect = "twoways"))
coefs <- do.call(rbind, lapply(names(fits), function(e) {
  s <- summary(fits[[e]]); cf <- as.data.frame(s$coefficients); names(cf) <- c("estimate", "se", "stat", "p")
  cf$term <- rownames(cf); cf$effect <- e
  cf$rsq <- s$r.squared[1]; cf$fstat <- s$fstatistic$statistic; cf$df2 <- s$fstatistic$parameter[2]; cf
}))
ft <- function(a, b, name) { x <- pFtest(a, b); data.frame(test = name, stat = unname(x$statistic), df1 = unname(x$parameter[1]), df2 = unname(x$parameter[2]), p = x$p.value) }
ftests <- rbind(ft(fits$individual, pool, "fe vs pooled"), ft(fits$twoways, fits$individual, "twoway vs fe"),
                ft(fits$time, pool, "time vs pooled"), ft(fits$twoways, fits$time, "twoway vs time"), ft(fits$twoways, pool, "twoway vs pooled"))
h <- phtest(fits$individual, re); hausman <- data.frame(stat = unname(h$statistic), df = unname(h$parameter), p = h$p.value)
cf <- as.data.frame(summary(re)$coefficients); names(cf) <- c("estimate", "se", "stat", "p"); cf$term <- rownames(cf); recoef <- cf
lw <- spdep::mat2listw(W, style = "W")
lm_tests <- data.frame(test = c("honda", "lml", "lme", "rlml", "rlme", "bp"),
  stat = c(plmtest(pool, effect = "individual", type = "honda")$statistic,
           sapply(c("lml", "lme", "rlml", "rlme"), function(t) unname(slmtest(pool, listw = lw, test = t)$statistic)),
           plmtest(pool, effect = "individual", type = "bp")$statistic))
"""
    return run_r(
        script,
        frames={"d": frame},
        matrices={"W": w},
        outputs=["coefs", "ftests", "hausman", "recoef", "lm_tests"],
        packages=["plm", "splm"],
    )


def _slopes(table):
    return table.loc[["x1", "x2"]]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "effects, label",
    [("entity", "individual"), ("time", "time"), ("twoway", "twoways")],
)
def test_fixed_effects_models_match_plm(data, reference, backend, effects, label):
    df, w, *_ = data
    (r,), (m,) = fit(df, w, "fe", backend=backend, effects=effects)
    ours = _slopes(plm_report([r], [m], decimal=None, add_stars=False)[0])
    ref = reference["coefs"].query("effect == @label").set_index("term")
    np.testing.assert_allclose(ours["coef"], ref["estimate"], rtol=1e-8)
    if (
        backend == "linearmodels" or effects == "time"
    ):  # (the time-effects model of spreg is an ordinary OLS)
        np.testing.assert_allclose(ours["se"], ref["se"], rtol=1e-7)
    else:  # spreg: N·T - N degrees of freedom instead of the residual degrees of freedom of R
        np.testing.assert_allclose(
            ours["se"], ref["se"] * np.sqrt(m["df_resid"] / (N * T - N)), rtol=1e-7
        )
    first = ref.iloc[0]
    assert m["r_sq"] == pytest.approx(first["rsq"], rel=1e-8)
    assert m["f_stat"] == pytest.approx(first["fstat"], rel=1e-8)
    assert m["df_resid"] == first["df2"]
    # the adjusted R² is not compared: plm uses (N·T - 1) / df_resid, we (N·T - effects) / df_resid, like linearmodels
    assert m["r_sq_adj"] == pytest.approx(
        1
        - (1 - m["r_sq"])
        * (N * T - (N * T - m["df_resid"] - m["n_params"]))
        / m["df_resid"]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_effects_match_plm(data, reference, backend):
    df, w, *_ = data
    (r,), (m,) = fit(df, w, "re", backend=backend)
    ours = plm_report([r], [m], decimal=None, add_stars=False)[0]
    ref = reference["recoef"].set_index("term")
    np.testing.assert_allclose(
        ours["coef"].to_numpy(), ref["estimate"].to_numpy(), rtol=1e-6
    )
    # standard errors: linearmodels identical, spreg a little smaller (its own degrees of freedom)
    ratio = ours["se"].to_numpy() / ref["se"].to_numpy()
    if backend == "linearmodels":
        np.testing.assert_allclose(ratio, 1, rtol=1e-6)
    else:
        assert np.all((ratio > 0.97) & (ratio <= 1.0 + 1e-9))


@pytest.mark.skipif(not HAS_LM, reason="no linearmodels")
def test_hausman_matches_phtest(data, reference):
    from apatools.plm import hausman

    df, w, *_ = data
    fe, re = (fit(df, w, name, backend="linearmodels")[0][0] for name in ("fe", "re"))
    stat, dof, p = hausman(fe, re)
    ref = reference["hausman"].iloc[0]
    assert (stat, dof, p) == pytest.approx((ref["stat"], ref["df"], ref["p"]), rel=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_f_tests_of_effects_match_pftest(data, reference, backend):
    df, w, *_ = data
    models = [("pooled", "entity"), ("fe", "entity"), ("fe", "time"), ("fe", "twoway")]
    results = [
        fit(df, w, name, backend=backend, effects=effects)[0][0]
        for name, effects in models
    ]
    tests = plm_tests(results, decimal=None).set_index("model")
    for _, ref in reference["ftests"].iterrows():
        label = (
            ref["test"]
            .replace("twoway vs", "fe (twoway) vs")
            .replace("time vs", "fe (time) vs")
        )
        label = label.replace("vs time", "vs fe (time)").replace(
            "fe vs pooled", "fe vs pooled"
        )
        row = tests.loc[label]
        assert row["statistic"] == pytest.approx(ref["stat"], rel=1e-7), label
        assert (row["df"], row["df_denom"]) == (ref["df1"], ref["df2"]), label
        assert row["p-value"] == pytest.approx(ref["p"], rel=1e-6, abs=1e-300), label


def test_lm_tests_of_the_pooled_model_match_plm_and_splm(data, reference):
    df, w, *_ = data
    tests = plm_tests([fit(df, w, "pooled")[0][0]], decimal=None).set_index("test")[
        "statistic"
    ]
    ref = reference["lm_tests"].set_index("test")["stat"]
    assert tests["LM Marginal RE (LM1)"] == pytest.approx(
        ref["honda"], rel=1e-7
    )  # the one-sided (Honda) form
    # Anselin's LM tests of the stacked pooled model: spreg's own panel versions differ from these (see plm.py)
    for label, name in (
        ("LM lag", "lml"),
        ("LM error", "lme"),
        ("Robust LM lag", "rlml"),
        ("Robust LM error", "rlme"),
    ):
        assert tests[label] == pytest.approx(ref[name], rel=1e-7), label
    # the marginal spatial test of Baltagi, Song & Koh is the one-sided form of the LM error test, and the joint
    # test the sum of the two marginal tests (the trace of W is zero)
    assert tests["LM Marginal Spatial (LM2)"] == pytest.approx(
        np.sqrt(ref["lme"]), rel=1e-7
    )
    assert tests["LM Joint (LMJ)"] == pytest.approx(ref["bp"] + ref["lme"], rel=1e-7)


@pytest.fixture(scope="module")
def spatial_reference(data):
    """splm::spml (ML, within) for the lag and error models with entity, time and two-way effects."""
    _, _, frame, w = data
    script = """
d$id <- factor(d$id); d$t <- factor(d$t)
pd <- pdata.frame(d, index = c("id", "t"))
lw <- spdep::mat2listw(W, style = "W")
runs <- list(c("lag", "individual"), c("lag", "twoways"), c("err", "individual"), c("err", "twoways"), c("err", "time"))
spat <- do.call(rbind, lapply(runs, function(r) {
  g <- spml(y ~ x1 + x2, pd, listw = lw, model = "within", effect = r[2], lag = r[1] == "lag", spatial.error = if (r[1] == "err") "b" else "none")
  cf <- as.data.frame(summary(g)$CoefTable); names(cf)[1:2] <- c("estimate", "se"); cf$term <- rownames(cf); cf$kind <- r[1]; cf$effect <- r[2]; cf
}))
"""
    return run_r(
        script,
        frames={"d": frame},
        matrices={"W": w},
        outputs=["spat"],
        packages=["plm", "splm"],
    )["spat"]


@pytest.mark.parametrize(
    "kind, model, effects, label",
    [
        ("lag", "slm_fe", "entity", "individual"),
        ("lag", "slm_fe", "twoway", "twoways"),
        ("err", "sem_fe", "entity", "individual"),
        ("err", "sem_fe", "twoway", "twoways"),
        ("err", "sem_fe", "time", "time"),
    ],
)
def test_spatial_fixed_effects_models_match_splm(
    data, spatial_reference, kind, model, effects, label
):
    df, w, *_ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (r,), (m,) = fit(df, w, model, effects=effects)
    ours = plm_report([r], [m], decimal=None, add_stars=False)[0]
    ref = spatial_reference.query("kind == @kind and effect == @label").set_index(
        "term"
    )
    spatial = (
        ("rho", "lambda") if kind == "lag" else ("lambda", "rho")
    )  # (ours, splm): the names are swapped
    # the spatial parameter within the tolerance of splm's optimizer, the slopes and all standard errors tightly
    assert ours.loc[spatial[0], "coef"] == pytest.approx(
        ref.loc[spatial[1], "estimate"], abs=1e-3
    )
    np.testing.assert_allclose(
        _slopes(ours)["coef"], _slopes(ref)["estimate"], atol=2e-4
    )
    assert ours.loc[spatial[0], "se"] == pytest.approx(
        ref.loc[spatial[1], "se"], rel=1e-2
    )
    np.testing.assert_allclose(_slopes(ours)["se"], _slopes(ref)["se"], rtol=1e-2)


@pytest.fixture(scope="module")
def impacts_reference(data):
    """splm's impacts() of the spatial lag model with individual effects (point estimates, simulated errors)."""
    _, _, frame, w = data
    script = f"""
d$id <- factor(d$id); d$t <- factor(d$t)
pd <- pdata.frame(d, index = c("id", "t"))
lw <- spdep::mat2listw(W, style = "W")
g <- spml(y ~ x1 + x2, pd, listw = lw, model = "within", effect = "individual", lag = TRUE, spatial.error = "none")
set.seed(1); imp <- impacts(g, listw = lw, time = {T}, R = 4000)
s <- summary(imp, zstats = TRUE, short = TRUE)
ref <- data.frame(variable = c("x1", "x2"), d = imp$res$direct, i = imp$res$indirect, t = imp$res$total,
  se_d = s$semat[, "Direct"], se_i = s$semat[, "Indirect"], se_t = s$semat[, "Total"])
"""
    out = run_r(
        script,
        frames={"d": frame},
        matrices={"W": w},
        outputs=["ref"],
        packages=["plm", "splm", "spdep"],
    )
    return out["ref"].set_index("variable")


@pytest.mark.parametrize("backend", ["spreg", "splm"])
def test_effects_of_the_spatial_lag_panel_model_match_splm(
    data, impacts_reference, backend
):
    from apatools.plm import plm_effects

    df, w, *_ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (r,), _ = fit(df, w, "slm_fe", backend=backend)
        analytic = plm_effects([r], inference="analytic").pivot(
            index="variable", columns="effect", values=["estimate", "se"]
        )
        simulated = plm_effects([r], inference="simulation", n_draws=4000).pivot(
            index="variable", columns="effect", values="se"
        )
    ref = impacts_reference
    for effect, column in (
        ("Direct", "d"),
        ("Indirect", "i"),
        ("Total", "t"),
    ):  # the same numbers as R's
        np.testing.assert_allclose(
            analytic["estimate"][effect],
            ref[column],
            atol=2e-3 if backend == "spreg" else 1e-6,
        )
    for effect, column in (("Direct", "se_d"), ("Indirect", "se_i"), ("Total", "se_t")):
        np.testing.assert_allclose(simulated[effect], ref[column], rtol=0.2)
    np.testing.assert_allclose(analytic["se"]["Direct"], ref["se_d"], rtol=0.12)
