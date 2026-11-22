"""plm(backend="splm"): the R packages plm and splm (with spdep) as the engine of the panel models.

Needs R with the packages plm, splm and spdep (the module is skipped without them). The R engine is compared with
the other backends (spreg, linearmodels), with OLS with dummies and with maximum-likelihood fits written out in numpy;
the comparison of apatools with R's own test functions is in test_r_plm.py. The conversion of the R tables
(``apatools/r/splm.py``) and the requests rejected before R is started are tested without R in test_r_bridge.py.
"""

import warnings

import numpy as np
import pytest
import scipy.optimize as so

from apatools.plm import base_metrics, make_panel, mundlak, plm, plm_report, plm_tests

from .r_reference import require_r
from .test_plm import HAS_LM, N, T, fit, simulate

require_r("plm", "splm", "spdep")

needs_lm = pytest.mark.skipif(not HAS_LM, reason="linearmodels is not installed")


###########################
#  helpers and references  #
###########################


def run(df, w, model, **kwargs):
    kwargs.setdefault("backend", "splm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(df, w, model, **kwargs)


def report(result_and_metrics):
    (r,), (m,) = result_and_metrics
    return plm_report([r], [m], decimal=None, add_stars=False)[0]


def slopes(result_and_metrics):
    return report(result_and_metrics).loc[["x1", "x2"]]


def _lag_reference(df, w, effects):
    """ρ and slopes of the spatial lag model with the effects as explicit parameters (profile likelihood)."""
    p = make_panel(df, "y", ["x1", "x2"], entity="county", time="year", w=w)
    n, wm = len(p.y), w.sparse.toarray()
    wb = np.kron(np.eye(T), wm)
    y, x = p.y.to_numpy(), p.x.to_numpy()
    dt = (p.tim[:, None] == np.arange(1, T)[None, :]).astype(float)
    de = (p.ent[:, None] == np.arange(N)[None, :]).astype(float)
    z = {
        "entity": np.column_stack([x, de]),
        "twoway": np.column_stack([x, de, dt]),
        "time": np.column_stack([x, np.ones(n), dt]),
    }[effects]

    def nll(r):
        v = y - r * wb @ y
        e = v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
        return n / 2 * np.log(e @ e / n) - T * np.linalg.slogdet(np.eye(N) - r * wm)[1]

    r = so.minimize_scalar(
        nll, bounds=(-0.95, 0.95), method="bounded", options={"xatol": 1e-10}
    ).x
    return r, np.linalg.lstsq(z, y - r * wb @ y, rcond=None)[0][:2]


def _error_random_effects_reference(df, w):
    """
    ML of the spatial error random-effects model, written out without any package (Baltagi, Song & Koh, 2003):
    ``y = Xb + mu + v``, ``v = lambda W v + e``, so the stacked errors have the covariance
    ``s_mu (J_T x I_N) + s_e I_T x (B'B)^-1`` with ``B = I - lambda W``. The coefficients are profiled out by
    generalized least squares. Returns ``(log-likelihood, coefficients, s_mu, s_e, lambda)``.
    """
    p = make_panel(df, "y", ["x1", "x2"], entity="county", time="year", w=w)
    n, wm, eye = N * T, w.sparse.toarray(), np.eye(N)
    y, x = p.y.to_numpy(), np.column_stack([np.ones(n), p.x.to_numpy()])
    common = np.kron(np.ones((T, T)), eye)

    def solve(theta):
        s_mu, s_e, lam = np.exp(theta[0]), np.exp(theta[1]), np.tanh(theta[2])
        b = eye - lam * wm
        omega = s_mu * common + s_e * np.kron(np.eye(T), np.linalg.inv(b.T @ b))
        weighted = np.linalg.solve(omega, x)
        beta = np.linalg.solve(x.T @ weighted, weighted.T @ y)
        e = y - x @ beta
        loglik = -0.5 * (
            n * np.log(2 * np.pi)
            + np.linalg.slogdet(omega)[1]
            + e @ np.linalg.solve(omega, e)
        )
        return loglik, beta, s_mu, s_e, lam

    best = so.minimize(
        lambda t: -solve(t)[0],
        [0.0, 0.0, 0.3],
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000},
    )
    return solve(best.x)


@pytest.fixture(scope="module")
def sim():
    return simulate(rho=0.4, seed=7)


@pytest.fixture(scope="module")
def non_spatial(sim):
    """The non-spatial models of the R engine: pooled, RE and the FE model with entity, time and two-way effects."""
    df, w = sim
    fits = {"pooled": run(df, w, "pooled"), "re": run(df, w, "re")}
    for effects in ("entity", "time", "twoway"):
        fits[effects] = run(df, w, "fe", effects=effects)
    return fits


####################################
#  non-spatial models (R: plm)     #
####################################


@needs_lm
@pytest.mark.parametrize(
    "key, model, effects",
    [
        ("pooled", "pooled", "entity"),
        ("re", "re", "entity"),
        ("entity", "fe", "entity"),
        ("time", "fe", "time"),
        ("twoway", "fe", "twoway"),
    ],
)
def test_non_spatial_models_equal_linearmodels(sim, non_spatial, key, model, effects):
    df, w = sim
    ours = non_spatial[key]
    ref = fit(df, w, model, backend="linearmodels", effects=effects)
    a, b = slopes(ours), slopes(ref)
    np.testing.assert_allclose(a["coef"], b["coef"], rtol=1e-6)
    np.testing.assert_allclose(
        a["se"], b["se"], rtol=1e-6
    )  # R's plm and linearmodels use the same covariance
    m, lm_m = ours[1][0], ref[1][0]
    for k in (
        "r_sq",
        "r_sq_adj",
        "f_stat",
        "df_model",
        "df_resid",
        "n_obs",
        "n_entities",
        "n_periods",
        "n_params",
    ):
        if k in lm_m and model != "re" and not (model == "pooled" and k == "r_sq_adj"):
            assert m[k] == pytest.approx(lm_m[k], rel=1e-6), k
    assert m["backend"] == "splm" and m["model"] == model


@pytest.mark.parametrize(
    "key, dummies, absorbed",
    [
        ("entity", "C(county)", N),
        ("time", "C(year)", T),
        ("twoway", "C(county) + C(year)", N + T - 1),
    ],
)
def test_fixed_effects_equal_ols_with_dummies(sim, non_spatial, key, dummies, absorbed):
    smf = pytest.importorskip("statsmodels.formula.api")
    df, _ = sim
    ols = smf.ols(f"y ~ x1 + x2 + {dummies}", df).fit()
    table = slopes(non_spatial[key])
    np.testing.assert_allclose(table["coef"], ols.params[["x1", "x2"]], rtol=1e-6)
    np.testing.assert_allclose(table["se"], ols.bse[["x1", "x2"]], rtol=1e-6)
    assert non_spatial[key][1][0]["df_resid"] == ols.df_resid == N * T - absorbed - 2


def test_residuals_are_in_the_layout_of_the_panel(sim, non_spatial):
    """R returns the observations sorted by entity; the result has them in period blocks, like the panel."""
    df, w = sim
    p = make_panel(df, "y", ["x1", "x2"], entity="county", time="year", w=w)
    y, x = p.y.to_numpy().reshape(T, N), p.x.to_numpy().reshape(T, N, 2)
    (pooled,), _ = non_spatial["pooled"]
    design = np.column_stack([np.ones(N * T), p.x.to_numpy()])
    np.testing.assert_allclose(
        pooled.resid,
        p.y.to_numpy() - design @ pooled.table["coef"].to_numpy(),
        atol=1e-8,
    )
    (fe,), _ = non_spatial["entity"]
    within = (y - y.mean(axis=0)).ravel() - (x - x.mean(axis=0)).reshape(
        -1, 2
    ) @ fe.table["coef"].to_numpy()
    np.testing.assert_allclose(fe.resid, within, atol=1e-8)
    assert fe.ssr == pytest.approx(np.sum(within**2)) and fe.n_obs == N * T


########################################
#  spatial models (R: splm::spml)      #
########################################


@pytest.mark.parametrize("effects", ["entity", "time", "twoway"])
def test_spatial_lag_fixed_effects_equal_the_reference_likelihood(sim, effects):
    df, w = sim
    fitted = run(df, w, "slm_fe", effects=effects)
    full = report(fitted)
    rho, beta = _lag_reference(df, w, effects)
    assert full.loc["rho", "coef"] == pytest.approx(
        rho, abs=1e-3
    )  # (the tolerance of splm's optimizer)
    np.testing.assert_allclose(full.loc[["x1", "x2"], "coef"], beta, atol=2e-4)
    (r,), (m,) = fitted
    assert r.effects == effects and m["r_sq_type"] == "pseudo" and np.isfinite(m["llf"])


def test_spatial_error_random_effects_equal_the_reference_likelihood(sim):
    # spreg's ML_ErrorRE fails with some numpy / scipy versions (see test_plm.py): the reference is the only comparison
    df, w = sim
    fitted = run(df, w, "sem_re")
    table, m = report(fitted), fitted[1][0]
    loglik, beta, s_mu, s_e, lam = _error_random_effects_reference(df, w)
    assert list(table.index) == ["Intercept", "x1", "x2", "phi", "lambda"]
    assert table.loc["lambda", "coef"] == pytest.approx(lam, abs=1e-3)
    np.testing.assert_allclose(
        table.loc[["Intercept", "x1", "x2"], "coef"], beta, atol=1e-3
    )
    assert table.loc["phi", "coef"] == pytest.approx(
        s_mu / s_e, rel=1e-3
    )  # splm's phi: the ratio of the variances
    assert m["llf"] == pytest.approx(loglik, abs=1e-2)
    assert m["aic"] == pytest.approx(
        2 * 5 - 2 * loglik, abs=2e-2
    )  # 3 coefficients, phi and lambda


@pytest.mark.parametrize(
    "model, effects, spatial",
    [
        ("slm_fe", "entity", "rho"),
        ("slm_fe", "twoway", "rho"),
        ("slm_re", "entity", "rho"),
        ("sem_pooled", "entity", "lambda"),
        ("sem_fe", "entity", "lambda"),
        ("sem_fe", "twoway", "lambda"),
        ("sem_fe", "time", "lambda"),
    ],
)
def test_spatial_models_agree_with_spreg(sim, model, effects, spatial):
    df, w = sim
    table = report(run(df, w, model, effects=effects))
    reference = report(run(df, w, model, backend="spreg", effects=effects))
    # the spatial parameter within the tolerance of splm's optimizer, the slopes and standard errors more tightly
    assert table.loc[spatial, "coef"] == pytest.approx(
        reference.loc[spatial, "coef"], abs=1e-3
    )
    tolerance = 2e-3 if model.endswith("_re") else 2e-4
    np.testing.assert_allclose(
        table.loc[["x1", "x2"], "coef"],
        reference.loc[["x1", "x2"], "coef"],
        atol=tolerance,
    )
    np.testing.assert_allclose(
        table.loc[["x1", "x2"], "se"], reference.loc[["x1", "x2"], "se"], rtol=3e-2
    )


@pytest.mark.parametrize("model, spatial", [("slm_fe", "rho"), ("sem_fe", "lambda")])
def test_weights_that_are_not_row_standardized_are_used_as_given(model, spatial):
    df, w = simulate(rho=0.4, seed=7)
    w.transform = "b"  # binary weights
    table = report(run(df, w, model))
    reference = report(run(df, w, model, backend="spreg"))
    assert table.loc[spatial, "coef"] == pytest.approx(
        reference.loc[spatial, "coef"], abs=1e-3
    )
    np.testing.assert_allclose(
        table.loc[["x1", "x2"], "coef"], reference.loc[["x1", "x2"], "coef"], atol=2e-4
    )
    if model == "slm_fe":
        rho, _ = _lag_reference(df, w, "entity")
        assert table.loc["rho", "coef"] == pytest.approx(rho, abs=1e-3)


#####################################
#  reports, metrics and tests       #
#####################################


def test_report_and_metrics_of_the_r_engine(sim, non_spatial):
    (r,), (m,) = non_spatial["twoway"]
    assert (m["n_obs"], m["n_entities"], m["n_periods"], m["n_params"]) == (
        N * T,
        N,
        T,
        2,
    )
    assert (
        base_metrics(r)["r_sq"] == m["r_sq"]
        and r.versions["plm"]
        and r.versions["splm"]
    )
    text = plm_report([r], [m])[0]["model"].iloc[0]
    assert text.startswith("R²within = .59") and text.endswith("N = 216, n = 36, T = 6")
    assert "x1" in r.summary  # a printable summary
    assert r.warnings == []


@pytest.mark.parametrize(
    "model, parameters", [("slm_fe", 3), ("slm_re", 5), ("sem_pooled", 4)]
)
def test_information_criteria_count_the_coefficients_and_the_spatial_parameter(
    sim, model, parameters
):
    df, w = sim
    (r,), (m,) = run(df, w, model)
    # coefficients (with the constant), the spatial parameter and, for random effects, phi; not the error variance
    assert len(r.table) == parameters
    assert m["aic"] == pytest.approx(2 * parameters - 2 * m["llf"])
    assert m["bic"] == pytest.approx(parameters * np.log(N * T) - 2 * m["llf"])


@needs_lm
def test_tests_of_the_r_engine_equal_those_of_the_other_backends(sim, non_spatial):
    df, w = sim
    results = [
        non_spatial[k][0][0] for k in ("pooled", "re", "entity", "time", "twoway")
    ]
    tests = plm_tests(results, w=w, decimal=None).set_index(["test", "model"])
    others = [
        fit(df, w, name, backend="linearmodels", effects=e)[0][0]
        for name, e in [
            ("pooled", "entity"),
            ("fe", "entity"),
            ("fe", "time"),
            ("fe", "twoway"),
        ]
    ]
    reference = plm_tests(others, decimal=None).set_index(["test", "model"])
    for index in reference.index:
        if index[0].startswith("F test"):
            assert tests.loc[index, "statistic"] == pytest.approx(
                reference.loc[index, "statistic"], rel=1e-7
            ), index
    spreg_tests = plm_tests([fit(df, w, "pooled")[0][0]], w=w, decimal=None).set_index(
        "test"
    )["statistic"]
    labels = (
        "LM lag",
        "LM error",
        "Robust LM lag",
        "Robust LM error",
        "LM Marginal RE (LM1)",
        "LM Marginal Spatial (LM2)",
        "LM Joint (LMJ)",
    )
    for label in labels:
        assert tests.loc[(label, "pooled"), "statistic"] == pytest.approx(
            spreg_tests[label], rel=1e-9
        ), label
    # R's phtest (checked in test_r_plm.py) equals the Hausman test of linearmodels
    fe_re = [fit(df, w, name, backend="linearmodels")[0][0] for name in ("fe", "re")]
    hausman_lm = (
        plm_tests(fe_re, decimal=None)
        .set_index("test")
        .loc["Hausman (FE vs RE)", "statistic"]
    )
    assert tests.loc[("Hausman (FE vs RE)", "fe vs re"), "statistic"] == pytest.approx(
        hausman_lm, rel=1e-6
    )
    assert ("Mundlak (correlated effects)", "re") in tests.index


def test_lr_test_of_time_effects_with_the_r_engine(sim):
    df, w = sim
    results = [run(df, w, "slm_fe", effects=e)[0][0] for e in ("entity", "twoway")]
    reference = [
        run(df, w, "slm_fe", backend="spreg", effects=e)[0][0]
        for e in ("entity", "twoway")
    ]
    ours = (
        plm_tests(results, decimal=None)
        .set_index("test")
        .loc["LR test of time effects"]
    )
    theirs = (
        plm_tests(reference, decimal=None)
        .set_index("test")
        .loc["LR test of time effects"]
    )
    assert ours["statistic"] == pytest.approx(theirs["statistic"], rel=2e-3)
    assert ours["df"] == theirs["df"] == T - 1


################################
#  options and error handling  #
################################


def test_results_carry_the_panel_and_the_fitting_options(sim):
    df, w = sim
    results, _ = plm(
        df,
        "y",
        ["x1", "x2"],
        w=w,
        entity="county",
        time="year",
        model=["fe", "slm_fe"],
        backend="splm",
        effects="twoway",
        verbose=False,
    )
    assert [type(r).__name__ for r in results] == ["SplmFit", "SplmFit"]
    for r in results:
        assert (
            r.plm_panel.N == N
            and r.plm_spec["backend"] == "splm"
            and r.plm_spec["kwargs"]["effects"] == "twoway"
        )
        assert r.effects == "twoway"


def test_mundlak_and_options_of_the_r_engine(sim):
    df, w = sim
    (re,), _ = run(df, w, "re")
    stat, dof, p = mundlak(
        re
    )  # refits the model with the entity means through the same backend
    assert dof == 2 and np.isfinite(stat) and 0 <= p <= 1
    # arguments of R functions are passed on as <model>_model_<argument>, and override the defaults
    (a,), _ = run(df, w, "fe", fe_model_effect="time")
    (b,), _ = run(df, w, "fe", effects="time")
    np.testing.assert_allclose(a.table["coef"], b.table["coef"])
    # the prediction metrics would start R for every refit: they are NaN, with a warning
    with pytest.warns(UserWarning, match="backend='splm'"):
        _, metrics = plm(
            df,
            "y",
            ["x1", "x2"],
            w=w,
            entity="county",
            time="year",
            model="fe",
            backend="splm",
            verbose=False,
            pred_metrics=True,
        )
    assert np.isnan(metrics[0]["pred_loo_r_sq"])


def test_invalid_effects_are_rejected(sim):
    df, w = sim
    with pytest.raises(ValueError, match="effects"):
        fit(df, w, "fe", backend="splm", effects="both")


def test_a_wrong_rscript_is_reported(sim):
    df, w = sim
    with pytest.raises(RuntimeError, match="not found"):
        fit(df, w, "fe", backend="splm", rscript="no-such-rscript-anywhere")


def test_r_timeout_stops_the_run(sim):
    df, w = sim
    with pytest.raises(RuntimeError, match="did not finish within 0.001 seconds"):
        fit(df, w, "fe", backend="splm", r_timeout=0.001)
