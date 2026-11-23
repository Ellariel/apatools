"""Direct, indirect and total effects of the spatial lag models: apatools._impacts, slm_effects and plm_effects.

The numbers are checked against the definition (the average diagonal and row sum of the inverse of I - rho W, computed
with a dense inverse), spreg's multipliers, finite differences of the effects (the gradients of the delta method) and
the simulation (the standard errors). The comparison with R (spatialreg, splm) is in test_r_slm.py / test_r_plm.py.
"""

import warnings

import libpysal
import numpy as np
import pandas as pd
import pytest
import spreg

from apatools._impacts import (
    WeightsTraces,
    draw_parameters,
    effects_gradients,
    lag_effects,
    point_effects,
    series_order,
    split_parameters,
)
from apatools.plm import plm_effects
from apatools.slm import slm, slm_effects

from .test_plm import fit, simulate


def weights(kind):
    if kind == "rook":
        w = libpysal.weights.lat2W(7, 7, rook=True)
        w.transform = "r"
    elif (
        kind == "queen-binary"
    ):  # not row-standardized: the spectral radius is far above 1
        w = libpysal.weights.lat2W(6, 6, rook=False)
        w.transform = "b"
    else:  # asymmetric: the 4 nearest neighbours of random points
        points = np.random.default_rng(5).uniform(size=(40, 2))
        w = libpysal.weights.KNN.from_array(points, k=4)
        w.transform = "r"
    return w


KINDS = [("rook", 0.45), ("queen-binary", 0.08), ("knn", 0.5)]


def brute_force(w, rho, beta, theta):
    """Average diagonal and average row sum of (I - rho W)^-1 (beta I + theta W), from the dense inverse."""
    matrix = w.sparse.toarray()
    n = len(matrix)
    effect = np.linalg.inv(np.eye(n) - rho * matrix) @ (
        beta * np.eye(n) + theta * matrix
    )
    return np.diag(effect).mean(), effect.sum(axis=1).mean()


def series(w, rho, order=None):
    traces = WeightsTraces(w, traces="exact")
    order = order or series_order(abs(rho) * traces.radius)
    return (*traces.get(order), order)


@pytest.mark.parametrize("kind, rho", KINDS)
def test_traces_are_those_of_the_matrix_powers(kind, rho):
    w = weights(kind)
    matrix = w.sparse.toarray()
    tr, s = WeightsTraces(w, traces="exact").get(6)
    for k in range(8):
        power = np.linalg.matrix_power(matrix, k)
        assert tr[k] == pytest.approx(np.trace(power), rel=1e-10, abs=1e-8)
        assert s[k] == pytest.approx(power.sum(), rel=1e-10, abs=1e-8)
    assert WeightsTraces(w, traces="exact").radius == pytest.approx(
        np.max(np.abs(np.linalg.eigvals(matrix)))
    )


def test_monte_carlo_traces_approximate_the_exact_ones():
    w = weights("rook")
    exact, _ = WeightsTraces(w, traces="exact").get(8)
    mc = WeightsTraces(w, traces="mc", n_probes=3000, seed=1)
    approx, _ = mc.get(8)
    np.testing.assert_allclose(
        approx[1:], exact[1:], atol=0.1 * w.n
    )  # the estimator is unbiased, with noise
    assert mc.radius == pytest.approx(WeightsTraces(w, traces="exact").radius, rel=1e-6)


@pytest.mark.parametrize("kind, rho", KINDS)
@pytest.mark.parametrize("theta", [0.0, 0.35])
def test_point_effects_equal_the_dense_inverse(kind, rho, theta):
    w = weights(kind)
    tr, s, order = series(w, rho)
    beta = np.array([1.0, -0.5])
    effects = point_effects(beta, theta * np.ones(2), rho, tr, s, order)
    for j, b in enumerate(beta):
        direct, total = brute_force(w, rho, b, theta)
        assert effects[0, j] == pytest.approx(direct, abs=1e-8)
        assert effects[2, j] == pytest.approx(total, abs=1e-8)
        assert effects[1, j] == pytest.approx(total - direct, abs=1e-8)


def test_total_effect_of_row_standardized_weights_is_the_closed_form():
    w = weights("rook")
    tr, s, order = series(w, 0.45)
    effects = point_effects(np.array([1.0]), np.array([0.3]), 0.45, tr, s, order)
    assert effects[2, 0] == pytest.approx((1.0 + 0.3) / (1 - 0.45), abs=1e-8)


def test_multipliers_equal_spreg():
    w = weights("rook")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(w.n, 2))
    y = np.linalg.solve(
        np.eye(w.n) - 0.4 * w.sparse.toarray(), x @ [1, -0.5] + rng.normal(size=w.n)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spreg.ML_Lag(y[:, None], x, w, spat_impacts=["full"])
    direct, indirect, total = model.sp_multipliers["full"]
    tr, s, order = series(w, float(model.rho))
    effects = point_effects(
        np.array([1.0]), np.array([0.0]), float(model.rho), tr, s, order
    )
    assert (effects[0, 0], effects[1, 0], effects[2, 0]) == pytest.approx(
        (direct, indirect, total), abs=1e-8
    )


@pytest.mark.parametrize("kind, rho", KINDS)
def test_gradients_equal_finite_differences(kind, rho):
    w = weights(kind)
    tr, s, order = series(w, rho + 0.05, order=200)
    beta, theta = np.array([1.0, -0.5]), np.array([0.3, 0.1])
    gradient = effects_gradients(beta, theta, rho, tr, s, order)
    h = 1e-6

    def effects(b, t, r):
        return point_effects(b, t, r, tr, s, order)

    for j in range(2):
        numeric = np.empty((3, 3))
        for position, (db, dt, dr) in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            bump = np.array([db if i == j else 0 for i in range(2)], dtype=float)
            tump = np.array([dt if i == j else 0 for i in range(2)], dtype=float)
            up = effects(beta + h * bump, theta + h * tump, rho + h * dr)[:, j]
            down = effects(beta - h * bump, theta - h * tump, rho - h * dr)[:, j]
            numeric[:, position] = (up - down) / (2 * h)
        np.testing.assert_allclose(gradient[:, j, :], numeric, rtol=1e-5, atol=1e-7)


def synthetic(durbin=False):
    names = ["const", "x1", "x2"] + (["W_x1", "W_x2"] if durbin else []) + ["rho"]
    coefficients = np.array([0.2, 1.0, -0.5] + ([0.3, 0.1] if durbin else []) + [0.45])
    stds = np.array([0.1, 0.03, 0.03] + ([0.05, 0.05] if durbin else []) + [0.03])
    correlation = np.eye(len(names))
    correlation[1, -1] = correlation[
        -1, 1
    ] = -0.3  # the coefficients and rho are correlated
    return names, coefficients, correlation * np.outer(stds, stds)


@pytest.mark.parametrize("durbin", [False, True])
def test_analytic_standard_errors_agree_with_the_simulation(durbin):
    names, coefficients, cov = synthetic(durbin)
    w = weights("rook")
    analytic = lag_effects(names, coefficients, cov, w, inference="analytic").set_index(
        ["variable", "effect"]
    )
    simulated = lag_effects(
        names, coefficients, cov, w, inference="simulation", n_draws=40000
    ).set_index(["variable", "effect"])
    np.testing.assert_allclose(simulated["se"], analytic["se"], rtol=0.03)
    np.testing.assert_allclose(
        simulated["estimate"], analytic["estimate"]
    )  # the same point estimates
    # the analytic interval is symmetric, the simulated one made of percentiles
    np.testing.assert_allclose(
        analytic["cir"] - analytic["estimate"], analytic["estimate"] - analytic["cil"]
    )
    assert np.all(simulated["cil"] < simulated["estimate"]) and np.all(
        simulated["estimate"] < simulated["cir"]
    )
    z = analytic["estimate"] / analytic["se"]
    assert analytic["z"].to_numpy() == pytest.approx(z.to_numpy())


def test_the_covariance_between_coefficients_and_rho_matters():
    names, coefficients, cov = synthetic()
    w = weights("rook")
    independent = cov.copy()
    independent[1, -1] = independent[-1, 1] = 0.0
    with_cov = lag_effects(names, coefficients, cov, w, inference="analytic")
    without = lag_effects(names, coefficients, independent, w, inference="analytic")
    assert not np.allclose(with_cov["se"], without["se"])


def test_conf_level_and_seed():
    names, coefficients, cov = synthetic()
    w = weights("rook")
    a = lag_effects(names, coefficients, cov, w, inference="simulation", seed=3)
    b = lag_effects(names, coefficients, cov, w, inference="simulation", seed=3)
    c = lag_effects(names, coefficients, cov, w, inference="simulation", seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["se"], c["se"])
    narrow = lag_effects(
        names, coefficients, cov, w, inference="analytic", conf_level=0.8
    )
    wide = lag_effects(
        names, coefficients, cov, w, inference="analytic", conf_level=0.99
    )
    assert np.all(narrow["cir"] - narrow["cil"] < wide["cir"] - wide["cil"])


def test_draws_with_rho_outside_the_admissible_range_are_replaced():
    names, coefficients, cov = synthetic()
    cov = cov.copy()
    cov[-1, -1] = 0.05  # rho = 0.45 +- 0.22: many draws above 1
    _, positions = split_parameters(names)
    draws = draw_parameters(coefficients, cov, positions, 1.0, 5000, seed=1)
    assert np.all(np.abs(draws[:, -1]) < 1.0) and len(draws) == 5000


def test_parameter_names_and_errors():
    variables, (beta, theta, rho) = split_parameters(
        ["const", "x1", "x2", "W_x1", "W_x2", "phi", "rho"]
    )
    assert (
        variables == ["x1", "x2"]
        and list(beta) == [1, 2]
        and list(theta) == [3, 4]
        and rho == 6
    )
    assert split_parameters(["CONSTANT", "x1", "rho"])[1][1] is None
    with pytest.raises(ValueError, match="rho"):
        split_parameters(["const", "x1", "lambda"])
    with pytest.raises(ValueError, match="W_<name>"):
        split_parameters(["x1", "x2", "W_x1", "rho"])
    names, coefficients, cov = synthetic()
    w = weights("rook")
    with pytest.raises(ValueError, match="inference"):
        lag_effects(names, coefficients, cov, w, inference="bootstrap")
    with pytest.raises(ValueError, match="conf_level"):
        lag_effects(names, coefficients, cov, w, conf_level=95)
    with pytest.raises(ValueError, match="traces"):
        lag_effects(names, coefficients, cov, w, traces="fast")


def test_rho_outside_the_admissible_range_is_reported():
    names, coefficients, cov = synthetic()
    coefficients = coefficients.copy()
    coefficients[-1] = 1.2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = lag_effects(
            names, coefficients, cov, weights("rook"), inference="analytic"
        )
    assert [str(c.message)[:31] for c in caught] == [
        "|rho| is outside the admissible"
    ]  # and no overflow warnings
    assert (
        len(table) == 6
        and table[["estimate", "se", "p-value", "cil", "cir"]].isna().all().all()
    )


###################################
#  slm_effects and plm_effects    #
###################################


@pytest.fixture(scope="module")
def cross_section():
    rng = np.random.default_rng(1)
    w = libpysal.weights.lat2W(8, 8)
    w.transform = "r"
    n = w.n
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = np.linalg.solve(
        np.eye(n) - 0.4 * w.sparse.toarray(),
        df["x1"] - 0.5 * df["x2"] + rng.normal(size=n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, metrics = slm(
            df,
            "y",
            ["x1", "x2"],
            w=w,
            model=["ols", "slm", "sem", "slm"],
            verbose=False,
        )
    return w, results


def test_slm_effects_skip_models_without_spillovers_and_label_repeated_models(
    cross_section,
):
    w, results = cross_section
    with pytest.warns(UserWarning) as caught:
        table = slm_effects(results, w, inference="analytic")
    messages = " ".join(str(c.message) for c in caught)
    assert (
        "OLS has no spatial lag parameter" in messages
        and "spatial error model has no spillover" in messages
    )
    assert list(table.columns) == [
        "model",
        "variable",
        "effect",
        "estimate",
        "se",
        "z",
        "p-value",
        "cil",
        "cir",
    ]
    assert list(table["model"].unique()) == ["slm", "slm_2"] and len(table) == 12
    assert list(table["effect"][:3]) == ["Direct", "Indirect", "Total"] and list(
        table["variable"][::3][:2]
    ) == ["x1", "x2"]
    one = table[table["model"] == "slm"].set_index(["variable", "effect"])
    np.testing.assert_allclose(
        one["estimate"].xs("Direct", level="effect")
        + one["estimate"].xs("Indirect", level="effect"),
        one["estimate"].xs("Total", level="effect"),
    )


def test_slm_effects_equal_the_definition(cross_section):
    w, results = cross_section
    model = results[1]
    table = slm_effects([model], w, inference="analytic").set_index(
        ["variable", "effect"]
    )
    for variable, beta in zip(["x1", "x2"], np.asarray(model.betas).ravel()[1:3]):
        direct, total = brute_force(w, float(model.rho), beta, 0.0)
        assert table.loc[(variable, "Direct"), "estimate"] == pytest.approx(
            direct, abs=1e-8
        )
        assert table.loc[(variable, "Total"), "estimate"] == pytest.approx(
            total, abs=1e-8
        )


def test_slm_effects_formatting_and_options(cross_section):
    w, results = cross_section
    table = slm_effects(
        results[1:2],
        w,
        inference="analytic",
        decimal=2,
        add_stars=True,
        format_pval=True,
    )
    assert (
        table["estimate"].equals(table["estimate"].round(2))
        and "sig" in table
        and table["p-value"].dtype == object
    )
    empty = slm_effects([], w)
    assert (
        list(empty.columns)
        == [
            "model",
            "variable",
            "effect",
            "estimate",
            "se",
            "z",
            "p-value",
            "cil",
            "cir",
        ]
        and empty.empty
    )
    wide = slm_effects(results[1:2], w, inference="analytic").pivot(
        index="variable", columns="effect", values="estimate"
    )
    assert list(wide.columns) == ["Direct", "Indirect", "Total"]


@pytest.fixture(scope="module")
def panel_fits():
    df, w = simulate(rho=0.4, seed=7)
    fits = {
        name: fit(df, w, name, **kw)[0][0]
        for name, kw in [("slm_fe", {}), ("slm_re", {}), ("sem_fe", {})]
    }
    fits["durbin"] = fit(df, w, "slm_fe", slx_lags=1)[0][0]
    fits["twoway"] = fit(df, w, "slm_fe", effects="twoway")[0][0]
    fits["fe"] = fit(df, w, "fe")[0][0]
    return df, w, fits


def test_plm_effects_use_the_weights_of_the_panel(panel_fits):
    df, w, fits = panel_fits
    r = fits["slm_fe"]
    table = plm_effects([r], inference="analytic").set_index(["variable", "effect"])
    same = plm_effects([r], w=w, inference="analytic").set_index(["variable", "effect"])
    pd.testing.assert_frame_equal(
        table.drop(columns="model"), same.drop(columns="model")
    )
    rho = float(r.rho)
    for variable, beta in zip(["x1", "x2"], r.betas.ravel()[:2]):
        direct, total = brute_force(w, rho, beta, 0.0)
        assert table.loc[(variable, "Direct"), "estimate"] == pytest.approx(
            direct, abs=1e-8
        )
        assert table.loc[(variable, "Total"), "estimate"] == pytest.approx(
            total, abs=1e-8
        )


def test_plm_effects_of_random_effects_period_effects_and_the_durbin_model(panel_fits):
    df, w, fits = panel_fits
    random = plm_effects([fits["slm_re"]], inference="analytic")
    assert list(random["variable"].unique()) == ["x1", "x2"]  # no constant, no phi
    twoway = plm_effects([fits["twoway"]], inference="analytic")
    assert list(twoway["variable"].unique()) == [
        "x1",
        "x2",
    ]  # the period dummies are not regressors
    durbin = plm_effects([fits["durbin"]], inference="analytic").set_index(
        ["variable", "effect"]
    )
    r = fits["durbin"]
    b, t = r.betas.ravel()[:2], r.betas.ravel()[2:4]
    for j, variable in enumerate(["x1", "x2"]):
        direct, total = brute_force(w, float(r.rho), b[j], t[j])
        assert durbin.loc[(variable, "Direct"), "estimate"] == pytest.approx(
            direct, abs=1e-8
        )
        assert durbin.loc[(variable, "Total"), "estimate"] == pytest.approx(
            total, abs=1e-8
        )


def test_plm_effects_skip_models_without_a_spatial_lag(panel_fits):
    df, w, fits = panel_fits
    with pytest.warns(UserWarning) as caught:
        table = plm_effects(
            [fits["fe"], fits["sem_fe"], fits["slm_fe"], fits["slm_fe"]],
            inference="analytic",
        )
    messages = " ".join(str(c.message) for c in caught)
    assert (
        "no spatial lag parameter" in messages
        and "spatial error model already are the direct and total" in messages
    )
    assert list(table["model"].unique()) == ["slm_fe", "slm_fe_2"]


def test_plm_effects_need_weights_when_the_results_do_not_carry_them(panel_fits):
    df, w, fits = panel_fits
    r = fits["slm_fe"]
    plm_panel = r.plm_panel
    try:
        del r.plm_panel
        with pytest.raises(ValueError, match="weights"):
            plm_effects([r])
        assert len(plm_effects([r], w=w, inference="analytic")) == 6
    finally:
        r.plm_panel = plm_panel
