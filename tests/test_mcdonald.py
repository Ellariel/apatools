"""McDonald's omega variants: robust (Huber M-estimated covariance), raw metric and bifactor (hierarchical, total).

The R tests (lavaan, coefficientalpha) are skipped without R and these packages.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from apatools.lib import mcdonald_omega, mcdonald_omega_bifactor
from apatools.lib._robust import huber_asymptotic_covariance, huber_location_scatter
from apatools.lib.mcdonald import (
    _bifactor_coefficients,
    _normal_gamma,
    _omega_delta_se,
    _one_factor_coefficients,
    _one_factor_ml,
    _phi_function,
    _sandwich_covariance,
)

from .r_reference import run_r

GENERAL = np.array([0.7, 0.6, 0.65, 0.55, 0.6, 0.7, 0.5, 0.6, 0.65])
SPECIFIC = np.array([0.4, 0.35, 0.3, 0.45, 0.4, 0.35, 0.3, 0.4, 0.45])
GROUPS = {"A": ["x1", "x2", "x3"], "B": ["x4", "x5", "x6"], "C": ["x7", "x8", "x9"]}
POSITIONS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def bifactor_loadings():
    loadings = np.zeros((9, 4))
    loadings[:, 0] = GENERAL
    for item, group in enumerate([0, 0, 0, 1, 1, 1, 2, 2, 2]):
        loadings[item, group + 1] = SPECIFIC[item]
    return loadings, 1 - (loadings**2).sum(axis=1)


def exact_sample(covariance, n=500, seed=0):
    """A sample whose covariance matrix (divisor n) is exactly ``covariance``."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, len(covariance)))
    q, _ = np.linalg.qr(z - z.mean(axis=0))
    return q * np.sqrt(n) @ np.linalg.cholesky(covariance).T


def bifactor_data(n=800, seed=1, scale=None):
    loadings, theta = bifactor_loadings()
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 4))
    data = factors @ loadings.T + rng.normal(size=(n, 9)) * np.sqrt(theta)
    if scale is not None:
        data = data * scale
    return pd.DataFrame(data, columns=[f"x{i + 1}" for i in range(9)])


def true_coefficients(standardized=True):
    loadings, theta = bifactor_loadings()
    return _bifactor_coefficients(loadings, theta, POSITIONS, standardized)


NAMES = (
    ["omega_h", "omega_t", "ecv"]
    + [f"omega_s[{g}]" for g in GROUPS]
    + [f"omega_hs[{g}]" for g in GROUPS]
)


###########################
# the Huber M-estimator   #
###########################


def test_huber_location_scatter_is_the_sample_estimate_without_downweighting_and_resists_outliers():
    rng = np.random.default_rng(0)
    x = (
        rng.normal(size=(400, 4))
        @ np.linalg.cholesky(
            np.array(
                [
                    [1, 0.5, 0.2, 0],
                    [0.5, 1, 0.3, 0.1],
                    [0.2, 0.3, 1, 0.4],
                    [0, 0.1, 0.4, 1],
                ]
            )
        ).T
    )
    mu, sigma = huber_location_scatter(x, varphi=0)
    assert mu == pytest.approx(x.mean(axis=0)) and sigma == pytest.approx(
        np.cov(x, rowvar=False, bias=True)
    )
    # clean normal data: consistent for the covariance matrix (the constant kappa corrects the downweighting)
    _mu10, sigma10 = huber_location_scatter(x, varphi=0.1)
    assert sigma10 == pytest.approx(sigma, abs=0.08)
    # with outliers the sample covariance matrix explodes, the M-estimate does not
    contaminated = x.copy()
    contaminated[:20] += 30
    classical = np.cov(contaminated, rowvar=False, bias=True)
    _, robust = huber_location_scatter(contaminated, varphi=0.1)
    assert np.abs(classical - sigma).max() > 20 and np.abs(robust - sigma).max() < 0.35
    # it is a fixed point of the estimating equations: re-estimating from the solution changes nothing
    again = huber_location_scatter(contaminated, 0.1, tolerance=1e-13)
    assert again[1] == pytest.approx(robust, abs=1e-7)
    # shifting and scaling the data shifts and scales the estimates (equivariance)
    shifted_mu, shifted_sigma = huber_location_scatter(2 * contaminated + 5, 0.1)
    assert shifted_mu == pytest.approx(
        2 * huber_location_scatter(contaminated, 0.1)[0] + 5, abs=1e-6
    )
    assert shifted_sigma == pytest.approx(4 * robust, abs=1e-5)


def test_huber_asymptotic_covariance_matches_the_sampling_variability():
    n, p = 400, 3
    cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    chol = np.linalg.cholesky(cov)
    rows, columns = np.triu_indices(p)
    estimates, predicted = [], []
    for seed in range(250):
        rng = np.random.default_rng(seed)
        x = rng.standard_t(8, size=(n, p)) @ chol.T
        mu, sigma = huber_location_scatter(x, 0.1)
        estimates.append(sigma[rows, columns])
        if seed < 15:
            predicted.append(
                np.diag(huber_asymptotic_covariance(x, mu, sigma, 0.1)) / n
            )
    observed_variance = np.var(estimates, axis=0, ddof=1)
    assert np.mean(predicted, axis=0) == pytest.approx(
        observed_variance, rel=0.3
    )  # the sandwich estimates the true variance
    # without downweighting the covariance matrix of the sample covariance matrix under normality is (I + K)(S kron S)
    normal = np.random.default_rng(7).normal(size=(5000, p)) @ chol.T
    mu, sigma = huber_location_scatter(normal, 0)
    gamma = huber_asymptotic_covariance(normal, mu, sigma, 0)
    d = np.zeros((p * p, len(rows)))
    d[rows * p + columns, np.arange(len(rows))] = 1
    d[columns * p + rows, np.arange(len(rows))] = 1
    vech_normal = np.linalg.pinv(d) @ _normal_gamma(sigma) @ np.linalg.pinv(d).T
    assert np.diag(gamma) == pytest.approx(np.diag(vech_normal), rel=0.2)


###################################
# one-factor omega: new options   #
###################################


def one_factor_data(n=400, seed=0, outliers=0):
    rng = np.random.default_rng(seed)
    loadings = np.array([0.8, 0.7, 0.6, 0.7, 0.5, 0.6])
    f = rng.normal(size=n)
    x = f[:, None] * loadings + rng.normal(size=(n, 6)) * np.sqrt(1 - loadings**2)
    x[:outliers] += rng.normal(size=(outliers, 6)) * 4
    return pd.DataFrame(x, columns=list("abcdef"))


def test_standardized_omega_is_invariant_to_the_scale_of_the_items_and_the_raw_one_is_not():
    data = one_factor_data(seed=1)
    standardized = mcdonald_omega(data)
    rescaled = data.assign(a=data["a"] * 5, d=data["d"] * 0.2)
    assert mcdonald_omega(rescaled) == pytest.approx(
        standardized, abs=1e-8
    )  # unchanged default
    raw, raw_rescaled = (
        mcdonald_omega(data, standardized=False),
        mcdonald_omega(rescaled, standardized=False),
    )
    assert raw[0] != pytest.approx(raw_rescaled[0], abs=1e-3)
    # for items with variance 1 both definitions give the same estimate; the raw omega is a different statistic (it
    # is not invariant to the scale of the items), so its interval is of a similar size, not identical
    z = (data - data.mean()) / data.std(ddof=0)
    same, other = mcdonald_omega(z), mcdonald_omega(z, standardized=False)
    assert other[0] == pytest.approx(same[0], abs=1e-7)
    assert (other[2] - other[1]) == pytest.approx(same[2] - same[1], rel=0.1)
    # the raw omega is the reliability of the raw sum score: (sum of loadings)^2 / (variance of the model-implied sum)
    loadings = (
        np.sqrt(np.diag(np.cov(rescaled, rowvar=False, bias=True)))
        * _one_factor_ml(np.corrcoef(rescaled, rowvar=False))[:6]
    )
    uniquenesses = (
        np.diag(np.cov(rescaled, rowvar=False, bias=True))
        * _one_factor_ml(np.corrcoef(rescaled, rowvar=False))[6:]
    )
    assert mcdonald_omega(rescaled, standardized=False)[0] == pytest.approx(
        loadings.sum() ** 2 / (loadings.sum() ** 2 + uniquenesses.sum())
    )


def test_generic_sandwich_reproduces_the_explicit_delta_method():
    data = one_factor_data(n=300, seed=2).to_numpy()
    n, p = data.shape
    theta = _one_factor_ml(np.corrcoef(data, rowvar=False))
    loadings, uniquenesses = theta[:p, None], theta[p:]
    mask = np.ones((p, 1), dtype=bool)

    def function(matrix, unique):
        return _one_factor_coefficients(matrix[:, 0], unique, True)

    gamma = _normal_gamma(loadings @ loadings.T + np.diag(uniquenesses))
    variance = _sandwich_covariance(
        loadings, uniquenesses, mask, gamma, n, _phi_function(mask, function)
    )
    assert np.sqrt(variance[0, 0]) == pytest.approx(_omega_delta_se(theta, n), rel=1e-6)


def test_robust_omega_resists_outliers():
    clean = one_factor_data(n=300, seed=3)
    contaminated = one_factor_data(n=300, seed=3, outliers=24)
    est_clean = mcdonald_omega(clean)[0]
    ordinary, robust = (
        mcdonald_omega(contaminated)[0],
        mcdonald_omega(contaminated, robust=True)[0],
    )
    assert ordinary < est_clean - 0.15  # outliers destroy the ordinary estimate
    assert (
        abs(robust - est_clean) < 0.06
        and abs(robust - est_clean) < abs(ordinary - est_clean) / 3
    )
    # varphi = 0 keeps the sample covariance matrix, with the distribution-free standard error
    plain = mcdonald_omega(clean)
    free = mcdonald_omega(clean, robust=True, varphi=0)
    assert free[0] == pytest.approx(plain[0], abs=1e-8) and free[1] < plain[0] < free[2]
    # robust omega depends on varphi, and on the scale option
    assert mcdonald_omega(contaminated, robust=True, varphi=0.2)[0] != pytest.approx(
        robust, abs=1e-4
    )
    assert mcdonald_omega(contaminated, robust=True, standardized=False)[
        0
    ] != pytest.approx(robust, abs=1e-4)


def test_robust_omega_has_the_right_behaviour_in_simulations():
    loadings = np.array([0.8, 0.7, 0.6, 0.7, 0.5, 0.6])
    truth = loadings.sum() ** 2 / (loadings.sum() ** 2 + (1 - loadings**2).sum())
    estimates = {"ordinary": [], "robust": []}
    covered = {"ordinary": 0, "robust": 0}
    clean_covered = 0
    reps = 60
    for seed in range(reps):
        for contaminate in (True, False):
            data = one_factor_data(
                n=300, seed=900 + seed, outliers=24 if contaminate else 0
            )
            for name, kwargs in (("ordinary", {}), ("robust", {"robust": True})):
                est, low, high = mcdonald_omega(data, **kwargs)
                if contaminate:
                    estimates[name].append(est)
                    covered[name] += low <= truth <= high
                elif name == "robust":
                    clean_covered += low <= truth <= high
    assert (
        np.mean(estimates["ordinary"]) < truth - 0.15
        and abs(np.mean(estimates["robust"]) - truth) < 0.06
    )
    assert (
        covered["ordinary"] == 0 and covered["robust"] / reps > 0.4
    )  # robust intervals are not perfect with 8 % outliers
    assert 0.88 <= clean_covered / reps <= 1.0  # and about right for clean data


def test_robust_and_raw_options_work_with_the_bootstrap():
    data = one_factor_data(n=150, seed=4, outliers=12)
    est, low, high = mcdonald_omega(data, robust=True, bootstrap_iter=40, seed=1)
    assert low < est < high and (est, low, high) == pytest.approx(
        mcdonald_omega(data, robust=True, bootstrap_iter=40, seed=1)
    )
    raw = mcdonald_omega(data, standardized=False, bootstrap_iter=40, seed=1)
    assert raw[1] < raw[0] < raw[2] and raw[0] != pytest.approx(est, abs=1e-3)
    # the analytic and bootstrap intervals of the robust omega are of the same size
    analytic = mcdonald_omega(data, robust=True)
    assert (analytic[2] - analytic[1]) == pytest.approx(high - low, rel=0.6)


def test_option_errors():
    data = one_factor_data(n=80, seed=5)
    for bad in (-0.1, 1.0, 2):
        with pytest.raises(ValueError, match="varphi"):
            mcdonald_omega(data, robust=True, varphi=bad)
        with pytest.raises(ValueError, match="varphi"):
            mcdonald_omega_bifactor(bifactor_data(n=100), GROUPS, varphi=bad)


###################
# bifactor omega  #
###################


def test_bifactor_coefficients_follow_their_definitions():
    loadings, theta = bifactor_loadings()
    values = true_coefficients(standardized=True)
    g = GENERAL
    group_sum = lambda k: SPECIFIC[POSITIONS[k]].sum()
    total = g.sum() ** 2 + sum(group_sum(k) ** 2 for k in range(3)) + theta.sum()
    assert values[0] == pytest.approx(g.sum() ** 2 / total)  # omega_h
    assert values[1] == pytest.approx(
        (g.sum() ** 2 + sum(group_sum(k) ** 2 for k in range(3))) / total
    )  # omega_t
    assert values[2] == pytest.approx(
        (g**2).sum() / ((g**2).sum() + (SPECIFIC**2).sum())
    )  # ecv
    for k in range(3):
        items = POSITIONS[k]
        variance_k = g[items].sum() ** 2 + group_sum(k) ** 2 + theta[items].sum()
        assert values[3 + k] == pytest.approx(
            (g[items].sum() ** 2 + group_sum(k) ** 2) / variance_k
        )  # omega_s
        assert values[6 + k] == pytest.approx(
            group_sum(k) ** 2 / variance_k
        )  # omega_hs
    assert values[0] < values[1] < 1 and np.all(values[6:] < values[3:6])
    # the model-implied variance of the total score is 1'Sigma 1
    sigma = loadings @ loadings.T + np.diag(theta)
    assert total == pytest.approx(np.ones(9) @ sigma @ np.ones(9))


def test_bifactor_omega_recovers_the_population_values_exactly_from_a_perfect_sample():
    loadings, theta = bifactor_loadings()
    sigma = loadings @ loadings.T + np.diag(theta)
    frame = pd.DataFrame(
        exact_sample(sigma, n=600, seed=3), columns=[f"x{i + 1}" for i in range(9)]
    )
    result = mcdonald_omega_bifactor(frame, GROUPS)
    assert list(result.index) == NAMES and list(result.columns) == [
        "estimate",
        "lower",
        "upper",
    ]
    assert result["estimate"].to_numpy() == pytest.approx(
        true_coefficients(True), abs=2e-5
    )
    assert (
        (result["lower"] < result["estimate"]) & (result["estimate"] < result["upper"])
    ).all()
    # raw metric: items with other variances; the standardized coefficients do not change
    scale = np.array([3, 1, 1, 1, 2, 1, 1, 0.5, 1])
    scaled = frame * scale
    assert mcdonald_omega_bifactor(scaled, GROUPS)[
        "estimate"
    ].to_numpy() == pytest.approx(true_coefficients(True), abs=2e-5)
    raw = mcdonald_omega_bifactor(scaled, GROUPS, standardized=False)[
        "estimate"
    ].to_numpy()
    scaled_loadings, scaled_theta = (
        bifactor_loadings()[0] * scale[:, None],
        bifactor_loadings()[1] * scale**2,
    )
    assert raw == pytest.approx(
        _bifactor_coefficients(scaled_loadings, scaled_theta, POSITIONS, False),
        abs=2e-5,
    )
    assert raw != pytest.approx(true_coefficients(True), abs=1e-3)


def test_bifactor_groups_can_be_given_in_several_ways_and_items_may_be_left_out():
    frame = bifactor_data(n=500, seed=2)
    by_dict = mcdonald_omega_bifactor(frame, GROUPS)
    by_list = mcdonald_omega_bifactor(frame, list(GROUPS.values()))
    assert list(by_list.index)[3:] == [
        "omega_s[1]",
        "omega_s[2]",
        "omega_s[3]",
        "omega_hs[1]",
        "omega_hs[2]",
        "omega_hs[3]",
    ]
    assert by_list["estimate"].to_numpy() == pytest.approx(
        by_dict["estimate"].to_numpy()
    )
    by_position = mcdonald_omega_bifactor(
        frame.to_numpy(), POSITIONS
    )  # arrays: positions
    assert by_position["estimate"].to_numpy() == pytest.approx(
        by_dict["estimate"].to_numpy()
    )
    # the columns of the data may be in any order, the names decide
    shuffled = mcdonald_omega_bifactor(
        frame[["x9", "x1", "x5", "x2", "x8", "x3", "x7", "x6", "x4"]], GROUPS
    )
    assert shuffled["estimate"].to_numpy() == pytest.approx(
        by_dict["estimate"].to_numpy(), abs=1e-5
    )
    # an item without a group loads on the general factor only (here: the last two items are left out)
    partial = mcdonald_omega_bifactor(
        frame, {"A": GROUPS["A"], "B": GROUPS["B"], "C": ["x7", "x8", "x9"][:3]}
    )
    assert partial.shape == (9, 3)
    left_out = mcdonald_omega_bifactor(
        frame.assign(x10=frame["x1"] * 0.6 + np.random.default_rng(0).normal(size=500)),
        GROUPS,
    )
    assert left_out.shape == (9, 3) and left_out.loc["omega_h", "estimate"] > 0.6


def test_bifactor_errors_and_warnings():
    frame = bifactor_data(n=300, seed=3)
    with pytest.raises(ValueError, match="not in the data.*'nope'"):
        mcdonald_omega_bifactor(
            frame, {"A": ["x1", "x2", "nope"], "B": ["x4", "x5", "x6"]}
        )
    with pytest.raises(ValueError, match="more than one group"):
        mcdonald_omega_bifactor(
            frame, {"A": ["x1", "x2", "x3"], "B": ["x3", "x4", "x5"]}
        )
    with pytest.raises(ValueError, match="At least two groups"):
        mcdonald_omega_bifactor(frame, {"A": GROUPS["A"]})
    with pytest.raises(ValueError, match=r"at least 3 items.*\['B'\]"):
        mcdonald_omega_bifactor(frame, {"A": GROUPS["A"], "B": ["x4", "x5"]})
    with pytest.raises(ValueError, match="At least 6 items"):
        mcdonald_omega_bifactor(
            frame[["x1", "x2", "x3", "x4", "x5"]],
            {"A": ["x1", "x2", "x3"], "B": ["x4", "x5", "x1"]},
        )
    with pytest.raises(ValueError, match="More respondents than items"):
        mcdonald_omega_bifactor(frame.iloc[:8], GROUPS)
    # reverse-keyed items are noticed
    reversed_item = frame.assign(x2=-frame["x2"], x5=-frame["x5"])
    with pytest.warns(UserWarning, match="different signs"):
        mcdonald_omega_bifactor(reversed_item, GROUPS)
    # rows with missing values are dropped
    gappy = frame.copy()
    gappy.iloc[::10, 2] = np.nan
    assert mcdonald_omega_bifactor(gappy, GROUPS)["estimate"].notna().all()


def test_bifactor_intervals_have_about_the_right_coverage():
    truth = true_coefficients(True)
    covered = np.zeros(9)
    reps = 100
    for seed in range(reps):
        result = mcdonald_omega_bifactor(bifactor_data(n=600, seed=2000 + seed), GROUPS)
        covered += ((result["lower"] <= truth) & (truth <= result["upper"])).to_numpy()
    rates = covered / reps
    assert rates.min() >= 0.85 and rates.mean() >= 0.91, dict(zip(NAMES, rates))
    # the bifactor ML estimates are unbiased for the population values
    estimates = np.array(
        [
            mcdonald_omega_bifactor(bifactor_data(n=600, seed=3000 + s), GROUPS)[
                "estimate"
            ].to_numpy()
            for s in range(30)
        ]
    )
    assert estimates.mean(axis=0) == pytest.approx(truth, abs=0.02)


def test_bifactor_bootstrap_robust_and_boundary_cases():
    frame = bifactor_data(n=250, seed=4)
    est = mcdonald_omega_bifactor(frame, GROUPS, bootstrap_iter=30, seed=5)
    again = mcdonald_omega_bifactor(frame, GROUPS, bootstrap_iter=30, seed=5)
    pd.testing.assert_frame_equal(est, again)  # the seed makes it reproducible
    assert ((est["lower"] <= est["estimate"]) & (est["estimate"] <= est["upper"])).all()
    analytic = mcdonald_omega_bifactor(frame, GROUPS)
    pd.testing.assert_series_equal(est["estimate"], analytic["estimate"])
    assert (
        ((est["upper"] - est["lower"]) / (analytic["upper"] - analytic["lower"]))
        .between(0.4, 2.5)
        .all()
    )
    # robust: outliers do not destroy omega_h, the ordinary estimates move a lot
    contaminated = frame.copy()
    contaminated.iloc[:25] += np.random.default_rng(0).normal(size=(25, 9)) * 4
    ordinary, robust = (
        mcdonald_omega_bifactor(contaminated, GROUPS),
        mcdonald_omega_bifactor(contaminated, GROUPS, robust=True),
    )
    assert abs(
        robust.loc["omega_h", "estimate"] - analytic.loc["omega_h", "estimate"]
    ) < abs(ordinary.loc["omega_h", "estimate"] - analytic.loc["omega_h", "estimate"])
    assert (
        abs(robust.loc["omega_h", "estimate"] - analytic.loc["omega_h", "estimate"])
        < 0.08
    )
    assert robust["lower"].notna().all()
    # a boundary solution (a uniqueness at its bound): no analytic interval, with a warning; the bootstrap still works
    rng = np.random.default_rng(6)
    base = rng.normal(size=300)
    degenerate = pd.DataFrame(
        {f"x{i + 1}": base * 0.7 + rng.normal(size=300) * 0.7 for i in range(9)}
    )
    degenerate["x1"] = base  # a perfect indicator: its uniqueness is zero
    with pytest.warns(UserWarning, match="boundary"):
        result = mcdonald_omega_bifactor(degenerate, GROUPS)
    assert result["estimate"].notna().all() and result["lower"].isna().all()


#############################
# against lavaan (R)        #
#############################


def test_bifactor_omega_agrees_with_lavaan():
    frame = bifactor_data(n=1500, seed=7, scale=np.array([3, 1, 1, 1, 2, 1, 1, 0.5, 1]))
    script = """
model <- 'g =~ a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+a7*x7+a8*x8+a9*x9
A =~ b1*x1+b2*x2+b3*x3
B =~ b4*x4+b5*x5+b6*x6
C =~ b7*x7+b8*x8+b9*x9
x1 ~~ e1*x1; x2 ~~ e2*x2; x3 ~~ e3*x3; x4 ~~ e4*x4; x5 ~~ e5*x5; x6 ~~ e6*x6; x7 ~~ e7*x7; x8 ~~ e8*x8; x9 ~~ e9*x9
sg := (a1+a2+a3+a4+a5+a6+a7+a8+a9)^2
sA := (b1+b2+b3)^2
sB := (b4+b5+b6)^2
sC := (b7+b8+b9)^2
V := sg + sA + sB + sC + e1+e2+e3+e4+e5+e6+e7+e8+e9
omega_h := sg / V
omega_t := (sg + sA + sB + sC) / V
VA := (a1+a2+a3)^2 + sA + e1+e2+e3
omega_hs_A := sA / VA
omega_s_A := ((a1+a2+a3)^2 + sA) / VA'
fit <- cfa(model, d, orthogonal = TRUE, std.lv = TRUE)
pe <- parameterEstimates(fit)
pe <- pe[pe$op == ":=" & pe$lhs %in% c("omega_h", "omega_t", "omega_hs_A", "omega_s_A"), c("lhs", "est", "se")]
"""
    pe = run_r(script, frames={"d": frame}, outputs=["pe"], packages=["lavaan"])["pe"]
    reference = {row.lhs: (row.est, row.se) for row in pe.itertuples()}
    ours = mcdonald_omega_bifactor(frame, GROUPS, standardized=False)
    z = norm.ppf(0.975)
    for lavaan_name, name in {
        "omega_h": "omega_h",
        "omega_t": "omega_t",
        "omega_hs_A": "omega_hs[A]",
        "omega_s_A": "omega_s[A]",
    }.items():
        est, low, high = ours.loc[name]
        se = (
            (np.log(high / (1 - high)) - np.log(low / (1 - low)))
            / (2 * z)
            * est
            * (1 - est)
        )  # the standard error behind the interval
        assert est == pytest.approx(reference[lavaan_name][0], abs=2e-5), name
        assert se == pytest.approx(reference[lavaan_name][1], rel=0.01), name


def test_robust_omega_agrees_with_the_r_package_coefficientalpha():
    rng = np.random.default_rng(3)
    n, p = 300, 5
    factor = rng.normal(size=n)
    x = np.column_stack(
        [
            lam * factor + rng.normal(size=n) * s
            for lam, s in zip([1, 0.8, 0.7, 0.9, 0.6], [1, 1, 1.2, 0.9, 1.1])
        ]
    )
    x[:12] += rng.normal(size=(12, p)) * 5
    frame = pd.DataFrame(x, columns=[f"v{i}" for i in range(1, p + 1)])
    script = """
rows <- lapply(c(0, 0.1), function(phi) { o <- omega(y, varphi = phi, se = TRUE, test = FALSE); data.frame(phi = phi, omega = o$omega, se = o$se) })
res <- do.call(rbind, rows)
"""
    rows = run_r(
        script, frames={"y": frame}, outputs=["res"], packages=["coefficientalpha"]
    )["res"].to_numpy()
    z = norm.ppf(0.975)
    for phi, omega, se in rows:
        est, low, high = mcdonald_omega(
            frame, robust=True, varphi=phi, standardized=False
        )
        our_se = (
            (np.log(high / (1 - high)) - np.log(low / (1 - low)))
            / (2 * z)
            * est
            * (1 - est)
        )
        assert est == pytest.approx(omega, abs=1e-5), phi
        assert our_se == pytest.approx(se, rel=0.02), (
            phi
        )  # Zhang & Yuan's robust standard error


def test_warnings_are_not_raised_for_well_behaved_data():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcdonald_omega(one_factor_data(n=300, seed=8), robust=True)
        mcdonald_omega_bifactor(bifactor_data(n=800, seed=8), GROUPS)
