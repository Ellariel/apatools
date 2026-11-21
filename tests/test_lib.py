import itertools
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
from statsmodels.multivariate.factor import Factor
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.stats.inter_rater import aggregate_raters

import apatools.lib as lib_package
import apatools.lib.cronbach as cronbach_module
import apatools.lib.mcdonald as mcdonald_module
from apatools.calc import ci_from_r, p_from_r
from apatools.lib.cronbach import _alpha_from_covariance, _alpha_standard_error
from apatools.lib.mcdonald import _omega_delta_se, _omega_from_theta, _one_factor_ml
from apatools.lib import (
    average_correlations,
    bootstrap,
    cronbachs_alpha,
    dependent_correlations,
    fleiss_kappa,
    gini,
    gini_from_percentiles,
    htmt_ratio,
    independent_correlations,
    krippendorffs_alpha,
    log_transform,
    mcdonald_omega,
    rho_rxy_rxz,
    rz_ci,
    z_transform,
)


def s(x):
    return f"{x:.3f}"


def test_lib():
    # fleiss_kappa
    df = pd.DataFrame(
        {
            "Expert1": ["for", "no decision", "for", "for", "no decision", np.nan],
            "Expert2": [
                "for",
                "no decision",
                "for",
                "no decision",
                "no decision",
                np.nan,
            ],
            "Expert3": ["for", "no decision", "for", "for", "for", np.nan],
            "Expert4": ["for", "no decision", "for", "for", "no decision", np.nan],
        }
    )
    kappa, z_score, p_value, per_item = fleiss_kappa(df, return_per_item_agreement=True)
    # the last item has no ratings at all and is left out (it is not a "missing" category)
    assert s(kappa) == "0.583"
    assert s(z_score) == "3.195"
    assert s(p_value) == "0.001"
    assert len(per_item) == len(df) and per_item.iloc[:5].notna().all() and np.isnan(per_item.iloc[5])


###########
# helpers #
###########


def correlated_sample(corr, n, rng):
    """Normal sample with exactly the sample correlation matrix ``corr`` (orthogonalized draws)."""
    k = len(corr)
    a = rng.normal(size=(n, k))
    q, _ = np.linalg.qr(a - a.mean(axis=0))  # orthonormal and still centered
    z = q / q.std(axis=0, ddof=0)
    return z @ np.linalg.cholesky(np.asarray(corr)).T


########
# gini #
########


def test_gini():
    assert gini([1, 1, 1, 1]) == pytest.approx(0, abs=1e-5)
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75, abs=1e-4)  # (n - 1) / n
    assert gini([0] * 9 + [1]) == pytest.approx(0.9, abs=1e-4)
    # scale and order invariant
    x = np.array([1.0, 3, 4, 10, 25])
    assert gini(x) == pytest.approx(gini(10 * x), rel=1e-4)
    assert gini(x) == pytest.approx(gini(x[::-1]))
    # negative values are shifted to start at zero
    assert gini([-1, 0, 1]) == pytest.approx(gini([0, 1, 2]), abs=1e-5)
    # zeros can be dropped
    assert gini([0, 0, 1, 1], drop_zeros=True) == pytest.approx(0, abs=1e-5)
    assert gini([0, 0, 1, 1]) > 0.4
    # more unequal -> larger
    assert gini([1, 1, 1, 10]) > gini([1, 2, 3, 4])


def test_gini_from_percentiles():
    pct = np.arange(10, 101, 10)
    # perfect equality: cumulative share equals the percentile (percentiles do not start at 0!)
    assert gini_from_percentiles(pct, pct, values_are_shares=True) == pytest.approx(
        0, abs=1e-12
    )
    assert gini_from_percentiles(pct, pct) == pytest.approx(
        0, abs=1e-12
    )  # cumulative values
    # one person has everything: 1 - 1/n
    assert gini_from_percentiles(
        [0] * 9 + [100], pct, values_are_shares=True
    ) == pytest.approx(0.9)
    # the same with percentiles that start at 0
    pct0 = np.arange(0, 101, 10)
    assert gini_from_percentiles(pct0, pct0, values_are_shares=True) == pytest.approx(
        0, abs=1e-12
    )
    assert gini_from_percentiles(
        [0] * 10 + [100], pct0, values_are_shares=True
    ) == pytest.approx(0.9)
    # agrees with the sample Gini of the underlying (decile) data: incomes 1..10 per decile
    incomes = np.arange(1, 11)
    cum = np.cumsum(incomes)
    assert gini_from_percentiles(cum, pct) == pytest.approx(gini(incomes), abs=0.02)
    # cumulative values and shares give the same result
    assert gini_from_percentiles(cum, pct) == pytest.approx(
        gini_from_percentiles(100 * cum / cum[-1], pct, values_are_shares=True)
    )


###################
# transformations #
###################


def test_log_and_z_transform():
    assert log_transform([-99, 0, 9, 99]).tolist() == pytest.approx([-2, 0, 1, 2])
    z = z_transform([1.0, 2, 3, 4, 10])
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std() == pytest.approx(1)
    z, m, sd = z_transform([2.0, 4, 6], return_params=True)
    assert (m, sd) == (
        4.0,
        pytest.approx(np.sqrt(8 / 3)),
    ) and z.tolist() == pytest.approx([-1.2247, 0, 1.2247], abs=1e-4)
    # NaNs are ignored for mean and sd and stay NaN
    z = z_transform([1.0, np.nan, 3.0])
    assert z[0] == pytest.approx(-1) and z[2] == pytest.approx(1) and np.isnan(z[1])
    # constant or empty input: sd is 0 / undefined; NaN (or an error), never a division by zero
    with pytest.raises(ValueError, match="zero"):
        z_transform([5, 5, 5], ignore_errors=False)
    with pytest.raises(ValueError, match="zero"):
        z_transform([np.nan, np.nan], ignore_errors=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning (division by zero, mean of empty slice)
        assert np.isnan(z_transform([5, 5, 5])).all()
        assert np.isnan(z_transform([np.nan, np.nan])).all()
        z, m, sd = z_transform([5.0, np.nan, 5.0], return_params=True)
        assert (m, sd) == (5.0, 0.0) and np.isnan(z).all()
        assert z_transform([]).size == 0


######################
# interrater measures #
######################

# Fleiss (1971), as used in the Wikipedia article: 10 items, 14 raters, 5 categories; kappa = 0.210
FLEISS_COUNTS = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


def counts_to_ratings(counts):
    rows = [[c for c, k in enumerate(row) for _ in range(k)] for row in counts]
    return pd.DataFrame(rows, columns=[f"rater{i}" for i in range(len(rows[0]))])


def test_fleiss_kappa_reference_values():
    df = counts_to_ratings(FLEISS_COUNTS)
    kappa, z, p = fleiss_kappa(df)
    assert kappa == pytest.approx(0.210, abs=5e-4)
    assert kappa == pytest.approx(
        sm_fleiss_kappa(np.array(FLEISS_COUNTS)), abs=1e-12
    )  # statsmodels
    assert p == pytest.approx(2 * norm.sf(abs(z))) and z > 0
    assert fleiss_kappa(df, method="one-tailed")[2] == pytest.approx(p / 2)
    # perfect agreement
    perfect = pd.DataFrame({"a": list("xyzx"), "b": list("xyzx"), "c": list("xyzx")})
    assert fleiss_kappa(perfect)[0] == pytest.approx(1.0)
    # random ratings: kappa near zero, and compared with statsmodels on random data
    rng = np.random.default_rng(1)
    rnd = pd.DataFrame(rng.integers(0, 3, size=(200, 5)))
    table = aggregate_raters(rnd.astype(str))[0]
    assert fleiss_kappa(rnd)[0] == pytest.approx(sm_fleiss_kappa(table), abs=1e-12)
    assert abs(fleiss_kappa(rnd)[0]) < 0.1
    # the standard error under H0 (Fleiss, Nee & Landis, 1979; irr::kappam.fleiss): z = .2099 / SE0
    assert z == pytest.approx(12.374, abs=1e-3)
    # ... and it is the right one: simulated kappas of independent raters scatter as the formula says
    sims = []
    for _ in range(300):
        x = pd.DataFrame(rng.choice(3, size=(60, 4), p=[0.5, 0.3, 0.2]))
        k_, z_, _ = fleiss_kappa(x)
        sims.append((k_, k_ / z_))
    kappas, ses = np.array(sims).T
    assert np.std(kappas) == pytest.approx(np.mean(ses), rel=0.15)
    assert np.mean(np.abs(kappas / ses) > 1.96) < 0.1  # about 5 % false positives, not 30 %
    # one-tailed p tests agreement better than chance: tiny for good agreement, large for negative kappa
    assert fleiss_kappa(df, method="one-tailed")[2] < 1e-6
    anti = pd.DataFrame({"a": list("xyxyxyxy"), "b": list("yxyxyxyx"), "c": list("xxyyxxyy")})
    k_anti, _, p_anti = fleiss_kappa(anti, method="one-tailed")
    assert k_anti < 0 and p_anti > 0.5
    # missing ratings: incomplete items are dropped, the others are untouched
    gappy = df.copy().astype(float)
    gappy.iloc[0, 0] = np.nan
    assert fleiss_kappa(gappy)[:3] == pytest.approx(fleiss_kappa(df.iloc[1:])[:3])
    _, _, _, per_item_gappy = fleiss_kappa(gappy, return_per_item_agreement=True)
    assert np.isnan(per_item_gappy.iloc[0]) and per_item_gappy.iloc[1:].notna().all()
    with pytest.raises(ValueError, match="completely rated"):
        fleiss_kappa(pd.DataFrame({"a": [np.nan], "b": [1.0]}))
    # per-item agreement is in [0, 1] and 1 for unanimous items
    _, _, _, per_item = fleiss_kappa(df, return_per_item_agreement=True)
    assert per_item.between(0, 1).all() and per_item.iloc[0] == pytest.approx(1.0)


NAN = np.nan
# Krippendorff (2011), "Computing Krippendorff's alpha-reliability": 4 observers (rows) rated 12 units
# (columns), with missing ratings. Published: nominal .743, ordinal .815, interval .849, ratio .797
KRIPPENDORFF_EXAMPLE = np.array(
    [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, NAN, NAN, NAN],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, NAN, 3],
        [NAN, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, NAN],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, NAN],
    ]
)
LEVELS = ("nominal", "ordinal", "interval", "ratio")


def alpha_only(data, level="nominal"):
    return krippendorffs_alpha(
        data, measurement=level, return_bootstraped_z_score=False
    )[0]


def definition_alpha(data, level):
    """Alpha straight from the definition: enumerate pairs of ratings (no coincidence matrix)."""
    x = np.asarray(data, dtype=float)
    units = [row[~np.isnan(row)] for row in x]
    units = [u for u in units if len(u) >= 2]  # only units with 2+ ratings are pairable
    pool = np.concatenate(units)
    n = len(pool)
    categories, counts = np.unique(pool, return_counts=True)
    frequency = dict(zip(categories, counts))

    def delta(a, b):
        if level == "nominal":
            return float(a != b)
        if level == "interval":
            return (a - b) ** 2
        if level == "ratio":
            return 0.0 if a + b == 0 else ((a - b) / (a + b)) ** 2
        low, high = min(a, b), max(a, b)  # ordinal
        between = sum(frequency[g] for g in categories if low <= g <= high)
        return (between - (frequency[a] + frequency[b]) / 2) ** 2

    observed = 0.0
    for (
        u
    ) in units:  # ordered pairs of ratings within a unit, weighted by 1 / (ratings - 1)
        for i, j in itertools.permutations(range(len(u)), 2):
            observed += delta(u[i], u[j]) / (len(u) - 1)
    observed /= n
    expected = sum(
        delta(pool[i], pool[j]) for i, j in itertools.permutations(range(n), 2)
    )
    return 1 - observed / (expected / (n * (n - 1)))


def random_ratings(rng, units, raters, categories, missing):
    support = np.sort(
        rng.choice(np.arange(0, 40), size=categories, replace=False)
    ).astype(float)
    base = rng.choice(support, size=(units, 1))
    agree = rng.uniform(size=(units, raters)) < rng.uniform(0.3, 0.9)
    data = np.where(agree, base, rng.choice(support, size=(units, raters)))
    return np.where(rng.uniform(size=data.shape) < missing, np.nan, data)


def usable(data):
    pairable = (np.sum(~np.isnan(data), axis=1) >= 2).sum() >= 2
    return pairable and len(np.unique(data[~np.isnan(data)])) >= 2


def test_krippendorffs_alpha_reproduces_the_published_example():
    frame = pd.DataFrame(KRIPPENDORFF_EXAMPLE.T)  # units in rows, raters in columns
    published = {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}
    for level, value in published.items():
        assert round(alpha_only(frame, level), 3) == value
    assert alpha_only(frame, "nominal") == pytest.approx(0.7434, abs=5e-5)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", range(6))
def test_krippendorffs_alpha_matches_the_definition(level, seed):
    rng = np.random.default_rng(seed)
    while True:  # small designs with missing ratings, and values such as 2, 10 and 100
        data = rng.choice(
            [1.0, 2.0, 10.0, 100.0],
            size=(int(rng.integers(5, 10)), int(rng.integers(2, 5))),
        )
        data = np.where(rng.uniform(size=data.shape) < 0.25, np.nan, data)
        if usable(data):
            break
    assert alpha_only(pd.DataFrame(data), level) == pytest.approx(
        definition_alpha(data, level), abs=1e-10
    )


def test_krippendorffs_alpha_known_cases():
    # perfect agreement
    perfect = pd.DataFrame({"a": list("xyzx"), "b": list("xyzx"), "c": list("xyzx")})
    assert alpha_only(perfect) == pytest.approx(1.0)
    # two raters who always disagree (labels swapped): alpha = (1 - U) / U
    for units in (4, 10, 25):
        first = np.tile([1, 2], units)[:units]
        assert alpha_only(
            pd.DataFrame({"r1": first, "r2": 3 - first})
        ) == pytest.approx((1 - units) / units)
    # all ratings identical: alpha is undefined, which is reported as nan without an error
    same = pd.DataFrame(np.ones((6, 3)))
    alpha, z, p = krippendorffs_alpha(same)
    assert np.isnan(alpha) and np.isnan(z) and np.isnan(p)


def test_krippendorffs_alpha_handles_missing_ratings_properly():
    frame = pd.DataFrame(KRIPPENDORFF_EXAMPLE.T)
    base = alpha_only(frame)
    # a unit with a single rating carries no information and is ignored
    single = pd.concat([frame, pd.DataFrame([[3, NAN, NAN, NAN]])], ignore_index=True)
    assert alpha_only(single) == pytest.approx(base)
    # NaN and None are both missing; the missing rating is not a category of its own
    assert alpha_only(frame.astype(object).where(frame.notna(), None)) == pytest.approx(
        base
    )
    # rows of units and columns of raters can be reordered
    assert alpha_only(frame.iloc[::-1, ::-1]) == pytest.approx(base)
    # no unit with two ratings: nothing to compare
    with pytest.raises(ValueError, match="at least two ratings"):
        krippendorffs_alpha(pd.DataFrame({"a": [1, NAN, 2], "b": [NAN, 3, NAN]}))


def test_krippendorffs_alpha_levels_use_the_values_not_the_category_positions():
    rng = np.random.default_rng(3)
    data = rng.choice([1.0, 2.0, 3.0], size=(40, 4))
    data = np.where(rng.uniform(size=data.shape) < 0.6, data[:, [0]], data)
    positions, values = (
        pd.DataFrame(data),
        pd.DataFrame(np.where(data == 3.0, 10.0, data)),
    )
    # ordinal depends on the order only; nominal on equality only
    for level in ("nominal", "ordinal"):
        assert alpha_only(positions, level) == pytest.approx(alpha_only(values, level))
    # interval and ratio depend on the actual distances: 3 -> 10 makes disagreements with 3 count more
    for level in ("interval", "ratio"):
        assert alpha_only(positions, level) != pytest.approx(
            alpha_only(values, level), abs=1e-3
        )
        assert alpha_only(values, level) == pytest.approx(
            definition_alpha(values.to_numpy(), level)
        )
    # numbers are ordered as numbers: 2 < 10 (as text, "10" would come before "2")
    tens = pd.DataFrame(
        {"a": [2, 10, 2, 10, 100], "b": [2, 10, 10, 2, 100], "c": [2, 2, 10, 10, 100]}
    )
    assert alpha_only(tens, "ordinal") == pytest.approx(
        definition_alpha(tens.to_numpy(float), "ordinal")
    )


def test_krippendorffs_alpha_invariances():
    frame = pd.DataFrame(random_ratings(np.random.default_rng(5), 50, 4, 5, 0.2))
    frame = frame + 1.0  # strictly positive, so that the ratio level is defined
    nominal = alpha_only(frame, "nominal")
    # nominal: any relabelling (also with text labels)
    labels = {
        v: f"label-{i}"
        for i, v in enumerate(np.unique(frame.to_numpy()[~frame.isna().to_numpy()]))
    }
    assert alpha_only(frame.apply(lambda c: c.map(labels)), "nominal") == pytest.approx(
        nominal
    )
    # ordinal: any strictly increasing transformation
    ordinal = alpha_only(frame, "ordinal")
    assert alpha_only(frame**3, "ordinal") == pytest.approx(ordinal)
    assert alpha_only(np.log(frame), "ordinal") == pytest.approx(ordinal)
    # interval: shift and positive scale, but not a non-linear transformation
    interval = alpha_only(frame, "interval")
    assert alpha_only(frame * 7.5 - 20, "interval") == pytest.approx(interval)
    assert alpha_only(frame**2, "interval") != pytest.approx(interval, abs=1e-3)
    # ratio: positive scale, but not a shift
    ratio = alpha_only(frame, "ratio")
    assert alpha_only(frame * 3.0, "ratio") == pytest.approx(ratio)
    assert alpha_only(frame + 10.0, "ratio") != pytest.approx(ratio, abs=1e-3)


def test_krippendorffs_alpha_input_types_and_per_item_agreement():
    frame = pd.DataFrame(KRIPPENDORFF_EXAMPLE.T, index=[f"unit{i}" for i in range(12)])
    base = alpha_only(frame)
    assert alpha_only(KRIPPENDORFF_EXAMPLE.T) == pytest.approx(base)
    assert alpha_only(KRIPPENDORFF_EXAMPLE.T.tolist()) == pytest.approx(base)
    alpha, z, p, per_item = krippendorffs_alpha(frame, return_per_item_agreement=True)
    assert alpha == pytest.approx(base) and list(per_item.index) == list(frame.index)
    assert per_item.name == "per_item_agreement"
    # share of the ratings that equal the most frequent one, worked out by hand from the example
    assert per_item["unit0"] == 1.0  # 1, 1, 1 (one rater did not rate it)
    assert per_item["unit1"] == 0.75  # 2, 2, 3, 2
    assert per_item["unit2"] == 1.0  # 3, 3, 3, 3
    assert per_item["unit5"] == 0.25  # 1, 2, 3, 4: no agreement at all
    assert per_item["unit7"] == 0.75  # 1, 1, 2, 1
    assert per_item["unit10"] == 1.0  # 1, 1 (two ratings are enough)
    assert np.isnan(per_item["unit11"])  # a single rating: not pairable


def test_krippendorffs_alpha_permutation_test():
    rng = np.random.default_rng(2)
    # raters who agree strongly: far from chance
    truth = rng.integers(1, 6, size=(60, 1)).astype(float)
    agree = pd.DataFrame(
        np.where(
            rng.uniform(size=(60, 4)) < 0.85, truth, rng.integers(1, 6, size=(60, 4))
        )
    )
    alpha, z, p = krippendorffs_alpha(agree, measurement="interval")
    assert alpha > 0.7 and z > 5 and p < 1e-6
    # independent raters: alpha near 0, not significant
    chance = pd.DataFrame(rng.integers(1, 6, size=(60, 4)).astype(float))
    alpha, z, p = krippendorffs_alpha(chance)
    assert abs(alpha) < 0.1 and p > 0.05
    # reproducible, other seeds differ, and one-tailed p is half of the two-tailed p
    first = krippendorffs_alpha(agree, n_iter_for_bootstrap=50)
    assert krippendorffs_alpha(agree, n_iter_for_bootstrap=50) == first
    assert (
        krippendorffs_alpha(agree, n_iter_for_bootstrap=50, bootstrap_seed=99)[1]
        != first[1]
    )
    one_tailed = krippendorffs_alpha(chance, method="one-tailed")[2]
    assert one_tailed == pytest.approx(krippendorffs_alpha(chance)[2] / 2)
    # no bootstrap: z and p are nan
    a, z, p = krippendorffs_alpha(agree, return_bootstraped_z_score=False)
    assert a == pytest.approx(first[0]) and np.isnan(z) and np.isnan(p)
    # works with missing ratings (the pattern of who rated what is kept)
    holes = agree.mask(rng.uniform(size=agree.shape) < 0.3)
    alpha, z, p = krippendorffs_alpha(holes, n_iter_for_bootstrap=50)
    assert np.isfinite([alpha, z, p]).all() and z > 3


def test_krippendorffs_alpha_does_not_touch_numpys_global_random_state():
    frame = pd.DataFrame(KRIPPENDORFF_EXAMPLE.T)
    np.random.seed(5)
    expected = np.random.rand()
    np.random.seed(5)
    krippendorffs_alpha(frame, n_iter_for_bootstrap=20)
    assert np.random.rand() == expected


def test_krippendorffs_alpha_validates_the_input():
    frame = pd.DataFrame(KRIPPENDORFF_EXAMPLE.T)
    with pytest.raises(ValueError, match="measurement must be one of"):
        krippendorffs_alpha(frame, measurement="categorical")
    with pytest.raises(ValueError, match="at least 2 raters"):
        krippendorffs_alpha(frame[[0]])
    with pytest.raises(ValueError, match="at least 2 raters"):
        krippendorffs_alpha(np.arange(5.0))
    text = pd.DataFrame({"a": list("xyzxy"), "b": list("xyzzy")})
    for level in ("ordinal", "interval", "ratio"):
        with pytest.raises(ValueError, match="needs numeric ratings"):
            krippendorffs_alpha(text, measurement=level)
    assert alpha_only(text, "nominal") == pytest.approx(
        definition_alpha(
            text.apply(lambda c: c.map({"x": 1.0, "y": 2.0, "z": 3.0})).to_numpy(),
            "nominal",
        )
    )
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="n_iter_for_bootstrap"):
            krippendorffs_alpha(frame, n_iter_for_bootstrap=bad)


###########################
# pooling of correlations #
###########################


def test_average_correlations():
    r, n = [0.3, 0.5], [103, 53]
    res = average_correlations(r, n)
    w = np.array(n) - 3
    z_bar = np.sum(w * np.arctanh(r)) / w.sum()
    se = 1 / np.sqrt(w.sum())
    assert res["z_mean"] == pytest.approx(z_bar) and res["se"] == pytest.approx(se)
    assert res["r_mean"] == pytest.approx(np.tanh(z_bar)) and res["z"] == pytest.approx(
        z_bar / se
    )
    assert res["p-value"] == pytest.approx(2 * norm.sf(abs(z_bar / se)))
    assert res["cil"] == pytest.approx(np.tanh(z_bar - 1.959964 * se), abs=1e-6)
    assert res["cir"] == pytest.approx(np.tanh(z_bar + 1.959964 * se), abs=1e-6)
    assert res["cil"] < res["r_mean"] < res["cir"]
    # the larger study dominates
    assert abs(res["r_mean"] - 0.3) < abs(res["r_mean"] - 0.5)
    # a single study returns itself
    one = average_correlations([0.4], [103])
    assert one["r_mean"] == pytest.approx(0.4) and one["se"] == pytest.approx(0.1)
    # one-tailed p is half of the two-tailed p; wider interval for a higher confidence level
    assert average_correlations(r, n, method="one-tailed")["p-value"] == pytest.approx(
        res["p-value"] / 2
    )
    narrow = average_correlations(r, n, conf_level=0.8)
    assert narrow["cil"] > res["cil"] and narrow["cir"] < res["cir"]


###########################################
# comparing dependent / independent r     #
###########################################


def test_rz_ci():
    lo, hi = rz_ci(0.5, 103)  # standard error of Fisher z is 0.1
    assert lo == pytest.approx(np.tanh(np.arctanh(0.5) - 1.959964 * 0.1), abs=1e-6)
    assert hi == pytest.approx(np.tanh(np.arctanh(0.5) + 1.959964 * 0.1), abs=1e-6)
    lo99, hi99 = rz_ci(0.5, 103, conf_level=0.99)
    assert lo99 < lo < 0.5 < hi < hi99
    lo_n, hi_n = rz_ci(0.5, 1003)
    assert hi_n - lo_n < hi - lo  # larger n, narrower interval


def test_rho_rxy_rxz_is_a_correlation():
    assert rho_rxy_rxz(0.0, 0.0, 0.0) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = rng.normal(size=(30, 3))
        c = np.corrcoef(a, rowvar=False)
        assert -1 <= rho_rxy_rxz(c[0, 1], c[0, 2], c[1, 2]) <= 1


def test_dependent_correlations_steiger():
    t, p = dependent_correlations(0.7, 0.2, 0.4, n=200)
    assert t > 0 and p < 0.001
    # identical correlations: no difference
    t0, p0 = dependent_correlations(0.4, 0.4, 0.3, n=100)
    assert t0 == pytest.approx(0) and p0 == pytest.approx(1)
    # antisymmetric in the two correlations, same p
    t_sw, p_sw = dependent_correlations(0.2, 0.7, 0.4, n=200)
    assert t_sw == pytest.approx(-t) and p_sw == pytest.approx(p)
    # one-tailed is half of the two-tailed p
    assert dependent_correlations(0.7, 0.2, 0.4, n=200, twotailed=False)[
        1
    ] == pytest.approx(p / 2)
    # more data -> stronger evidence
    assert (
        dependent_correlations(0.5, 0.4, 0.3, n=1000)[1]
        < dependent_correlations(0.5, 0.4, 0.3, n=100)[1]
    )

    # Monte Carlo: under H0 (equal true correlations) the test rejects in about 5% of the samples
    rng = np.random.default_rng(42)
    cov = np.array(
        [[1, 0.3, 0.3], [0.3, 1, 0.5], [0.3, 0.5, 1]]
    )  # x with y and z correlate .3
    rejections = 0
    reps = 600
    for _ in range(reps):
        a = rng.multivariate_normal(np.zeros(3), cov, size=100)
        c = np.corrcoef(a, rowvar=False)
        rejections += dependent_correlations(c[0, 1], c[0, 2], c[1, 2], n=100)[1] < 0.05
    assert 0.025 < rejections / reps < 0.085


def test_dependent_correlations_zou():
    lo, hi = dependent_correlations(0.7, 0.2, 0.4, n=200, method="zou")
    assert 0 < lo < 0.5 < hi  # the difference .5 lies inside, zero outside
    lo0, hi0 = dependent_correlations(0.30, 0.28, 0.4, n=100, method="zou")
    assert lo0 < 0 < hi0  # not significant
    # swapping the two correlations mirrors the interval
    lo_sw, hi_sw = dependent_correlations(0.2, 0.7, 0.4, n=200, method="zou")
    assert (lo_sw, hi_sw) == pytest.approx((-hi, -lo))
    # a lower confidence level gives a narrower interval
    lo90, hi90 = dependent_correlations(0.7, 0.2, 0.4, n=200, method="zou", conf_level=0.90)
    assert lo < lo90 < hi90 < hi
    with pytest.raises(ValueError, match="Wrong method"):
        dependent_correlations(0.7, 0.2, 0.4, n=200, method="x")


def test_independent_correlations_fisher():
    z, p = independent_correlations(0.5, 0.3, 103)
    expected = abs(np.arctanh(0.5) - np.arctanh(0.3)) / np.sqrt(2 / 100)
    assert z == pytest.approx(expected) and p == pytest.approx(2 * norm.sf(expected))
    # different sample sizes; n2 defaults to n1; one-tailed
    z2, _ = independent_correlations(0.5, 0.3, 103, 53)
    assert z2 == pytest.approx(
        abs(np.arctanh(0.5) - np.arctanh(0.3)) / np.sqrt(1 / 100 + 1 / 50)
    )
    assert independent_correlations(0.5, 0.3, 103, 103)[0] == pytest.approx(z)
    assert independent_correlations(0.5, 0.3, 103, twotailed=False)[1] == pytest.approx(
        p / 2
    )
    # symmetric, and equal correlations give p = 1
    assert independent_correlations(0.3, 0.5, 103)[0] == pytest.approx(z)
    assert independent_correlations(0.4, 0.4, 50)[1] == pytest.approx(1)

    # Monte Carlo under H0 (both true correlations .4): about 5% rejections
    rng = np.random.default_rng(7)
    cov = np.array([[1, 0.4], [0.4, 1]])
    rejections, reps, n = 0, 600, 60
    for _ in range(reps):
        r = [
            np.corrcoef(rng.multivariate_normal([0, 0], cov, size=n), rowvar=False)[
                0, 1
            ]
            for _ in range(2)
        ]
        rejections += independent_correlations(r[0], r[1], n)[1] < 0.05
    assert 0.025 < rejections / reps < 0.085


def test_correlation_comparisons_take_the_confidence_level_as_conf_level():
    import inspect

    for func in (dependent_correlations, independent_correlations):
        assert "conf_level" in inspect.signature(func).parameters
        assert "alpha" not in inspect.signature(func).parameters
    lo95, hi95 = independent_correlations(0.6, 0.2, 100, 100, method="zou")
    lo99, hi99 = independent_correlations(0.6, 0.2, 100, 100, method="zou", conf_level=0.99)
    assert lo99 < lo95 < hi95 < hi99
    # n2 defaults to n1 for the Zou interval, too
    assert independent_correlations(0.6, 0.2, 100, method="zou") == pytest.approx((lo95, hi95))


def test_independent_correlations_zou():
    lo, hi = independent_correlations(0.6, 0.2, 100, 100, method="zou")
    assert 0 < lo < 0.4 < hi
    lo0, hi0 = independent_correlations(0.31, 0.30, 100, 100, method="zou")
    assert lo0 < 0 < hi0
    lo_sw, hi_sw = independent_correlations(0.2, 0.6, 100, 100, method="zou")
    assert (lo_sw, hi_sw) == pytest.approx((-hi, -lo))
    # decision agrees with the Fisher test on a clear case
    assert independent_correlations(0.6, 0.2, 100, 100)[1] < 0.05 and lo > 0
    with pytest.raises(ValueError, match="Wrong method"):
        independent_correlations(0.6, 0.2, 100, method="x")


########
# HTMT #
########


def test_htmt_ratio():
    rng = np.random.default_rng(3)
    # exact sample correlations: .6 within A, .5 within B, .3 between -> HTMT = .3 / sqrt(.6 * .5)
    corr = np.full((5, 5), 0.3)
    corr[:3, :3] = 0.6
    corr[3:, 3:] = 0.5
    np.fill_diagonal(corr, 1)
    data = pd.DataFrame(
        correlated_sample(corr, 400, rng), columns=["a1", "a2", "a3", "b1", "b2"]
    )
    assert htmt_ratio(
        data, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"
    ) == pytest.approx(0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # item order does not matter; constructs are interchangeable
    assert htmt_ratio(
        data, ["a3", "a1", "a2"], ["b2", "b1"], method="pearson"
    ) == pytest.approx(
        htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson")
    )
    assert htmt_ratio(
        data, ["b1", "b2"], ["a1", "a2", "a3"], method="pearson"
    ) == pytest.approx(0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # signs do not matter (absolute correlations are used)
    flipped = data.assign(a1=-data["a1"], b2=-data["b2"])
    assert htmt_ratio(
        flipped, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"
    ) == pytest.approx(0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # Spearman is invariant to monotone transformations, Pearson is not
    mono = np.exp(data)
    sp = htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="spearman")
    assert htmt_ratio(
        mono, ["a1", "a2", "a3"], ["b1", "b2"], method="spearman"
    ) == pytest.approx(sp)
    assert htmt_ratio(
        mono, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"
    ) != pytest.approx(
        htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"), abs=1e-3
    )
    # rows with missing values are dropped
    with_na = pd.concat([data, data.iloc[:5].assign(a1=np.nan)], ignore_index=True)
    assert htmt_ratio(
        with_na, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"
    ) == pytest.approx(0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # discriminant validity: indistinguishable constructs have HTMT close to 1
    same = np.full((4, 4), 0.5)
    np.fill_diagonal(same, 1)
    d2 = pd.DataFrame(
        correlated_sample(same, 400, rng), columns=["a1", "a2", "b1", "b2"]
    )
    assert htmt_ratio(
        d2, ["a1", "a2"], ["b1", "b2"], method="pearson"
    ) == pytest.approx(1.0, abs=1e-9)


##################
# McDonald omega #
##################


def one_factor_corr(loadings):
    l = np.asarray(loadings, float)
    r = np.outer(l, l)
    np.fill_diagonal(r, 1)
    return r


def true_omega(loadings):
    l = np.asarray(loadings, float)
    return l.sum() ** 2 / (l.sum() ** 2 + (1 - l**2).sum())


def exact_items(corr, n=600, seed=0):
    """Items whose sample correlation matrix is exactly ``corr``."""
    x = correlated_sample(corr, n, np.random.default_rng(seed))
    return pd.DataFrame(x, columns=[f"i{j + 1}" for j in range(len(corr))])


def simulated_items(loadings, n, seed=1):
    rng = np.random.default_rng(seed)
    l = np.asarray(loadings, float)
    x = rng.normal(size=(n, 1)) * l + rng.normal(size=(n, len(l))) * np.sqrt(1 - l**2)
    return pd.DataFrame(x, columns=[f"i{j + 1}" for j in range(len(l))])


def cronbach_alpha(df):
    c = np.cov(np.asarray(df), rowvar=False)
    k = len(c)
    return k / (k - 1) * (1 - np.trace(c) / c.sum())


def omega_only(data, **kwargs):
    return mcdonald_omega(data, **kwargs)[0]


@pytest.mark.parametrize(
    "loadings",
    [
        [0.8, 0.7, 0.6, 0.5, 0.4],
        [0.9] * 6,
        [0.5] * 4,
        [0.95, 0.6, 0.3],
        [0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4],
    ],
)
def test_omega_equals_the_population_value_for_a_one_factor_structure(loadings):
    assert omega_only(exact_items(one_factor_corr(loadings))) == pytest.approx(
        true_omega(loadings), abs=1e-6
    )


def test_omega_with_three_items_uses_the_exact_solution():
    # a just-identified one-factor model: l1^2 = r12 * r13 / r23, ...
    r = np.array([[1, 0.42, 0.35], [0.42, 1, 0.30], [0.35, 0.30, 1]])
    l = np.sqrt(
        [
            r[0, 1] * r[0, 2] / r[1, 2],
            r[0, 1] * r[1, 2] / r[0, 2],
            r[0, 2] * r[1, 2] / r[0, 1],
        ]
    )
    assert omega_only(exact_items(r)) == pytest.approx(true_omega(l), abs=1e-6)
    theta = _one_factor_ml(r)
    assert theta[:3] == pytest.approx(l, abs=1e-5)
    assert theta[3:] == pytest.approx(1 - l**2, abs=1e-5)


def test_omega_and_cronbachs_alpha():
    # tau-equivalent items (equal loadings): omega == alpha
    tau = exact_items(one_factor_corr([0.6] * 6))
    assert omega_only(tau) == pytest.approx(cronbach_alpha(tau), abs=1e-6)
    # congeneric items: alpha underestimates the reliability
    cong = exact_items(one_factor_corr([0.9, 0.8, 0.5, 0.4, 0.3]))
    assert cronbach_alpha(cong) < omega_only(cong) - 0.01


def test_omega_agrees_with_an_independent_ml_factor_analysis():
    for seed in range(5):
        df = simulated_items([0.8, 0.7, 0.6, 0.5, 0.4], 2000, seed=seed)
        fit = Factor(df.to_numpy(), n_factor=1, method="ml").fit()
        l, u = fit.loadings.ravel(), fit.uniqueness
        theta = _one_factor_ml(np.corrcoef(df.to_numpy(), rowvar=False))
        assert theta[:5] == pytest.approx(l * np.sign(l.sum()), abs=1e-4)
        assert theta[5:] == pytest.approx(u, abs=1e-4)
        assert omega_only(df) == pytest.approx(l.sum() ** 2 / (l.sum() ** 2 + u.sum()), abs=1e-5)


def test_omega_returns_the_estimate_with_its_interval():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 300, seed=4)
    result = mcdonald_omega(df)
    assert isinstance(result, tuple) and len(result) == 3
    omega, low, high = result
    assert all(isinstance(v, float) for v in result)
    assert 0 < low < omega < high < 1
    assert mcdonald_omega(df, bootstrap_iter=100)[0] == omega  # same estimate, whatever the interval


def test_omega_is_invariant_to_scale_shift_sign_and_input_type():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 1000, seed=6)
    base = mcdonald_omega(df)
    for other in (df * 10 + 5, -df, df.to_numpy(), df.to_numpy().tolist()):
        assert mcdonald_omega(other) == pytest.approx(base, abs=1e-7)  # estimate and interval
    assert 0 < base[0] < 1
    with pytest.raises(TypeError):  # the former `scale` option had no effect: removed
        mcdonald_omega(df, scale=True)


def test_omega_warns_about_reverse_keyed_items():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 1000, seed=6)
    base = omega_only(df)
    flipped = df.assign(i4=-df["i4"])
    with pytest.warns(UserWarning, match="different signs"):
        too_low = omega_only(flipped)
    assert too_low < base - 0.1
    # recoding the item restores the value, without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recoded = omega_only(flipped.assign(i4=-flipped["i4"]))
    assert recoded == pytest.approx(base, abs=1e-7)


def test_omega_drops_incomplete_rows():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 500, seed=7)
    holes = df.copy()
    holes.iloc[[3, 50, 51], 0] = np.nan
    holes.iloc[10, 2] = np.nan
    assert len(holes.dropna()) == 496
    assert mcdonald_omega(holes) == pytest.approx(mcdonald_omega(holes.dropna()), abs=1e-12)


@pytest.mark.parametrize(
    "data, match",
    [
        (pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 4, 3]}), "At least 3 items"),
        (np.arange(10.0), "two-dimensional"),
        (pd.DataFrame(np.ones((10, 3))), "constant"),
        (pd.DataFrame(np.random.default_rng(0).normal(size=(3, 3))), "More respondents"),
        (
            pd.DataFrame({"a": list("xyzw"), "b": [1.0, 2, 3, 4], "c": [4.0, 3, 2, 1]}),
            "could not convert",
        ),
    ],
)
def test_omega_validates_the_data(data, match):
    with pytest.raises(ValueError, match=match):
        mcdonald_omega(data)


def test_omega_validates_the_options():
    df = simulated_items([0.8, 0.7, 0.6], 100)
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError, match="bootstrap_iter"):
            mcdonald_omega(df, bootstrap_iter=bad)
    for bad in (0, 1, 1.5, 95):
        with pytest.raises(ValueError, match="conf_level"):
            mcdonald_omega(df, conf_level=bad)


def test_omega_in_heywood_cases():
    # r13 * r23 / r12 > 1: the exact solution needs a uniqueness below zero; it is held on the bound
    r = np.array([[1, 0.3, 0.6], [0.3, 1, 0.7], [0.6, 0.7, 1]])
    items = exact_items(r)
    theta = _one_factor_ml(r)
    assert theta[3:].min() == pytest.approx(0.005, abs=1e-9)
    # the estimate is fine, the delta method is not available and says so; the bootstrap works
    with pytest.warns(UserWarning, match="boundary"):
        omega, low, high = mcdonald_omega(items)
    assert 0 < omega < 1 and np.isnan(low) and np.isnan(high)
    omega_boot, low_boot, high_boot = mcdonald_omega(items, bootstrap_iter=50)
    assert omega_boot == omega and 0 < low_boot <= high_boot <= 1


###################################
# omega: delta-method interval    #
###################################


def test_omega_delta_interval_is_a_logit_interval():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 300, seed=4)
    omega, low, high = mcdonald_omega(df)
    logit = lambda v: np.log(v / (1 - v))
    # symmetric on the logit scale, with the half-width z * SE / (omega * (1 - omega))
    assert logit(high) - logit(omega) == pytest.approx(logit(omega) - logit(low), abs=1e-9)
    theta = _one_factor_ml(np.corrcoef(df.to_numpy(), rowvar=False))
    se = _omega_delta_se(theta, len(df))
    assert logit(high) - logit(omega) == pytest.approx(
        1.959964 * se / (omega * (1 - omega)), abs=1e-5
    )
    # a higher confidence level gives a wider interval, more respondents a narrower one
    wide, narrow = mcdonald_omega(df, conf_level=0.99), mcdonald_omega(df, conf_level=0.80)
    assert wide[2] - wide[1] > high - low > narrow[2] - narrow[1]
    big = simulated_items([0.8, 0.7, 0.6, 0.5], 3000, seed=4)
    assert np.ptp(mcdonald_omega(big)[1:]) < high - low
    # stays inside (0, 1) even for a very reliable scale in a small sample
    small = simulated_items([0.95] * 6, 40, seed=5)
    _, low, high = mcdonald_omega(small)
    assert 0 < low < high < 1


def test_omega_delta_se_matches_the_observed_information():
    """Independent route: the numerical Hessian of the likelihood instead of the expected information."""
    n = 4000
    x = simulated_items([0.8, 0.7, 0.6, 0.5], n, seed=3).to_numpy()
    r = np.corrcoef(x, rowvar=False)
    theta = _one_factor_ml(r)

    def discrepancy(t):
        s = np.outer(t[:4], t[:4]) + np.diag(t[4:])
        return np.linalg.slogdet(s)[1] + np.trace(np.linalg.solve(s, r))

    observed = n / 2 * approx_hess(theta, discrepancy)
    gradient = np.ravel(approx_fprime(theta, _omega_from_theta, centered=True))
    se_observed = np.sqrt(gradient @ np.linalg.inv(observed) @ gradient)
    assert _omega_delta_se(theta, n) == pytest.approx(se_observed, rel=0.02)


def test_omega_delta_se_has_the_right_size_and_the_interval_the_right_coverage():
    """Monte Carlo with normal items: mean SE ~ spread of the estimate, coverage ~ 95%."""
    rng = np.random.default_rng(21)
    loadings = np.array([0.8, 0.7, 0.6, 0.5])
    truth, n, sims = true_omega(loadings), 200, 400
    estimates, ses, covered = [], [], 0
    for _ in range(sims):
        f = rng.normal(size=(n, 1))
        x = f * loadings + rng.normal(size=(n, 4)) * np.sqrt(1 - loadings**2)
        omega, low, high = mcdonald_omega(x)
        estimates.append(omega)
        ses.append(_omega_delta_se(_one_factor_ml(np.corrcoef(x, rowvar=False)), n))
        covered += low <= truth <= high
    assert np.mean(estimates) == pytest.approx(truth, abs=0.005)
    assert np.mean(ses) == pytest.approx(np.std(estimates, ddof=1), rel=0.12)
    assert 0.91 <= covered / sims <= 0.98  # nominal 95%, simulation error about 1 point


def test_omega_delta_and_bootstrap_intervals_agree_for_normal_items():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 500, seed=12)
    _, low, high = mcdonald_omega(df)
    _, low_boot, high_boot = mcdonald_omega(df, bootstrap_iter=500)
    assert low == pytest.approx(low_boot, abs=0.01) and high == pytest.approx(high_boot, abs=0.01)


###################################
# omega: bootstrap interval       #
###################################


def test_omega_bootstrap():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 300, seed=8)
    point = omega_only(df)
    result = mcdonald_omega(df, bootstrap_iter=200)
    # the estimate of the original sample together with the limits of the interval
    assert isinstance(result, tuple) and len(result) == 3
    omega, low, high = result
    assert all(isinstance(v, float) for v in result)
    assert omega == point  # not the bootstrap median
    assert low <= omega <= high
    # reproducible; other seeds (and seed=None) give other resamples, but the same estimate
    assert mcdonald_omega(df, bootstrap_iter=200) == result
    other = mcdonald_omega(df, bootstrap_iter=200, seed=99)
    assert other[0] == omega and other[1:] != result[1:]
    unseeded = [mcdonald_omega(df, bootstrap_iter=50, seed=None) for _ in range(2)]
    assert unseeded[0][0] == unseeded[1][0] and unseeded[0][1:] != unseeded[1][1:]
    # a higher confidence level gives a wider interval, more respondents a narrower one
    wide = mcdonald_omega(df, bootstrap_iter=200, conf_level=0.99)
    narrow = mcdonald_omega(df, bootstrap_iter=200, conf_level=0.80)
    assert wide[2] - wide[1] > high - low > narrow[2] - narrow[1]
    assert wide[0] == narrow[0] == omega  # the estimate does not depend on the level
    big = simulated_items([0.8, 0.7, 0.6, 0.5], 3000, seed=8)
    _, big_low, big_high = mcdonald_omega(big, bootstrap_iter=200)
    assert big_high - big_low < high - low
    # a single resample is fine: both limits are that resample's omega
    _, only_low, only_high = mcdonald_omega(df, bootstrap_iter=1)
    assert only_low == only_high


def test_omega_bootstrap_interval_is_a_percentile_interval():
    """The limits are the (1 - level) / 2 and (1 + level) / 2 percentiles of the resampled omegas."""
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 200, seed=2)
    rng = np.random.default_rng(13)  # the generator and the draws of mcdonald_omega
    x = df.to_numpy()
    omegas = [
        _omega_from_theta(
            _one_factor_ml(np.corrcoef(x[rng.integers(0, len(x), size=len(x))], rowvar=False))
        )
        for _ in range(100)
    ]
    _, low, high = mcdonald_omega(df, bootstrap_iter=100, conf_level=0.90, seed=13)
    assert low == pytest.approx(np.percentile(omegas, 5))
    assert high == pytest.approx(np.percentile(omegas, 95))


def test_omega_bootstrap_resamples_are_all_different(monkeypatch):
    """Regression: re-seeding in a recursive call used to give two resamples the same data."""
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 300, seed=8)
    values = []
    original = mcdonald_module._omega_from_theta

    def record(theta):
        values.append(original(theta))
        return values[-1]

    monkeypatch.setattr(mcdonald_module, "_omega_from_theta", record)
    mcdonald_omega(df, bootstrap_iter=40)
    assert len(values) == 41  # the estimate of the original sample, then 40 resamples
    assert len(set(values[1:])) == 40


@pytest.mark.parametrize("kwargs", [{}, {"bootstrap_iter": 10}])
def test_omega_does_not_touch_numpys_global_random_state(kwargs):
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 200)
    np.random.seed(5)
    expected = np.random.rand()
    np.random.seed(5)
    mcdonald_omega(df, **kwargs)
    assert np.random.rand() == expected


def test_omega_bootstrap_warns_once_about_the_original_sample():
    df = simulated_items([0.8, 0.7, 0.6, 0.5], 400, seed=6)
    flipped = df.assign(i4=-df["i4"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mcdonald_omega(flipped, bootstrap_iter=30)
    assert sum("different signs" in str(w.message) for w in caught) == 1


#############
# bootstrap #
#############


def test_bootstrap():
    cil, m, cir = bootstrap(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        func=np.mean,
        n_rep=1000,
        seed=13,
    )
    assert s(cil) == "3.333"
    assert s(m) == "5.000"
    assert s(cir) == "6.667"

    def f(*args):
        return spearmanr(*args).statistic

    x, y = [1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 8, 4, 5, 6, 7, 8, 9]
    cil, m, cir = bootstrap(x, y, func=f, n_rep=1000, seed=13)
    # a bootstrap with 1000 resamples is only accurate to about 0.1 in the tail of such a small
    # sample, so the exact digits depend on the random stream; compare with a very large bootstrap
    # (95% interval about [0.047, 1.0], median 0.745) instead of pinning them
    assert 0.0 <= cil <= 0.2 and s(cir) == "1.000"
    assert m == pytest.approx(0.745, abs=0.03)
    assert cil < m < cir

    # confidence intervals / p-value derived from the bootstrap median
    p = p_from_r(m, 9)
    lo, hi = ci_from_r(m, 9)
    assert 0.01 < p < 0.04
    assert lo < m < hi and hi == pytest.approx(0.943, abs=0.01)

    # spearmanr
    # SignificanceResult(statistic=np.float64(0.7173570516842638),
    # pvalue=np.float64(0.02958657594620456))


def test_bootstrap_random_state_and_seed():
    data = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0]
    first = bootstrap(data, n_rep=200, seed=13)
    assert bootstrap(data, n_rep=200, seed=13) == pytest.approx(first)  # reproducible
    assert not np.allclose(
        bootstrap(data, n_rep=200, seed=14), first
    )  # another seed, other resamples
    # seed=None works and gives different resamples on every call
    a, b = bootstrap(data, n_rep=200, seed=None), bootstrap(data, n_rep=200, seed=None)
    assert a.shape == (3,) and a[0] <= a[1] <= a[2] and not np.allclose(a, b)
    # numpy's global random state is not touched
    np.random.seed(5)
    expected = np.random.rand()
    np.random.seed(5)
    bootstrap(data, n_rep=50)
    assert np.random.rand() == expected
    # a higher confidence level gives a wider interval
    wide, narrow = (
        bootstrap(data, n_rep=500, conf_level=0.99),
        bootstrap(data, n_rep=500, conf_level=0.8),
    )
    assert wide[2] - wide[0] > narrow[2] - narrow[0]


def test_bootstrap_properties():
    x = np.arange(1, 10)
    lo, m, hi = bootstrap(x, func=np.mean, n_rep=500, seed=1)
    assert lo < m < hi
    assert bootstrap(x, func=np.mean, n_rep=500, seed=1).tolist() == [
        lo,
        m,
        hi,
    ]  # reproducible
    lo99, _, hi99 = bootstrap(x, func=np.mean, n_rep=500, seed=1, conf_level=0.99)
    assert lo99 <= lo and hi99 >= hi  # a higher confidence level gives a wider interval
    # a constant sample has a degenerate interval
    assert bootstrap([3, 3, 3, 3], func=np.mean, n_rep=50).tolist() == [3.0, 3.0, 3.0]


####################
# Cronbach's alpha #
####################


def compound_symmetry(k, rho):
    r = np.full((k, k), rho)
    np.fill_diagonal(r, 1)
    return r


def population_alpha(sigma):
    k = len(sigma)
    return k / (k - 1) * (1 - np.trace(sigma) / np.sum(sigma))


def normal_items(sigma, n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.multivariate_normal(np.zeros(len(sigma)), sigma, size=n))


def textbook_alpha(x):
    """Alpha from the variances of the items and of the row sums (independent of the code under test)."""
    x = np.asarray(x, float)
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def cronbach_only(data, **kwargs):
    return cronbachs_alpha(data, **kwargs)[0]


def test_cronbachs_alpha_by_hand():
    # item variances 2.5 each (sum 7.5), variance of the sum score 20: alpha = 3/2 * (1 - 7.5/20)
    x = pd.DataFrame([[1, 2, 3], [2, 3, 5], [3, 5, 4], [4, 4, 6], [5, 6, 7]])
    alpha, low, high = cronbachs_alpha(x)
    assert alpha == pytest.approx(0.9375, abs=1e-12)
    assert low < alpha < high <= 1.0


def test_cronbachs_alpha_matches_the_textbook_formula():
    rng = np.random.default_rng(3)
    for _ in range(50):
        k, n = int(rng.integers(2, 10)), int(rng.integers(20, 300))
        x = rng.normal(size=(n, k)) + rng.normal(size=(n, 1)) * rng.uniform(0.2, 1.5, size=k)
        x = x + rng.uniform(-5, 5, size=k)
        assert cronbach_only(x) == pytest.approx(textbook_alpha(x), abs=1e-10)


@pytest.mark.parametrize("k, rho", [(3, 0.3), (6, 0.5), (10, 0.2), (4, 0.9)])
def test_cronbachs_alpha_for_compound_symmetry(k, rho):
    # equal variances and covariances: alpha = k * rho / (1 + (k - 1) * rho) (Spearman-Brown)
    expected = k * rho / (1 + (k - 1) * rho)
    items = exact_items(compound_symmetry(k, rho))
    assert cronbach_only(items) == pytest.approx(expected, abs=1e-10)
    assert population_alpha(compound_symmetry(k, rho)) == pytest.approx(expected)


def test_cronbachs_alpha_and_omega():
    # tau-equivalent items: alpha == omega; congeneric items: alpha underestimates the reliability
    tau = exact_items(compound_symmetry(6, 0.36))
    assert cronbach_only(tau) == pytest.approx(omega_only(tau), abs=1e-6)
    cong = exact_items(one_factor_corr([0.9, 0.8, 0.5, 0.4, 0.3]))
    assert cronbach_only(cong) < omega_only(cong) - 0.01


def test_cronbachs_alpha_scale_and_input_type():
    df = normal_items(compound_symmetry(5, 0.4) + np.diag([0.1, 0.5, 1.0, 0.2, 0.3]), 300, seed=5)
    base = cronbachs_alpha(df)
    # the same shift and scale for all items, a sign flip of all items, other input types
    for other in (df * 7 + 3, -df, df.to_numpy(), df.to_numpy().tolist()):
        assert cronbachs_alpha(other) == pytest.approx(base, abs=1e-9)
    # rescaling a single item changes alpha: it is computed from the raw covariances
    scaled = df.copy()
    scaled[0] = scaled[0] * 10
    assert cronbach_only(scaled) != pytest.approx(base[0], abs=1e-3)
    assert isinstance(base, tuple) and len(base) == 3 and all(isinstance(v, float) for v in base)


def test_cronbachs_alpha_drops_incomplete_rows():
    df = normal_items(compound_symmetry(4, 0.4), 200, seed=2)
    holes = df.copy()
    holes.iloc[[3, 50, 51], 0] = np.nan
    holes.iloc[10, 2] = np.nan
    assert len(holes.dropna()) == 196
    assert cronbachs_alpha(holes) == pytest.approx(cronbachs_alpha(holes.dropna()), abs=1e-12)


def test_cronbachs_alpha_warns_when_it_is_negative():
    items = exact_items(np.array([[1.0, -0.5], [-0.5, 1.0]]))  # alpha = 2 * (1 - 2 / 1) = -2
    with pytest.warns(UserWarning, match="negative"):
        alpha, low, high = cronbachs_alpha(items)
    assert alpha == pytest.approx(-2.0, abs=1e-9)
    # two of four items reverse-keyed and not recoded: the covariances cancel and alpha is below zero
    df = normal_items(compound_symmetry(4, 0.5), 300, seed=6)
    df[[0, 1]] = -df[[0, 1]]
    with pytest.warns(UserWarning, match="negative"):
        assert cronbach_only(df) < 0


def test_cronbachs_alpha_with_items_that_cancel_each_other_out():
    x = np.arange(10.0)
    with pytest.warns(UserWarning, match="undefined"):
        result = cronbachs_alpha(np.column_stack([x, -x, x**2, -(x**2)]))
    assert all(np.isnan(v) for v in result)


@pytest.mark.parametrize(
    "data, match",
    [
        (pd.DataFrame({"a": [1.0, 2, 3, 4]}), "At least 2 items"),
        (np.arange(10.0), "two-dimensional"),
        (pd.DataFrame({"a": np.arange(10.0), "b": 1.0}), "constant"),
        (pd.DataFrame(np.random.default_rng(0).normal(size=(2, 2))), "More respondents"),
        (pd.DataFrame({"a": list("xyzw"), "b": [1.0, 2, 3, 4]}), "could not convert"),
    ],
)
def test_cronbachs_alpha_validates_the_data(data, match):
    with pytest.raises(ValueError, match=match):
        cronbachs_alpha(data)


def test_cronbachs_alpha_validates_the_options():
    df = normal_items(compound_symmetry(3, 0.4), 100)
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError, match="bootstrap_iter"):
            cronbachs_alpha(df, bootstrap_iter=bad)
    for bad in (0, 1, 1.5, 95):
        with pytest.raises(ValueError, match="conf_level"):
            cronbachs_alpha(df, conf_level=bad)


def test_cronbachs_alpha_analytic_se_matches_an_independent_delta_method():
    """Numerical gradient of alpha with respect to vech(S) and the explicit normal-theory covariance of S."""
    k, n = 5, 300
    sigma = 0.3 * np.ones((k, k)) + np.diag([0.4, 0.6, 0.8, 1.0, 1.2])
    s = np.cov(normal_items(sigma, n, seed=4).to_numpy(), rowvar=False)
    rows, cols = np.triu_indices(k)

    def alpha_of_vech(v):
        m = np.zeros((k, k))
        m[rows, cols] = v
        return _alpha_from_covariance(m + m.T - np.diag(np.diag(m)))

    gradient = np.ravel(approx_fprime(s[rows, cols], alpha_of_vech, centered=True))
    pairs = list(zip(rows, cols))
    covariance = np.array(
        [[(s[i, a] * s[j, b] + s[i, b] * s[j, a]) / n for (a, b) in pairs] for (i, j) in pairs]
    )
    assert _alpha_standard_error(s, n) == pytest.approx(np.sqrt(gradient @ covariance @ gradient), rel=1e-6)


def test_cronbachs_alpha_analytic_interval():
    df = normal_items(compound_symmetry(5, 0.4), 200, seed=7)
    alpha, low, high = cronbachs_alpha(df)
    se = _alpha_standard_error(np.cov(df.to_numpy(), rowvar=False), len(df))
    assert low == pytest.approx(alpha - 1.959964 * se, abs=1e-5)
    assert high == pytest.approx(alpha + 1.959964 * se, abs=1e-5)
    # a higher confidence level gives a wider interval, more respondents a narrower one
    wide, narrow = cronbachs_alpha(df, conf_level=0.99), cronbachs_alpha(df, conf_level=0.80)
    assert wide[2] - wide[1] > high - low > narrow[2] - narrow[1]
    assert wide[0] == narrow[0] == alpha
    big = normal_items(compound_symmetry(5, 0.4), 3000, seed=7)
    assert np.ptp(cronbachs_alpha(big)[1:]) < high - low
    # the upper limit is capped at 1 (alpha cannot exceed it)
    near_perfect = exact_items(compound_symmetry(4, 0.9), n=6)
    se = _alpha_standard_error(np.cov(near_perfect.to_numpy(), rowvar=False), 6)
    alpha, _, high = cronbachs_alpha(near_perfect)
    assert alpha + 1.959964 * se > 1  # without the cap the limit would be 1.008
    assert high == 1.0


def test_cronbachs_alpha_analytic_interval_has_the_right_size_and_coverage():
    """Monte Carlo with normal items: mean ASE ~ spread of alpha, coverage ~ 95% for population alpha."""
    rng = np.random.default_rng(31)
    loadings = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
    sigma = np.outer(loadings, loadings) + np.diag(1 - loadings**2)  # congeneric items
    truth, n, sims = population_alpha(sigma), 150, 400
    chol = np.linalg.cholesky(sigma)
    estimates, ses, covered = [], [], 0
    for _ in range(sims):
        x = rng.normal(size=(n, 5)) @ chol.T
        alpha, low, high = cronbachs_alpha(x)
        estimates.append(alpha)
        ses.append(_alpha_standard_error(np.cov(x, rowvar=False), n))
        covered += low <= truth <= high
    assert np.mean(estimates) == pytest.approx(truth, abs=0.01)
    assert np.mean(ses) == pytest.approx(np.std(estimates, ddof=1), rel=0.12)
    assert 0.91 <= covered / sims <= 0.98  # nominal 95%, simulation error about 1 point


def test_cronbachs_alpha_analytic_interval_agrees_with_feldts_exact_interval():
    """Under compound symmetry (equal variances and covariances) Feldt (1965) gives an exact interval."""
    from scipy.stats import f

    k, n = 6, 400
    df = normal_items(compound_symmetry(k, 0.4), n, seed=9)
    alpha, low, high = cronbachs_alpha(df)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    feldt_low = 1 - (1 - alpha) / f.ppf(0.025, df1, df2)
    feldt_high = 1 - (1 - alpha) / f.ppf(0.975, df1, df2)
    assert low == pytest.approx(feldt_low, abs=0.012)
    assert high == pytest.approx(feldt_high, abs=0.012)


def test_cronbachs_alpha_analytic_and_bootstrap_intervals_agree_for_normal_items():
    df = normal_items(compound_symmetry(5, 0.4), 500, seed=12)
    _, low, high = cronbachs_alpha(df)
    _, low_boot, high_boot = cronbachs_alpha(df, bootstrap_iter=1000)
    assert low == pytest.approx(low_boot, abs=0.012) and high == pytest.approx(high_boot, abs=0.012)


def test_cronbachs_alpha_bootstrap():
    df = normal_items(compound_symmetry(5, 0.4), 300, seed=8)
    point = cronbach_only(df)
    result = cronbachs_alpha(df, bootstrap_iter=200)
    assert isinstance(result, tuple) and len(result) == 3 and all(isinstance(v, float) for v in result)
    alpha, low, high = result
    assert alpha == point and low <= alpha <= high  # the estimate of the sample, not the bootstrap median
    # reproducible; other seeds (and seed=None) give other resamples, but the same estimate
    assert cronbachs_alpha(df, bootstrap_iter=200) == result
    other = cronbachs_alpha(df, bootstrap_iter=200, seed=99)
    assert other[0] == alpha and other[1:] != result[1:]
    unseeded = [cronbachs_alpha(df, bootstrap_iter=50, seed=None) for _ in range(2)]
    assert unseeded[0][0] == unseeded[1][0] and unseeded[0][1:] != unseeded[1][1:]
    # confidence level and sample size change the width
    wide = cronbachs_alpha(df, bootstrap_iter=200, conf_level=0.99)
    narrow = cronbachs_alpha(df, bootstrap_iter=200, conf_level=0.80)
    assert wide[2] - wide[1] > high - low > narrow[2] - narrow[1]
    big = normal_items(compound_symmetry(5, 0.4), 3000, seed=8)
    assert np.ptp(cronbachs_alpha(big, bootstrap_iter=200)[1:]) < high - low
    # a single resample is fine
    _, only_low, only_high = cronbachs_alpha(df, bootstrap_iter=1)
    assert only_low == only_high


def test_cronbachs_alpha_bootstrap_interval_is_a_percentile_interval():
    df = normal_items(compound_symmetry(5, 0.4), 200, seed=2)
    x = df.to_numpy()
    rng = np.random.default_rng(13)  # the generator and the draws of the bootstrap
    alphas = [
        _alpha_from_covariance(np.cov(x[rng.integers(0, len(x), size=len(x))], rowvar=False))
        for _ in range(100)
    ]
    _, low, high = cronbachs_alpha(df, bootstrap_iter=100, conf_level=0.90, seed=13)
    assert low == pytest.approx(np.percentile(alphas, 5))
    assert high == pytest.approx(np.percentile(alphas, 95))


def test_cronbachs_alpha_bootstrap_resamples_are_all_different(monkeypatch):
    df = normal_items(compound_symmetry(5, 0.4), 300, seed=8)
    values = []
    original = cronbach_module._alpha_from_covariance

    def record(covariance):
        values.append(original(covariance))
        return values[-1]

    monkeypatch.setattr(cronbach_module, "_alpha_from_covariance", record)
    cronbachs_alpha(df, bootstrap_iter=40)
    assert len(values) == 41  # the estimate of the original sample, then 40 resamples
    assert len(set(values[1:])) == 40


@pytest.mark.parametrize("kwargs", [{}, {"bootstrap_iter": 10}])
def test_cronbachs_alpha_does_not_touch_numpys_global_random_state(kwargs):
    df = normal_items(compound_symmetry(4, 0.4), 200)
    np.random.seed(5)
    expected = np.random.rand()
    np.random.seed(5)
    cronbachs_alpha(df, **kwargs)
    assert np.random.rand() == expected


def test_cronbachs_alpha_bootstrap_warns_once_about_the_original_sample():
    items = exact_items(np.array([[1.0, -0.5], [-0.5, 1.0]]), n=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cronbachs_alpha(items, bootstrap_iter=30)
    assert sum("negative" in str(w.message) for w in caught) == 1


def test_lib_is_a_package_that_exports_everything_from_one_place():
    public = {
        "bootstrap", "gini", "gini_from_percentiles", "log_transform", "z_transform", "htmt_ratio",
        "fleiss_kappa", "krippendorffs_alpha", "mcdonald_omega", "cronbachs_alpha",
        "average_correlations", "dependent_correlations", "independent_correlations", "rz_ci", "rho_rxy_rxz",
    }  # fmt: skip
    assert set(lib_package.__all__) == public
    expected_module = {
        "fleiss_kappa": "fleiss",
        "krippendorffs_alpha": "krippendorff",
        "mcdonald_omega": "mcdonald",
        "cronbachs_alpha": "cronbach",
        "average_correlations": "correlations",
        "dependent_correlations": "correlations",
        "independent_correlations": "correlations",
        "rz_ci": "correlations",
        "rho_rxy_rxz": "correlations",
    }
    for name in public:
        function = getattr(lib_package, name)
        assert callable(function)
        assert function.__module__ == f"apatools.lib.{expected_module.get(name, 'misc')}"
    # private helpers and the dependency check do not leak into the package namespace
    private = [
        name
        for name, value in vars(lib_package).items()
        if name.startswith("_") and not name.startswith("__") and not inspect.ismodule(value)
    ]
    assert not private  # (the private submodule _common is fine, helper functions are not)
    assert not {"pandas", "statsmodels", "np", "pd"} & set(vars(lib_package))


if __name__ == "__main__":
    test_lib()
    print("Tests for apatools.lib are PASSED!")
