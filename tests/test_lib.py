import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
from statsmodels.stats.inter_rater import aggregate_raters

from apatools.lib import (
    average_correlations,
    average_morans_i,
    dependent_correlations,
    fleiss_kappa,
    gini,
    gini_from_percentiles,
    htmt_ratio,
    independent_correlations,
    krippendorffs_alpha,
    log_transform,
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
            "Expert2": ["for", "no decision", "for", "no decision", "no decision", np.nan],
            "Expert3": ["for", "no decision", "for", "for", "for", np.nan],
            "Expert4": ["for", "no decision", "for", "for", "no decision", np.nan],
        }
    )
    kappa, z_score, p_value, per_item = fleiss_kappa(df, return_per_item_agreement=True)
    assert s(kappa) == "0.727"
    assert s(z_score) == "10.808"
    assert s(p_value) == "0.000"
    assert len(per_item) == len(df)


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
    assert gini_from_percentiles(pct, pct, values_are_shares=True) == pytest.approx(0, abs=1e-12)
    assert gini_from_percentiles(pct, pct) == pytest.approx(0, abs=1e-12)  # cumulative values
    # one person has everything: 1 - 1/n
    assert gini_from_percentiles([0] * 9 + [100], pct, values_are_shares=True) == pytest.approx(0.9)
    # the same with percentiles that start at 0
    pct0 = np.arange(0, 101, 10)
    assert gini_from_percentiles(pct0, pct0, values_are_shares=True) == pytest.approx(0, abs=1e-12)
    assert gini_from_percentiles([0] * 10 + [100], pct0, values_are_shares=True) == pytest.approx(0.9)
    # agrees with the sample Gini of the underlying (decile) data: incomes 1..10 per decile
    incomes = np.arange(1, 11)
    cum = np.cumsum(incomes)
    assert gini_from_percentiles(cum, pct) == pytest.approx(gini(incomes), abs=0.02)
    # cumulative values and shares give the same result
    assert gini_from_percentiles(cum, pct) == pytest.approx(
        gini_from_percentiles(100 * cum / cum[-1], pct, values_are_shares=True))


###################
# transformations #
###################


def test_log_and_z_transform():
    assert log_transform([-99, 0, 9, 99]).tolist() == pytest.approx([-2, 0, 1, 2])
    z = z_transform([1.0, 2, 3, 4, 10])
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std() == pytest.approx(1)
    z, m, sd = z_transform([2.0, 4, 6], return_params=True)
    assert (m, sd) == (4.0, pytest.approx(np.sqrt(8 / 3))) and z.tolist() == pytest.approx([-1.2247, 0, 1.2247], abs=1e-4)
    # NaNs are ignored for mean and sd and stay NaN
    z = z_transform([1.0, np.nan, 3.0])
    assert z[0] == pytest.approx(-1) and z[2] == pytest.approx(1) and np.isnan(z[1])
    # constant input
    with pytest.raises(ValueError, match="zero"):
        z_transform([5, 5, 5], ignore_errors=False)
    with np.errstate(all="ignore"):
        assert np.isnan(z_transform([5, 5, 5])).all()


######################
# interrater measures #
######################

# Fleiss (1971), as used in the Wikipedia article: 10 items, 14 raters, 5 categories; kappa = 0.210
FLEISS_COUNTS = [
    [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
]


def counts_to_ratings(counts):
    rows = [[c for c, k in enumerate(row) for _ in range(k)] for row in counts]
    return pd.DataFrame(rows, columns=[f"rater{i}" for i in range(len(rows[0]))])


def test_fleiss_kappa_reference_values():
    df = counts_to_ratings(FLEISS_COUNTS)
    kappa, z, p = fleiss_kappa(df)
    assert kappa == pytest.approx(0.210, abs=5e-4)
    assert kappa == pytest.approx(sm_fleiss_kappa(np.array(FLEISS_COUNTS)), abs=1e-12)  # statsmodels
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
    # per-item agreement is in [0, 1] and 1 for unanimous items
    _, _, _, per_item = fleiss_kappa(df, return_per_item_agreement=True)
    assert per_item.between(0, 1).all() and per_item.iloc[0] == pytest.approx(1.0)


def test_krippendorffs_alpha():
    krippendorff = pytest.importorskip("krippendorff")
    df = counts_to_ratings(FLEISS_COUNTS)
    alpha, z, p = krippendorffs_alpha(df, measurement="nominal", n_iter_for_bootstrap=50)
    # same value as the reference implementation applied to the raw (rater x item) data
    ref = krippendorff.alpha(reliability_data=df.to_numpy().T, level_of_measurement="nominal")
    assert alpha == pytest.approx(ref, abs=1e-9)
    assert 0 < alpha < 0.4 and z > 0 and 0 <= p <= 1
    # reproducible (seeded bootstrap) and optional bootstrap
    again = krippendorffs_alpha(df, n_iter_for_bootstrap=50)
    assert (alpha, z, p) == again
    a2, z2, p2 = krippendorffs_alpha(df, return_bootstraped_z_score=False)
    assert a2 == pytest.approx(alpha) and np.isnan(z2) and np.isnan(p2)
    perfect = pd.DataFrame({"a": list("xyzx"), "b": list("xyzx")})
    assert krippendorffs_alpha(perfect, return_bootstraped_z_score=False)[0] == pytest.approx(1.0)


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
    assert res["r_mean"] == pytest.approx(np.tanh(z_bar)) and res["z"] == pytest.approx(z_bar / se)
    assert res["p-value"] == pytest.approx(2 * norm.sf(abs(z_bar / se)))
    assert res["cil"] == pytest.approx(np.tanh(z_bar - 1.959964 * se), abs=1e-6)
    assert res["cir"] == pytest.approx(np.tanh(z_bar + 1.959964 * se), abs=1e-6)
    assert res["cil"] < res["r_mean"] < res["cir"]
    # the larger study dominates
    assert abs(res["r_mean"] - 0.3) < abs(res["r_mean"] - 0.5)
    # a single study returns itself
    one = average_correlations([0.4], [103])
    assert one["r_mean"] == pytest.approx(0.4) and one["se"] == pytest.approx(0.1)
    # one-tailed p is half of the two-tailed p; wider interval for a smaller alpha
    assert average_correlations(r, n, method="one-tailed")["p-value"] == pytest.approx(res["p-value"] / 2)
    narrow = average_correlations(r, n, alpha=0.2)
    assert narrow["cil"] > res["cil"] and narrow["cir"] < res["cir"]


def test_average_morans_i():
    i, var = [0.2, 0.4, 0.3], [0.01, 0.04, 0.02]
    res = average_morans_i(i, var)
    w = 1 / np.array(var)
    mean = np.sum(w * i) / w.sum()
    assert res["i_mean"] == pytest.approx(mean) and res["se"] == pytest.approx(np.sqrt(1 / w.sum()))
    assert res["z"] == pytest.approx(mean / res["se"])
    assert res["p-value"] == pytest.approx(2 * norm.sf(abs(res["z"])))
    assert average_morans_i(i, var, method="one-tailed")["p-value"] == pytest.approx(res["p-value"] / 2)
    # equal variances: plain mean; one value: its own z
    assert average_morans_i([0.1, 0.3], [0.04, 0.04])["i_mean"] == pytest.approx(0.2)
    assert average_morans_i([0.2], [0.01])["z"] == pytest.approx(2.0)
    # the most precise value dominates
    assert abs(res["i_mean"] - 0.2) < abs(res["i_mean"] - 0.4)


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
    assert dependent_correlations(0.7, 0.2, 0.4, n=200, twotailed=False)[1] == pytest.approx(p / 2)
    # more data -> stronger evidence
    assert dependent_correlations(0.5, 0.4, 0.3, n=1000)[1] < dependent_correlations(0.5, 0.4, 0.3, n=100)[1]

    # Monte Carlo: under H0 (equal true correlations) the test rejects in about 5% of the samples
    rng = np.random.default_rng(42)
    cov = np.array([[1, 0.3, 0.3], [0.3, 1, 0.5], [0.3, 0.5, 1]])  # x with y and z correlate .3
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
    lo90, hi90 = dependent_correlations(0.7, 0.2, 0.4, n=200, method="zou", alpha=0.90)
    assert lo < lo90 < hi90 < hi
    with pytest.raises(ValueError, match="Wrong method"):
        dependent_correlations(0.7, 0.2, 0.4, n=200, method="x")


def test_independent_correlations_fisher():
    z, p = independent_correlations(0.5, 0.3, 103)
    expected = abs(np.arctanh(0.5) - np.arctanh(0.3)) / np.sqrt(2 / 100)
    assert z == pytest.approx(expected) and p == pytest.approx(2 * norm.sf(expected))
    # different sample sizes; n2 defaults to n1; one-tailed
    z2, _ = independent_correlations(0.5, 0.3, 103, 53)
    assert z2 == pytest.approx(abs(np.arctanh(0.5) - np.arctanh(0.3)) / np.sqrt(1 / 100 + 1 / 50))
    assert independent_correlations(0.5, 0.3, 103, 103)[0] == pytest.approx(z)
    assert independent_correlations(0.5, 0.3, 103, twotailed=False)[1] == pytest.approx(p / 2)
    # symmetric, and equal correlations give p = 1
    assert independent_correlations(0.3, 0.5, 103)[0] == pytest.approx(z)
    assert independent_correlations(0.4, 0.4, 50)[1] == pytest.approx(1)

    # Monte Carlo under H0 (both true correlations .4): about 5% rejections
    rng = np.random.default_rng(7)
    cov = np.array([[1, 0.4], [0.4, 1]])
    rejections, reps, n = 0, 600, 60
    for _ in range(reps):
        r = [np.corrcoef(rng.multivariate_normal([0, 0], cov, size=n), rowvar=False)[0, 1] for _ in range(2)]
        rejections += independent_correlations(r[0], r[1], n)[1] < 0.05
    assert 0.025 < rejections / reps < 0.085


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
    data = pd.DataFrame(correlated_sample(corr, 400, rng), columns=["a1", "a2", "a3", "b1", "b2"])
    assert htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson") == pytest.approx(
        0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # item order does not matter; constructs are interchangeable
    assert htmt_ratio(data, ["a3", "a1", "a2"], ["b2", "b1"], method="pearson") == pytest.approx(
        htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"))
    assert htmt_ratio(data, ["b1", "b2"], ["a1", "a2", "a3"], method="pearson") == pytest.approx(
        0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # signs do not matter (absolute correlations are used)
    flipped = data.assign(a1=-data["a1"], b2=-data["b2"])
    assert htmt_ratio(flipped, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson") == pytest.approx(
        0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # Spearman is invariant to monotone transformations, Pearson is not
    mono = np.exp(data)
    sp = htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="spearman")
    assert htmt_ratio(mono, ["a1", "a2", "a3"], ["b1", "b2"], method="spearman") == pytest.approx(sp)
    assert htmt_ratio(mono, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson") != pytest.approx(
        htmt_ratio(data, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson"), abs=1e-3)
    # rows with missing values are dropped
    with_na = pd.concat([data, data.iloc[:5].assign(a1=np.nan)], ignore_index=True)
    assert htmt_ratio(with_na, ["a1", "a2", "a3"], ["b1", "b2"], method="pearson") == pytest.approx(
        0.3 / np.sqrt(0.6 * 0.5), abs=1e-9)
    # discriminant validity: indistinguishable constructs have HTMT close to 1
    same = np.full((4, 4), 0.5)
    np.fill_diagonal(same, 1)
    d2 = pd.DataFrame(correlated_sample(same, 400, rng), columns=["a1", "a2", "b1", "b2"])
    assert htmt_ratio(d2, ["a1", "a2"], ["b1", "b2"], method="pearson") == pytest.approx(1.0, abs=1e-9)


if __name__ == "__main__":
    test_lib()
    print("Tests for apatools.lib are PASSED!")
