import numpy as np
import pytest

from apatools.calc import (
    ci_from_p,
    ci_from_r,
    cohen_d_from_logit,
    cohen_d_from_p,
    cohen_d_from_r,
    cohen_d_from_t,
    cohen_d_from_z,
    d_from_f,
    d_from_p,
    d_from_r,
    d_from_t,
    d_from_z,
    logit_from_cohen_d,
    logit_from_OR,
    logit_from_probit,
    OR_from_logit,
    p_from_ci,
    p_from_d,
    p_from_f,
    p_from_r,
    p_from_t,
    p_from_z,
    r_from_d,
    r_from_f,
    r_from_p,
    r_from_t,
    r_from_z,
    t_from_b,
    t_from_d,
    t_from_p,
    t_from_r,
    t_from_z,
    unbiased_z,
    z_from_d,
    z_from_p,
    z_from_r,
    z_from_t,
)


def s(x):
    return f"{x:.3f}"


def test_conversion():
    ##############
    # conversion #
    ##############

    p = p_from_f(32.82, 9, 33)
    assert s(p) == "0.000"

    r = r_from_f(32.82, 9, 33)
    assert s(r) == "0.948"

    d = d_from_f(32.82, 9, 33)
    assert s(d) == "5.984"

    r = r_from_d(d)
    assert s(r) == "0.948"

    r = 0.75
    d = d_from_r(r)
    assert s(d) == "2.268"
    r = r_from_d(d)
    assert s(r) == "0.750"

    r = 0.5
    z = z_from_r(r)
    r = r_from_z(z)
    assert s(z) == "0.549"
    assert s(r) == "0.500"

    r = 0.9
    n = 100
    t = t_from_r(r, n)
    z = z_from_r(r)
    assert s(t) == "20.440"
    assert s(z) == "1.472"
    r = r_from_t(t, n)
    assert s(r) == "0.900"
    r = r_from_z(z)
    assert s(r) == "0.900"
    d = d_from_t(t, n)
    assert s(d) == "4.129"
    r = r_from_d(d)
    assert s(r) == "0.900"
    d = d_from_z(z)
    assert s(d) == "4.129"
    r = r_from_d(d)
    assert s(r) == "0.900"
    z = z_from_d(d)
    assert s(z) == "1.472"
    t = t_from_d(d, n)
    assert s(t) == "20.440"

    ###########
    # p-value #
    ###########

    r = -0.181
    n = 181
    p = p_from_r(r, n, method="two-tailed")
    assert s(p) == "0.015"
    r = r_from_p(p, n, method="two-tailed")
    assert s(r) == "0.181"  # does not keep sign

    d = -0.41
    n = 181
    p = p_from_d(d, n, method="two-tailed")
    assert s(p) == "0.007"
    d = d_from_p(p, n, method="two-tailed")
    assert s(d) == "0.410"  # does not keep sign

    t = -2.16
    n = 163
    p = p_from_t(t, n, method="two-tailed")
    assert s(p) == "0.032"
    t = t_from_p(p, n, method="two-tailed")
    assert s(t) == "2.160"
    t = 2.16
    p = p_from_t(t, n, method="two-tailed")
    assert s(p) == "0.032"
    t = t_from_p(p, n, method="two-tailed")
    assert s(t) == "2.160"

    p = 0.5
    z = z_from_p(p, method="two-tailed")
    p = p_from_z(z, method="two-tailed")
    assert s(z) == "0.674"
    assert s(p) == "0.500"
    z = -2.335
    p = p_from_z(z, method="two-tailed")
    assert s(p) == "0.020"
    z = 2.052
    p = p_from_z(z, method="two-tailed")
    assert s(p) == "0.040"
    z = z_from_p(p, method="two-tailed")
    assert s(z) == "2.052"

    n = 163
    z = -2.335
    p = p_from_z(z, method="two-tailed")
    t = t_from_p(p, n, method="two-tailed")
    p = p_from_t(t, n, method="two-tailed")
    assert s(p) == "0.020"
    z = z_from_p(p)
    assert s(z) == "2.335"

    t = -2.16
    p = p_from_t(t, n, method="two-tailed")
    z = z_from_p(p, method="two-tailed")
    p = p_from_z(z, method="two-tailed")
    assert s(p) == "0.032"
    t = t_from_p(p, n, method="two-tailed")
    assert s(t) == "2.160"

    d = 2.5
    n = 10
    t = t_from_d(d, n)
    assert s(t) == "3.536"
    d = d_from_t(t, n)
    assert s(d) == "2.500"
    z = z_from_t(t, n)
    d = d_from_z(z)
    assert s(d) == "2.500"

    d = 2.5
    n = 10
    z = z_from_d(d)
    assert s(z) == "1.048"
    d = d_from_z(z)
    assert s(d) == "2.500"
    t = t_from_z(z, n)
    d = d_from_t(t, n)
    assert s(d) == "2.500"

    r = 0.9
    n = 10
    t = t_from_r(r, n)
    assert s(t) == "5.840"
    r = r_from_t(t, n)
    assert s(r) == "0.900"
    z = z_from_t(t, n)
    r = r_from_z(z)
    assert s(r) == "0.900"

    r = 0.90
    n = 10
    z = z_from_r(r)
    assert s(z) == "1.472"
    r = r_from_z(z)
    assert s(r) == "0.900"
    t = t_from_z(z, n)
    r = r_from_t(t, n)
    assert s(r) == "0.900"

    p = 0.010
    n = 10
    t = t_from_p(p, n, method="two-tailed")
    assert s(t) == "3.355"
    p = p_from_t(t, n, method="two-tailed")
    assert s(p) == "0.010"
    z = z_from_t(t, n, method="two-tailed")
    p = p_from_z(z, method="two-tailed")
    assert s(p) == "0.010"

    p = 0.010
    n = 10
    z = z_from_p(p, method="two-tailed")
    assert s(z) == "2.576"
    p = p_from_z(z, method="two-tailed")
    assert s(p) == "0.010"
    t = t_from_z(z, n, method="two-tailed")
    p = p_from_t(t, n)
    assert s(p) == "0.010"

    # sign check

    r = -0.9
    n = 10
    t = t_from_r(r, n)
    assert s(t) == "-5.840"
    r = r_from_t(t, n)
    assert s(r) == "-0.900"
    z = z_from_t(t, n)
    assert s(z) == "-1.472"
    r = r_from_z(z)
    assert s(r) == "-0.900"

    r = -0.90
    n = 10
    z = z_from_r(r)
    assert s(z) == "-1.472"
    r = r_from_z(z)
    assert s(r) == "-0.900"
    t = t_from_z(z, n)
    assert s(t) == "-5.840"
    r = r_from_t(t, n)
    assert s(r) == "-0.900"

    d = -2.5
    n = 10
    t = t_from_d(d, n)
    assert s(t) == "-3.536"
    d = d_from_t(t, n)
    assert s(d) == "-2.500"
    z = z_from_t(t, n)
    assert s(z) == "-1.048"
    d = d_from_z(z)
    assert s(d) == "-2.500"

    d = -2.5
    n = 10
    z = z_from_d(d)
    assert s(z) == "-1.048"
    d = d_from_z(z)
    assert s(d) == "-2.500"
    t = t_from_z(z, n)
    assert s(t) == "-3.536"
    d = d_from_t(t, n)
    assert s(d) == "-2.500"

    p = p_from_r(-0.79056, 5)
    assert (
        s(p) == "0.111"
    )  # spearman(statistic=-0.7905694150420948, pvalue=0.11136715471408384)

    p = p_from_ci(
        6.37,
        0.773,
        11.959,
    )
    assert s(p) == "0.026"  # 0.028
    p = p_from_ci(6.37, 0.773, 11.959, 88)
    assert s(p) == "0.026"  # 0.028

    cil, cir = ci_from_r(0.95, 100)
    assert s(cil) == "0.926"
    assert s(cir) == "0.966"

    cil, cir = ci_from_r(0.053, 198)
    p = p_from_ci(0.053, -0.087, 0.191, 198)
    assert s(cil) == "-0.087"
    assert s(cir) == "0.191"  # 0.206
    assert s(p) == "0.453"  # 0.553

    cil, cir = ci_from_p(p, 198)
    assert s(cil) == "-0.086"
    assert s(cir) == "0.192"


def test_regressions():
    # CI width must be positive and the p-value must not depend on its sign
    assert abs(p_from_ci(0.5, 0.1, 0.9) - 0.0143) < 1e-3
    assert abs(p_from_ci(-0.5, -0.9, -0.1) - 0.0143) < 1e-3
    # probit -> logit scaling is positive
    assert logit_from_probit(1.0) > 0
    # unbiased Fisher's z is slightly larger than the plain one for small n
    assert unbiased_z(0.5, 10) > z_from_r(0.5)
    assert abs(unbiased_z(0.5, 10_000) - z_from_r(0.5)) < 1e-3
    # r <-> t <-> p round trip
    assert abs(p_from_r(r_from_p(0.01, 40), 40) - 0.01) < 1e-9


@pytest.mark.parametrize("r", [-0.9, -0.3, 0.0, 0.1, 0.5, 0.95])
def test_round_trips(r):
    n = 50
    assert r_from_z(z_from_r(r)) == pytest.approx(r)
    assert r_from_d(d_from_r(r)) == pytest.approx(r)
    assert r_from_t(t_from_r(r, n), n) == pytest.approx(r)
    assert cohen_d_from_t(t_from_d(d_from_r(r), n), n) == pytest.approx(d_from_r(r))
    assert d_from_z(z_from_d(d_from_r(r))) == pytest.approx(d_from_r(r))
    assert t_from_z(z_from_t(t_from_r(r, n), n), n) == pytest.approx(t_from_r(r, n))


@pytest.mark.parametrize("p", [0.001, 0.01, 0.05, 0.3])
@pytest.mark.parametrize("method", ["two-tailed", "one-tailed"])
def test_p_round_trips(p, method):
    n = 40
    assert p_from_z(z_from_p(p, method=method), method=method) == pytest.approx(p)
    assert p_from_t(t_from_p(p, n, method=method), n, method=method) == pytest.approx(p)
    assert p_from_r(r_from_p(p, n, method=method), n, method=method) == pytest.approx(p)
    assert p_from_d(d_from_p(p, n, method=method), n, method=method) == pytest.approx(p)
    # the "normal deviate" branch of z_from_t / t_from_z is consistent as well
    t = t_from_p(p, n, method=method)
    assert t_from_z(z_from_t(t, n, method=method), n, method=method) == pytest.approx(t)


def test_known_values():
    assert z_from_p(0.05) == pytest.approx(1.959964, abs=1e-5)
    assert z_from_p(0.05, method="one-tailed") == pytest.approx(1.644854, abs=1e-5)
    assert p_from_t(2.0, 12) == pytest.approx(0.0734, abs=1e-3)  # df = 10
    assert r_from_d(1.0) == pytest.approx(1 / np.sqrt(5))
    assert d_from_r(r_from_d(0.8)) == pytest.approx(0.8)
    assert OR_from_logit(logit_from_OR(2.5)) == pytest.approx(2.5)
    assert cohen_d_from_logit(logit_from_cohen_d(0.4)) == pytest.approx(0.4)
    assert logit_from_probit(1.0) == pytest.approx(1.7)
    assert t_from_b(2.0, 0.5) == 4.0
    # F(1, df) is t squared
    t, df = 2.5, 20
    assert p_from_f(t**2, 1, df) == pytest.approx(p_from_t(t, df + 2))
    assert r_from_f(t**2, 1, df) == pytest.approx(r_from_t(t, df + 2))
    assert d_from_f(t**2, 1, df) == pytest.approx(d_from_t(t, df + 2))


def test_signs_and_symmetry():
    assert r_from_z(-1.0) == pytest.approx(-r_from_z(1.0))
    assert r_from_d(-0.5) < 0 and d_from_r(-0.5) < 0
    assert p_from_t(-2.0, 30) == pytest.approx(p_from_t(2.0, 30))
    assert p_from_r(-0.4, 60) == pytest.approx(p_from_r(0.4, 60))
    # p-value based reconstructions are always non-negative
    assert r_from_p(0.01, 40) > 0 and d_from_p(0.01, 40) > 0
    lo, hi = ci_from_r(-0.5, 40)
    assert lo < -0.5 < hi


def test_confidence_intervals():
    lo, hi = ci_from_r(0.5, 103)  # se of Fisher z = 0.1
    assert lo == pytest.approx(np.tanh(np.atanh(0.5) - 1.959964 * 0.1), abs=1e-5)
    assert hi == pytest.approx(np.tanh(np.atanh(0.5) + 1.959964 * 0.1), abs=1e-5)
    # narrower interval for a lower confidence level
    lo90, hi90 = ci_from_r(0.5, 103, conf_level=0.90)
    assert lo < lo90 < 0.5 < hi90 < hi
    # a normal-theory 95% CI that just excludes zero gives p ~ .05
    est, se = 0.98, 0.5  # z = 1.96, so the interval is (0, 1.96)
    assert p_from_ci(est, est - 1.96 * se, est + 1.96 * se) == pytest.approx(
        0.05, abs=2e-3
    )
    assert p_from_ci(2.0, 1.02, 2.98) == pytest.approx(p_from_z(2.0 / 0.5), abs=1e-3)
    # the t-based version is more conservative
    assert p_from_ci(est, est - 1.96 * se, est + 1.96 * se, n=20) > p_from_ci(
        est, est - 1.96 * se, est + 1.96 * se
    )
    # ci_from_p returns an interval around the r implied by p
    lo, hi = ci_from_p(0.01, 40)
    assert lo < r_from_p(0.01, 40) < hi


if __name__ == "__main__":
    test_conversion()
    test_regressions()
    print("Tests for apatools.calc are PASSED!")


def test_d_r_z_p_conversions_round_trip_for_any_number_of_groups():
    for k in (2, 3, 4):
        for d in (-1.2, 0.0, 0.3, 2.0):
            r = r_from_d(d, n_groups=k)
            assert d_from_r(r, n_groups=k) == pytest.approx(d)
            assert d_from_z(z_from_d(d, n_groups=k), n_groups=k) == pytest.approx(d)
            assert cohen_d_from_r(r, n_groups=k) == pytest.approx(d)
            assert cohen_d_from_z(z_from_r(r), n_groups=k) == pytest.approx(d)
            assert d_from_p(p_from_d(d, 80, n_groups=k), 80, n_groups=k) == pytest.approx(abs(d), abs=1e-6) or d == 0
            assert cohen_d_from_p(p_from_d(1.0, 80, n_groups=k), 80, n_groups=k) == pytest.approx(1.0)
    # the default is two groups: d = 2r / sqrt(1 - r^2)
    assert d_from_r(0.5) == pytest.approx(2 * 0.5 / np.sqrt(0.75))
    assert d_from_r(0.5) == d_from_r(0.5, n_groups=2)
    assert d_from_z(0.5) == d_from_z(0.5, n_groups=2) and z_from_d(0.5) == z_from_d(0.5, n_groups=2)
    # more groups: the same r belongs to a larger d
    assert d_from_r(0.5, n_groups=4) > d_from_r(0.5, n_groups=2)
