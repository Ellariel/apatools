import numpy as np
from scipy.stats import spearmanr

from apatools.calc import *


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

    cil, m, cir = bootstrap(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 8, 4, 5, 6, 7, 8, 9],
        func=f,
        n_rep=1000,
        seed=13,
    )
    assert s(cil) == "0.029"
    assert s(m) == "0.745"
    assert s(cir) == "1.000"

    p = p_from_r(m, 9)
    cil, cir = ci_from_r(m, 9)
    assert s(cil) == "0.159"
    assert s(p) == "0.021"
    assert s(cir) == "0.943"

    # spearmanr
    # SignificanceResult(statistic=np.float64(0.7173570516842638),
    # pvalue=np.float64(0.02958657594620456))


if __name__ == "__main__":
    test_conversion()
    test_bootstrap()
    print("Tests for metatools.calc are PASSED!")
