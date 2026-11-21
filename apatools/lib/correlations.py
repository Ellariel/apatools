"""Pooling and comparing correlation coefficients.

Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!

Author: Philipp Singer (www.philippsinger.info)
"""

import numpy as np
from scipy.stats import norm, t


def average_correlations(r, n, alpha=0.05, method="two-tailed"):
    """
    r : list/array of correlations
    n : list/array of sample sizes
    """
    r = np.array(r, dtype=float)
    n = np.array(n, dtype=float)
    z = np.arctanh(r)  # 1) Fisher z-transform
    w = n - 3  # 2) Weights (inverse variance)
    z_bar = np.sum(w * z) / np.sum(
        w
    )  # 3) Pooled effect (mean z weighted by sample size)
    se = 1 / np.sqrt(np.sum(w))  # 4) Standard error of pooled effect
    z = z_bar / se  # 5) Test statistic
    p = 1 - norm.cdf(np.abs(z))  # 6) Two-tailed p-value
    p = p * 2 if method == "two-tailed" else p
    r_bar = np.tanh(z_bar)  # 7) Convert back to correlation
    # 8) CI in z-space, then back-transform
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)
    a = alpha / 2 if method == "two-tailed" else alpha
    z_crit = norm.ppf(1 - a)
    z_ci_low = z_bar - z_crit * se
    z_ci_high = z_bar + z_crit * se
    r_ci_low = np.tanh(z_ci_low)
    r_ci_high = np.tanh(z_ci_high)
    return {
        "r_mean": r_bar,
        "z_mean": z_bar,
        "se": se,
        "z": z,
        "p-value": p,
        "cil": r_ci_low,
        "cir": r_ci_high,
    }


def rz_ci(r, n, conf_level=0.95):
    zr_se = np.pow(1 / (n - 3), 0.5)
    moe = norm.ppf(1 - (1 - conf_level) / float(2)) * zr_se
    zu = np.atanh(r) + moe
    zl = np.atanh(r) - moe
    return np.tanh((zl, zu))


def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz - 1 / 2.0 * rxy * rxz) * (
        1 - np.pow(rxy, 2) - np.pow(rxz, 2) - np.pow(ryz, 2)
    ) + np.pow(ryz, 3)
    den = (1 - np.pow(rxy, 2)) * (1 - np.pow(rxz, 2))
    return num / float(den)


def dependent_correlations(xy, xz, yz, n, method="steiger", twotailed=True, alpha=0.95):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    xy: correlation coefficient between x and y
    xz: correlation coefficient between x and z
    yz: correlation coefficient between y and z
    n: number of elements in x, y and z
    twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    alpha / conf_level: confidence level, only works for 'zou' method
    method: defines the method uses, 'steiger' or 'zou'
    t and p-val
    """
    if method == "steiger":
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz) / 2
        cube = (1 - yz) * (1 - yz) * (1 - yz)
        t2 = d * np.sqrt(
            (n - 1) * (1 + yz) / ((2 * (n - 1) / (n - 3)) * determin + av * av * cube)
        )
        p = 1 - t.cdf(abs(t2), n - 3)
        if twotailed:
            p *= 2
        return t2, p
    elif method == "zou":
        L1 = rz_ci(xy, n, conf_level=alpha)[0]
        U1 = rz_ci(xy, n, conf_level=alpha)[1]
        L2 = rz_ci(xz, n, conf_level=alpha)[0]
        U2 = rz_ci(xz, n, conf_level=alpha)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = (
            xy
            - xz
            - np.pow(
                (
                    np.pow((xy - L1), 2)
                    + np.pow((U2 - xz), 2)
                    - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)
                ),
                0.5,
            )
        )
        upper = (
            xy
            - xz
            + np.pow(
                (
                    np.pow((U1 - xy), 2)
                    + np.pow((xz - L2), 2)
                    - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)
                ),
                0.5,
            )
        )
        return lower, upper
    else:
        raise ValueError(f"Wrong method {method!r}: use 'steiger' or 'zou'.")


def independent_correlations(
    xy, ab, n1, n2=None, method="fisher", twotailed=True, alpha=0.95
):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n1: number of elements in xy
    @param n2: number of elements in ab (if distinct from n1)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param alpha: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == "fisher":
        xy_z = 0.5 * np.log((1 + xy) / (1 - xy))
        ab_z = 0.5 * np.log((1 + ab) / (1 - ab))
        if n2 is None:
            n2 = n1
        se_diff_r = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        diff = xy_z - ab_z
        z = np.abs(diff / se_diff_r)
        p = 1 - norm.cdf(z)
        if twotailed:
            p *= 2
        return z, p
    elif method == "zou":
        L1 = rz_ci(xy, n1, conf_level=alpha)[0]
        U1 = rz_ci(xy, n1, conf_level=alpha)[1]
        L2 = rz_ci(ab, n2, conf_level=alpha)[0]
        U2 = rz_ci(ab, n2, conf_level=alpha)[1]
        lower = xy - ab - np.pow((np.pow((xy - L1), 2) + np.pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + np.pow((np.pow((U1 - xy), 2) + np.pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise ValueError(f"Wrong method {method!r}: use 'fisher' or 'zou'.")
