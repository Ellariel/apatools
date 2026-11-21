"""Pooling and comparing correlation coefficients.

* :func:`average_correlations`: fixed-effect pooling of correlations from several samples.
* :func:`independent_correlations`: test / confidence interval for the difference of two correlations
  from independent samples (Fisher's z test; Zou's interval).
* :func:`dependent_correlations`: the same for two overlapping correlations from one sample
  (Williams' / Steiger's t test; Zou's interval).
* :func:`rz_ci`, :func:`rho_rxy_rxz`: building blocks (confidence interval of a correlation,
  correlation between two overlapping correlations).

The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!

References
----------
Fisher, R. A. (1921). On the "probable error" of a coefficient of correlation deduced from a small
sample. Metron, 1, 3-32.
Williams, E. J. (1959). The comparison of regression variables. Journal of the Royal Statistical
Society B, 21(2), 396-399.
Steiger, J. H. (1980). Tests for comparing elements of a correlation matrix. Psychological
Bulletin, 87(2), 245-251.
Zou, G. Y. (2007). Toward using confidence intervals to compare correlations. Psychological
Methods, 12(4), 399-413. https://doi.org/10.1037/1082-989X.12.4.399
Hedges, L. V., & Olkin, I. (1985). Statistical Methods for Meta-Analysis. Academic Press.

Author: Philipp Singer (www.philippsinger.info)
"""

import numpy as np
from scipy.stats import norm, t


def average_correlations(r, n, conf_level=0.95, method="two-tailed"):
    """
    Pool correlations from several samples (fixed-effect meta-analysis on Fisher's z scale).

    The correlations are Fisher-transformed (``z = atanh(r)``), averaged with the inverse-variance
    weights ``n - 3`` (the variance of ``z`` is ``1 / (n - 3)``), tested against zero and
    back-transformed. The model assumes one common population correlation: heterogeneity between
    the samples is neither tested nor modelled (use a random-effects model for that).

    Parameters
    ----------
    r : array-like of float
        Correlations, one per sample.
    n : array-like of int
        Sample sizes, same length as ``r``.
    conf_level : float
        Confidence level of the interval, e.g. 0.95 (not a significance level). With
        ``method="one-tailed"`` both limits use the one-sided critical value for this level.
    method : {"two-tailed", "one-tailed"}
        Two- or one-tailed p-value (one-tailed: half of the two-tailed value).

    Returns
    -------
    dict
        ``r_mean`` pooled correlation, ``z_mean`` pooled Fisher z, ``se`` its standard error,
        ``z`` test statistic (``z_mean / se``), ``p-value``, ``cil`` / ``cir`` limits of the
        confidence interval of the pooled correlation.

    References
    ----------
    Hedges, L. V., & Olkin, I. (1985). Statistical Methods for Meta-Analysis. Academic Press.
    Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009). Introduction to
    Meta-Analysis. Wiley (ch. 6: correlations on Fisher's z scale; ch. 11: fixed-effect model).
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
    a = (1 - conf_level) / 2 if method == "two-tailed" else 1 - conf_level
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
    """
    Confidence interval of a correlation via Fisher's z transformation.

    ``tanh(atanh(r) +- z_crit / sqrt(n - 3))``: asymmetric around ``r`` and always within (-1, 1).
    Assumes bivariate normality and a sample (not population) correlation.

    Parameters
    ----------
    r : float
        Sample correlation.
    n : int
        Sample size (``n > 3``).
    conf_level : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    numpy.ndarray
        ``[lower, upper]``.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of correlation deduced from a
    small sample. Metron, 1, 3-32.
    """
    zr_se = np.pow(1 / (n - 3), 0.5)
    moe = norm.ppf(1 - (1 - conf_level) / float(2)) * zr_se
    zu = np.atanh(r) + moe
    zl = np.atanh(r) - moe
    return np.tanh((zl, zu))


def rho_rxy_rxz(rxy, rxz, ryz):
    """
    Estimated correlation between two overlapping correlations ``rxy`` and ``rxz``.

    Needed for the confidence interval of the difference of two dependent correlations
    (:func:`dependent_correlations`, ``method="zou"``): the covariance of the two estimates, scaled
    to a correlation (Zou, 2007; after Dunn & Clark, 1969, and Steiger, 1980).

    Parameters
    ----------
    rxy, rxz : float
        The two correlations that share the variable x.
    ryz : float
        Correlation between y and z.

    Returns
    -------
    float

    References
    ----------
    Zou, G. Y. (2007). Toward using confidence intervals to compare correlations. Psychological
    Methods, 12(4), 399-413.
    Dunn, O. J., & Clark, V. A. (1969). Correlation coefficients measured on the same individuals.
    Journal of the American Statistical Association, 64(325), 366-377.
    """
    num = (ryz - 1 / 2.0 * rxy * rxz) * (
        1 - np.pow(rxy, 2) - np.pow(rxz, 2) - np.pow(ryz, 2)
    ) + np.pow(ryz, 3)
    den = (1 - np.pow(rxy, 2)) * (1 - np.pow(rxz, 2))
    return num / float(den)


def dependent_correlations(
    xy, xz, yz, n, method="steiger", twotailed=True, conf_level=0.95
):
    """
    Compare two dependent (overlapping) correlations, r(x, y) and r(x, z), from one sample.

    Parameters
    ----------
    xy, xz : float
        The two correlations to compare; they share the variable x.
    yz : float
        Correlation between y and z.
    n : int
        Number of observations (the same for all three correlations).
    method : {"steiger", "zou"}
        ``"steiger"``: Williams' (1959) t test as given by Steiger (1980), df = n - 3.
        ``"zou"``: confidence interval of the difference (Zou, 2007).
    twotailed : bool
        Two- or one-tailed p-value; only used by ``"steiger"``.
    conf_level : float
        Confidence level of the interval, e.g. 0.95 (not a significance level); only used by ``"zou"``.

    Returns
    -------
    (t, p) for ``"steiger"``, (lower, upper) limit of the difference ``xy - xz`` for ``"zou"``.
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
        L1, U1 = rz_ci(xy, n, conf_level=conf_level)
        L2, U2 = rz_ci(xz, n, conf_level=conf_level)
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
    xy, ab, n1, n2=None, method="fisher", twotailed=True, conf_level=0.95
):
    """
    Compare two independent correlations (from different samples).

    Parameters
    ----------
    xy, ab : float
        The two correlations.
    n1, n2 : int
        Sample sizes of ``xy`` and ``ab``; ``n2`` defaults to ``n1``.
    method : {"fisher", "zou"}
        ``"fisher"``: z test on the Fisher-transformed correlations.
        ``"zou"``: confidence interval of the difference (Zou, 2007).
    twotailed : bool
        Two- or one-tailed p-value; only used by ``"fisher"``.
    conf_level : float
        Confidence level of the interval, e.g. 0.95 (not a significance level); only used by ``"zou"``.

    Returns
    -------
    (z, p) for ``"fisher"`` (z is the absolute value), (lower, upper) limit of ``xy - ab`` for ``"zou"``.
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
        if n2 is None:
            n2 = n1
        L1, U1 = rz_ci(xy, n1, conf_level=conf_level)
        L2, U2 = rz_ci(ab, n2, conf_level=conf_level)
        lower = xy - ab - np.pow((np.pow((xy - L1), 2) + np.pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + np.pow((np.pow((U1 - xy), 2) + np.pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise ValueError(f"Wrong method {method!r}: use 'fisher' or 'zou'.")
