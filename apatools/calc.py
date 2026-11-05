"""Statistical conversions (r, d, t, z, F, p, odds ratios), confidence intervals and bootstrapping.

Conventions
-----------
* ``n`` is the *total* sample size; degrees of freedom for t are ``n - 2``.
* ``d`` conversions assume two equal-sized groups unless stated otherwise.
* ``method`` is ``"two-tailed"`` (default) or any other string for one-tailed.
* ``z`` denotes a standard-normal deviate unless the name says "Fisher's z".
* All conversions are approximate.
"""
import numpy as np
import scipy
import scipy.stats


###############
# bootstrap   #
###############


def bootstrap(*args, func=np.mean, alpha=0.95, n_rep=1000, seed=13):
    """Percentile bootstrap confidence interval for a statistic.

    Parameters
    ----------
    *args : array-like
        One or more equally long samples (resampled jointly by row).
    func : callable
        Statistic computed on the resampled arrays, called as ``func(*samples)``.
    alpha : float
        Confidence level (e.g. 0.95), not a significance level.
    n_rep : int
        Number of bootstrap replications.
    seed : int
        Seed for numpy's global random generator.

    Returns
    -------
    numpy.ndarray
        ``[lower, median, upper]`` percentiles of the bootstrap distribution.
    """
    np.random.seed(seed)
    data = np.asanyarray(args)
    idx = np.arange(0, data.shape[1], 1)
    sample = [
        func(*np.take(data, np.random.choice(idx, size=data.shape[1], replace=True), 1))
        for _ in range(n_rep)
    ]
    lp, m, rp = (1 - alpha) / 2, 0.5, 1 - (1 - alpha) / 2
    return np.percentile(sample, [lp * 100, m * 100, rp * 100])


##############
# conversion #
##############


def d_from_r(r):
    """Cohen's d from correlation r (equal groups): ``2r / sqrt(1 - r^2)``.

    https://stats.stackexchange.com/questions/390549
    """
    return 2 * r / np.sqrt(1 - r * r)


def r_from_d(d, n_groups=2):
    """Correlation r from Cohen's d (equal groups): ``d / sqrt(d^2 + 2 * n_groups)``.

    https://easystats.github.io/effectsize/reference/d_to_r.html
    """
    return d / np.sqrt(d * d + 2 * n_groups)


def t_from_r(r, n):
    """t-statistic of a correlation: ``r / sqrt((1 - r^2) / (n - 2))``."""
    df = n - 2
    return r / np.sqrt((1 - r * r) / df)


def r_from_t(t, n):
    """Correlation r from a t-statistic: ``t / sqrt(t^2 + df)``, df = n - 2."""
    df = n - 2
    return t / np.sqrt(t * t + df)


def d_from_t(t, n):
    """Cohen's d from a t-statistic (equal groups): ``2t / sqrt(df)``, df = n - 2.

    Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Journal of Experimental
    Psychology: General, 141(1), 2-18. https://doi.org/10.1037/a0024338
    """
    df = n - 2
    return 2 * t / np.sqrt(df)


def t_from_d(d, n):
    """t-statistic from Cohen's d (equal groups): ``d * sqrt(df) / 2``, df = n - 2.

    Inverse of :func:`d_from_t` (Fritz et al., 2012).
    """
    df = n - 2
    return d * np.sqrt(df) / 2


def z_from_r(r):
    """Fisher's z-transformation of r: ``atanh(r)``."""
    return np.atanh(r)


def r_from_z(z):
    """Correlation r (Cohen's r) from Fisher's z: ``tanh(z)``."""
    r = np.sqrt(1 - np.square(1 / np.cosh(z)))
    return np.copysign(r, z)


def d_from_f(f, df_num, df_denom, n_groups=2):
    """Cohen's d from an F-statistic (Fritz et al., 2012).

    Goes via eta squared and Cohen's f squared: ``d = sqrt(2 * n_groups * f_sq)``.
    """
    eta_sq = (f * df_num) / (f * df_num + df_denom)
    f_sq = eta_sq / (1 - eta_sq)
    return np.sqrt(f_sq * 2 * n_groups)


def r_from_f(f, df_num, df_denom):
    """Correlation r from an F-statistic (``R^2 = 1 - 1 / (1 + F * df_num / df_denom)``).

    https://statproofbook.github.io/P/fstat-rsq.html
    """
    r_sq = 1 - (1 / (1 + f * df_num / df_denom))
    return np.sqrt(r_sq)


###########
# p-value #
###########


def p_from_f(f, df_num, df_denom):
    """Upper-tail p-value of an F-statistic."""
    return scipy.stats.f.sf(f, df_num, df_denom)


def z_from_p(p, method="two-tailed"):
    """Absolute standard-normal deviate (not Fisher's z) for a p-value."""
    p = p / 2 if method == "two-tailed" else p
    return np.abs(scipy.stats.norm.ppf(p))


def p_from_z(z, method="two-tailed"):
    """p-value of a standard-normal deviate ``z`` (sign is ignored)."""
    p = scipy.stats.norm.sf(np.abs(z))
    return p * 2 if method == "two-tailed" else p


def p_from_t(t, n, method="two-tailed"):
    """p-value of a t-statistic with df = n - 2 (sign is ignored)."""
    df = n - 2
    p = scipy.stats.t.sf(np.abs(t), df)
    return p * 2 if method == "two-tailed" else p


def t_from_p(p, n, method="two-tailed"):
    """Absolute t-statistic (df = n - 2) for a p-value; the sign is lost."""
    df = n - 2
    p = p / 2 if method == "two-tailed" else p
    return np.abs(scipy.stats.t.ppf(p, df))


def r_from_p(p, n, method="two-tailed"):
    """Correlation r from a p-value via t. Always non-negative (the sign is lost).

    Cohen's guidelines: .1 small, .3 medium, .5 large. See Fritz et al. (2012).
    """
    return r_from_t(t_from_p(p, n, method=method), n)


def p_from_r(r, n, method="two-tailed"):
    """p-value of a correlation r with sample size n."""
    return p_from_t(t_from_r(r, n), n, method=method)


def d_from_p(p, n, method="two-tailed"):
    """Cohen's d from a p-value (via r). Always non-negative."""
    return d_from_r(r_from_p(p, n, method=method))


def p_from_d(d, n, method="two-tailed"):
    """p-value of Cohen's d (via r), equal groups."""
    return p_from_r(r_from_d(d), n, method=method)


##############
# conflicted #
##############


def z_from_t(t, n, method=None):
    """z from a t-statistic.

    Without ``method``: Fisher's z of the corresponding r.
    With ``method``: the standard-normal deviate with the same p-value.
    The two branches are *not* interchangeable.
    """
    if method:
        p = p_from_t(t, n, method=method)
        return z_from_p(p, method=method)
    return z_from_r(r_from_t(t, n))


def t_from_z(z, n, method=None):
    """t-statistic from z; inverse of :func:`z_from_t` (same two interpretations)."""
    if method:
        p = p_from_z(z, method=method)
        return t_from_p(p, n, method=method)
    return t_from_r(r_from_z(z), n)


def d_from_z(z):
    """Cohen's d from Fisher's z (via r), equal groups."""
    return d_from_r(r_from_z(z))


def z_from_d(d):
    """Fisher's z from Cohen's d (via r), equal groups."""
    return z_from_r(r_from_d(d))


########################
# Confidence intervals #
########################


def p_from_ci(estimate, cil, cir, n=None, alpha=0.95, method="two-tailed"):
    """p-value from an estimate and its confidence interval.

    ``alpha`` is the confidence level of the interval (e.g. 0.95). With ``n`` a
    t-distribution (df = n - 2) is used, otherwise the normal distribution.
    For ratios (OR, RR) pass log-transformed values.
    https://genometoolbox.blogspot.com/2013/11/how-to-estimate-p-value-from-confidence.html
    """
    se = (cir - cil) / 2
    if n is None:
        se /= z_from_p(1 - alpha, method=method)
        return p_from_z(estimate / se, method=method)
    se /= t_from_p(1 - alpha, n, method=method)
    return p_from_t(estimate / se, n, method=method)


def ci_from_r(r, n, alpha=0.95, method="two-tailed"):
    """Confidence interval of a correlation via Fisher's transformation.

    ``alpha`` is the confidence level (e.g. 0.95). Returns ``(lower, upper)``.
    https://en.wikipedia.org/wiki/Fisher_transformation
    """
    se = z_from_p(1 - alpha, method=method) / np.sqrt(n - 3)
    return np.tanh(np.atanh(r) - se), np.tanh(np.atanh(r) + se)


def ci_from_p(p, n, alpha=0.95, method="two-tailed"):
    """Confidence interval of r reconstructed from a p-value (r is non-negative)."""
    r = r_from_p(p, n, method=method)
    return ci_from_r(r, n, alpha=alpha, method=method)


##############
# Cohen r, d #
##############


def cohen_r_from_p(p, n, method="two-tailed"):
    """Alias of :func:`r_from_p`."""
    return r_from_p(p, n, method=method)


def cohen_r_from_d(d, n_groups=2):
    """Alias of :func:`r_from_d`."""
    return r_from_d(d, n_groups=n_groups)


def cohen_r_from_z(z):
    """Alias of :func:`r_from_z`."""
    return r_from_z(z)


def cohen_r_from_t(t, n):
    """Alias of :func:`r_from_t`."""
    return r_from_t(t, n)


def cohen_r_from_f(f, df_num, df_denom):
    """Alias of :func:`r_from_f`."""
    return r_from_f(f, df_num, df_denom)


def cohen_d_from_p(p, n, method="two-tailed"):
    """Alias of :func:`d_from_p`."""
    return d_from_p(p, n, method=method)


def cohen_d_from_r(r):
    """Alias of :func:`d_from_r`."""
    return d_from_r(r)


def cohen_d_from_z(z):
    """Alias of :func:`d_from_z`."""
    return d_from_z(z)


def cohen_d_from_t(t, n):
    """Alias of :func:`d_from_t`."""
    return d_from_t(t, n)


def cohen_d_from_f(f, df_num, df_denom, n_groups=2):
    """Alias of :func:`d_from_f`."""
    return d_from_f(f, df_num, df_denom, n_groups=n_groups)


###########
#  other  #
###########


def t_from_b(b, se):
    """t-statistic of a regression coefficient: ``b / se``."""
    return b / se


def unbiased_z(r, n):
    """Fisher's z of the bias-corrected correlation (Hedges & Olkin, 1985).

    ``r_u = r * (1 + (1 - r^2) / (2 * (n - 3)))``, then ``atanh(r_u)``.
    Recommended for small samples (< 20, Nakagawa & Cuthill, 2007).
    """
    r_u = r * (1 + (1 - r * r) / (2 * (n - 3)))
    return np.atanh(r_u)


def logit_from_probit(x):
    """Approximate logit coefficient from a probit coefficient (``1.7 * x``).

    https://statmodeling.stat.columbia.edu/2006/06/06/take_logit_coef/
    """
    return 1.7 * x


def OR_from_logit(x):
    """Odds ratio from a log odds ratio: ``exp(x)``."""
    return np.exp(x)


def logit_from_cohen_d(d):
    """Log odds ratio from Cohen's d: ``pi * d / sqrt(3)``."""
    return np.pi * d / np.sqrt(3)


def cohen_d_from_logit(x):
    """Cohen's d from a log odds ratio: ``x * sqrt(3) / pi``."""
    return x * np.sqrt(3) / np.pi


def logit_from_OR(x):
    """Log odds ratio from an odds ratio: ``log(x)``."""
    return np.log(x)
