"""Statistical conversions (r, d, t, z, F, p, odds ratios) and confidence intervals.

Conventions
-----------
* ``n`` is the *total* sample size; degrees of freedom for t are ``n - 2``.
* ``d`` conversions assume two equal-sized groups unless stated otherwise.
* ``method`` is ``"two-tailed"`` (default) or any other string for one-tailed.
* ``z`` denotes a standard-normal deviate unless the name says "Fisher's z".
* All conversions are approximate.

References
----------
Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum
Associates. (Reprinted 2013 by Routledge, https://doi.org/10.4324/9780203771587)
"""

import numpy as np
import scipy
import scipy.stats

##############
# conversion #
##############


def d_from_r(r, n_groups=2):
    """Cohen's d from a correlation r, assuming equal-sized groups.

    Formula: ``d = sqrt(2 * n_groups) * r / sqrt(1 - r^2)``, the inverse of :func:`r_from_d`. For the
    default two groups this is ``d = 2r / sqrt(1 - r^2)``.

    Parameters
    ----------
    r : float or array_like
        Correlation coefficient (point-biserial r), in ``(-1, 1)``.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Cohen's d. The sign of ``r`` is preserved.

    Notes
    -----
    This is an approximation that assumes equal group sizes.

    References
    ----------
    Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009). Introduction to
    Meta-Analysis, ch. 7. Wiley. https://doi.org/10.1002/9780470743386

    https://stats.stackexchange.com/questions/390549
    """
    return np.sqrt(2 * n_groups) * r / np.sqrt(1 - r * r)


def r_from_d(d, n_groups=2):
    """Correlation r from Cohen's d, assuming equal-sized groups.

    Formula: ``r = d / sqrt(d^2 + 2 * n_groups)``. For the default two groups
    this is ``d / sqrt(d^2 + 4)``.

    Parameters
    ----------
    d : float or array_like
        Cohen's d.
    n_groups : int, default 2
        Number of (equal-sized) groups, used in the correction term
        ``2 * n_groups``.

    Returns
    -------
    float or ndarray
        Correlation r. The sign of ``d`` is preserved.

    Notes
    -----
    Approximation that assumes equal group sizes.

    References
    ----------
    Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009). Introduction to
    Meta-Analysis, ch. 7. Wiley. https://doi.org/10.1002/9780470743386

    https://easystats.github.io/effectsize/reference/d_to_r.html
    """
    return d / np.sqrt(d * d + 2 * n_groups)


def t_from_r(r, n):
    """t-statistic of a correlation.

    Formula: ``t = r / sqrt((1 - r^2) / df)`` with ``df = n - 2``.

    Parameters
    ----------
    r : float or array_like
        Correlation coefficient.
    n : int or array_like
        Total sample size (number of pairs); ``df = n - 2``.

    Returns
    -------
    float or ndarray
        t-statistic; the sign of ``r`` is preserved.
    """
    df = n - 2
    return r / np.sqrt((1 - r * r) / df)


def r_from_t(t, n):
    """Correlation r from a t-statistic.

    Formula: ``r = t / sqrt(t^2 + df)`` with ``df = n - 2``.

    Parameters
    ----------
    t : float or array_like
        t-statistic.
    n : int or array_like
        Total sample size; ``df = n - 2``.

    Returns
    -------
    float or ndarray
        Correlation r; the sign of ``t`` is preserved.
    """
    df = n - 2
    return t / np.sqrt(t * t + df)


def d_from_t(t, n):
    """Cohen's d from a t-statistic of two equal-sized groups.

    Formula: ``d = 2t / sqrt(df)`` with ``df = n - 2``.

    Parameters
    ----------
    t : float or array_like
        t-statistic of an independent-samples comparison.
    n : int or array_like
        Total sample size (both groups); ``df = n - 2``.

    Returns
    -------
    float or ndarray
        Cohen's d; the sign of ``t`` is preserved.

    Notes
    -----
    Approximation that assumes equal group sizes.

    References
    ----------
    Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates:
    Current use, calculations, and interpretation. Journal of Experimental
    Psychology: General, 141(1), 2-18. https://doi.org/10.1037/a0024338
    """
    df = n - 2
    return 2 * t / np.sqrt(df)


def t_from_d(d, n):
    """t-statistic from Cohen's d of two equal-sized groups.

    Formula: ``t = d * sqrt(df) / 2`` with ``df = n - 2``. Inverse of
    :func:`d_from_t`.

    Parameters
    ----------
    d : float or array_like
        Cohen's d.
    n : int or array_like
        Total sample size (both groups); ``df = n - 2``.

    Returns
    -------
    float or ndarray
        t-statistic; the sign of ``d`` is preserved.

    Notes
    -----
    Approximation that assumes equal group sizes (Fritz et al., 2012).
    """
    df = n - 2
    return d * np.sqrt(df) / 2


def z_from_r(r):
    """Fisher's z-transformation of a correlation.

    Formula: ``z = atanh(r) = 0.5 * ln((1 + r) / (1 - r))``.

    Parameters
    ----------
    r : float or array_like
        Correlation coefficient, in ``(-1, 1)``.

    Returns
    -------
    float or ndarray
        Fisher's z (not a standard-normal deviate).

    References
    ----------
    https://en.wikipedia.org/wiki/Fisher_transformation
    """
    return np.atanh(r)


def r_from_z(z):
    """Correlation r from Fisher's z (inverse Fisher transformation).

    Mathematically ``r = tanh(z)``; computed as ``sign(z) * sqrt(1 - sech(z)^2)``.

    Parameters
    ----------
    z : float or array_like
        Fisher's z (not a standard-normal deviate).

    Returns
    -------
    float or ndarray
        Correlation r, with the sign of ``z``.
    """
    r = np.sqrt(1 - np.square(1 / np.cosh(z)))
    return np.copysign(r, z)


def d_from_f(f, df_num, df_denom, n_groups=2):
    """Cohen's d from an F-statistic.

    Goes via eta squared and Cohen's f squared::

        eta_sq = F * df_num / (F * df_num + df_denom)
        f_sq   = eta_sq / (1 - eta_sq)
        d      = sqrt(2 * n_groups * f_sq)

    Parameters
    ----------
    f : float or array_like
        F-statistic.
    df_num : int or array_like
        Numerator degrees of freedom.
    df_denom : int or array_like
        Denominator degrees of freedom.
    n_groups : int, default 2
        Number of (equal-sized) groups. For two groups this reduces to
        ``d = 2 * sqrt(f_sq)``.

    Returns
    -------
    float or ndarray
        Cohen's d. Always non-negative (the direction of the effect is lost).

    Notes
    -----
    Approximation that assumes equal group sizes. It is only meaningful for a
    two-group comparison (``df_num = 1``), where ``F = t^2``.

    References
    ----------
    Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates:
    Current use, calculations, and interpretation. Journal of Experimental
    Psychology: General, 141(1), 2-18. https://doi.org/10.1037/a0024338
    """
    eta_sq = (f * df_num) / (f * df_num + df_denom)
    f_sq = eta_sq / (1 - eta_sq)
    return np.sqrt(f_sq * 2 * n_groups)


def r_from_f(f, df_num, df_denom):
    """Correlation r from an F-statistic.

    Formula: ``R^2 = 1 - 1 / (1 + F * df_num / df_denom)``, ``r = sqrt(R^2)``.

    Parameters
    ----------
    f : float or array_like
        F-statistic.
    df_num : int or array_like
        Numerator degrees of freedom.
    df_denom : int or array_like
        Denominator degrees of freedom.

    Returns
    -------
    float or ndarray
        Correlation r. Always non-negative (the sign is lost); for
        ``df_num > 1`` it is a multiple correlation coefficient.

    References
    ----------
    https://statproofbook.github.io/P/fstat-rsq.html
    """
    r_sq = 1 - (1 / (1 + f * df_num / df_denom))
    return np.sqrt(r_sq)


###########
# p-value #
###########


def p_from_f(f, df_num, df_denom):
    """Upper-tail p-value of an F-statistic.

    Parameters
    ----------
    f : float or array_like
        F-statistic.
    df_num : int or array_like
        Numerator degrees of freedom.
    df_denom : int or array_like
        Denominator degrees of freedom.

    Returns
    -------
    float or ndarray
        ``P(F >= f)`` (survival function of the F distribution).
    """
    return scipy.stats.f.sf(f, df_num, df_denom)


def z_from_p(p, method="two-tailed"):
    """Absolute standard-normal deviate (not Fisher's z) for a p-value.

    Parameters
    ----------
    p : float or array_like
        p-value, in ``(0, 1)``.
    method : str, default "two-tailed"
        ``"two-tailed"`` halves ``p`` before the lookup; any other string is
        treated as one-tailed (``p`` is used as is).

    Returns
    -------
    float or ndarray
        ``|z|`` such that the tail probability beyond it equals ``p`` (or
        ``p / 2`` per tail). Always non-negative.
    """
    p = p / 2 if method == "two-tailed" else p
    return np.abs(scipy.stats.norm.ppf(p))


def p_from_z(z, method="two-tailed"):
    """p-value of a standard-normal deviate.

    Parameters
    ----------
    z : float or array_like
        Standard-normal deviate (not Fisher's z). The sign is ignored.
    method : str, default "two-tailed"
        ``"two-tailed"`` doubles the tail probability; any other string is
        treated as one-tailed.

    Returns
    -------
    float or ndarray
        p-value.
    """
    p = scipy.stats.norm.sf(np.abs(z))
    return p * 2 if method == "two-tailed" else p


def p_from_t(t, n, method="two-tailed"):
    """p-value of a t-statistic with ``df = n - 2``.

    Parameters
    ----------
    t : float or array_like
        t-statistic. The sign is ignored.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str, default "two-tailed"
        ``"two-tailed"`` doubles the tail probability; any other string is
        treated as one-tailed.

    Returns
    -------
    float or ndarray
        p-value.
    """
    df = n - 2
    p = scipy.stats.t.sf(np.abs(t), df)
    return p * 2 if method == "two-tailed" else p


def t_from_p(p, n, method="two-tailed"):
    """Absolute t-statistic (``df = n - 2``) for a p-value.

    Parameters
    ----------
    p : float or array_like
        p-value, in ``(0, 1)``.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str, default "two-tailed"
        ``"two-tailed"`` halves ``p`` before the lookup; any other string is
        treated as one-tailed.

    Returns
    -------
    float or ndarray
        ``|t|``. Always non-negative; the sign is lost.
    """
    df = n - 2
    p = p / 2 if method == "two-tailed" else p
    return np.abs(scipy.stats.t.ppf(p, df))


def r_from_p(p, n, method="two-tailed"):
    """Correlation r from a p-value, via the t-statistic.

    Computed as ``r_from_t(t_from_p(p, n), n)``.

    Parameters
    ----------
    p : float or array_like
        p-value, in ``(0, 1)``.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed.

    Returns
    -------
    float or ndarray
        Correlation r. Always non-negative (the sign is lost).

    Notes
    -----
    Cohen's guidelines for r: .1 small, .3 medium, .5 large (Cohen, 1988;
    see also Fritz et al., 2012).
    """
    return r_from_t(t_from_p(p, n, method=method), n)


def p_from_r(r, n, method="two-tailed"):
    """p-value of a correlation, via its t-statistic.

    Computed as ``p_from_t(t_from_r(r, n), n)``.

    Parameters
    ----------
    r : float or array_like
        Correlation coefficient.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed.

    Returns
    -------
    float or ndarray
        p-value (the sign of ``r`` is ignored).
    """
    return p_from_t(t_from_r(r, n), n, method=method)


def d_from_p(p, n, method="two-tailed", n_groups=2):
    """Cohen's d from a p-value, via r.

    Computed as ``d_from_r(r_from_p(p, n))``.

    Parameters
    ----------
    p : float or array_like
        p-value, in ``(0, 1)``.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Cohen's d. Always non-negative (the sign is lost).

    Notes
    -----
    Approximation that assumes equal-sized groups (see :func:`d_from_r`).
    """
    return d_from_r(r_from_p(p, n, method=method), n_groups=n_groups)


def p_from_d(d, n, method="two-tailed", n_groups=2):
    """p-value of Cohen's d, via r.

    Computed as ``p_from_r(r_from_d(d, n_groups), n)``.

    Parameters
    ----------
    d : float or array_like
        Cohen's d.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        p-value (the sign of ``d`` is ignored).

    Notes
    -----
    Approximation that assumes equal-sized groups (see :func:`r_from_d`).
    """
    return p_from_r(r_from_d(d, n_groups=n_groups), n, method=method)


##############
# conflicted #
##############


def z_from_t(t, n, method=None):
    """z from a t-statistic; two different interpretations.

    Parameters
    ----------
    t : float or array_like
        t-statistic.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str or None, default None
        If falsy (``None``): return Fisher's z of the corresponding
        correlation, ``atanh(r_from_t(t, n))`` (sign preserved).
        If given (``"two-tailed"`` or any other string for one-tailed):
        return the standard-normal deviate with the same p-value as ``t``
        (always non-negative, the sign is lost).

    Returns
    -------
    float or ndarray
        Fisher's z or a standard-normal deviate, depending on ``method``.

    Notes
    -----
    The two branches measure different things and are *not* interchangeable.
    The p-value branch is a normal approximation of the t distribution.
    """
    if method:
        p = p_from_t(t, n, method=method)
        return z_from_p(p, method=method)
    return z_from_r(r_from_t(t, n))


def t_from_z(z, n, method=None):
    """t-statistic from z; inverse of :func:`z_from_t`.

    Parameters
    ----------
    z : float or array_like
        Fisher's z if ``method`` is falsy, otherwise a standard-normal deviate.
    n : int or array_like
        Total sample size; ``df = n - 2``.
    method : str or None, default None
        If falsy (``None``): ``z`` is Fisher's z; return
        ``t_from_r(r_from_z(z), n)`` (sign preserved).
        If given (``"two-tailed"`` or any other string for one-tailed):
        ``z`` is a standard-normal deviate; return the t-statistic with the
        same p-value (always non-negative, the sign is lost).

    Returns
    -------
    float or ndarray
        t-statistic.

    Notes
    -----
    See :func:`z_from_t`; the two interpretations are not interchangeable.
    """
    if method:
        p = p_from_z(z, method=method)
        return t_from_p(p, n, method=method)
    return t_from_r(r_from_z(z), n)


def d_from_z(z, n_groups=2):
    """Cohen's d from Fisher's z, via r.

    Computed as ``d_from_r(r_from_z(z), n_groups)``.

    Parameters
    ----------
    z : float or array_like
        Fisher's z (not a standard-normal deviate).
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Cohen's d; the sign of ``z`` is preserved.

    Notes
    -----
    Approximation that assumes equal-sized groups (see :func:`d_from_r`).
    """
    return d_from_r(r_from_z(z), n_groups=n_groups)


def z_from_d(d, n_groups=2):
    """Fisher's z from Cohen's d, via r.

    Computed as ``z_from_r(r_from_d(d, n_groups))``.

    Parameters
    ----------
    d : float or array_like
        Cohen's d.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Fisher's z (not a standard-normal deviate); the sign of ``d`` is
        preserved.

    Notes
    -----
    Approximation that assumes equal-sized groups (see :func:`r_from_d`).
    """
    return z_from_r(r_from_d(d, n_groups=n_groups))


########################
# Confidence intervals #
########################


def p_from_ci(estimate, cil, cir, n=None, conf_level=0.95, method="two-tailed"):
    """p-value from an estimate and its confidence interval.

    The standard error is recovered from the interval half-width,
    ``se = (cir - cil) / 2 / crit``, where ``crit`` is the critical value for
    ``conf_level``; the p-value is then that of ``estimate / se``.

    Parameters
    ----------
    estimate : float or array_like
        Point estimate (tested against zero).
    cil, cir : float or array_like
        Lower and upper confidence limits.
    n : int or None, default None
        Total sample size. If given, a t distribution with ``df = n - 2`` is
        used; if ``None``, the standard normal distribution is used.
    conf_level : float, default 0.95
        Confidence level of the interval (e.g. 0.95), *not* a significance
        level.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed; applied both to
        the critical value and to the resulting p-value.

    Returns
    -------
    float or ndarray
        p-value.

    Notes
    -----
    Assumes a symmetric interval on the scale of ``estimate``. For ratios
    (OR, RR, HR) pass log-transformed estimate and limits.

    References
    ----------
    Altman, D. G., & Bland, J. M. (2011). How to obtain the P value from a confidence interval. BMJ,
    343, d2304. https://doi.org/10.1136/bmj.d2304

    https://genometoolbox.blogspot.com/2013/11/how-to-estimate-p-value-from-confidence.html
    """
    se = (cir - cil) / 2
    if n is None:
        se /= z_from_p(1 - conf_level, method=method)
        return p_from_z(estimate / se, method=method)
    se /= t_from_p(1 - conf_level, n, method=method)
    return p_from_t(estimate / se, n, method=method)


def ci_from_r(r, n, conf_level=0.95, method="two-tailed"):
    """Confidence interval of a correlation via Fisher's z transformation.

    The interval is computed on the Fisher z scale with
    ``se = z_crit / sqrt(n - 3)`` and back-transformed with ``tanh``::

        (tanh(atanh(r) - se), tanh(atanh(r) + se))

    Parameters
    ----------
    r : float or array_like
        Correlation coefficient.
    n : int or array_like
        Total sample size (must exceed 3).
    conf_level : float, default 0.95
        Confidence level (e.g. 0.95), *not* a significance level.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed; selects how
        ``1 - conf_level`` is turned into the critical z value (a one-tailed
        method yields a narrower, still symmetric interval on the z scale).

    Returns
    -------
    tuple
        ``(lower, upper)`` limits of r.

    Notes
    -----
    Assumes bivariate normality. The interval is asymmetric around ``r`` on
    the r scale.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of correlation deduced from a
    small sample. Metron.

    https://en.wikipedia.org/wiki/Fisher_transformation
    """
    se = z_from_p(1 - conf_level, method=method) / np.sqrt(n - 3)
    return np.tanh(np.atanh(r) - se), np.tanh(np.atanh(r) + se)


def ci_from_p(p, n, conf_level=0.95, method="two-tailed"):
    """Confidence interval of r reconstructed from a p-value.

    First recovers r with :func:`r_from_p`, then applies :func:`ci_from_r`.

    Parameters
    ----------
    p : float or array_like
        p-value, in ``(0, 1)``.
    n : int or array_like
        Total sample size (must exceed 3).
    conf_level : float, default 0.95
        Confidence level (e.g. 0.95), *not* a significance level.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed; used for both
        steps.

    Returns
    -------
    tuple
        ``(lower, upper)`` limits of r.

    Notes
    -----
    The recovered r is non-negative (the sign is lost), so the interval
    describes ``|r|``.
    """
    r = r_from_p(p, n, method=method)
    return ci_from_r(r, n, conf_level=conf_level, method=method)


##############
# Cohen r, d #
##############


def cohen_r_from_p(p, n, method="two-tailed"):
    """Alias of :func:`r_from_p`.

    Parameters
    ----------
    p : float or array_like
        p-value.
    n : int or array_like
        Total sample size.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed.

    Returns
    -------
    float or ndarray
        Non-negative correlation r; see :func:`r_from_p`.
    """
    return r_from_p(p, n, method=method)


def cohen_r_from_d(d, n_groups=2):
    """Alias of :func:`r_from_d`.

    Parameters
    ----------
    d : float or array_like
        Cohen's d.
    n_groups : int, default 2
        Number of (equal-sized) groups.

    Returns
    -------
    float or ndarray
        Correlation r; see :func:`r_from_d`.
    """
    return r_from_d(d, n_groups=n_groups)


def cohen_r_from_z(z):
    """Alias of :func:`r_from_z`.

    Parameters
    ----------
    z : float or array_like
        Fisher's z.

    Returns
    -------
    float or ndarray
        Correlation r; see :func:`r_from_z`.
    """
    return r_from_z(z)


def cohen_r_from_t(t, n):
    """Alias of :func:`r_from_t`.

    Parameters
    ----------
    t : float or array_like
        t-statistic.
    n : int or array_like
        Total sample size; ``df = n - 2``.

    Returns
    -------
    float or ndarray
        Correlation r; see :func:`r_from_t`.
    """
    return r_from_t(t, n)


def cohen_r_from_f(f, df_num, df_denom):
    """Alias of :func:`r_from_f`.

    Parameters
    ----------
    f : float or array_like
        F-statistic.
    df_num, df_denom : int or array_like
        Numerator and denominator degrees of freedom.

    Returns
    -------
    float or ndarray
        Non-negative correlation r; see :func:`r_from_f`.
    """
    return r_from_f(f, df_num, df_denom)


def cohen_d_from_p(p, n, method="two-tailed", n_groups=2):
    """Alias of :func:`d_from_p`.

    Parameters
    ----------
    p : float or array_like
        p-value.
    n : int or array_like
        Total sample size.
    method : str, default "two-tailed"
        ``"two-tailed"`` or any other string for one-tailed.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Non-negative Cohen's d; see :func:`d_from_p`.
    """
    return d_from_p(p, n, method=method, n_groups=n_groups)


def cohen_d_from_r(r, n_groups=2):
    """Alias of :func:`d_from_r`.

    Parameters
    ----------
    r : float or array_like
        Correlation coefficient.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Cohen's d (equal groups); see :func:`d_from_r`.
    """
    return d_from_r(r, n_groups=n_groups)


def cohen_d_from_z(z, n_groups=2):
    """Alias of :func:`d_from_z`.

    Parameters
    ----------
    z : float or array_like
        Fisher's z.
    n_groups : int, default 2
        Number of (equal-sized) groups, as in :func:`r_from_d`.

    Returns
    -------
    float or ndarray
        Cohen's d (equal groups); see :func:`d_from_z`.
    """
    return d_from_z(z, n_groups=n_groups)


def cohen_d_from_t(t, n):
    """Alias of :func:`d_from_t`.

    Parameters
    ----------
    t : float or array_like
        t-statistic.
    n : int or array_like
        Total sample size; ``df = n - 2``.

    Returns
    -------
    float or ndarray
        Cohen's d (equal groups); see :func:`d_from_t`.
    """
    return d_from_t(t, n)


def cohen_d_from_f(f, df_num, df_denom, n_groups=2):
    """Alias of :func:`d_from_f`.

    Parameters
    ----------
    f : float or array_like
        F-statistic.
    df_num, df_denom : int or array_like
        Numerator and denominator degrees of freedom.
    n_groups : int, default 2
        Number of (equal-sized) groups.

    Returns
    -------
    float or ndarray
        Non-negative Cohen's d; see :func:`d_from_f`.
    """
    return d_from_f(f, df_num, df_denom, n_groups=n_groups)


###########
#  other  #
###########


def t_from_b(b, se):
    """t-statistic of a regression coefficient: ``t = b / se``.

    Parameters
    ----------
    b : float or array_like
        Regression coefficient.
    se : float or array_like
        Standard error of the coefficient.

    Returns
    -------
    float or ndarray
        t-statistic (Wald statistic).
    """
    return b / se


def unbiased_z(r, n):
    """Fisher's z of the bias-corrected correlation.

    The sample r is first corrected with the approximation
    ``r_u = r * (1 + (1 - r^2) / (2 * (n - 3)))`` and then transformed,
    ``z = atanh(r_u)``.

    Parameters
    ----------
    r : float or array_like
        Sample correlation coefficient.
    n : int or array_like
        Total sample size (must exceed 3).

    Returns
    -------
    float or ndarray
        Fisher's z of the bias-corrected r (not a standard-normal deviate).

    Notes
    -----
    Recommended for small samples (n < 20; Nakagawa & Cuthill, 2007). The
    correction approximates the Olkin-Pratt unbiased estimator.

    References
    ----------
    Olkin, I., & Pratt, J. W. (1958). Unbiased estimation of certain correlation coefficients. The
    Annals of Mathematical Statistics, 29(1), 201-211. https://doi.org/10.1214/aoms/1177706717

    Hedges, L. V., & Olkin, I. (1985). Statistical Methods for Meta-Analysis.
    Academic Press.

    Nakagawa, S., & Cuthill, I. C. (2007). Effect size, confidence interval and statistical
    significance: a practical guide for biologists. Biological Reviews, 82(4), 591-605.
    https://doi.org/10.1111/j.1469-185x.2007.00027.x
    """
    r_u = r * (1 + (1 - r * r) / (2 * (n - 3)))
    return np.atanh(r_u)


def logit_from_probit(x):
    """Approximate logit coefficient from a probit coefficient.

    Formula: ``logit = 1.7 * probit``.

    Parameters
    ----------
    x : float or array_like
        Probit coefficient.

    Returns
    -------
    float or ndarray
        Logit coefficient (log odds ratio).

    Notes
    -----
    Rule-of-thumb approximation; the scale factor 1.7 is a heuristic (the
    theoretical ratio of the logistic and normal scales, ``pi / sqrt(3)``, is
    about 1.81).

    References
    ----------
    https://statmodeling.stat.columbia.edu/2006/06/06/take_logit_coef/
    """
    return 1.7 * x


def OR_from_logit(x):
    """Odds ratio from a log odds ratio (logit coefficient): ``OR = exp(x)``.

    Parameters
    ----------
    x : float or array_like
        Log odds ratio.

    Returns
    -------
    float or ndarray
        Odds ratio.
    """
    return np.exp(x)


def logit_from_cohen_d(d):
    """Log odds ratio from Cohen's d.

    Formula: ``ln(OR) = pi * d / sqrt(3)``.

    Parameters
    ----------
    d : float or array_like
        Cohen's d.

    Returns
    -------
    float or ndarray
        Log odds ratio.

    Notes
    -----
    Approximation that assumes the underlying continuous variable follows a
    logistic distribution, whose standard deviation is ``pi / sqrt(3)``.

    References
    ----------
    Hasselblad, V., & Hedges, L. V. (1995). Meta-analysis of screening and diagnostic tests.
    Psychological Bulletin, 117(1), 167-178. https://doi.org/10.1037//0033-2909.117.1.167

    Chinn, S. (2000). A simple method for converting an odds ratio to effect size for use in
    meta-analysis. Statistics in Medicine, 19(22), 3127-3131.
    https://doi.org/10.1002/1097-0258(20001130)19:22<3127::aid-sim784>3.0.co;2-m
    """
    return np.pi * d / np.sqrt(3)


def cohen_d_from_logit(x):
    """Cohen's d from a log odds ratio.

    Formula: ``d = ln(OR) * sqrt(3) / pi``. Inverse of
    :func:`logit_from_cohen_d`.

    Parameters
    ----------
    x : float or array_like
        Log odds ratio.

    Returns
    -------
    float or ndarray
        Cohen's d.

    Notes
    -----
    Approximation that assumes a logistic distribution of the underlying
    continuous variable (Hasselblad & Hedges, 1995; Chinn, 2000).
    """
    return x * np.sqrt(3) / np.pi


def logit_from_OR(x):
    """Log odds ratio from an odds ratio: ``ln(OR)``.

    Parameters
    ----------
    x : float or array_like
        Odds ratio (positive).

    Returns
    -------
    float or ndarray
        Natural logarithm of the odds ratio.
    """
    return np.log(x)
