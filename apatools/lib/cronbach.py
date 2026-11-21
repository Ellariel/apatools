"""Cronbach's alpha with confidence intervals."""

import warnings

import numpy as np
from scipy.stats import norm

from ._common import check_interval_options, item_matrix, percentile_bootstrap


def _alpha_from_covariance(s):
    """Alpha from the item covariance matrix; ``nan`` if the total score has no variance."""
    k = len(s)
    total = s.sum()  # variance of the sum score
    if total <= 1e-12 * np.trace(s):  # the items cancel each other out
        return np.nan
    return k / (k - 1) * (1 - np.trace(s) / total)


def _alpha_standard_error(s, n):
    """
    Asymptotic standard error of alpha under normal theory for a general covariance matrix.

    With ``G = d alpha / d S`` and the normal-theory covariance of a sample covariance matrix,
    ``var(alpha) = 2/n * tr(G S G S)``, which gives (van Zyl et al., 2000; Duhachek &
    Iacobucci, 2004): ``2 k^2 / ((k - 1)^2 j'Sj^3 n) * (j'Sj (tr S^2 + (tr S)^2) - 2 tr S j'S^2j)``.
    """
    k = len(s)
    total, trace = s.sum(), np.trace(s)  # j'Sj and tr S
    squared = s @ s
    bracket = total * (np.trace(squared) + trace**2) - 2 * trace * squared.sum()
    variance = 2 * k**2 / ((k - 1) ** 2 * total**3 * n) * bracket
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def cronbachs_alpha(data, bootstrap_iter=None, conf_level=0.95, seed=13):
    """
    Cronbach's alpha with a confidence interval.

    ``alpha = k / (k - 1) * (1 - sum(item variances) / variance of the sum score)`` for ``k``
    items, computed from the sample covariance matrix. Alpha equals the reliability of the sum
    score only if the items are tau-equivalent (equal loadings on one factor) and their errors
    are uncorrelated; otherwise it underestimates the reliability (use :func:`mcdonald_omega`,
    which does not assume equal loadings). Kelley and Pornprasertmanit (2016) therefore advise
    against alpha as an estimate of population reliability and, if alpha is required, to report
    it along with a more appropriate coefficient; see also Sijtsma (2009). Alpha is computed from
    the raw covariances, so it changes when single items are rescaled (it does not change when
    all items are shifted or rescaled by the same factor).

    Two confidence intervals are available:

    * **Analytic** (default, ``bootstrap_iter=None``): ``alpha ± z * ASE`` with the asymptotic
      standard error of alpha under normal theory (van Zyl et al., 2000; Duhachek & Iacobucci,
      2004), which does not assume equal item variances or covariances. The upper limit is
      capped at 1. It relies on normally distributed items: for clearly non-normal items (skewed,
      few categories, heavy tails) or small samples use ``bootstrap_iter``, see Padilla et al.
      (2012).
    * **Percentile bootstrap** (``bootstrap_iter`` = number of resamples of respondents).

    Items must be keyed in the same direction: a warning is given if alpha is negative, which
    means that the items correlate negatively on average (e.g. reverse-keyed items that were not
    recoded).

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        Item scores, one row per respondent and one column per item (at least 2 items and more
        respondents than items). Rows with missing values are dropped (listwise).
    bootstrap_iter : int, optional
        Number of bootstrap resamples. If None, the analytic interval is used.
    conf_level : float
        Confidence level of the interval, e.g. 0.95 (not a significance level).
    seed : int, optional
        Seed of the random generator used for the bootstrap (local; numpy's global random
        state is not touched). ``None`` gives different resamples on every call.

    Returns
    -------
    alpha, lower, upper : float
        The estimate from the original sample and the limits of its confidence interval. All are
        ``nan`` (with a warning) if the sum score has no variance; the limits are ``nan`` if the
        analytic standard error cannot be computed.

    Raises
    ------
    ValueError
        For data that are not two-dimensional or numeric, fewer than 2 items, not more
        respondents than items, constant items, or invalid ``bootstrap_iter`` / ``conf_level``.

    References
    ----------
    Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.
    Psychometrika, 16(3), 297-334. https://doi.org/10.1007/BF02310555

    van Zyl, J. M., Neudecker, H., & Nel, D. G. (2000). On the distribution of the maximum
    likelihood estimator of Cronbach's alpha. Psychometrika, 65(3), 271-280.
    https://doi.org/10.1007/BF02296146

    Duhachek, A., & Iacobucci, D. (2004). Alpha's standard error (ASE): An accurate and precise
    confidence interval estimate. Journal of Applied Psychology, 89(5), 792-808.
    https://doi.org/10.1037/0021-9010.89.5.792

    Padilla, M. A., Divers, J., & Newton, M. (2012). Coefficient alpha bootstrap confidence
    interval under nonnormality. Applied Psychological Measurement, 36, 331-348.
    https://doi.org/10.1177/0146621612445470

    Sijtsma, K. (2009). On the use, the misuse, and the very limited usefulness of Cronbach's
    alpha. Psychometrika, 74(1), 107-120. https://doi.org/10.1007/s11336-008-9101-0

    Kelley, K., & Pornprasertmanit, S. (2016). Confidence intervals for population reliability
    coefficients: Evaluation of methods, recommendations, and software for composite measures.
    Psychological Methods, 21(1), 69-92. https://doi.org/10.1037/a0040086
    """
    check_interval_options(bootstrap_iter, conf_level)
    x = item_matrix(data, 2)
    n = len(x)
    covariance = np.cov(x, rowvar=False)
    alpha = float(_alpha_from_covariance(covariance))
    if not np.isfinite(alpha):
        warnings.warn(
            "The sum score has no variance (the items cancel each other out): alpha is undefined.",
            UserWarning,
        )
        return np.nan, np.nan, np.nan
    if alpha < 0:
        warnings.warn(
            "Alpha is negative: the items correlate negatively on average. Reverse-keyed items "
            "must be recoded before alpha is computed.",
            UserWarning,
        )

    if bootstrap_iter is None:  # analytic interval, alpha ± z * ASE
        se = _alpha_standard_error(covariance, n)
        if not np.isfinite(se):
            return alpha, np.nan, np.nan
        half_width = norm.ppf(1 - (1 - conf_level) / 2) * se
        return alpha, float(alpha - half_width), float(min(alpha + half_width, 1.0))

    low, high = percentile_bootstrap(
        x,
        lambda sample: _alpha_from_covariance(np.cov(sample, rowvar=False)),
        bootstrap_iter,
        conf_level,
        seed,
    )
    return alpha, low, high
