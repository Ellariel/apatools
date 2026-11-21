"""McDonald's omega: reliability of a scale under a one-factor model, with confidence intervals."""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_fprime


def _item_matrix(data):
    """Item scores as a float matrix without incomplete rows; validated for the one-factor model."""
    x = np.asarray(data, dtype=float)  # raises a ValueError for non-numeric items
    if x.ndim != 2:
        raise ValueError(
            "data must be two-dimensional: one row per respondent, one column per item."
        )
    x = x[~np.isnan(x).any(axis=1)]  # listwise deletion
    n, p = x.shape
    if p < 3:
        raise ValueError(
            f"At least 3 items are needed: a one-factor model with {p} items is not identified."
        )
    if n <= p:
        raise ValueError(
            f"More respondents than items are needed (got {n} rows for {p} items)."
        )
    if np.any(x.std(axis=0) == 0):
        raise ValueError("An item is constant: its correlations are undefined.")
    return x


_MIN_UNIQUENESS = 0.005


def _one_factor_ml(r, min_uniqueness=_MIN_UNIQUENESS):
    """
    Maximum likelihood estimates ``theta = (loadings, uniquenesses)`` of the one-factor model
    ``Sigma = loadings loadings' + diag(uniquenesses)`` for the correlation matrix ``r``.

    The model does not depend on the scale of the items, so fitting the correlation matrix gives
    the same standardized solution as fitting the covariance matrix. Uniquenesses are kept above
    ``min_uniqueness``: Heywood cases (improper solutions) end on this bound. The sign is chosen
    so that the loadings sum to a positive number.
    """
    p = len(r)
    values, vectors = np.linalg.eigh(r)
    start = np.clip(vectors[:, -1] * np.sqrt(max(values[-1], 1e-3)), -0.9, 0.9)

    def sigma(theta):
        return np.outer(theta[:p], theta[:p]) + np.diag(theta[p:])

    def discrepancy(theta):  # log|Sigma| + tr(Sigma^-1 R), the ML discrepancy up to a constant
        sign, logdet = np.linalg.slogdet(sigma(theta))
        return logdet + np.trace(np.linalg.solve(sigma(theta), r)) if sign > 0 else np.inf

    def gradient(theta):
        inverse = np.linalg.inv(sigma(theta))
        g = inverse - inverse @ r @ inverse
        return np.concatenate([2 * g @ theta[:p], np.diag(g)])

    result = minimize(
        discrepancy,
        np.concatenate([start, 1 - start**2]),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(None, None)] * p + [(min_uniqueness, None)] * p,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success:
        warnings.warn(
            f"The one-factor model did not converge ({result.message}).", UserWarning
        )
    theta = result.x
    theta[:p] *= -1 if theta[:p].sum() < 0 else 1
    return theta


def _omega_from_theta(theta):
    """Omega of the standardized items from the model-implied loadings and uniquenesses."""
    p = len(theta) // 2
    loadings, uniquenesses = theta[:p], theta[p:]
    variances = loadings**2 + uniquenesses  # equal to 1 for a correlation matrix at an exact fit
    total = np.sum(loadings / np.sqrt(variances)) ** 2
    return total / (total + np.sum(uniquenesses / variances))


def _omega_delta_se(theta, n):
    """
    Delta-method standard error of omega under normal theory: the covariance matrix of the ML
    estimates is the inverse of the expected information ``n/2 * D' (S^-1 kron S^-1) D``, where
    ``D = d vec(Sigma) / d theta'``, and the gradient of omega is taken numerically.
    """
    p = len(theta) // 2
    loadings = theta[:p]
    inverse = np.linalg.inv(np.outer(loadings, loadings) + np.diag(theta[p:]))
    jacobian = np.zeros((p * p, 2 * p))
    unit = np.eye(p)
    for k in range(p):
        jacobian[:, k] = (np.outer(unit[k], loadings) + np.outer(loadings, unit[k])).ravel()
        jacobian[:, p + k] = np.outer(unit[k], unit[k]).ravel()
    information = n / 2 * jacobian.T @ np.kron(inverse, inverse) @ jacobian
    covariance = np.linalg.inv(information)
    gradient = np.ravel(approx_fprime(theta, _omega_from_theta, centered=True))
    variance = gradient @ covariance @ gradient
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def mcdonald_omega(data, bootstrap_iter=None, conf_level=0.95, seed=13):
    """
    McDonald's omega (omega total) with a confidence interval: the reliability of the sum score
    of a scale under a one-factor model with unequal loadings (congeneric items).

    ``omega = (sum(loadings))**2 / ((sum(loadings))**2 + sum(uniquenesses))`` for the standardized
    items, with the loadings and uniquenesses of a one-factor model estimated by maximum
    likelihood (multivariate normal theory). Rescaling or shifting items does not change the
    result. Unlike Cronbach's alpha, omega does not assume equal loadings: for equal loadings
    (tau-equivalent items) they coincide, otherwise alpha is lower.

    Two confidence intervals are available:

    * **Delta method** (default, ``bootstrap_iter=None``): omega is a smooth function of the model
      parameters, so its standard error follows from the asymptotic covariance matrix of the ML
      estimates (Raykov, 2002; for the structural equation modelling formulation see Raykov &
      Shrout, 2002). The interval is computed on the logit scale and transformed back, so that it
      stays within (0, 1); Kelley and Pornprasertmanit (2016) found that a logistic
      transformation improved the coverage of standard-error based intervals. The interval relies
      on normally distributed items. In their simulations, normal-theory intervals did not
      perform well for non-normal items, and they recommend bootstrap intervals when coefficient
      omega is used: use ``bootstrap_iter`` for items that are clearly not normal (skewed,
      few categories, heavy tails) or for small samples.
    * **Percentile bootstrap** (``bootstrap_iter`` = number of resamples of respondents).

    Assumptions: the items measure a single factor (unidimensional scale) and are continuous
    (for ordinal items with few categories, polychoric correlations would be needed), and are
    keyed in the same direction. Reverse-keyed items must be recoded before: a warning is given if
    the loadings have different signs, and omega is then too low. The one-factor model is assumed
    to fit; with population model error omega no longer equals the reliability and its intervals
    cover less often (Kelley & Pornprasertmanit, 2016).

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        Item scores, one row per respondent and one column per item (at least 3 items and more
        respondents than items). Rows with missing values are dropped (listwise).
    bootstrap_iter : int, optional
        Number of bootstrap resamples. If None, the delta-method interval is used.
    conf_level : float
        Confidence level of the interval, e.g. 0.95 (not a significance level).
    seed : int, optional
        Seed of the random generator used for the bootstrap (local; numpy's global random
        state is not touched). ``None`` gives different resamples on every call.

    Returns
    -------
    omega, lower, upper : float
        The estimate from the original sample and the limits of its confidence interval. The
        limits are ``nan`` (with a warning) if the delta method cannot be used, i.e. for a
        Heywood case (a uniqueness on its lower bound of .005) or a singular information matrix.

    Raises
    ------
    ValueError
        For data that are not two-dimensional or numeric, fewer than 3 items, not more
        respondents than items, constant items, or invalid ``bootstrap_iter`` / ``conf_level``.

    References
    ----------
    McDonald, R. P. (1999). Test theory: A unified treatment. Erlbaum.

    Raykov, T. (2002). Analytic estimation of standard error and confidence interval for scale
    reliability. Multivariate Behavioral Research, 37(1), 89-103.
    https://doi.org/10.1207/s15327906mbr3701_04

    Raykov, T., & Shrout, P. E. (2002). Reliability of scales with general structure: Point and
    interval estimation using a structural equation modeling approach. Structural Equation
    Modeling, 9(2), 195-212. https://doi.org/10.1207/S15328007SEM0902_3

    Kelley, K., & Pornprasertmanit, S. (2016). Confidence intervals for population reliability
    coefficients: Evaluation of methods, recommendations, and software for composite measures.
    Psychological Methods, 21(1), 69-92. https://doi.org/10.1037/a0040086

    Flora, D. B. (2020). Your coefficient alpha is probably wrong, but which coefficient omega
    is right? A tutorial on using R to obtain better reliability estimates. Advances in Methods
    and Practices in Psychological Science, 3(4), 484-501.
    https://doi.org/10.1177/2515245920951747
    """
    if bootstrap_iter is not None and (
        int(bootstrap_iter) != bootstrap_iter or bootstrap_iter < 1
    ):
        raise ValueError("bootstrap_iter must be a positive integer or None.")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1, e.g. 0.95.")
    x = _item_matrix(data)
    n, p = x.shape
    theta = _one_factor_ml(np.corrcoef(x, rowvar=False))
    omega = float(_omega_from_theta(theta))
    if (theta[:p] > 0).any() and (theta[:p] < 0).any():
        warnings.warn(
            "The factor loadings have different signs. Reverse-keyed items must be recoded "
            "before omega is computed, otherwise omega is too low.",
            UserWarning,
        )

    if bootstrap_iter is None:  # delta method, on the logit scale
        se = np.nan
        if np.any(theta[p:] <= _MIN_UNIQUENESS * (1 + 1e-6)) or not 0 < omega < 1:
            warnings.warn(
                "The solution is on the boundary (a uniqueness is at its lower bound, a Heywood "
                "case): the delta-method interval is not available, use bootstrap_iter.",
                UserWarning,
            )
        else:
            try:
                se = _omega_delta_se(theta, n)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "The information matrix is singular: the delta-method interval is not "
                    "available, use bootstrap_iter.",
                    UserWarning,
                )
        if not np.isfinite(se):
            return omega, np.nan, np.nan
        half_width = norm.ppf(1 - (1 - conf_level) / 2) * se / (omega * (1 - omega))
        center = np.log(omega / (1 - omega))
        return omega, float(expit(center - half_width)), float(expit(center + half_width))

    rng = np.random.default_rng(seed)
    omegas = np.full(int(bootstrap_iter), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the warnings concern the original sample
        for i in range(len(omegas)):
            sample = x[rng.integers(0, n, size=n)]
            if np.any(sample.std(axis=0) == 0):  # a resample without variation in an item
                continue
            omegas[i] = _omega_from_theta(
                _one_factor_ml(np.corrcoef(sample, rowvar=False))
            )
    failed = int(np.isnan(omegas).sum())
    if failed:
        warnings.warn(
            f"{failed} of {len(omegas)} bootstrap resamples could not be used.", UserWarning
        )
    lower = 100 * (1 - conf_level) / 2
    low, high = np.nanpercentile(omegas, [lower, 100 - lower])
    return omega, float(low), float(high)
