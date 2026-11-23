"""McDonald's omega coefficients: reliability of a scale and general factor saturation, with confidence intervals.

* :func:`mcdonald_omega`: omega of a unidimensional scale (one-factor model); optionally robust to
  outlying respondents (Huber-type M-estimated covariance matrix) and for the raw sum score.
* :func:`mcdonald_omega_bifactor`: omega hierarchical, omega total and subscale omegas of a scale with
  subscales (bifactor model).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_fprime

from ._common import check_interval_options, item_matrix, percentile_bootstrap
from ._robust import huber_asymptotic_covariance, huber_location_scatter

_MIN_UNIQUENESS = 0.005


###############################################################################
# maximum likelihood factor model with a given pattern of loadings
###############################################################################


def _ml_factor_model(r, mask, starts, name="one-factor", min_uniqueness=_MIN_UNIQUENESS):
    """
    Maximum likelihood estimates of ``Sigma = L L' + diag(theta)`` for the correlation matrix ``r``.

    ``mask`` (``p x F``, boolean) marks the free loadings of ``L``; the others are zero, and the factors are
    uncorrelated with unit variance. ``starts`` are the starting values ``(free loadings, theta)`` (the
    best of the converged fits is kept: models with several factors have local optima). Uniquenesses
    are kept above ``min_uniqueness``: Heywood cases (improper solutions) end on this bound. The model
    does not depend on the scale of the items, so fitting the correlation matrix gives the standardized
    solution of the covariance matrix. The signs of the factors are chosen so that every column of
    loadings has a positive sum. Returns ``(loadings, uniquenesses)`` with ``L`` of shape ``p x F``.
    """
    p, factors = mask.shape
    where = np.argwhere(mask)
    k = len(where)

    def unpack(phi):
        loadings = np.zeros((p, factors))
        loadings[where[:, 0], where[:, 1]] = phi[:k]
        return loadings, phi[k:]

    def sigma(phi):
        loadings, theta = unpack(phi)
        return loadings @ loadings.T + np.diag(theta)

    def discrepancy(phi):  # log|Sigma| + tr(Sigma^-1 R), the ML discrepancy up to a constant
        sign, logdet = np.linalg.slogdet(sigma(phi))
        return logdet + np.trace(np.linalg.solve(sigma(phi), r)) if sign > 0 else np.inf

    def gradient(phi):
        inverse = np.linalg.inv(sigma(phi))
        g = inverse - inverse @ r @ inverse
        loadings, _ = unpack(phi)
        return np.concatenate([(2 * g @ loadings)[where[:, 0], where[:, 1]], np.diag(g)])

    bounds = [(None, None)] * k + [(min_uniqueness, None)] * p
    best = None
    for start in starts:
        result = minimize(
            discrepancy,
            start,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        if best is None or (result.success, -result.fun) > (best.success, -best.fun):
            best = result
    if not best.success:
        warnings.warn(f"The {name} model did not converge ({best.message}).", UserWarning)
    loadings, theta = unpack(best.x)
    loadings = loadings * np.where(loadings.sum(axis=0) < 0, -1.0, 1.0)
    return loadings, theta


def _one_factor_start(r):
    values, vectors = np.linalg.eigh(r)
    start = np.clip(vectors[:, -1] * np.sqrt(max(values[-1], 1e-3)), -0.9, 0.9)
    return np.concatenate([start, 1 - start**2])


def _one_factor_ml(r, min_uniqueness=_MIN_UNIQUENESS):
    """
    Maximum likelihood estimates ``theta = (loadings, uniquenesses)`` of the one-factor model
    ``Sigma = loadings loadings' + diag(uniquenesses)`` for the correlation matrix ``r``.

    The model does not depend on the scale of the items, so fitting the correlation matrix gives
    the same standardized solution as fitting the covariance matrix. Uniquenesses are kept above
    ``min_uniqueness``: Heywood cases (improper solutions) end on this bound. The sign is chosen
    so that the loadings sum to a positive number.
    """
    mask = np.ones((len(r), 1), dtype=bool)
    loadings, theta = _ml_factor_model(r, mask, [_one_factor_start(r)], min_uniqueness=min_uniqueness)
    return np.concatenate([loadings[:, 0], theta])


def _bifactor_starts(r, mask, seed=0):
    """Starting values of a bifactor model: from the first principal component and from random loadings."""
    p, factors = mask.shape
    values, vectors = np.linalg.eigh(r)
    general = vectors[:, -1] * np.sqrt(max(values[-1], 1e-3))
    general = np.clip(general * (1 if general.sum() >= 0 else -1), 0.1, 0.8)
    rng = np.random.default_rng(seed)
    starts = []
    for kind in range(4):
        loadings = np.zeros((p, factors))
        loadings[:, 0] = general if kind == 0 else rng.uniform(0.3, 0.7, size=p)
        for f in range(1, factors):
            loadings[mask[:, f], f] = 0.3 if kind == 0 else rng.uniform(0.1, 0.4, size=mask[:, f].sum())
        theta = np.clip(1 - (loadings**2).sum(axis=1), 0.1, None)
        starts.append(np.concatenate([loadings[np.nonzero(mask)], theta]))
    return starts


###############################################################################
# coefficients from the loadings and uniquenesses
###############################################################################


def _omega_from_theta(theta):
    """Omega of the standardized items from the model-implied loadings and uniquenesses."""
    p = len(theta) // 2
    loadings, uniquenesses = theta[:p], theta[p:]
    variances = loadings**2 + uniquenesses  # equal to 1 for a correlation matrix at an exact fit
    total = np.sum(loadings / np.sqrt(variances)) ** 2
    return total / (total + np.sum(uniquenesses / variances))


def _one_factor_coefficients(loadings, theta, standardized):
    """Omega of the standardized items (``standardized``) or of the sum score of the items as they are."""
    if standardized:
        return np.array([_omega_from_theta(np.concatenate([loadings, theta]))])
    total = np.sum(loadings) ** 2
    return np.array([total / (total + np.sum(theta))])


def _bifactor_coefficients(loadings, theta, group_items, standardized):
    """
    ``[omega_h, omega_t, ecv, omega_s of every group, omega_hs of every group]`` of a bifactor model.

    ``loadings`` has the general factor in column 0 and one column per group (``group_items`` lists the item
    positions of the groups). With ``standardized`` the items are standardized first (model-implied
    variances), otherwise the raw sum scores are used.
    """
    if standardized:
        variances = (loadings**2).sum(axis=1) + theta
        loadings, theta = loadings / np.sqrt(variances)[:, None], theta / variances
    general = loadings[:, 0]
    group_sums = np.array([loadings[items, k + 1].sum() for k, items in enumerate(group_items)])
    general_sums = np.array([general[items].sum() for items in group_items])
    unique_sums = np.array([theta[items].sum() for items in group_items])
    general_part = general.sum() ** 2
    total_variance = general_part + np.sum(group_sums**2) + theta.sum()
    subscale_variance = general_sums**2 + group_sums**2 + unique_sums
    ecv = np.sum(general**2) / np.sum(loadings**2)
    return np.concatenate(
        [
            [general_part / total_variance, (general_part + np.sum(group_sums**2)) / total_variance, ecv],
            (general_sums**2 + group_sums**2) / subscale_variance,
            group_sums**2 / subscale_variance,
        ]
    )


###############################################################################
# standard errors
###############################################################################


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


def _duplication_matrix(p):
    """Matrix ``D`` with ``vec(S) = D vech(S)`` for symmetric ``S`` (``vech``: upper triangle by rows)."""
    rows, columns = np.triu_indices(p)
    d = np.zeros((p * p, len(rows)))
    d[rows * p + columns, np.arange(len(rows))] = 1.0
    d[columns * p + rows, np.arange(len(rows))] = 1.0
    return d


def _normal_gamma(sigma):
    """Asymptotic covariance ``(I + K)(Sigma kron Sigma)`` of ``sqrt(n) vec(S)`` under normality."""
    p = len(sigma)
    kron = np.kron(sigma, sigma)
    swapped = kron.reshape(p, p, p * p).transpose(1, 0, 2).reshape(p * p, p * p)  # K (Sigma kron Sigma)
    return kron + swapped


def _sandwich_covariance(loadings, theta, mask, gamma, n, function):
    """
    Covariance matrix of ``function(free loadings, theta)`` for ML estimates of the factor model fitted to a
    covariance matrix whose sample version has the asymptotic covariance ``gamma`` of ``sqrt(n) vec(S)``.

    ``theta_hat - theta = B (s - sigma)`` with ``B = (J'WJ)^-1 J'W``, ``J = d vec(Sigma) / d phi'`` and
    ``W = (Sigma^-1 kron Sigma^-1) / 2`` (Browne, 1984): the covariance matrix of the estimates is
    ``B gamma B' / n`` (with the normal-theory ``gamma`` it reduces to the inverse of the expected
    information). The gradient of ``function`` is taken numerically.
    """
    p = len(theta)
    where = np.argwhere(mask)
    sigma = loadings @ loadings.T + np.diag(theta)
    inverse = np.linalg.inv(sigma)
    q = len(where) + p
    jacobian = np.zeros((p * p, q))
    unit = np.eye(p)
    for j, (i, f) in enumerate(where):
        jacobian[:, j] = (np.outer(unit[i], loadings[:, f]) + np.outer(loadings[:, f], unit[i])).ravel()
    for i in range(p):
        jacobian[:, len(where) + i] = np.outer(unit[i], unit[i]).ravel()
    weight = np.kron(inverse, inverse) / 2
    information = jacobian.T @ weight @ jacobian
    if np.linalg.matrix_rank(information) < q:
        raise np.linalg.LinAlgError("singular information matrix")
    projection = np.linalg.solve(information, jacobian.T @ weight)
    covariance = projection @ gamma @ projection.T / n
    phi = np.concatenate([loadings[tuple(where.T)], theta])
    gradient = np.atleast_2d(approx_fprime(phi, function, centered=True))
    return gradient @ covariance @ gradient.T


def _logit_interval(estimates, variances, conf_level):
    """Limits on the logit scale (they stay within (0, 1)); NaN where the standard error or estimate is unusable."""
    estimates = np.asarray(estimates, dtype=float)
    se = np.sqrt(np.where(variances > 0, variances, np.nan))
    valid = (estimates > 0) & (estimates < 1)
    half = norm.ppf(1 - (1 - conf_level) / 2) * se / np.where(valid, estimates * (1 - estimates), np.nan)
    center = np.log(np.where(valid, estimates, np.nan) / np.where(valid, 1 - estimates, np.nan))
    return expit(center - half), expit(center + half)


###############################################################################
# estimation pipeline shared by the public functions
###############################################################################


def _sigma_hat(x, robust, varphi):
    """Covariance matrix of the items (divisor n): the sample one, or the Huber-type M-estimate."""
    return huber_location_scatter(x, varphi)[1] if robust else np.cov(x, rowvar=False, bias=True)


def _fit_raw(sigma, mask, bifactor):
    """ML fit of the factor model to ``sigma``; the loadings and uniquenesses in the metric of the items."""
    sd = np.sqrt(np.diag(sigma))
    r = sigma / np.outer(sd, sd)
    starts = _bifactor_starts(r, mask) if bifactor else [_one_factor_start(r)]
    loadings, theta = _ml_factor_model(r, mask, starts, "bifactor" if bifactor else "one-factor")
    return loadings * sd[:, None], theta * sd**2, theta


def _estimate(x, mask, function, robust, varphi, bifactor):
    """Coefficients ``function(loadings, theta)`` from the data, and the fitted parameters."""
    sigma = _sigma_hat(x, robust, varphi)
    loadings, theta, theta_standardized = _fit_raw(sigma, mask, bifactor)
    return function(loadings, theta), loadings, theta, theta_standardized


def _analytic_covariance(x, loadings, theta, mask, function, robust, varphi):
    """Sandwich covariance matrix of the coefficients (robust: from the estimating equations of the M-estimator)."""
    n, p = x.shape
    if robust:
        mu, sigma = huber_location_scatter(x, varphi)
        gamma = _duplication_matrix(p) @ huber_asymptotic_covariance(x, mu, sigma, varphi) @ _duplication_matrix(p).T
    else:
        gamma = _normal_gamma(loadings @ loadings.T + np.diag(theta))
    phi_function = _phi_function(mask, function)
    return _sandwich_covariance(loadings, theta, mask, gamma, n, phi_function)


def _phi_function(mask, function):
    where = np.argwhere(mask)
    p, factors = mask.shape

    def evaluate(phi):
        loadings = np.zeros((p, factors))
        loadings[tuple(where.T)] = phi[: len(where)]
        return function(loadings, phi[len(where):])

    return evaluate


def _coefficient_table(x, mask, function, robust, varphi, bifactor, bootstrap_iter, conf_level, seed):
    """Estimates and confidence limits of the coefficients and the loadings: ``(estimate, lower, upper, loadings)``."""
    estimate, loadings, theta, theta_standardized = _estimate(x, mask, function, robust, varphi, bifactor)
    if bootstrap_iter is None:  # sandwich (delta) method, on the logit scale
        lower = upper = np.full(len(estimate), np.nan)
        if np.any(theta_standardized <= _MIN_UNIQUENESS * (1 + 1e-6)):
            warnings.warn(
                "The solution is on the boundary (a uniqueness is at its lower bound, a Heywood "
                "case): the delta-method interval is not available, use bootstrap_iter.",
                UserWarning,
            )
        else:
            try:
                covariance = _analytic_covariance(x, loadings, theta, mask, function, robust, varphi)
                lower, upper = _logit_interval(estimate, np.diag(covariance), conf_level)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "The information matrix is singular: the delta-method interval is not "
                    "available, use bootstrap_iter.",
                    UserWarning,
                )
        return estimate, lower, upper, loadings
    low, high = percentile_bootstrap(
        x,
        lambda sample: _estimate(sample, mask, function, robust, varphi, bifactor)[0],
        bootstrap_iter,
        conf_level,
        seed,
    )
    return estimate, np.atleast_1d(low), np.atleast_1d(high), loadings


###############################################################################
# public functions
###############################################################################


def mcdonald_omega(
    data,
    bootstrap_iter=None,
    conf_level=0.95,
    seed=13,
    robust=False,
    varphi=0.1,
    standardized=True,
):
    """
    McDonald's omega (omega total) with a confidence interval: the reliability of the sum score
    of a scale under a one-factor model with unequal loadings (congeneric items).

    ``omega = (sum(loadings))**2 / ((sum(loadings))**2 + sum(uniquenesses))`` with the loadings and
    uniquenesses of a one-factor model estimated by maximum likelihood (multivariate normal theory).
    For a scale with subscales (several factors) use :func:`mcdonald_omega_bifactor`. Unlike
    Cronbach's alpha, omega does not assume equal loadings: for equal loadings (tau-equivalent items)
    they coincide, otherwise alpha is lower.

    By default omega is that of the *standardized* items, which does not change when items are rescaled
    or shifted. With ``standardized=False`` it is the reliability of the sum score of the items as
    they are (loadings and uniquenesses in the metric of the items), the definition of Zhang & Yuan
    (2016) and of the R packages lavaan / semTools; the two differ when the items have different variances.

    With ``robust=True`` the covariance matrix of the items is replaced by a Huber-type M-estimate that
    gives cases far from the center (in the Mahalanobis distance) weights below one (Yuan & Bentler,
    1998; Zhang & Yuan, 2016), so that a few outlying respondents cannot drive omega. ``varphi`` is the
    share of the cases that is downweighted under normality (0.1: a tenth). The delta-method interval
    then uses the sandwich covariance matrix of the estimator (Yuan & Bentler, 2002; Zhang & Yuan, 2016),
    which is also valid for non-normal data. ``varphi=0`` keeps the sample covariance matrix but gives
    this distribution-free (robust) standard error.

    Two confidence intervals are available:

    * **Delta method** (default, ``bootstrap_iter=None``): omega is a smooth function of the model
      parameters, so its standard error follows from the asymptotic covariance matrix of the ML
      estimates (Raykov, 2002; for the structural equation modelling formulation see Raykov &
      Shrout, 2002). The interval is computed on the logit scale and transformed back, so that it
      stays within (0, 1); Kelley and Pornprasertmanit (2016) found that a logistic
      transformation improved the coverage of standard-error based intervals. Without ``robust`` the
      interval relies on normally distributed items. In their simulations, normal-theory intervals
      did not perform well for non-normal items, and they recommend bootstrap intervals when
      coefficient omega is used: use ``bootstrap_iter`` for items that are clearly not normal
      (skewed, few categories, heavy tails) or for small samples.
    * **Percentile bootstrap** (``bootstrap_iter`` = number of resamples of respondents; with
      ``robust`` the robust estimate is recomputed in every resample).

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
    robust : bool, default False
        Use the Huber-type M-estimate of the covariance matrix and the sandwich standard error.
    varphi : float, default 0.1
        Share of cases downweighted under normality, in [0, 1) (only used with ``robust``; 0 is the
        sample covariance matrix with the distribution-free standard error).
    standardized : bool, default True
        Omega of the standardized items (True) or of the raw sum score (False).

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
        respondents than items, constant items, or invalid ``bootstrap_iter`` / ``conf_level`` /
        ``varphi``.

    References
    ----------
    Raykov, T. (2002). Analytic estimation of standard error and confidence interval for scale
    reliability. Multivariate Behavioral Research, 37(1), 89-103.
    https://doi.org/10.1207/s15327906mbr3701_04

    Raykov, T., & Shrout, P. E. (2002). Reliability of scales with general structure: Point and
    interval estimation using a structural equation modeling approach. Structural Equation Modeling:
    A Multidisciplinary Journal, 9(2), 195-212. https://doi.org/10.1207/S15328007SEM0902_3

    Kelley, K., & Pornprasertmanit, S. (2016). Confidence intervals for population reliability
    coefficients: Evaluation of methods, recommendations, and software for composite measures.
    Psychological Methods, 21(1), 69-92. https://doi.org/10.1037/a0040086

    Flora, D. B. (2020). Your coefficient alpha is probably wrong, but which coefficient omega
    is right? A tutorial on using R to obtain better reliability estimates. Advances in Methods
    and Practices in Psychological Science, 3(4), 484-501.
    https://doi.org/10.1177/2515245920951747

    Yuan, K.-H., & Bentler, P. M. (1998). Structural equation modeling with robust covariances.
    Sociological Methodology, 28(1), 363-396. https://doi.org/10.1111/0081-1750.00052

    Yuan, K.-H., & Bentler, P. M. (2002). On robustness of the normal-theory based asymptotic
    distributions of three reliability coefficient estimates. Psychometrika, 67(2), 251-259.
    https://doi.org/10.1007/BF02294845

    Zhang, Z., & Yuan, K.-H. (2016). Robust coefficients alpha and omega and confidence intervals
    with outlying observations and missing data: Methods and software. Educational and
    Psychological Measurement, 76(3), 387-411. https://doi.org/10.1177/0013164415594658
    """
    check_interval_options(bootstrap_iter, conf_level)
    _check_varphi(varphi)
    x = item_matrix(data, 3, ": a one-factor model with fewer items is not identified")
    n, p = x.shape
    generic = robust or not standardized
    if generic:
        mask = np.ones((p, 1), dtype=bool)

        def function(loadings, theta):
            return _one_factor_coefficients(loadings[:, 0], theta, standardized)

        estimate, low, high, loadings = _coefficient_table(
            x, mask, function, robust, varphi, False, bootstrap_iter, conf_level, seed
        )
        omega, loadings = float(estimate[0]), loadings[:, 0]
    else:
        theta = _one_factor_ml(np.corrcoef(x, rowvar=False))
        omega = float(_omega_from_theta(theta))
        loadings = theta[:p]
    if (loadings > 0).any() and (loadings < 0).any():
        warnings.warn(
            "The factor loadings have different signs. Reverse-keyed items must be recoded "
            "before omega is computed, otherwise omega is too low.",
            UserWarning,
        )

    if generic:
        return omega, float(low[0]), float(high[0])

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

    low, high = percentile_bootstrap(
        x,
        lambda sample: _omega_from_theta(_one_factor_ml(np.corrcoef(sample, rowvar=False))),
        bootstrap_iter,
        conf_level,
        seed,
    )
    return omega, low, high


def _check_varphi(varphi):
    if not 0 <= varphi < 1:
        raise ValueError("varphi must be in [0, 1): the share of cases that is downweighted.")


def _parse_groups(data, groups, p):
    """Names and item positions of the groups: ``groups`` is a dict or a list of lists of item names (positions)."""
    names = list(data.columns) if hasattr(data, "columns") else list(range(p))
    position = {name: i for i, name in enumerate(names)}
    items = groups.items() if isinstance(groups, dict) else ((str(k + 1), v) for k, v in enumerate(groups))
    labels, positions = [], []
    for label, members in items:
        members = list(members)
        unknown = [m for m in members if m not in position]
        if unknown:
            raise ValueError(f"Group '{label}' contains items that are not in the data: {unknown}.")
        positions.append([position[m] for m in members])
        labels.append(str(label))
    flat = [i for group in positions for i in group]
    if len(set(flat)) != len(flat):
        raise ValueError("An item belongs to more than one group.")
    if len(positions) < 2:
        raise ValueError("At least two groups of items are needed (for a single factor use mcdonald_omega).")
    small = [label for label, group in zip(labels, positions) if len(group) < 3]
    if small:
        raise ValueError(f"Every group needs at least 3 items for the bifactor model to be identified (groups {small}).")
    return labels, positions


def mcdonald_omega_bifactor(
    data,
    groups,
    bootstrap_iter=None,
    conf_level=0.95,
    seed=13,
    robust=False,
    varphi=0.1,
    standardized=True,
):
    """
    Omega hierarchical, omega total and subscale omegas of a scale with subscales, from a bifactor model.

    A confirmatory bifactor model is fitted by maximum likelihood: every item loads on a *general factor*
    and on the *group factor* of its subscale; all factors are uncorrelated. With the loadings
    ``g`` on the general factor, ``s_k`` on the group factor ``k`` and the uniquenesses ``u`` of the items
    (standardized, or in the metric of the items with ``standardized=False``), and the total variance
    ``V = (sum g)^2 + sum_k (sum_{i in k} s_ki)^2 + sum u`` of the sum score:

    * ``omega_h`` = ``(sum g)^2 / V``: the share of the variance of the total score that is due to the
      general factor (omega hierarchical; McDonald, 1999; Zinbarg et al., 2005): the reliability of the
      total score as a measure of the *general* construct. Compare it with ``omega_t``: the closer they
      are, the less the group factors contribute to the total score.
    * ``omega_t`` = ``((sum g)^2 + sum_k (sum s_k)^2) / V``: the reliability of the total score, general and
      group factors together (omega total of a multidimensional scale).
    * ``ecv``: the explained common variance, ``sum g^2 / (sum g^2 + sum s^2)``: the share of the common
      variance of the items that is due to the general factor (Rodriguez et al., 2016).
    * ``omega_s[k]`` = ``((sum_k g)^2 + (sum_k s)^2) / V_k`` and ``omega_hs[k]`` = ``(sum_k s)^2 / V_k`` for the
      subscale ``k`` with the variance ``V_k`` of its sum score: the reliability of the subscale score, and
      its reliability *after* removing the general factor, i.e. the reliable variance that is specific
      to the subscale (Reise, 2012; Rodriguez et al., 2016). A low ``omega_hs`` means that the subscale
      adds little beyond the total score.

    Items that are not listed in ``groups`` load only on the general factor.

    Confidence intervals: with ``bootstrap_iter=None`` the sandwich (delta-method) interval of the ML
    estimates on the logit scale (normal theory; with ``robust`` the covariance matrix of the
    M-estimator is used, as in :func:`mcdonald_omega`), otherwise the percentile bootstrap. Bifactor
    models often give improper solutions (a uniqueness at zero, a group factor without loadings) in
    small samples or when the subscales are not distinct: the analytic interval is then not available
    (``nan`` with a warning) and the bootstrap is the alternative. The estimates assume that the
    bifactor model fits; omega_h and the subscale coefficients are sensitive to model misfit.

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        Item scores, one row per respondent and one column per item (more respondents than items).
        Rows with missing values are dropped (listwise).
    groups : dict or list of lists
        The items of the subscales: ``{"name": ["item1", "item2", ...], ...}`` or a list of lists of
        column names (positions for arrays), at least two groups of at least three items each.
        An item may belong to one group only. Items must be keyed in the same direction.
    bootstrap_iter, conf_level, seed, robust, varphi, standardized
        As in :func:`mcdonald_omega`.

    Returns
    -------
    pandas.DataFrame
        One row per coefficient (``omega_h``, ``omega_t``, ``ecv``, then ``omega_s[name]`` and
        ``omega_hs[name]`` of every group) with the columns ``estimate``, ``lower`` and ``upper``.

    Raises
    ------
    ValueError
        For invalid ``groups`` (unknown or repeated items, fewer than two groups or fewer than three items
        in a group) or data and options as in :func:`mcdonald_omega`.

    References
    ----------
    McDonald, R. P. (1999). *Test Theory: A Unified Treatment*. Lawrence Erlbaum.
    https://doi.org/10.4324/9781410601087

    Zinbarg, R. E., Revelle, W., Yovel, I., & Li, W. (2005). Cronbach's alpha, Revelle's beta, and
    McDonald's omega h: Their relations with each other and two alternative conceptualizations of
    reliability. Psychometrika, 70(1), 123-133. https://doi.org/10.1007/s11336-003-0974-7

    Revelle, W., & Zinbarg, R. E. (2009). Coefficients alpha, beta, omega, and the glb: Comments on
    Sijtsma. Psychometrika, 74(1), 145-154. https://doi.org/10.1007/s11336-008-9102-z

    Reise, S. P. (2012). The rediscovery of bifactor measurement models. Multivariate Behavioral
    Research, 47(5), 667-696. https://doi.org/10.1080/00273171.2012.715555

    Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and
    interpreting statistical indices. Psychological Methods, 21(2), 137-150.
    https://doi.org/10.1037/met0000045

    Browne, M. W. (1984). Asymptotically distribution-free methods for the analysis of covariance
    structures. British Journal of Mathematical and Statistical Psychology, 37(1), 62-83.
    https://doi.org/10.1111/j.2044-8317.1984.tb00789.x

    Zhang, Z., & Yuan, K.-H. (2016). Robust coefficients alpha and omega and confidence intervals
    with outlying observations and missing data: Methods and software. Educational and
    Psychological Measurement, 76(3), 387-411. https://doi.org/10.1177/0013164415594658
    """
    check_interval_options(bootstrap_iter, conf_level)
    _check_varphi(varphi)
    x = item_matrix(data, 6, ": a bifactor model needs at least two groups of three items")
    p = x.shape[1]
    labels, positions = _parse_groups(data, groups, p)
    mask = np.zeros((p, 1 + len(positions)), dtype=bool)
    mask[:, 0] = True
    for k, items in enumerate(positions):
        mask[items, k + 1] = True

    def function(loadings, theta):
        return _bifactor_coefficients(loadings, theta, positions, standardized)

    names = ["omega_h", "omega_t", "ecv"] + [f"omega_s[{g}]" for g in labels] + [f"omega_hs[{g}]" for g in labels]
    estimate, low, high, loadings = _coefficient_table(
        x, mask, function, robust, varphi, True, bootstrap_iter, conf_level, seed
    )
    if (loadings[:, 0] > 0).any() and (loadings[:, 0] < 0).any():
        warnings.warn(
            "The loadings on the general factor have different signs. Reverse-keyed items must be recoded "
            "before omega is computed.",
            UserWarning,
        )
    return pd.DataFrame({"estimate": estimate, "lower": low, "upper": high}, index=names)
