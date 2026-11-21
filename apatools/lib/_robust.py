"""Huber-type M-estimation of the mean vector and covariance matrix, and its asymptotic covariance.

Cases far from the center in the Mahalanobis distance get weights below one, so that a few outlying
cases do not drive the covariance matrix (and the reliability coefficients computed from it).
Private: used by the robust option of the omega coefficients in :mod:`apatools.lib.mcdonald`.

References
----------
Maronna, R. A. (1976). Robust M-estimators of multivariate location and scatter. *The Annals of
Statistics, 4*(1), 51-67. https://doi.org/10.1214/aos/1176343347

Yuan, K.-H., & Bentler, P. M. (1998). Structural equation modeling with robust covariances.
*Sociological Methodology, 28*(1), 363-396. https://doi.org/10.1111/0081-1750.00052

Yuan, K.-H., Bentler, P. M., & Chan, W. (2004). Structural equation modeling with heavy tailed
distributions. *Psychometrika, 69*(3), 421-436. https://doi.org/10.1007/BF02295644

Zhang, Z., & Yuan, K.-H. (2016). Robust coefficients alpha and omega and confidence intervals with
outlying observations and missing data: Methods and software. *Educational and Psychological
Measurement, 76*(3), 387-411. https://doi.org/10.1177/0013164415594658
"""

import numpy as np
from scipy.stats import chi2
from statsmodels.robust.norms import HuberT


def _constants(p, varphi):
    """Cutoff ``rho`` of the Mahalanobis distance and the consistency constant ``kappa`` (normal data)."""
    if varphi == 0:
        return np.inf, 1.0  # no downweighting: the sample mean and covariance matrix
    rho2 = chi2.ppf(1 - varphi, p)
    kappa = (p * chi2.cdf(rho2, p + 2) + rho2 * varphi) / p
    return float(np.sqrt(rho2)), float(kappa)


def _weights(d2, rho, kappa):
    """Huber weights of the mean (``w1 = min(1, rho / d)``, statsmodels' ``HuberT``) and of the covariance (``w1^2 / kappa``)."""
    w1 = np.ones_like(d2) if np.isinf(rho) else HuberT(t=rho).weights(np.sqrt(d2))  # inf: no downweighting
    return w1, w1**2 / kappa


def huber_location_scatter(x, varphi=0.1, tolerance=1e-10, max_iterations=2000):
    """
    Huber-type M-estimates of the mean vector and the covariance matrix of the rows of ``x``.

    The estimates solve ``mu = sum(w1 x) / sum(w1)`` and ``Sigma = mean(w2 (x - mu)(x - mu)')`` with the
    weights of :func:`_weights`; they are computed by iterative reweighting from the sample mean and
    covariance matrix. ``varphi`` is the share of cases that are downweighted under normality
    (``varphi = 0``: the sample mean and the covariance matrix with divisor ``n``).

    Returns
    -------
    mu, sigma : numpy.ndarray
        Mean vector and covariance matrix.
    """
    x = np.asarray(x, dtype=float)
    p = x.shape[1]
    rho, kappa = _constants(p, varphi)
    mu, sigma = x.mean(axis=0), np.cov(x, rowvar=False, bias=True)
    for _ in range(max_iterations):
        centered = x - mu
        d2 = np.einsum("ij,ij->i", centered @ np.linalg.inv(sigma), centered)
        w1, w2 = _weights(d2, rho, kappa)
        new_mu = (w1[:, None] * x).sum(axis=0) / w1.sum()
        centered = x - new_mu
        new_sigma = (w2[:, None, None] * (centered[:, :, None] * centered[:, None, :])).mean(axis=0)
        change = max(np.abs(new_mu - mu).max(), np.abs(new_sigma - sigma).max())
        mu, sigma = new_mu, new_sigma
        if change < tolerance:
            break
    return mu, sigma


def _estimating_functions(x, xi, p, rho, kappa, rows, columns):
    """Rows ``psi_i(xi)`` of the estimating functions of ``xi = (mu, vech Sigma)`` (one row per case)."""
    mu = xi[:p]
    sigma = np.zeros((p, p))
    sigma[rows, columns] = xi[p:]
    sigma[columns, rows] = xi[p:]
    centered = x - mu
    d2 = np.einsum("ij,ij->i", centered @ np.linalg.inv(sigma), centered)
    w1, w2 = _weights(d2, rho, kappa)
    products = centered[:, rows] * centered[:, columns]
    return np.column_stack([w1[:, None] * centered, w2[:, None] * products - sigma[rows, columns]])


def huber_asymptotic_covariance(x, mu, sigma, varphi=0.1):
    """
    Asymptotic covariance matrix of ``sqrt(n) * vech(Sigma_hat)`` of the Huber-type M-estimator.

    ``vech`` stacks the upper triangle row by row (``numpy.triu_indices``). The estimator solves
    estimating equations ``mean(psi_i(xi)) = 0`` for ``xi = (mu, vech Sigma)``, so the sandwich
    ``A^-1 B A^-T`` applies, with ``A = -mean(d psi_i / d xi')`` (numerical derivative) and
    ``B = mean(psi_i psi_i')`` (Yuan & Bentler, 1998; Zhang & Yuan, 2016). For ``varphi = 0`` this is the
    distribution-free (fourth moment) covariance matrix of the sample covariance matrix.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    rho, kappa = _constants(p, varphi)
    rows, columns = np.triu_indices(p)
    xi = np.concatenate([mu, sigma[rows, columns]])
    psi = _estimating_functions(x, xi, p, rho, kappa, rows, columns)
    derivative = np.empty((len(xi), len(xi)))
    for k in range(len(xi)):
        step = 1e-6 * max(1.0, abs(xi[k]))
        up, down = xi.copy(), xi.copy()
        up[k] += step
        down[k] -= step
        mean_up = _estimating_functions(x, up, p, rho, kappa, rows, columns).mean(axis=0)
        mean_down = _estimating_functions(x, down, p, rho, kappa, rows, columns).mean(axis=0)
        derivative[:, k] = (mean_up - mean_down) / (2 * step)
    a_inverse = np.linalg.inv(-derivative)
    b = psi.T @ psi / n
    return (a_inverse @ b @ a_inverse.T)[p:, p:]
