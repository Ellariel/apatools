"""Direct, indirect and total effects of the spatial lag model and its Durbin extension (private core of
:func:`apatools.slm.slm_effects` and :func:`apatools.plm.plm_effects`).

In the spatial lag model ``y = ρWy + Xβ + WXθ + ε`` (``θ = 0`` without the spatially lagged regressors) a change of
regressor ``k`` in one location spills over to the others: the effect matrix is
``S_k = (I - ρW)⁻¹ (β_k I + θ_k W)``. LeSage & Pace (2009) summarize it by the *direct effect* (the average of the
diagonal, the effect of a change in a location on that location itself), the *total effect* (the average of the row
sums) and the *indirect effect* (their difference, the spillover).

Both averages are series in the traces of the powers of ``W``: with ``tr_k = tr(Wᵏ)`` and ``s_k = 1'Wᵏ1``,

    direct = β_k d(ρ) + θ_k e(ρ),   total = β_k t(ρ) + θ_k u(ρ),
    d = Σ ρᵏ tr_k / n,   e = Σ ρᵏ tr_{k+1} / n,   t = Σ ρᵏ s_k / n,   u = Σ ρᵏ s_{k+1} / n,

so neither the inverse of ``I - ρW`` nor a matrix product with it is needed, and the traces are computed once.
Two ways to the standard errors are offered (both use the covariance matrix of the estimates):

* ``"analytic"``: the delta method, with the gradient of the three effects with respect to ``(β_k, θ_k, ρ)``, which
  follows from differentiating the series; the confidence interval is symmetric (estimate ± z · se).
* ``"simulation"``: LeSage & Pace's simulation: ``(β, θ, ρ)`` are drawn from their estimated joint normal
  distribution (draws with ``|ρ|`` at or above ``1/`` the spectral radius of ``W`` are replaced), the effects are
  computed for every draw, the standard error is the standard deviation of the draws and the confidence interval is
  made of their percentiles.

References
----------
LeSage, J., & Pace, R. K. (2009). *Introduction to spatial econometrics*. CRC Press.
Elhorst, J. P. (2014). *Spatial econometrics: From cross-sectional data to spatial panels*. Springer.
Bivand, R. S., Hauke, J., & Kossowski, T. (2013). Computing the Jacobian in Gaussian spatial autoregressive models:
An illustrated comparison of available methods. *Geographical Analysis, 45*(2), 150-179.
"""

import warnings

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.stats

from ._common import INTERCEPT_NAMES

EFFECTS = ("Direct", "Indirect", "Total")
INFERENCE = ("simulation", "analytic")
TOLERANCE = 1e-10  # of the truncated series
MAX_ORDER = 5000


########################
#  traces of the weights  #
########################


class WeightsTraces:
    """
    ``tr(Wᵏ)`` and ``1'Wᵏ1`` of the weights ``w`` (libpysal ``W``) and their spectral radius, computed once.

    ``traces="exact"`` takes the traces from the eigenvalues of the dense matrix (cost ``n³``), ``"mc"`` estimates
    them with ``n_probes`` random ±1 vectors (Hutchinson's estimator, cost ``order · nnz · n_probes``), ``"auto"`` the
    exact ones for up to ``n_exact`` units.
    """

    def __init__(self, w, traces="auto", n_probes=50, seed=13, n_exact=2000):
        if traces not in ("auto", "exact", "mc"):
            raise ValueError("traces must be 'auto', 'exact' or 'mc'.")
        self.matrix = sps.csr_matrix(w.sparse, dtype=float)
        self.n = self.matrix.shape[0]
        self.exact = traces == "exact" or (traces == "auto" and self.n <= n_exact)
        self.n_probes, self.seed = n_probes, seed
        if self.exact:
            self.eigenvalues = np.linalg.eigvals(self.matrix.toarray())
            self.radius = float(np.max(np.abs(self.eigenvalues)))
        else:
            self.radius = _spectral_radius(self.matrix)

    def get(self, order):
        """``(tr, s)``: arrays of length ``order + 2`` (the powers ``0 .. order + 1``)."""
        steps = order + 2
        tr, s = np.empty(steps), np.empty(steps)
        vector = np.ones(self.n)
        for k in range(steps):  # 1'W^k 1 by repeated products with a vector
            s[k] = vector.sum()
            vector = self.matrix @ vector
        if self.exact:
            power = np.ones_like(self.eigenvalues)
            for k in range(steps):
                tr[k] = float(np.real(power.sum()))
                power = power * self.eigenvalues
        else:
            rng = np.random.default_rng(self.seed)
            probes = rng.choice([-1.0, 1.0], size=(self.n, self.n_probes))
            current = probes.copy()
            for k in range(steps):
                tr[k] = float(np.mean(np.sum(probes * current, axis=0)))
                current = self.matrix @ current
        tr[0] = self.n
        return tr, s


def _spectral_radius(matrix):
    try:
        return float(
            np.abs(spla.eigs(matrix, k=1, which="LM", return_eigenvectors=False)[0])
        )
    except (spla.ArpackNoConvergence, spla.ArpackError):  # pragma: no cover
        return 1.0


def series_order(rho_bound):
    """Number of terms after which the series of the effects is accurate (relative to ``TOLERANCE``)."""
    bound = min(
        max(abs(rho_bound), 1e-3), 0.9999
    )  # ratio of the series: |rho| times the spectral radius of W
    order = int(np.ceil(np.log(TOLERANCE * (1 - bound)) / np.log(bound)))
    if order > MAX_ORDER:
        warnings.warn(
            f"|rho| = {abs(rho_bound):.4f} is so close to the bound that the series of the effects is cut at "
            f"{MAX_ORDER} terms; the effects are approximate.",
            UserWarning,
            stacklevel=4,
        )
    return int(min(max(order, 10), MAX_ORDER))


#################################
#  effects for given parameters #
#################################


def _multipliers(rho, tr, s, order):
    """``d, e, t, u`` and their derivatives with respect to ``rho`` for every value of ``rho`` (arrays)."""
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    k = np.arange(order + 1)
    powers = rho[:, None] ** k[None, :]
    slope = np.where(
        k[None, :] > 0, k[None, :] * rho[:, None] ** np.maximum(k - 1, 0)[None, :], 0.0
    )
    n = tr[0]
    out = {}
    for name, series in (
        ("d", tr[: order + 1]),
        ("e", tr[1 : order + 2]),
        ("t", s[: order + 1]),
        ("u", s[1 : order + 2]),
    ):
        out[name] = powers @ series / n
        out["d" + name] = slope @ series / n  # derivative with respect to rho
    return out


def point_effects(beta, theta, rho, tr, s, order):
    """Direct, indirect and total effects ``(3, K)`` for the coefficients ``beta`` and ``theta`` (``K`` regressors)."""
    m = _multipliers(rho, tr, s, order)
    direct = beta * m["d"][0] + theta * m["e"][0]
    total = beta * m["t"][0] + theta * m["u"][0]
    return np.vstack([direct, total - direct, total])


def effects_gradients(beta, theta, rho, tr, s, order):
    """
    Gradient of every effect of every regressor with respect to ``(β_k, θ_k, ρ)``: array ``(3, K, 3)`` (the
    effects in the order direct, indirect, total).
    """
    m = _multipliers(rho, tr, s, order)
    out = np.zeros((3, len(beta), 3))
    out[0, :, 0], out[0, :, 1], out[0, :, 2] = (
        m["d"][0],
        m["e"][0],
        beta * m["dd"][0] + theta * m["de"][0],
    )
    out[2, :, 0], out[2, :, 1], out[2, :, 2] = (
        m["t"][0],
        m["u"][0],
        beta * m["dt"][0] + theta * m["du"][0],
    )
    out[1] = out[2] - out[0]
    return out


def analytic_se(beta, theta, rho, cov, positions, tr, s, order):
    """
    Delta-method standard errors ``(3, K)``. ``cov`` is the covariance matrix of all estimates, ``positions`` the
    rows of ``(beta_idx, theta_idx or None, rho_idx)`` in it.
    """
    beta_idx, theta_idx, rho_idx = positions
    gradients = effects_gradients(beta, theta, rho, tr, s, order)
    se = np.empty((3, len(beta)))
    for j in range(len(beta)):
        index = [beta_idx[j], theta_idx[j] if theta_idx is not None else None, rho_idx]
        keep = [i for i in range(3) if index[i] is not None]
        sub = cov[np.ix_([index[i] for i in keep], [index[i] for i in keep])]
        for e in range(3):
            g = gradients[e, j, keep]
            se[e, j] = np.sqrt(max(float(g @ sub @ g), 0.0))
    return se


def draw_parameters(estimates, cov, positions, rho_bound, n_draws, seed):
    """
    ``n_draws`` draws of ``(beta, theta, rho)`` from their joint normal distribution (LeSage & Pace's simulation);
    draws with ``|rho| >= rho_bound`` are drawn again. Returns an array ``(n_draws, 2K + 1)`` or ``(n_draws, K + 1)``.
    """
    beta_idx, theta_idx, rho_idx = positions
    rng = np.random.default_rng(seed)
    used = (
        list(beta_idx) + (list(theta_idx) if theta_idx is not None else []) + [rho_idx]
    )
    mean, sub = estimates[used], cov[np.ix_(used, used)]
    draws = rng.multivariate_normal(mean, sub, size=n_draws, method="eigh")
    for _ in range(50):
        bad = np.abs(draws[:, -1]) >= rho_bound
        if not bad.any():
            break
        draws[bad] = rng.multivariate_normal(
            mean, sub, size=int(bad.sum()), method="eigh"
        )
    else:
        warnings.warn(
            "Some draws of rho stayed outside the admissible range; they were kept.",
            UserWarning,
            stacklevel=4,
        )
    return draws


def simulated_effects(draws, has_theta, tr, s, order):
    """Effects of every draw: array ``(3, n_draws, K)`` (direct, indirect, total)."""
    k = (draws.shape[1] - 1) // (2 if has_theta else 1)
    m = _multipliers(draws[:, -1], tr, s, order)
    beta = draws[:, :k]
    theta = draws[:, k : 2 * k] if has_theta else np.zeros_like(beta)
    direct = beta * m["d"][:, None] + theta * m["e"][:, None]
    total = beta * m["t"][:, None] + theta * m["u"][:, None]
    return np.stack([direct, total - direct, total])


########################
#  parameters of a fit  #
########################


def split_parameters(names):
    """
    Positions of the coefficients, the spatially lagged coefficients and ``rho`` in the list of parameter names
    (``"rho"`` is the spatial lag parameter, ``"W_<x>"`` the lag of regressor ``x``, the constant and other
    parameters such as ``phi`` are ignored). Returns ``(variables, (beta_idx, theta_idx or None, rho_idx))``.
    """
    names = [str(n) for n in names]
    if "rho" not in names:
        raise ValueError("The model has no spatial lag parameter 'rho'.")
    lagged = {
        n[2:]: i for i, n in enumerate(names) if n.startswith("W_") and n[2:] in names
    }
    variables = [
        n
        for n in names
        if n not in INTERCEPT_NAMES
        and n not in ("rho", "phi", "lambda")
        and not (n.startswith("W_") and n[2:] in names)
    ]
    beta_idx = [names.index(v) for v in variables]
    theta_idx = (
        [lagged[v] for v in variables]
        if lagged and all(v in lagged for v in variables)
        else None
    )
    if lagged and theta_idx is None:
        raise ValueError(
            "Every regressor needs its spatial lag W_<name> in a Durbin model (or none)."
        )
    return variables, (
        np.array(beta_idx),
        None if theta_idx is None else np.array(theta_idx),
        names.index("rho"),
    )


def lag_effects(
    names,
    coefficients,
    cov,
    w,
    inference="simulation",
    n_draws=1000,
    seed=13,
    conf_level=0.95,
    traces="auto",
    n_probes=50,
):
    """
    Direct, indirect and total effects of a spatial lag (or Durbin) model with the parameters ``names`` /
    ``coefficients`` and their covariance matrix ``cov``, for the weights ``w``.

    Returns a DataFrame with the columns ``variable``, ``effect``, ``estimate``, ``se``, ``z``, ``p-value``, ``cil``,
    ``cir`` (the confidence interval of level ``conf_level``).
    """
    if inference not in INFERENCE:
        raise ValueError(f"inference must be one of {INFERENCE}.")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1.")
    coefficients, cov = (
        np.asarray(coefficients, dtype=float),
        np.asarray(cov, dtype=float),
    )
    variables, positions = split_parameters(names)
    beta_idx, theta_idx, rho_idx = positions
    beta = coefficients[beta_idx]
    theta = coefficients[theta_idx] if theta_idx is not None else np.zeros_like(beta)
    rho = float(coefficients[rho_idx])
    traces_of_w = WeightsTraces(w, traces=traces, n_probes=n_probes, seed=seed)
    bound = (
        1.0 / traces_of_w.radius
    )  # |rho| must stay below it for (I - rho W) to be invertible with a series
    if abs(rho) >= bound:
        warnings.warn(
            "|rho| is outside the admissible range of the weights; the effects are not defined.",
            UserWarning,
            stacklevel=3,
        )
        nan = np.full((3, len(variables)), np.nan)
        return _assemble(variables, nan, nan, nan, nan, nan)
    if inference == "simulation":
        draws = draw_parameters(coefficients, cov, positions, bound, n_draws, seed)
        order = series_order(
            max(abs(rho), float(np.max(np.abs(draws[:, -1])))) * traces_of_w.radius
        )
    else:
        order = series_order(abs(rho) * traces_of_w.radius)
    tr, s = traces_of_w.get(order)
    estimate = point_effects(beta, theta, rho, tr, s, order)
    if inference == "analytic":
        se = analytic_se(beta, theta, rho, cov, positions, tr, s, order)
        z_crit = scipy.stats.norm.ppf(1 - (1 - conf_level) / 2)
        lower, upper = estimate - z_crit * se, estimate + z_crit * se
    else:
        sample = simulated_effects(draws, theta_idx is not None, tr, s, order)
        alpha = (1 - conf_level) / 2
        se, lower, upper = (
            sample.std(axis=1, ddof=1),
            np.quantile(sample, alpha, axis=1),
            np.quantile(sample, 1 - alpha, axis=1),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = estimate / se
    p = 2 * scipy.stats.norm.sf(np.abs(z))
    return _assemble(variables, estimate, se, z, p, lower, upper)


def _assemble(variables, estimate, se, z, p, lower, upper=None):
    """Tidy table from the arrays ``(3, K)`` of the effects, their standard errors, z values, p-values and limits."""
    import pandas as pd

    if upper is None:  # the NaN table: all arrays are the same
        upper = lower
    rows = [
        {
            "variable": variable,
            "effect": EFFECTS[e],
            "estimate": estimate[e, j],
            "se": se[e, j],
            "z": z[e, j],
            "p-value": p[e, j],
            "cil": lower[e, j],
            "cir": upper[e, j],
        }
        for j, variable in enumerate(variables)
        for e in range(3)
    ]
    return pd.DataFrame(rows)


def effects_table(
    entries,
    w,
    inference="simulation",
    n_draws=1000,
    seed=13,
    conf_level=0.95,
    traces="auto",
    n_probes=50,
    format_pval=False,
    add_stars=False,
    decimal=None,
):
    """
    Effects of several models in one tidy table: ``entries`` is a list of ``(model label, parameter names,
    coefficients, covariance matrix)``; the options are those of :func:`apatools.slm.slm_effects`.
    """
    import pandas as pd

    from ._common import format_significance, round_columns

    columns = [
        "model",
        "variable",
        "effect",
        "estimate",
        "se",
        "z",
        "p-value",
        "cil",
        "cir",
    ]
    frames = []
    for label, names, coefficients, cov in entries:
        table = lag_effects(
            names,
            coefficients,
            cov,
            w,
            inference,
            n_draws,
            seed,
            conf_level,
            traces,
            n_probes,
        )
        table.insert(0, "model", label)
        frames.append(table)
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame({c: [] for c in columns})
    )
    for column in ("estimate", "se", "z", "p-value", "cil", "cir"):
        result[column] = result[column].astype(
            float
        )  # also for an empty table (object columns)
    round_columns(result, decimal, columns=("estimate", "se", "cil", "cir"))
    format_significance(result, add_stars, format_pval)
    return result
