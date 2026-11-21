import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t

from ._deps import missing_dependency

try:
    import pandas as pd
    from statsmodels.stats.inter_rater import aggregate_raters
except ImportError as e:  # pragma: no cover
    missing_dependency("lib", "lib", e)



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
    seed : int, optional
        Seed of the random generator used for the resampling (a local generator: numpy's global
        random state is not touched). ``None`` gives different resamples on every call.

    Returns
    -------
    numpy.ndarray
        ``[lower, median, upper]`` percentiles of the bootstrap distribution.
    """
    rng = np.random.default_rng(seed)  # local generator: numpy's global random state is untouched
    data = np.asanyarray(args)
    idx = np.arange(0, data.shape[1], 1)
    sample = [
        func(*np.take(data, rng.choice(idx, size=data.shape[1], replace=True), 1))
        for _ in range(n_rep)
    ]
    lp, m, rp = (1 - alpha) / 2, 0.5, 1 - (1 - alpha) / 2
    return np.nanpercentile(sample, [lp * 100, m * 100, rp * 100])


def gini(values, drop_zeros=False, tolerance=10**-6):
    """Calculate the Gini coefficient"""
    # https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    x = (
        np.array(list(values)).astype("float64").flatten()
    )  # all values are treated equally, arrays must be 1d
    x_min = np.amin(x)
    if x_min < tolerance:
        x += np.abs(x_min)  # values cannot be negative
    if drop_zeros:
        x = x[x >= tolerance]
    else:
        x += tolerance  # values cannot be 0
    n, x = x.shape[0], np.sort(x)  # values must be sorted
    idx = np.arange(1, n + 1)  # index per array element
    return np.sum((2 * idx - n - 1) * x) / (n * np.sum(x))  # Gini coefficient


def gini_from_percentiles(values, percentiles, values_are_shares=False):
    """
    Calculate the Gini coefficient from percentile-based cumulative data.
    Parameters:
    - percentiles: array of population percentiles (e.g., [10, 20, ..., 100])
    - values: array of corresponding cumulative shares/proportion (e.g., [1.5, 5.0, ..., 100])
        or cumulative values themselves (e.g., [100, 150, 350, ..., 1200])
    """

    x = np.array(list(percentiles)).astype("float64").flatten() / 100
    y = np.array(list(values)).astype("float64").flatten()
    if values_are_shares:
        y = y / 100
    else:
        y = y / np.max(y)
    if (
        x[0] > 0
    ):  # the Lorenz curve always starts in the origin (0% of the population has 0% of the total)
        x, y = np.insert(x, 0, 0.0), np.insert(y, 0, 0.0)
    # Trapezoidal approximation of area under Lorenz curve
    b = (y[1:] + y[:-1]) * np.diff(x) / 2

    return 1 - 2 * b.sum()  # Gini coefficient is 1 - 2 * area under Lorenz curve


def log_transform(values):
    """Pseudo log-transformation, keeping sign"""

    x = np.array(list(values)).astype("float64").flatten()
    return np.sign(x) * np.log10(1 + np.abs(x))


def z_transform(values, return_params=False, ignore_errors=True):
    """z-transformation or standardization"""

    x = np.array(list(values)).astype("float64").flatten()
    m, s = np.nanmean(x), np.nanstd(x)
    if not s and not ignore_errors:
        raise ValueError(
            "Standard deviation is zero or undefined. Z-transformation is not defined."
        )

    z = (x - m) / s

    if return_params:
        return z, m, s
    return z


def fleiss_kappa(df, return_per_item_agreement=False, method="two-tailed"):
    """
    Compute Fleiss' kappa, z-score, and p-value, per-item agreement
    Null hypothesis: Agreement is due to chance (kappa = 0)
    0.01-0.02   Slight agreement
    0.21-0.40   Fair Agreement
    0.41-0.60   Moderate Agreement
    0.61-0.80   Substantial Agreement
    0.81-1.00   Almost Perfect Agreement
    Negative (kappa < 0): Agreement less than that expected by chance
    """
    # https://en.wikipedia.org/wiki/Fleiss%27s_kappa
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.aggregate_raters.html

    df = df.astype(str).where(
        pd.notnull(df), "None"
    )  # be careful about NaNs, they are transformed to "None" category
    n_items, n_raters = df.shape
    m = aggregate_raters(df, n_cat=None)[0]
    P_i = (np.sum(m**2, axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )  # Per-item agreement
    P_j = np.sum(m, axis=0) / (n_items * n_raters)  # Category proportions
    P_e = np.sum(P_j**2)  # Expected agreement
    kappa = (
        (np.mean(P_i) - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan
    )  # Fleiss' kappa

    # variance of kappa (approximation)
    term1 = np.sum(P_j**2 * (1 - P_j) ** 2)
    term2 = (1 - P_e) * (np.sum(P_j**3) - P_e * np.sum(P_j**2))
    var_kappa = (term1 - term2) / (n_items * n_raters * (n_raters - 1) * (1 - P_e) ** 2)

    # z-score and p-value
    z, p = np.nan, np.nan
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p = norm.sf(np.abs(z))  # one-tailed, agreement > chance
        p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        return (
            kappa,
            z,
            p,
            pd.Series(P_i, index=df.index, name="per_item_agreement"),
        )

    return kappa, z, p


_KRIPPENDORFF_LEVELS = ("nominal", "ordinal", "interval", "ratio")


def _krippendorff_codes(data, measurement):
    """
    Ratings as integer codes (units x raters, -1 = missing) and the sorted category values.

    Nominal ratings may be of any type; ordinal, interval and ratio ratings must be numeric,
    because their differences are computed from the values.
    """
    if measurement not in _KRIPPENDORFF_LEVELS:
        raise ValueError(
            f"measurement must be one of {_KRIPPENDORFF_LEVELS}, got {measurement!r}."
        )
    frame = pd.DataFrame(data)
    if frame.ndim != 2 or frame.shape[1] < 2:
        raise ValueError(
            "data must have one row per unit and one column per rater (at least 2 raters)."
        )
    present = frame.notna().to_numpy()
    flat = frame.to_numpy(dtype=object)[present]
    if measurement == "nominal":
        inverse, values = pd.factorize(flat, sort=False)
        values = np.asarray(values, dtype=object)
    else:
        try:
            numbers = flat.astype(float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"The {measurement} level of measurement needs numeric ratings: {e}"
            ) from e
        values, inverse = np.unique(numbers, return_inverse=True)
    codes = np.full(frame.shape, -1, dtype=int)
    codes[present] = inverse
    return codes, values


def _krippendorff_difference(values, marginals, measurement):
    """Squared differences between all pairs of categories (Krippendorff, 2011)."""
    k = len(values)
    if measurement == "nominal":
        return 1.0 - np.eye(k)
    v = values.astype(float)
    if measurement == "interval":
        return (v[:, None] - v[None, :]) ** 2
    if measurement == "ratio":
        total = v[:, None] + v[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.where(total == 0, 0.0, ((v[:, None] - v[None, :]) / total) ** 2)
        return diff
    # ordinal: the metric uses the frequencies of all categories between the two compared ones
    cumulative = np.cumsum(marginals)
    low = np.minimum(np.arange(k)[:, None], np.arange(k)[None, :])
    high = np.maximum(np.arange(k)[:, None], np.arange(k)[None, :])
    between = cumulative[high] - cumulative[low] + marginals[low]
    return (between - (marginals[:, None] + marginals[None, :]) / 2.0) ** 2


def _krippendorff_alpha(codes, n_values, values, measurement):
    """Alpha from the coded ratings, through the coincidence matrix; ``nan`` if undefined."""
    counts = np.stack([(codes == k).sum(axis=1) for k in range(n_values)], axis=1)
    per_unit = counts.sum(axis=1)
    usable = per_unit >= 2  # units with fewer than two ratings are not pairable
    if not usable.any():
        raise ValueError("No unit has at least two ratings: agreement cannot be computed.")
    c = counts[usable].astype(float)
    weights = c / (per_unit[usable] - 1.0)[:, None]
    coincidence = c.T @ weights
    coincidence[np.diag_indices(n_values)] -= weights.sum(axis=0)
    marginals = coincidence.sum(axis=1)
    total = marginals.sum()
    difference = _krippendorff_difference(values, marginals, measurement)
    expected = np.sum(np.outer(marginals, marginals) * difference)
    if expected == 0:  # all pairable ratings are identical: no variation, alpha is undefined
        return np.nan
    return 1.0 - (total - 1.0) * np.sum(coincidence * difference) / expected


def krippendorffs_alpha(
    df,
    measurement="nominal",
    return_per_item_agreement=False,
    return_bootstraped_z_score=True,
    n_iter_for_bootstrap=100,
    bootstrap_seed=13,
    method="two-tailed",
):
    """
    Krippendorff's alpha, a permutation z-score and p-value, and optionally the agreement per item.

    Alpha is ``1 - D_o / D_e``: the observed disagreement relative to the disagreement expected if
    the ratings were paired by chance. It is 1 for perfect agreement, about 0 for chance agreement
    and negative for systematic disagreement. It works with any number of raters, with missing
    ratings (they are excluded, not counted as a category; units with fewer than two ratings are
    ignored) and for four levels of measurement. Krippendorff recommends alpha >= .80 for firm
    conclusions and alpha >= .667 for tentative ones.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        Ratings with **one row per unit (item) and one column per rater** (the transpose of the
        layout of the ``krippendorff`` package). Missing ratings are NaN/None.
    measurement : {"nominal", "ordinal", "interval", "ratio"}
        Level of measurement, i.e. which differences between two ratings are counted.
        ``nominal``: equal or not (ratings may be of any type); ``ordinal``: depends on the
        ranks and frequencies of the categories between the two ratings; ``interval``:
        squared difference of the values; ``ratio``: squared relative difference
        ``((a - b) / (a + b))**2`` (non-negative values). Ordinal, interval and ratio ratings must
        be numeric: the *values* are used (for ordinal, their order), not the position of a
        category.
    return_per_item_agreement : bool
        Also return the share of the ratings of each item that equals its most frequent rating
        (``nan`` for items with fewer than two ratings).
    return_bootstraped_z_score : bool
        Compute ``z`` and ``p``. Otherwise they are ``nan``.
    n_iter_for_bootstrap : int
        Number of permutations for the null distribution.
    bootstrap_seed : int, optional
        Seed of the (local) random generator; numpy's global random state is not touched.
    method : {"two-tailed", ...}
        ``"two-tailed"`` doubles the p-value, anything else gives the one-sided value.

    Returns
    -------
    alpha, z, p : float
        ``z = (alpha - mean) / sd`` of the null distribution of alpha, obtained by permuting the
        ratings of every rater among the units that rater rated (so the design, i.e. who rated
        what, and the distribution of each rater's ratings are kept); ``p`` from the normal
        distribution. Null hypothesis: agreement is due to chance. ``alpha`` is ``nan`` if all
        ratings are identical (undefined).
    per_item_agreement : pandas.Series
        Only if ``return_per_item_agreement`` is True (as fourth element).

    Raises
    ------
    ValueError
        For an unknown ``measurement``, fewer than 2 raters, no unit with two ratings, or
        non-numeric ratings with the ordinal, interval and ratio levels.

    References
    ----------
    Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. Annenberg School for
    Communication Departmental Papers. https://repository.upenn.edu/asc_papers/43
    """
    if int(n_iter_for_bootstrap) != n_iter_for_bootstrap or n_iter_for_bootstrap < 1:
        raise ValueError("n_iter_for_bootstrap must be a positive integer.")
    codes, values = _krippendorff_codes(df, measurement)
    n_values = len(values)
    alpha = float(_krippendorff_alpha(codes, n_values, values, measurement))

    z, p = np.nan, np.nan
    if return_bootstraped_z_score and np.isfinite(alpha):
        rng = np.random.default_rng(bootstrap_seed)
        rated = [np.flatnonzero(codes[:, j] >= 0) for j in range(codes.shape[1])]
        null = np.empty(int(n_iter_for_bootstrap))
        for i in range(len(null)):
            shuffled = codes.copy()
            for j, rows in enumerate(rated):  # permute within each rater's rated units
                shuffled[rows, j] = rng.permutation(codes[rows, j])
            null[i] = _krippendorff_alpha(shuffled, n_values, values, measurement)
        mu, sigma = np.nanmean(null), np.nanstd(null)
        if sigma > 0:
            z = float((alpha - mu) / sigma)
            p = float(norm.sf(np.abs(z)))  # one-tailed, agreement > chance
            p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        shares = np.full(codes.shape[0], np.nan)
        for u, row in enumerate(codes):
            rated_values = row[row >= 0]
            if len(rated_values) >= 2:
                shares[u] = np.bincount(rated_values).max() / len(rated_values)
        index = df.index if hasattr(df, "index") else None
        return alpha, z, p, pd.Series(shares, index=index, name="per_item_agreement")

    return alpha, z, p


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


"""
Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!

Author: Philipp Singer (www.philippsinger.info)
"""


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


def htmt_ratio(data, construct_a, construct_b, method="spearman"):
    """
    The Heterotrait-Monotrait (HTMT) ratio
    measures the similarity between different constructs (heterotrait) compared to
    the similarity of items measuring the exact same construct (monotrait)
    Conservative Threshold: < 0.85 (typically used for strict discriminant validity).
    Liberal Threshold: < 0.90 (widely accepted in most social and business sciences).
    Note: If an HTMT value is above the threshold (e.g., 0.95), it suggests that the two constructs are too similar.
    data: DataFrame with item scores
    construct_a: list of columns for latent variable A
    construct_b: list of columns for latent variable B
    method: correlation method to use ("spearman" or "pearson")
    """
    corr = (
        data[construct_a + construct_b]
        .dropna()
        .astype("float64")
        .corr(method=method)
        .abs()
    )
    hetero = [
        corr.loc[i, j] for i in construct_a for j in construct_b
    ]  # Heterotrait correlations
    mono_a = [  # Monotrait correlations within A
        corr.loc[construct_a[i], construct_a[j]]
        for i in range(len(construct_a))
        for j in range(i + 1, len(construct_a))
    ]
    mono_b = [  # Monotrait correlations within B
        corr.loc[construct_b[i], construct_b[j]]
        for i in range(len(construct_b))
        for j in range(i + 1, len(construct_b))
    ]
    return np.mean(hetero) / np.sqrt(np.mean(mono_a) * np.mean(mono_b))


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


def _one_factor_loadings(r, min_uniqueness=0.005):
    """
    Minimum residual (ordinary least squares) loadings of the one-factor model of the
    correlation matrix ``r``: minimizes the squared differences between the off-diagonal
    elements of ``r`` and ``loadings * loadings'``. Uniquenesses are kept above
    ``min_uniqueness``, i.e. loadings are bounded by ``sqrt(1 - min_uniqueness)``: Heywood cases
    (improper solutions with a loading above 1, which e.g. ``factor_analyzer`` returns) end on
    this bound instead. The sign is chosen so that the loadings sum to a positive number.
    """
    p = r.shape[0]
    upper = np.sqrt(1.0 - min_uniqueness)
    values, vectors = np.linalg.eigh(r)
    start = np.clip(vectors[:, -1] * np.sqrt(max(values[-1], 1e-3)), -upper, upper)
    off_diagonal = r - np.diag(np.diag(r))

    def residuals(loadings):  # off-diagonal residuals, zero on the diagonal
        return off_diagonal - (np.outer(loadings, loadings) - np.diag(loadings**2))

    result = minimize(
        lambda loadings: 0.5 * np.sum(residuals(loadings) ** 2),
        start,
        jac=lambda loadings: -2.0 * residuals(loadings) @ loadings,
        method="L-BFGS-B",
        bounds=[(-upper, upper)] * p,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        warnings.warn(
            f"The one-factor model did not converge ({result.message}).", UserWarning
        )
    loadings = result.x
    return -loadings if loadings.sum() < 0 else loadings


def _omega_from_loadings(loadings):
    total = np.sum(loadings) ** 2
    return total / (total + np.sum(1.0 - loadings**2))


def mcdonald_omega(data, bootstrap_iter=None, conf_level=0.95, seed=13):
    """
    McDonald's omega (omega total): reliability of the sum score of a scale under a one-factor
    model with unequal loadings (congeneric items).

    ``omega = (sum(loadings))**2 / ((sum(loadings))**2 + sum(uniquenesses))`` with the loadings
    and uniquenesses of a one-factor model of the *correlation* matrix of the items (minimum
    residual estimation, as ``psych::fa`` and ``factor_analyzer``). Because correlations are
    used, rescaling or shifting items does not change the result. Unlike Cronbach's alpha, omega
    does not assume equal loadings; for equal loadings (tau-equivalent items) they coincide, and
    otherwise alpha is lower.

    Assumptions: the items measure a single factor (unidimensional scale), are continuous
    (for ordinal items with few categories use polychoric correlations instead) and are keyed
    in the same direction. Reverse-keyed items must be recoded before: a warning is given if
    the loadings have different signs, and omega is then too low.

    Parameters
    ----------
    data : pandas.DataFrame or array-like
        Item scores, one row per respondent and one column per item (at least 3 items and more
        respondents than items). Rows with missing values are dropped (listwise).
    bootstrap_iter : int, optional
        Number of bootstrap resamples (of respondents). If given, a percentile confidence
        interval is returned instead of the point estimate.
    conf_level : float
        Confidence level of the bootstrap interval, e.g. 0.95 (not a significance level).
    seed : int, optional
        Seed of the random generator used for the bootstrap (local; numpy's global random
        state is not touched). ``None`` gives different resamples on every call.

    Returns
    -------
    float
        Omega, if ``bootstrap_iter`` is None.
    numpy.ndarray
        ``[lower, median, upper]`` of the bootstrap distribution of omega otherwise (the same
        layout as :func:`bootstrap`).

    Raises
    ------
    ValueError
        For data that are not two-dimensional or numeric, fewer than 3 items, not more
        respondents than items, constant items, or invalid ``bootstrap_iter`` / ``conf_level``.

    References
    ----------
    McDonald, R. P. (1999). Test theory: A unified treatment. Erlbaum.

    Flora, D. B. (2020). Your coefficient alpha is probably wrong, but which coefficient
    omega is right? Advances in Methods and Practices in Psychological Science, 3(4), 484-501.
    https://doi.org/10.1177/2515245920951747
    """
    if bootstrap_iter is not None and (
        int(bootstrap_iter) != bootstrap_iter or bootstrap_iter < 1
    ):
        raise ValueError("bootstrap_iter must be a positive integer or None.")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1, e.g. 0.95.")
    x = _item_matrix(data)
    loadings = _one_factor_loadings(np.corrcoef(x, rowvar=False))
    if (loadings > 0).any() and (loadings < 0).any():
        warnings.warn(
            "The factor loadings have different signs. Reverse-keyed items must be recoded "
            "before omega is computed, otherwise omega is too low.",
            UserWarning,
        )
    if bootstrap_iter is None:
        return float(_omega_from_loadings(loadings))

    rng = np.random.default_rng(seed)
    omegas = np.full(int(bootstrap_iter), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the warnings concern the original sample
        for i in range(len(omegas)):
            sample = x[rng.integers(0, len(x), size=len(x))]
            if np.any(
                sample.std(axis=0) == 0
            ):  # a resample without variation in an item
                continue
            omegas[i] = _omega_from_loadings(
                _one_factor_loadings(np.corrcoef(sample, rowvar=False))
            )
    failed = int(np.isnan(omegas).sum())
    if failed:
        warnings.warn(
            f"{failed} of {len(omegas)} bootstrap resamples could not be used.",
            UserWarning,
        )
    lower = 100 * (1 - conf_level) / 2
    return np.nanpercentile(omegas, [lower, 50, 100 - lower])
