"""Standalone helpers: bootstrap interval, Gini coefficient, transformations, HTMT ratio."""

import numpy as np


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
