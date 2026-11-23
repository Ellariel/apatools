"""Standalone helpers: bootstrap interval, Gini coefficient, transformations, HTMT ratio."""

import numpy as np


def bootstrap(*args, func=np.mean, conf_level=0.95, n_rep=1000, seed=13):
    """Percentile bootstrap confidence interval for a statistic.

    Parameters
    ----------
    *args : array-like
        One or more equally long samples (resampled jointly by row).
    func : callable
        Statistic computed on the resampled arrays, called as ``func(*samples)``.
    conf_level : float
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

    Notes
    -----
    The percentile method is simple but can be biased and too narrow for small samples or skewed
    statistics; the interval is taken from the non-missing replications (``nanpercentile``).

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
    https://doi.org/10.1007/978-1-4899-4541-9
    """
    rng = np.random.default_rng(seed)  # local generator: numpy's global random state is untouched
    data = np.asanyarray(args)
    idx = np.arange(0, data.shape[1], 1)
    sample = [
        func(*np.take(data, rng.choice(idx, size=data.shape[1], replace=True), 1))
        for _ in range(n_rep)
    ]
    lp, m, rp = (1 - conf_level) / 2, 0.5, 1 - (1 - conf_level) / 2
    return np.nanpercentile(sample, [lp * 100, m * 100, rp * 100])


def gini(values, drop_zeros=False, tolerance=10**-6):
    """
    Gini coefficient of inequality of a sample of non-negative values.

    ``G = sum_i (2 i - n - 1) x_(i) / (n * sum(x))`` for the sorted values ``x_(1) <= ... <= x_(n)``:
    0 means perfect equality, values near 1 mean that one unit holds almost everything (the maximum
    for a sample of ``n`` is ``(n - 1) / n``). This is the sample Gini without the ``n / (n - 1)``
    small-sample correction, identical to ``ineq::Gini(x, corr=FALSE)`` in R.

    Parameters
    ----------
    values : iterable of float
        The values (e.g. incomes); flattened, all treated alike.
    drop_zeros : bool
        Leave values below ``tolerance`` (zeros) out of the calculation instead of counting them as
        units with (almost) nothing.
    tolerance : float
        Values below it count as zero. If any value is below it, *all* values are shifted up by
        the absolute minimum so that none is negative (the Gini coefficient is not defined for
        negative values; the shift changes the result, so negative input should be avoided).
        Unless ``drop_zeros``, ``tolerance`` is then added to every value so that no value is
        exactly zero (a bias of the order of ``tolerance``).

    Returns
    -------
    float

    References
    ----------
    Gini, C. (1912). Variabilità e mutabilità. Bologna: Tipografia di Paolo Cuppini.
    Lorenz, M. O. (1905). Methods of measuring the concentration of wealth. Publications of the
    American Statistical Association, 9(70), 209-219. https://doi.org/10.1080/15225437.1905.10503443
    Sen, A. (1973). On Economic Inequality. Oxford: Clarendon Press.
    """
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
    Gini coefficient from grouped data: points of the Lorenz curve (e.g. income deciles).

    The Lorenz curve is interpolated linearly between the points (and from the origin to the first
    point) and ``G = 1 - 2 * area under the curve`` is computed with the trapezoid rule. Because
    the true Lorenz curve is convex, the straight segments lie above it: the result is a lower
    bound of the Gini coefficient of the underlying data and gets closer with more points.

    Parameters
    ----------
    values : array-like
        The *cumulative* amount held by the poorest ``percentiles`` percent of the population:
        cumulative shares in percent (``values_are_shares=True``, e.g. ``[1.5, 5.0, ..., 100]``)
        or cumulative amounts (e.g. ``[100, 150, 350, ..., 1200]``, divided by their maximum, so the
        last point has to be the total).
    percentiles : array-like
        Cumulative share of the population in percent, increasing, ending at 100
        (e.g. ``[10, 20, ..., 100]``). A starting point at 0 is added if missing.
    values_are_shares : bool
        ``values`` are cumulative shares in percent (divided by 100) instead of cumulative amounts.

    Returns
    -------
    float
        Gini coefficient between 0 (equality) and 1.

    References
    ----------
    Lorenz, M. O. (1905). Methods of measuring the concentration of wealth. Publications of the
    American Statistical Association, 9(70), 209-219. https://doi.org/10.1080/15225437.1905.10503443
    Gini, C. (1912). Variabilità e mutabilità. Bologna: Tipografia di Paolo Cuppini.
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
    """
    Signed pseudo-logarithm ``sign(x) * log10(1 + |x|)``.

    Defined for zero and negative values (``log_transform(0) == 0``), odd (``f(-x) == -f(x)``) and
    close to ``log10(x)`` for large ``|x|``; useful for skewed data that contain zeros or negative
    numbers (e.g. changes, balances). The shift by 1 makes the transform depend on the unit of
    measurement.

    Parameters
    ----------
    values : iterable of float
        Flattened to one dimension.

    Returns
    -------
    numpy.ndarray
        Transformed values, same length.
    """

    x = np.array(list(values)).astype("float64").flatten()
    return np.sign(x) * np.log10(1 + np.abs(x))


def z_transform(values, return_params=False, ignore_errors=True):
    """
    z-transformation (standardization) with the mean and the population standard deviation
    (``ddof=0``) of the non-missing values; NaNs stay NaN.

    If the standard deviation is zero (constant values) or undefined (no values, only NaNs) the
    z-scores do not exist: a ``ValueError`` is raised, or, with ``ignore_errors=True``, an array of
    NaN is returned (without any division by zero). With ``return_params`` the mean and the standard
    deviation are returned as well.
    """

    x = np.array(list(values)).astype("float64").flatten()
    if np.isnan(x).all():  # nothing to standardize (also avoids "mean of empty slice" warnings)
        m = s = np.nan
    else:
        m, s = np.nanmean(x), np.nanstd(x)
    undefined = not (np.isfinite(s) and s > 0)
    if undefined and not ignore_errors:
        raise ValueError(
            "Standard deviation is zero or undefined. Z-transformation is not defined."
        )

    z = np.full(x.shape, np.nan) if undefined else (x - m) / s

    if return_params:
        return z, m, s
    return z


def htmt_ratio(data, construct_a, construct_b, method="spearman"):
    """
    Heterotrait-monotrait (HTMT) ratio of correlations, a criterion of discriminant validity.

    ``HTMT = mean(|r| between the items of A and the items of B) /
    sqrt(mean(|r| among the items of A) * mean(|r| among the items of B))``: the similarity of
    items of different constructs (heterotrait) relative to the similarity of items of the same
    construct (monotrait). Values near 1 mean that the two constructs are not empirically distinct.

    Henseler et al. (2015) propose a threshold of .85 (Kline, 2011; conservative) or .90 (Gold et
    al., 2001; liberal); an HTMT significantly below the threshold (bootstrap confidence interval,
    not provided here) supports discriminant validity. The ratio needs at least two items per
    construct (otherwise it is NaN).

    Parameters
    ----------
    data : pandas.DataFrame
        Item scores, one column per item. Rows with a missing value in any of the items are dropped.
    construct_a, construct_b : list of str
        Column names of the items of constructs A and B.
    method : {"spearman", "pearson"}
        Correlation coefficient (``DataFrame.corr``); Henseler et al. use Pearson correlations,
        Spearman is more robust for ordinal items.

    Returns
    -------
    float

    References
    ----------
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for assessing discriminant
    validity in variance-based structural equation modeling. Journal of the Academy of Marketing
    Science, 43(1), 115-135. https://doi.org/10.1007/s11747-014-0403-8
    Kline, R. B. (2011). Principles and Practice of Structural Equation Modeling (3rd ed.). Guilford.
    Gold, A. H., Malhotra, A., & Segars, A. H. (2001). Knowledge management: An organizational
    capabilities perspective. Journal of Management Information Systems, 18(1), 185-214.
    https://doi.org/10.1080/07421222.2001.11045669
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
