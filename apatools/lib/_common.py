"""Helpers shared by the reliability coefficients (Cronbach's alpha and McDonald's omega)."""

import warnings

import numpy as np


def check_interval_options(bootstrap_iter, conf_level):
    """Validate the arguments that choose and size the confidence interval."""
    if bootstrap_iter is not None and (
        int(bootstrap_iter) != bootstrap_iter or bootstrap_iter < 1
    ):
        raise ValueError("bootstrap_iter must be a positive integer or None.")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1, e.g. 0.95.")


def item_matrix(data, min_items, reason=""):
    """Item scores as a float matrix without incomplete rows (listwise deletion), validated."""
    x = np.asarray(data, dtype=float)  # raises a ValueError for non-numeric items
    if x.ndim != 2:
        raise ValueError(
            "data must be two-dimensional: one row per respondent, one column per item."
        )
    x = x[~np.isnan(x).any(axis=1)]
    n, p = x.shape
    if p < min_items:
        raise ValueError(f"At least {min_items} items are needed{reason} (got {p}).")
    if n <= p:
        raise ValueError(
            f"More respondents than items are needed (got {n} rows for {p} items)."
        )
    if np.any(x.std(axis=0) == 0):
        raise ValueError("An item is constant: it carries no information.")
    return x


def percentile_bootstrap(x, statistic, iterations, conf_level, seed):
    """
    Percentile bootstrap interval of ``statistic(sample)`` over resamples of the rows of ``x``.

    The generator is local (numpy's global random state is not touched). Resamples in which an
    item is constant are skipped, as are warnings raised while fitting the resamples (they concern
    the original sample); a warning reports how many resamples could not be used. Returns the
    lower and upper limit as floats.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    values = np.full(int(iterations), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(values)):
            sample = x[rng.integers(0, n, size=n)]
            if np.any(sample.std(axis=0) == 0):
                continue
            values[i] = statistic(sample)
    failed = int(np.isnan(values).sum())
    if failed:
        warnings.warn(
            f"{failed} of {len(values)} bootstrap resamples could not be used.", UserWarning
        )
    lower = 100 * (1 - conf_level) / 2
    low, high = np.nanpercentile(values, [lower, 100 - lower])
    return float(low), float(high)
