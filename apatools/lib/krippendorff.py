"""Krippendorff's alpha: agreement for any number of raters, missing ratings and four levels of measurement."""

import numpy as np
import pandas as pd
from scipy.stats import norm

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
