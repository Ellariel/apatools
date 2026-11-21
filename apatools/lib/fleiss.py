"""Fleiss' kappa: agreement of several raters on categorical ratings."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.inter_rater import aggregate_raters


def fleiss_kappa(df, return_per_item_agreement=False, method="two-tailed"):
    """
    Fleiss' kappa for the agreement of several raters, with a test against chance agreement.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per rated item, one column per rater, the category of each rating as value.
        Every item has to be rated by the same number of raters (Fleiss, 1971): items with
        a missing rating (NaN / None) are left out completely (listwise deletion, as in the
        R package ``irr``) because treating "missing" as a category would inflate agreement.
    return_per_item_agreement : bool
        Also return the agreement ``P_i`` of every item (NaN for the items that were left out).
    method : {"two-tailed", "one-tailed"}
        p-value of the z test of H0: kappa = 0 (one-tailed: agreement better than chance).

    Returns
    -------
    kappa, z, p (and the per-item agreement as a Series if requested)
        ``kappa = (mean(P_i) - P_e) / (1 - P_e)`` (Fleiss, 1971). ``z = kappa / SE0`` where
        ``SE0`` is the standard error of kappa under H0 (all ratings independent, Fleiss,
        Nee & Landis, 1979; the formula of ``irr::kappam.fleiss``)::

            Var0(kappa) = 2 / (N n (n - 1)) * [(sum p_j q_j)^2 - sum p_j q_j (q_j - p_j)] / (sum p_j q_j)^2

        with ``N`` items, ``n`` raters, ``p_j`` the proportion of all ratings in category
        ``j`` and ``q_j = 1 - p_j``. ``z`` and ``p`` are NaN if kappa is undefined (all ratings in
        one category).

    Notes
    -----
    Landis & Koch (1977) call kappa < 0 poor, .00-.20 slight, .21-.40 fair, .41-.60 moderate,
    .61-.80 substantial and .81-1.00 almost perfect agreement.
    The z test only tells whether agreement exceeds chance, not how large it is.

    References
    ----------
    Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological
    Bulletin, 76(5), 378-382.
    Fleiss, J. L., Nee, J. C. M., & Landis, J. R. (1979). Large sample variance of kappa in the
    case of different sets of raters. Psychological Bulletin, 86(5), 974-977.
    Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical
    data. Biometrics, 33(1), 159-174.
    """
    n_raters = df.shape[1]
    complete = df.notna().all(axis=1).to_numpy()
    per_item = pd.Series(np.nan, index=df.index, name="per_item_agreement")
    n_items = int(complete.sum())
    if n_items == 0 or n_raters < 2:
        raise ValueError("Fleiss' kappa needs at least one completely rated item and two raters.")

    m = aggregate_raters(df[complete].astype(str), n_cat=None)[0]  # items x categories (counts)
    P_i = (np.sum(m**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))  # per-item agreement
    P_j = np.sum(m, axis=0) / (n_items * n_raters)  # category proportions
    P_e = np.sum(P_j**2)  # expected agreement
    kappa = (np.mean(P_i) - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan

    # variance of kappa under H0 (Fleiss, Nee & Landis, 1979)
    pq = P_j * (1 - P_j)
    var_kappa = (
        2
        / (n_items * n_raters * (n_raters - 1))
        * (pq.sum() ** 2 - np.sum(pq * (1 - 2 * P_j)))
        / pq.sum() ** 2
        if pq.sum() > 0
        else np.nan
    )

    z, p = np.nan, np.nan
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p = 2 * norm.sf(np.abs(z)) if method == "two-tailed" else norm.sf(z)  # one-tailed: agreement > chance

    if return_per_item_agreement:
        per_item[complete] = P_i
        return kappa, z, p, per_item

    return kappa, z, p
