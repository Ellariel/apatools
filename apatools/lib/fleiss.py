"""Fleiss' kappa: agreement of several raters on categorical ratings."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.inter_rater import aggregate_raters


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
