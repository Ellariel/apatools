import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.inter_rater import aggregate_raters
import krippendorff



def gini(values, 
         drop_zeros=False, 
         tolerance=10**-6):
    """Calculate the Gini coefficient"""
    # https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    x = np.array(list(values)).astype('float64').flatten() # all values are treated equally, arrays must be 1d
    x_min = np.amin(x)
    if x_min < tolerance:
        x += np.abs(x_min) # values cannot be negative
    if drop_zeros:
        x = x[x >= tolerance]
    else:
        x += tolerance # values cannot be 0
    n, x = x.shape[0], np.sort(x) # values must be sorted
    idx = np.arange(1, n + 1) # index per array element
    return np.sum((2 * idx - n  - 1) * x) / (n * np.sum(x)) # Gini coefficient


def gini_from_percentiles(values, 
                          percentiles, 
                          values_are_shares=False):
    """
    Calculate the Gini coefficient from percentile-based cumulative data.
    Parameters:
    - percentiles: array of population percentiles (e.g., [10, 20, ..., 100])
    - values: array of corresponding cumulative shares/proportion (e.g., [1.5, 5.0, ..., 100])
        or cumulative values themselves (e.g., [100, 150, 350, ..., 1200])
    """

    x = np.array(list(percentiles)).astype('float64').flatten() / 100
    y = np.array(list(values)).astype('float64').flatten()
    if values_are_shares:
        y = y / 100
    else:
        y = y / np.max(y)
    # Trapezoidal approximation of area under Lorenz curve
    b = (y[1:] + np.roll(y, 1)[1:]) * (x[1:] - np.roll(x, 1)[1:]) / 2

    return 1 - 2 * b.sum() # Gini coefficient is 1 - 2 * area under Lorenz curve


def log_transform(values):
    """Pseudo log-transformation, keeping sign"""

    x = np.array(list(values)).astype('float64').flatten()
    return np.sign(x) * np.log10(1 + np.abs(x))


def z_transform(values, return_params=False, ignore_errors=True):
    """z-transformation or standardization"""

    x = np.array(list(values)).astype('float64').flatten()
    m, s = np.nanmean(x), np.nanstd(x)
    if not s and not ignore_errors:
          raise ValueError("Standard deviation is zero or undefined. Z-transformation is not defined.")

    z = (x - m) / s

    if return_params:
        return z, m, s
    return z


def fleiss_kappa(df, 
                 return_per_item_agreement=False, 
                 method="two-tailed"):
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

    df = df.astype(str) # be careful about NaNs, they are transformed to a category via .astype(str)
    n_items, n_raters = df.shape
    m = aggregate_raters(df, n_cat=None)[0] 
    P_i = (np.sum(m**2, axis=1) - n_raters) / (n_raters * (n_raters - 1)) # Per-item agreement
    P_j = np.sum(m, axis=0) / (n_items * n_raters) # Category proportions
    P_e = np.sum(P_j**2) # Expected agreement
    kappa = (np.mean(P_i) - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan # Fleiss' kappa

    # variance of kappa (approximation)
    term1 = np.sum(P_j**2 * (1 - P_j)**2)
    term2 = (1 - P_e) * (np.sum(P_j**3) - P_e * np.sum(P_j**2))
    var_kappa = (term1 - term2) / (n_items * n_raters * (n_raters - 1) * (1 - P_e)**2)

    # z-score and p-value
    z, p = np.nan, np.nan
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p = norm.sf(np.abs(z)) # one-tailed, agreement > chance
        p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        return kappa, z, p, pd.Series(P_i, index=df.index, 
                                            name="per_item_agreement"), 

    return kappa, z, p


def krippendorffs_alpha(df, measurement="nominal", 
                        return_per_item_agreement=False, 
                        return_bootstraped_z_score=True,
                        n_iter_for_bootstrap=100,
                        bootstrap_seed=13,
                        method="two-tailed"):
    """
    Compute Krippendorff' alpha, z-score, and p-value, per-item agreement
    Level of measurement = "nominal", "ordinal", "interval", "ratio"
    Null hypothesis: Agreement is due to chance
    """
    # https://en.wikipedia.org/wiki/Krippendorff's_Alpha
    # https://github.com/pln-fing-udelar/fast-krippendorff
    
    df = df.astype(str) # be careful about NaNs, they are transformed to a category via .astype(str)
    m = aggregate_raters(df, n_cat=None)[0]#[:,:-1]
    alpha = krippendorff.alpha(value_counts=m, 
                               level_of_measurement=measurement)
    
    # z-score and p-value
    z, p = np.nan, np.nan
    if return_bootstraped_z_score:
        np.random.seed(bootstrap_seed)
        null_dist = []
        for _ in range(n_iter_for_bootstrap):
            m = aggregate_raters(df.apply(np.random.permutation, axis=0), 
                                 n_cat=None)[0]#[:,:-1]
            null_alpha = krippendorff.alpha(value_counts=m, 
                               level_of_measurement=measurement)
            null_dist.append(null_alpha)
        mu, sigma = np.mean(null_dist), np.std(null_dist)
        if sigma > 0:
            z = (alpha - mu) / sigma
            p = norm.sf(np.abs(z)) # one-tailed, agreement > chance
            p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        agreements = {}
        for idx, row in df.astype(str).iterrows():
            row = row.dropna()
            agreements[idx] = np.nan if len(row) <= 1 else row.value_counts().max() / len(row)
        return alpha, z, p, pd.Series(agreements, index=df.index, name="per_item_agreement")

    return alpha, z, p 