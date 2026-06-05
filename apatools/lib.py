import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.stats.inter_rater import aggregate_raters
import krippendorff


def bootstrap(*args, func=np.mean, alpha=0.95, n_rep=1000, seed=13):
    """
    Bootstraping confidence intervals for the mean/median value
    https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    """

    np.random.seed(seed)
    data = np.asanyarray(args)
    idx = np.arange(0, data.shape[1], 1)
    sample = [
        func(*np.take(data, np.random.choice(idx, size=data.shape[1], replace=True), 1))
        for _ in range(n_rep)
    ]
    lp, m, rp = (1 - alpha) / 2, 0.5, 1 - (1 - alpha) / 2
    return np.percentile(sample, [lp * 100, m * 100, rp * 100])


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
    # Trapezoidal approximation of area under Lorenz curve
    b = (y[1:] + np.roll(y, 1)[1:]) * (x[1:] - np.roll(x, 1)[1:]) / 2

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

    df = df.astype(
        str
    )  # be careful about NaNs, they are transformed to a category via .astype(str)
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
    Compute Krippendorff' alpha, z-score, and p-value, per-item agreement
    Level of measurement = "nominal", "ordinal", "interval", "ratio"
    Null hypothesis: Agreement is due to chance
    """
    # https://en.wikipedia.org/wiki/Krippendorff's_Alpha
    # https://github.com/pln-fing-udelar/fast-krippendorff

    df = df.astype(
        str
    )  # be careful about NaNs, they are transformed to a category via .astype(str)
    m = aggregate_raters(df, n_cat=None)[0]  # [:,:-1]
    alpha = krippendorff.alpha(value_counts=m, level_of_measurement=measurement)

    # z-score and p-value
    z, p = np.nan, np.nan
    if return_bootstraped_z_score:
        np.random.seed(bootstrap_seed)
        null_dist = []
        for _ in range(n_iter_for_bootstrap):
            m = aggregate_raters(df.apply(np.random.permutation, axis=0), n_cat=None)[
                0
            ]  # [:,:-1]
            null_alpha = krippendorff.alpha(
                value_counts=m, level_of_measurement=measurement
            )
            null_dist.append(null_alpha)
        mu, sigma = np.mean(null_dist), np.std(null_dist)
        if sigma > 0:
            z = (alpha - mu) / sigma
            p = norm.sf(np.abs(z))  # one-tailed, agreement > chance
            p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        agreements = {}
        for idx, row in df.astype(str).iterrows():
            row = row.dropna()
            agreements[idx] = (
                np.nan if len(row) <= 1 else row.value_counts().max() / len(row)
            )
        return (
            alpha,
            z,
            p,
            pd.Series(agreements, index=df.index, name="per_item_agreement"),
        )

    return alpha, z, p


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
    zr_se = pow(1 / (n - 3), 0.5)
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


def dependent_corr(xy, xz, yz, n, method="steiger", twotailed=True, alpha=0.95):
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
        raise Exception("Wrong method!")


def independent_corr(xy, ab, n1, n2=None, method="fisher", twotailed=True, alpha=0.95):
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
        z = abs(diff / se_diff_r)
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
        raise Exception("Wrong method!")


def htmt(data, construct_a, construct_b, method="spearman"):
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
