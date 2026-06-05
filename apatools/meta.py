import numpy as np
import pandas as pd
from scipy.stats import norm


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


def average_morans_i(i, i_var, method="two-tailed"):
    """
    I: array of Moran's I values
    var_I: array of variances of Moran's I
    """
    i = np.array(i, dtype=float)
    var_i = np.array(i_var, dtype=float)
    w = 1 / var_i  # weights = inverse variance
    i_bar = np.sum(w * i) / np.sum(w)  # weighted mean
    se = np.sqrt(1 / np.sum(w))  # standard error
    z = i_bar / se  # test statistic
    p = 1 - norm.cdf(np.abs(z))  # p-value (two-tailed)
    p = p * 2 if method == "two-tailed" else p
    return {
        "i_mean": i_bar,
        "z": z,
        "se": se,
        "p-value": p,
    }
