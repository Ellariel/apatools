import numpy as np
import pandas as pd



def gini(values, drop_zeros=False, tolerance=0.0000001):
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


def gini_from_percentiles(values, percentiles, values_are_shares=False):
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


def z_transform(values, return_params=False):
    """z-transformation or standardization"""

    x = np.array(list(values)).astype('float64').flatten()
    m, s = np.mean(x), np.std(x)
    if s == 0:
        raise ValueError("Standard deviation is zero. Z-transformation is not defined.")

    z = (x - m) / s

    if return_params:
        return z, m, s
    return z



