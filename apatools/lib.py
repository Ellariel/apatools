import numpy as np
import pandas as pd



def gini_coefficient(values, drop_zeros=False, tolerance=0.0000001):
    """Calculate the Gini coefficient"""
    # https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    x = np.array(values).flatten() # all values are treated equally, arrays must be 1d
    x_min = np.amin(x)
    if x_min < tolerance:
        x += np.abs(x_min) # values cannot be negative
    if drop_zeros:
        x = x[x >= tolerance]
    else:
        x += tolerance # values cannot be 0
    n = x.shape[0] # number of array elements
    x = np.sort(x) # values must be sorted
    idx = np.arange(1, n + 1) # index per array element
    return ((np.sum((2 * idx - n  - 1) * x)) / (n * np.sum(x))) # Gini coefficient
