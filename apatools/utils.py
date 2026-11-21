
import numpy as np
import scipy

from ._deps import missing_dependency

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    missing_dependency("utils", None, e, hint="pip install pandas")


def w_subset_rows(w, positions):
    """
    Spatial weights restricted to the observations at ``positions``.

    Unlike ``libpysal.weights.w_subset`` the original weight values are kept (not
    replaced by binary ones), the transformation of ``w`` (e.g. row-standardization,
    ``w.transform``) is re-applied to the subset and the ids of the returned
    object are the new row positions ``0..len(positions)-1``.
    Observations that lose all their neighbours become islands (see ``.islands``).

    Parameters
    ----------
    w : libpysal.weights.W
        Weights whose ``id_order`` corresponds to the rows of the data.
    positions : array-like of int
        Row positions to keep (in the order of the data).
    """
    # imported here: only the spatial modules need libpysal
    from libpysal.weights import WSP

    positions = np.asarray(positions, dtype=int)
    sub = WSP(w.sparse[positions][:, positions]).to_W(silence_warnings=True)
    sub.transform = w.transform
    return sub


def df_check_intercept(df=None, columns=None):  # look for the intercept column name
    if columns is None and isinstance(df, pd.DataFrame):
        columns = df.columns
    if "CONSTANT" in columns:
        return "CONSTANT"
    if "const" in columns:
        return "const"
    elif "Intercept" in columns:
        return "Intercept"
    else:
        return None


def df_standardize(df, func="z"):
    df = df.select_dtypes(include=[np.number, "bool"])
    const_name = df_check_intercept(df)

    if const_name is not None:  # rem const
        if df.shape[1] == 1:  # if only const in the data frame
            return df
        const_pos = df.columns.get_loc(const_name)
        const = df[const_name].copy()
        df.drop(const_name, axis=1, inplace=True)

    if callable(func):
        df = df.apply(func)
    elif func == "z" or (isinstance(func, bool) and func):
        df = df.apply(scipy.stats.zscore)
    else:
        raise NotImplementedError(
            f"Func '{func}' is not implemented, try df.apply-compatible method or use 'z'."
        )

    if const_name is not None:  # insert const back
        df.insert(const_pos, const_name, const)

    return df
