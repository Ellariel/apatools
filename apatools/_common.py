"""Helpers shared by the regression and sem modules (lm, slm, plm, sem): intercept handling,
standardization, subsetting of spatial weights, rounding and significance formatting of report
tables. Private: not part of the documented API."""

import numpy as np
from scipy.stats import zscore

from ._deps import missing_dependency
from .format import format_p, get_stars

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    missing_dependency("_common", None, e, hint="pip install pandas")

# labels of the intercept in the coefficient tables of spreg, statsmodels / linearmodels and patsy
INTERCEPT_NAMES = ("CONSTANT", "const", "Intercept")


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


def df_check_intercept(df=None, columns=None):
    """
    Label of the intercept among the columns of ``df`` (or among ``columns``, e.g. the index of a
    coefficient table): the first of ``INTERCEPT_NAMES`` that occurs, ``None`` if there is none.
    """
    if columns is None and isinstance(df, pd.DataFrame):
        columns = df.columns
    for name in INTERCEPT_NAMES:
        if name in columns:
            return name
    return None


def rename_intercept(index, name="Intercept"):
    """
    The labels with the intercept (an exact match of one of ``INTERCEPT_NAMES``) renamed to ``name``.

    Other labels are left unchanged, also those that merely contain such a name (``"const_income"``),
    and an index without intercept is returned as it is.
    """
    return pd.Index(index).map(
        lambda label: name if label in INTERCEPT_NAMES else label
    )


def round_columns(table, decimal, columns=("coef", "se", "cil", "cir")):
    """Round the ``columns`` of ``table`` (in place) to ``decimal`` places; no-op if ``decimal`` is falsy."""
    if decimal:
        for column in columns:
            table[column] = table[column].round(decimal)
    return table


def format_significance(
    table,
    add_stars=True,
    format_pval=True,
    p_column="p-value",
    stars_column="sig",
    na_rep=None,
):
    """
    Add significance stars and format the p-values of a report table (in place; returns ``table``).

    ``add_stars`` / ``format_pval`` are False (skip), True (:func:`apatools.format.get_stars`,
    APA-style :func:`apatools.format.format_p` without the ``p =`` prefix) or a callable that
    takes one p-value. The stars are computed from the unformatted p-values, so the order of the
    two steps does not matter to the caller. ``na_rep`` replaces missing p-values before both
    steps (``None``: leave them as they are).
    """
    pvalues = table[p_column] if na_rep is None else table[p_column].fillna(na_rep)
    if add_stars:
        stars = add_stars if callable(add_stars) else get_stars
        table[stars_column] = [stars(p) for p in pvalues]
    if format_pval:
        fmt = (
            format_pval
            if callable(format_pval)
            else lambda p: format_p(
                p, use_letter=False, keep_spaces=False, no_equals=True
            )
        )
        table[p_column] = [fmt(p) for p in pvalues]
    return table


def df_standardize(df, func="z"):
    """
    Numeric and boolean columns of ``df``, standardized (``"z"`` or ``True``: z-scores) or transformed
    by ``func``, a callable that is applied to every column. Other columns are dropped; an intercept
    column is kept unchanged at its position.
    """
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
        df = df.apply(zscore)
    else:
        raise NotImplementedError(
            f"Func '{func}' is not implemented, try df.apply-compatible method or use 'z'."
        )

    if const_name is not None:  # insert const back
        df.insert(const_pos, const_name, const)

    return df
