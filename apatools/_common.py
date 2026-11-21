"""Helpers shared by the regression and sem modules (lm, slm, plm, sem).

Private: not part of the documented API and may change without notice.

* Intercept handling: :data:`INTERCEPT_NAMES`, :func:`df_check_intercept`,
  :func:`rename_intercept` (the labels differ between spreg, statsmodels / linearmodels and
  patsy).
* :func:`df_standardize`: z-scores or another transformation of the numeric columns of a
  design matrix, leaving the intercept column alone.
* :func:`w_subset_rows`: subsetting of spatial weights (needs ``libpysal``, imported lazily).
* :func:`round_columns` and :func:`format_significance`: rounding, significance stars and
  APA formatting of p-values in coefficient / report tables.

Requires pandas (``missing_dependency`` raises an ``ImportError`` naming the package if it is
absent); scipy is needed for the z-scores.
"""

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

    Typical use: after listwise deletion of rows with missing values, the weights
    matrix has to be reduced to the rows that remain in the data.

    Parameters
    ----------
    w : libpysal.weights.W
        Weights whose ``id_order`` corresponds to the rows of the data.
    positions : array-like of int
        Row positions to keep (in the order of the data); the rows and columns of the
        sparse weights matrix are taken in this order.

    Returns
    -------
    libpysal.weights.W
        New weights object for the subset, with the same ``transform`` as ``w``. Island
        warnings of libpysal are silenced.

    Notes
    -----
    Requires the optional dependency ``libpysal`` (``pip install "apatools[slm]"``).
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

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Table whose columns are searched; used only if ``columns`` is None.
    columns : iterable of str, optional
        Labels to search, e.g. ``coef_table.index``. Takes precedence over ``df``. One of the
        two must be given (if ``df`` is not a DataFrame and ``columns`` is None a ``TypeError``
        is raised).

    Returns
    -------
    str or None
        ``"CONSTANT"`` (spreg), ``"const"`` (statsmodels / linearmodels) or ``"Intercept"``
        (patsy), in this order of precedence (not in the order of the columns), or None.
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

    Parameters
    ----------
    index : iterable of str
        Labels, e.g. the index of a coefficient table or the columns of a design matrix.
    name : str, default "Intercept"
        The unified label for the intercept in the reports.

    Returns
    -------
    pandas.Index
        New index of the same length and order; the input is not modified.
    """
    return pd.Index(index).map(
        lambda label: name if label in INTERCEPT_NAMES else label
    )


def round_columns(table, decimal, columns=("coef", "se", "cil", "cir")):
    """Round the ``columns`` of ``table`` (in place) to ``decimal`` places; no-op if ``decimal`` is falsy.

    Parameters
    ----------
    table : pandas.DataFrame
        Coefficient table; modified in place.
    decimal : int or falsy
        Number of decimals; ``0``, ``None`` or False leaves the table unrounded.
    columns : iterable of str, default ("coef", "se", "cil", "cir")
        Columns to round (estimate, standard error and the confidence limits). Each must
        exist in ``table`` (``KeyError`` otherwise); other columns, such as p-values, are
        not touched.

    Returns
    -------
    pandas.DataFrame
        ``table`` itself (the same object), for chaining.
    """
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

    Parameters
    ----------
    table : pandas.DataFrame
        Report table with a column of numeric p-values; modified in place.
    add_stars : bool or callable, default True
        Write a significance marker per row to ``stars_column``. True uses
        :func:`apatools.format.get_stars` (``***`` < .001, ``**`` < .01, ``*`` < .05,
        ``⁺`` < .10); a callable is applied to every p-value instead.
    format_pval : bool or callable, default True
        Replace ``p_column`` by strings. True gives the bare APA number without ``p``, equals
        sign and spaces (``".009"``, ``"<.001"``, i.e. ``format_p(p, use_letter=False,
        keep_spaces=False, no_equals=True)``); a callable is applied to every p-value instead.
    p_column : str, default "p-value"
        Name of the column with the p-values.
    stars_column : str, default "sig"
        Name of the column created for the stars (appended at the end, or overwritten).
    na_rep : object, optional
        Value substituted for missing p-values (e.g. ``""``) before both steps; the
        substitute is passed through ``get_stars`` / ``format_p`` like any other value, which
        return non-numeric input unchanged. None keeps the missing values (NaN then gets
        no star and is formatted as ``NaN``).

    Returns
    -------
    pandas.DataFrame
        ``table`` itself. If both options are falsy it is returned unchanged.
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

    Parameters
    ----------
    df : pandas.DataFrame
        Design matrix or data; not modified (a filtered copy is processed).
    func : {"z", True} or callable, default "z"
        ``"z"`` / True: z-scores (``scipy.stats.zscore``: mean 0, population standard deviation
        1, ``ddof=0``; a constant column gives NaN). A callable is passed to ``DataFrame.apply``
        and so receives each column as a Series.

    Returns
    -------
    pandas.DataFrame
        The numeric/boolean columns in their original order, transformed; the intercept column
        (see :func:`df_check_intercept`) is excluded from the transformation and keeps its
        position. If the intercept is the only column the frame is returned as it is.

    Raises
    ------
    NotImplementedError
        If ``func`` is neither ``"z"``, True nor a callable (including False and other strings).
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
