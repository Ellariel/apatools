"""APA-style formatting of significance levels and effect sizes.

* :func:`get_stars`: significance markers (``***``, ``**``, ``*``, ``⁺``) for a p-value.
* :func:`format_p`: p-values as ``p = .009`` / ``p < .001``.
* :func:`format_r`: correlations and R² without a leading zero, e.g. ``r = -.34``.

All functions pass non-numeric input (strings, ``None``, ...) through unchanged, so they can
be applied to mixed table columns. They need only numpy.

References
----------
American Psychological Association. (2020). *Publication manual of the American
Psychological Association* (7th ed.). https://doi.org/10.1037/0000165-000
(statistics reporting: no leading zero for values that cannot exceed 1, such as p and r;
exact p-values to two or three decimals, and p < .001 for smaller values).
"""
from numbers import Number

import numpy as np

###############
#  formating  #
###############


def get_stars(p, p001="***", p01="**", p05="*", p10="⁺", p_=""):
    """Significance marker for a p-value.

    Parameters
    ----------
    p : float or any
        The p-value. Non-numeric input (e.g. a string such as ``"< .001"`` or ``None``) is
        returned unchanged.
    p001 : str, default "***"
        Marker for ``p < .001``.
    p01 : str, default "**"
        Marker for ``.001 <= p < .01``.
    p05 : str, default "*"
        Marker for ``.01 <= p < .05``.
    p10 : str, default "⁺"
        Marker for ``.05 <= p < .10`` (marginal significance, superscript plus
        U+207A).
    p_ : str, default ""
        Marker for everything else, i.e. ``p >= .10`` and also NaN.

    Returns
    -------
    str or any
        One of the markers, or ``p`` itself if it is not a number.

    Notes
    -----
    The thresholds are the conventional .001, .01, .05 and .10 levels; they are
    strict upper bounds (``p = .05`` gets ``p10``, not ``p05``). Negative values and
    ``-inf`` are treated as ``p < .001``. Table notes should state the thresholds used.

    Examples
    --------
    >>> get_stars(0.0004)
    '***'
    >>> get_stars(0.03)
    '*'
    >>> get_stars(0.5)
    ''
    >>> get_stars("n/a")
    'n/a'
    """
    if not isinstance(p, Number):
        return p
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def format_p(p, use_letter="p", keep_spaces=True, no_equals=False):
    """Format a p-value in APA style, e.g. ``p = .009``, ``p < .001``.

    Parameters
    ----------
    p : float or any
        The p-value. Non-numeric input is returned unchanged.
    use_letter : str or False, default "p"
        Symbol put in front, separated by a space (e.g. ``"p"`` or ``"P"``); a falsy value
        omits it (the result then starts with the operator, e.g. ``"= .009"``).
    keep_spaces : bool, default True
        ``False`` removes all spaces (``p=.009``, ``p<.001``), as used in table cells or
        figure labels.
    no_equals : bool, default False
        ``True`` removes the ``=`` sign (``p .009``); the ``<`` and ``>`` signs are kept
        (``p < .001``). Combined with ``use_letter=False, keep_spaces=False`` it yields the
        bare number (``.009``) or ``<.001``, which is what the report tables use.

    Returns
    -------
    str or any
        The formatted string, or ``p`` unchanged if it is not a number.

    Notes
    -----
    Rules (APA 7th ed., 2020): the leading zero is dropped because p cannot exceed 1;
    the exact value is given with three decimals (``.009``, ``.050``); values below .001
    are reported as ``< .001`` (never ``p = .000``); values that would round to 1.000
    (``p >= .9995``) are reported as ``> .999``. NaN, infinite and negative values are
    reported as ``= NaN``. The output is a plain string, so it is meant for the final
    report, not for further calculation. With ``use_letter=False``, ``no_equals=True``
    and ``keep_spaces=True`` the result keeps a leading space (``" .009"``).

    References
    ----------
    American Psychological Association. (2020). *Publication manual of the American
    Psychological Association* (7th ed.). https://doi.org/10.1037/0000165-000

    Examples
    --------
    >>> format_p(0.009)
    'p = .009'
    >>> format_p(0.0004)
    'p < .001'
    >>> format_p(0.0004, keep_spaces=False)
    'p<.001'
    >>> format_p(0.009, no_equals=True)
    'p .009'
    >>> format_p(0.009, use_letter=False, keep_spaces=False, no_equals=True)
    '.009'
    """
    if not isinstance(p, Number):
        return p
    if not np.isfinite(p) or p < 0.0:
        p = "= NaN"
    elif p < 0.001:
        p = "< .001"  # APA: never report p = .000
    elif p >= 0.9995:  # would round to 1.000; a p-value is never exactly 1
        p = "> .999"
    else:
        p = "= " + f"{p:.3f}"[1:]
    if use_letter:
        p = use_letter + " " + p
    if not keep_spaces:
        p = p.replace(" ", "")
    if no_equals:
        p = p.replace(" =", "").replace("=", "")
    return p


def format_r(r, use_letter="r", keep_spaces=True, no_equals=False):
    """Format a correlation / R² in APA style without a leading zero, e.g. ``r = -.34``.

    Parameters
    ----------
    r : float or any
        The coefficient (correlation, R², standardized coefficient, ...). Non-numeric
        input is returned unchanged.
    use_letter : str or False, default "r"
        Symbol put in front (e.g. ``"r"``, ``"R²"``, ``"β"``); a falsy value omits it.
    keep_spaces : bool, default True
        ``False`` removes all spaces (``r=-.34``).
    no_equals : bool, default False
        ``True`` removes the ``=`` sign (``r -.34``).

    Returns
    -------
    str or any
        The formatted string, or ``r`` unchanged if it is not a number.

    Notes
    -----
    The value is rounded to two decimals, the leading zero is dropped and the sign is
    kept (APA 7th ed., 2020: no zero before the decimal point for statistics that cannot
    exceed 1). Values with ``|r| < .01`` are shown as ``0.`` and values with
    ``|r| > .99`` as ``1.`` (``-1.``) instead of ``.00`` / ``1.00``; this also applies to
    values outside [-1, 1], which are *not* flagged. NaN and infinite values are reported
    as ``= NaN``. Like :func:`format_p` the result is a string for the final report.

    References
    ----------
    American Psychological Association. (2020). *Publication manual of the American
    Psychological Association* (7th ed.). https://doi.org/10.1037/0000165-000

    Examples
    --------
    >>> format_r(-0.341)
    'r = -.34'
    >>> format_r(0.5, use_letter="R²", keep_spaces=False)
    'R²=.50'
    >>> format_r(0.004)
    'r = 0.'
    """
    if not isinstance(r, Number):
        return r
    sign = "-" if r < 0.0 else ""
    if not np.isfinite(r):
        r = "= NaN"
    elif np.abs(r) < 0.01:
        r = "= 0."
    elif np.abs(r) > 0.99:
        r = "= " + sign + "1."
    else:
        r = "= " + sign + f"{np.abs(r):.2f}"[1:]
    if use_letter:
        r = use_letter + " " + r
    if not keep_spaces:
        r = r.replace(" ", "")
    if no_equals:
        r = r.replace(" =", "").replace("=", "")
    return r
