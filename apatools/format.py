import numpy as np
from numbers import Number


###############
#  formating  #
###############


def get_stars(p, p001="***", p01="**", p05="*", p10="⁺", p_=""):
    """Significance marker for a p-value.

    Thresholds: p < .001, p < .01, p < .05, p < .10, otherwise ``p_``.
    Non-numeric input is returned unchanged.
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
    use_letter : str or False
        Prefix such as ``"p"``; falsy omits it.
    keep_spaces : bool
        ``False`` removes all spaces (``p<.001``).
    no_equals : bool
        ``True`` drops the ``=`` sign (``p .009``).

    Values below .001 are reported as ``< .001`` and values that would round to
    1.000 as ``> .999``. Non-numeric input is returned unchanged; NaN, inf and
    negative values are reported as ``= NaN``.
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

    Values with |r| < .01 are shown as ``0.`` and |r| > .99 as ``1.`` (or ``-1.``).
    Parameters are as in :func:`format_p`. Non-numeric input is returned unchanged;
    NaN and inf are reported as ``= NaN``.
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
