import numpy as np
import pytest

from apatools.format import get_stars, format_p, format_r


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.5, "p = .500"),
        (0.049, "p = .049"),
        (0.009, "p = .009"),
        (0.001, "p = .001"),  # lower edge of exact reporting
        (0.00099, "p < .001"),  # APA: never p = .000
        (0.0004, "p < .001"),
        (1e-12, "p < .001"),
        (0.0, "p < .001"),
        (0.99, "p = .990"),
        (0.9994, "p = .999"),
        (0.9995, "p > .999"),  # would round to 1.000
        (1.0, "p > .999"),
        (np.float64(0.0123), "p = .012"),
        (np.nan, "p = NaN"),
        (np.inf, "p = NaN"),
        (-0.1, "p = NaN"),
    ],
)
def test_format_p_values(p, expected):
    assert format_p(p) == expected


def test_format_p_options():
    assert format_p(0.009, use_letter=False) == "= .009"
    assert format_p(0.009, use_letter="P") == "P = .009"
    assert format_p(0.009, keep_spaces=False) == "p=.009"
    assert format_p(0.0004, keep_spaces=False) == "p<.001"
    assert format_p(0.009, no_equals=True) == "p .009"
    assert format_p(0.009, keep_spaces=False, no_equals=True, use_letter=False) == ".009"
    # "<" and ">" are kept when the equals sign is dropped
    assert format_p(0.0004, no_equals=True) == "p < .001"
    assert format_p(0.0004, use_letter=False, keep_spaces=False, no_equals=True) == "<.001"


@pytest.mark.parametrize("value", ["n/a", "", None, [0.1]])
def test_non_numeric_input_is_passed_through(value):
    assert format_p(value) is value
    assert format_r(value) is value
    assert get_stars(value) is value


@pytest.mark.parametrize(
    "r, expected",
    [
        (0.5, "r = .50"),
        (-0.34, "r = -.34"),
        (0.01, "r = .01"),
        (0.0099, "r = 0."),
        (-0.001, "r = 0."),
        (0.0, "r = 0."),
        (0.99, "r = .99"),
        (0.991, "r = 1."),
        (0.999, "r = 1."),
        (-0.999, "r = -1."),  # the sign must survive rounding to +-1
        (-1.0, "r = -1."),
        (np.nan, "r = NaN"),
        (np.inf, "r = NaN"),
    ],
)
def test_format_r_values(r, expected):
    assert format_r(r) == expected


def test_format_r_options():
    assert format_r(0.999, keep_spaces=False) == "r=1."
    assert format_r(0.999, use_letter=False) == "= 1."
    assert format_r(0.999, no_equals=True) == "r 1."
    assert format_r(0.01, use_letter="R") == "R = .01"
    assert format_r(0.01, use_letter="R", no_equals=True) == "R .01"
    assert format_r(0.01, use_letter="R", keep_spaces=False, no_equals=True) == "R.01"
    assert format_r(np.nan, use_letter="R:", keep_spaces=False, no_equals=True) == "R:NaN"
    assert format_r(-0.5, use_letter="r", keep_spaces=False) == "r=-.50"


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, "***"),
        (0.00099, "***"),
        (0.001, "**"),  # thresholds are strict "<"
        (0.0099, "**"),
        (0.01, "*"),
        (0.0499, "*"),
        (0.05, "⁺"),
        (0.0999, "⁺"),
        (0.1, ""),
        (1.0, ""),
    ],
)
def test_get_stars(p, expected):
    assert get_stars(p) == expected


def test_get_stars_custom_symbols():
    kw = dict(p001="a", p01="b", p05="c", p10="d", p_="e")
    assert [get_stars(p, **kw) for p in (0.0001, 0.005, 0.03, 0.07, 0.5)] == list("abcde")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
    print("Tests for apatools.format are PASSED!")
