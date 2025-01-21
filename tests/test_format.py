import numpy as np

from apatools.format import *


def s(x):
    return f"{x:.2f}"


def test_format():
    r = 0.999
    fr = format_r(r, keep_spaces=True, use_letter="r", no_equals=False)
    assert fr == "r = 1."
    fr = format_r(r, keep_spaces=False, use_letter="r", no_equals=False)
    assert fr == "r=1."
    fr = format_r(r, keep_spaces=True, use_letter=False, no_equals=False)
    assert fr == "= 1."
    fr = format_r(r, keep_spaces=True, use_letter="r", no_equals=True)
    assert fr == "r 1."
    fr = format_r(r, keep_spaces=True, use_letter="R", no_equals=False)
    assert fr == "R = 1."
    r = 0.01
    fr = format_r(r, keep_spaces=True, use_letter="R", no_equals=False)
    assert fr == "R = .01"
    fr = format_r(r, keep_spaces=True, use_letter="R", no_equals=True)
    assert fr == "R .01"
    fr = format_r(r, keep_spaces=False, use_letter="R", no_equals=True)
    assert fr == "R.01"
    r = np.nan
    fr = format_r(r, keep_spaces=True, use_letter="R", no_equals=False)
    assert fr == "R = inf"
    fr = format_r(r, keep_spaces=False, use_letter="R:", no_equals=True)
    assert fr == "R:inf"
    r = 0.5
    fr = format_r(r, keep_spaces=True, no_equals=False)
    assert fr == "r = .50"


if __name__ == "__main__":
    test_format()
    print("Tests for metatools.format are PASSED!")
