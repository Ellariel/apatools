"""Tests of the private helpers shared by lm, slm, plm and sem (apatools/_common.py)."""

import numpy as np
import pandas as pd
import pytest

from apatools._common import (
    INTERCEPT_NAMES,
    df_check_intercept,
    df_standardize,
    format_significance,
    rename_intercept,
    round_columns,
    w_subset_rows,
)


def test_intercept_names_are_those_of_the_supported_packages():
    # spreg: CONSTANT, statsmodels / linearmodels: const, patsy and the reports: Intercept
    assert INTERCEPT_NAMES == ("CONSTANT", "const", "Intercept")


@pytest.mark.parametrize("name", INTERCEPT_NAMES)
def test_df_check_intercept_finds_every_name(name):
    df = pd.DataFrame({"x": [1.0, 2.0], name: [1.0, 1.0]})
    assert df_check_intercept(df) == name
    assert df_check_intercept(columns=df.columns) == name  # e.g. the index of a coefficient table
    assert df_check_intercept(columns=list(df.columns)) == name


def test_df_check_intercept_without_or_with_several_intercepts():
    assert df_check_intercept(pd.DataFrame({"x": [1.0, 2.0]})) is None
    # only exact names count, not names that contain one
    assert df_check_intercept(columns=["const_income", "Intercept2", "x"]) is None
    # the first of CONSTANT, const, Intercept wins
    assert df_check_intercept(columns=["Intercept", "const"]) == "const"
    assert df_check_intercept(columns=["Intercept", "const", "CONSTANT"]) == "CONSTANT"


@pytest.mark.parametrize("name", INTERCEPT_NAMES)
def test_rename_intercept(name):
    renamed = rename_intercept(pd.Index([name, "x1", "x2"]), "Intercept")
    assert list(renamed) == ["Intercept", "x1", "x2"]
    assert list(rename_intercept([name, "x"], "(Intercept)")) == ["(Intercept)", "x"]  # any sequence


def test_rename_intercept_leaves_everything_else_alone():
    # no intercept: the labels must come back unchanged (they used to become NaN)
    index = pd.Index(["x1", "x2"], name="term")
    renamed = rename_intercept(index)
    assert list(renamed) == ["x1", "x2"] and renamed.name == "term"
    # exact matches only: names that merely contain an intercept name are not touched
    tricky = ["const_income", "Intercept2", "myconst", "CONSTANT_x", "const"]
    assert list(rename_intercept(tricky, "Intercept")) == ["const_income", "Intercept2", "myconst", "CONSTANT_x", "Intercept"]
    # non-string labels
    assert list(rename_intercept([0, 1, "const"])) == [0, 1, "Intercept"]
    assert list(rename_intercept(pd.Index([]))) == []


def test_df_standardize():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 40.0, 80.0], "flag": [True, False, True, True], "text": list("wxyz")}
    )
    before = df.copy()
    z = df_standardize(df)
    pd.testing.assert_frame_equal(df, before)  # the input is not modified
    assert list(z.columns) == ["a", "b", "flag"]  # numeric and boolean columns only
    assert z[["a", "b"]].mean().abs().max() < 1e-12
    assert np.allclose(z[["a", "b"]].std(ddof=0), 1.0)
    assert z["a"].tolist() == pytest.approx([-1.3416, -0.4472, 0.4472, 1.3416], abs=1e-4)
    # True means "z", a callable is applied to every column
    pd.testing.assert_frame_equal(df_standardize(df, func=True), z)
    centred = df_standardize(df[["a", "b"]], func=lambda column: column - column.mean())
    assert centred.mean().abs().max() < 1e-12 and centred["b"].std(ddof=0) > 1
    for bad in ("minmax", False):
        with pytest.raises(NotImplementedError, match="not implemented"):
            df_standardize(df[["a", "b"]], func=bad)


def test_df_standardize_keeps_the_intercept_untouched_at_its_position():
    df = pd.DataFrame({"const": [1.0] * 4, "a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 1.0, 3.0, 9.0]})
    z = df_standardize(df)
    assert list(z.columns) == ["const", "a", "b"] and z["const"].tolist() == [1.0] * 4
    assert z["a"].std(ddof=0) == pytest.approx(1.0)
    middle = df_standardize(df[["a", "const", "b"]])
    assert list(middle.columns) == ["a", "const", "b"] and middle["const"].tolist() == [1.0] * 4
    # a frame that only holds the intercept is returned as it is
    only = pd.DataFrame({"CONSTANT": [1.0, 1.0, 1.0]})
    pd.testing.assert_frame_equal(df_standardize(only), only)


def test_w_subset_rows_is_importable_from_the_common_module():
    pytest.importorskip("libpysal")
    import libpysal

    w = libpysal.weights.lat2W(3, 3)
    w.transform = "r"
    sub = w_subset_rows(w, [0, 1, 3, 4])
    assert sub.n == 4 and sub.transform == "R"
    assert np.allclose(sub.sparse.sum(axis=1).A.ravel(), 1.0)  # row-standardized again


def _table():
    return pd.DataFrame(
        {"coef": [1.23456, 2.0], "se": [0.1111, 0.2222], "cil": [1.0123, 1.5], "cir": [1.4567, 2.5], "p-value": [0.0004, 0.2]},
        index=["a", "b"],
    )


def test_round_columns():
    t = round_columns(_table(), 2)
    assert t["coef"].tolist() == [1.23, 2.0] and t["se"].tolist() == [0.11, 0.22]
    assert t["p-value"].tolist() == [0.0004, 0.2]  # other columns are untouched
    before = _table()
    for falsy in (None, 0, False):  # no rounding
        pd.testing.assert_frame_equal(round_columns(_table(), falsy), before)
    assert round_columns(_table(), 1, columns=["se"])["coef"].tolist() == [1.23456, 2.0]


def test_format_significance_defaults_use_apa_style():
    t = format_significance(_table())
    assert t["p-value"].tolist() == ["<.001", ".200"]
    assert t["sig"].tolist() == ["***", ""]  # computed from the numbers, not from the formatted text


def test_format_significance_switches_and_callables():
    raw = format_significance(_table(), add_stars=False, format_pval=False)
    pd.testing.assert_frame_equal(raw, _table())
    stars_only = format_significance(_table(), format_pval=False)
    assert stars_only["p-value"].tolist() == [0.0004, 0.2] and "sig" in stars_only
    custom = format_significance(_table(), add_stars=lambda p: "!" if p < 0.05 else "", format_pval=lambda p: f"{p:.1e}")
    assert custom["sig"].tolist() == ["!", ""] and custom["p-value"].tolist() == ["4.0e-04", "2.0e-01"]
    renamed = format_significance(
        _table().rename(columns={"p-value": "p"}), p_column="p", stars_column="stars", format_pval=False
    )
    assert renamed["stars"].tolist() == ["***", ""]


def test_format_significance_works_in_place_and_returns_the_table():
    t = _table()
    assert format_significance(t) is t and "sig" in t


def test_format_significance_na_rep_and_empty_tables():
    t = pd.DataFrame({"p-value": [0.01, np.nan, 0.5]})
    with_na = format_significance(t.copy(), na_rep="")
    assert with_na["p-value"].tolist()[1] == "" and with_na["sig"].tolist()[1] == ""
    assert with_na["p-value"].tolist()[0] == ".010"
    seen = []
    format_significance(t.copy(), add_stars=lambda p: seen.append(p) or "", format_pval=False, na_rep="-")
    assert seen == [0.01, "-", 0.5]  # missing values are replaced before the callables see them
    empty = pd.DataFrame({"p-value": pd.Series([], dtype=float)})
    assert format_significance(empty)["sig"].tolist() == []
