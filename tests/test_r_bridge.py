"""The R bridge (apatools/_lavaan.py) without R: literals, lookup of Rscript, estimator names, the conversion of
the lavaan tables, argument checks. The tests that run R are in test_lavaan.py."""

import sys

import numpy as np
import pandas as pd
import pytest

import apatools._lavaan as bridge
from apatools._lavaan import (
    LAVAAN_ESTIMATORS,
    convert_lavaan,
    find_rscript,
    lavaan_estimator,
    r_literal,
)

###########################
# unit tests (no R)       #
###########################


def test_r_literal():
    assert r_literal(None) == "NULL"
    assert (r_literal(True), r_literal(False), r_literal(np.bool_(True))) == ("TRUE", "FALSE", "TRUE")
    assert (r_literal(5), r_literal(np.int64(-2))) == ("5L", "-2L")
    assert r_literal(0.5) == "0.5" and r_literal(np.float64(1e-9)) == "1e-09"
    assert (r_literal(float("nan")), r_literal(float("inf")), r_literal(-np.inf)) == ("NA_real_", "Inf", "-Inf")
    assert r_literal("x1") == '"x1"'
    assert r_literal('a "quoted"\nname \\ é') == '"a \\"quoted\\"\\nname \\\\ \\u00e9"'  # valid R escapes
    assert r_literal(["x1", "x2"]) == 'c("x1", "x2")' and r_literal((1, 2)) == "c(1L, 2L)"
    assert r_literal(np.array([0.5, 1.5])) == "c(0.5, 1.5)"
    assert r_literal([["a", "b"], ["c"]]) == 'list(c("a", "b"), c("c"))'
    assert r_literal({"a": 1, "b.c": [True]}) == 'list("a" = 1L, "b.c" = c(TRUE))'
    with pytest.raises(TypeError, match="object"):
        r_literal(object())


def test_find_rscript_order_and_errors(monkeypatch):
    monkeypatch.delenv(bridge.ENV_RSCRIPT, raising=False)
    # an explicit argument wins over the environment variable
    monkeypatch.setenv(bridge.ENV_RSCRIPT, sys.executable)
    assert find_rscript() == find_rscript(sys.executable)
    other = find_rscript(sys.executable)
    monkeypatch.setenv(bridge.ENV_RSCRIPT, "no-such-rscript-anywhere")
    assert find_rscript(sys.executable) == other
    # a path that does not exist is an error naming both places to fix it
    with pytest.raises(RuntimeError, match=r"no-such-rscript-anywhere.*APATOOLS_RSCRIPT"):
        find_rscript()
    with pytest.raises(RuntimeError, match="'nope' was not found"):
        find_rscript("nope")
    # nothing anywhere: installation hints
    monkeypatch.delenv(bridge.ENV_RSCRIPT)
    monkeypatch.delenv("R_HOME", raising=False)
    monkeypatch.setattr(bridge.shutil, "which", lambda *_: None)
    monkeypatch.setattr(bridge.glob, "glob", lambda *_: [])
    monkeypatch.setattr(bridge.os.path, "isfile", lambda *_: False)
    with pytest.raises(RuntimeError, match=r"R was not found.*install.packages"):
        find_rscript()


def test_lavaan_estimator_translates_the_names_of_semopy():
    assert {k: lavaan_estimator(k, {})[0] for k in LAVAAN_ESTIMATORS} == LAVAAN_ESTIMATORS
    assert lavaan_estimator("mlw", {}) == ("ML", {})  # case does not matter
    assert lavaan_estimator("FIML", {}) == ("ML", {"missing": "fiml"})
    assert lavaan_estimator("FIML", {"missing": "listwise"}) == ("ML", {})  # the user's choice wins
    assert lavaan_estimator("WLSMV", {}) == ("WLSMV", {})  # other lavaan estimators pass through
    assert lavaan_estimator("MLR", {}) == ("MLR", {})


def _lavaan_frames():
    pe = pd.DataFrame(
        {
            "lhs": ["f", "f", "y", "f", "y", "y", "g", "y"],
            "op": ["=~", "=~", "~", "~~", "~~", "~1", ":=", "|"],
            "rhs": ["a", "b", "f", "f", "y", "", "", "t1"],
            "est": [1.0, 0.8, 0.5, 1.2, 0.7, 0.1, 9.0, 9.0],
            "se": [0.0, 0.1, 0.05, 0.2, 0.1, 0.05, 1.0, 1.0],
            "z": [np.nan, 8.0, 10.0, 6.0, 7.0, 2.0, 1.0, 1.0],
            "pvalue": [np.nan, 0.0, 0.0, 1e-9, 1e-12, 0.04, 0.5, 0.5],
        }
    )
    ss = pe.copy()
    ss["est.std"] = [0.9, 0.7, 0.4, 1.0, 0.5, 0.2, 9.0, 9.0]
    ss["se"] = [0.02, 0.05, 0.03, 0.0, 0.04, 0.03, 1.0, 1.0]
    ss["z"] = ss["est.std"] / ss["se"].replace(0, np.nan)
    ss["pvalue"] = [0.0, 0.0, 1e-5, np.nan, 0.01, 0.2, 0.5, 0.5]
    measures = pd.DataFrame(
        {
            "name": ["df", "chisq", "pvalue", "cfi", "rmsea", "srmr", "aic", "gfi", "npar"],
            "value": [5.0, 7.5, 0.19, 0.99, 0.04, 0.03, 100.0, np.nan, 9.0],
        }
    )
    return pe, ss, measures


def test_convert_lavaan_gives_the_format_of_sem():
    pe, ss, measures = _lavaan_frames()
    stats, metrics = convert_lavaan(pe, ss, measures, nobs=120, latent=["f"])
    assert list(stats.columns) == [
        "lval", "op", "rval", "estimate", "est_std", "std_err", "z-value", "p-value", "se_std", "z_std", "p_std",
    ]  # fmt: skip
    assert stats.attrs["latent"] == ["f"]
    # loadings point from the latent variable to the indicator as semopy reports them: indicator ~ latent
    assert stats[["lval", "op", "rval"]].values.tolist() == [
        ["a", "~", "f"],
        ["b", "~", "f"],
        ["y", "~", "f"],
        ["f", "~~", "f"],
        ["y", "~~", "y"],
        ["y", "~", "1"],  # the intercept pseudo-variable of semopy
    ]  # thresholds ("|") and defined parameters (":=") are left out
    # the loading fixed to 1 has no standard error, z-value and p-value, but a standardized one
    first = stats.iloc[0]
    assert first["estimate"] == 1.0 and first["est_std"] == 0.9
    assert np.isnan([first["std_err"], first["z-value"], first["p-value"]]).all()
    assert first["se_std"] == 0.02 and first["p_std"] == 0.0
    assert stats.iloc[1]["std_err"] == 0.1 and stats.iloc[1]["p-value"] == 0.0
    # the standardized variance of a latent variable is a constant: se 0 is reported as missing...
    assert stats.iloc[3]["est_std"] == 1.0 and stats.iloc[3]["se_std"] == 0.0
    # fit measures under the names of semopy; unavailable ones (NaN) and unknown ones are left out
    assert metrics.index.tolist() == ["N", "DoF", "chi2", "chi2 p-value", "CFI", "RMSEA", "SRMR", "AIC"]
    assert metrics.loc["N", "value"] == 120 and metrics.loc["chi2", "value"] == 7.5 and metrics.loc["DoF", "value"] == 5

    pe["group"] = [1, 1, 1, 1, 2, 2, 2, 2]
    with pytest.raises(NotImplementedError, match="several groups"):
        convert_lavaan(pe, ss, measures, 120, ["f"])


def test_plot_arguments_are_checked_before_r_is_started(monkeypatch):
    monkeypatch.setattr(bridge, "run_r", lambda *a, **k: pytest.fail("R must not be started"))
    args = (["x", "f"], {"f"}, [("f", "x", "0.5")], [])
    with pytest.raises(ValueError, match="Unsupported figure format 'gif'"):
        bridge.semplot(*args, format_fig="gif")
    with pytest.raises(ValueError, match="Unknown layout 'nope'"):
        bridge.semplot(*args, layout="nope")
