"""The R bridge (apatools/r/bridge.py and apatools/r/lavaan.py) without R: literals, lookup of Rscript, estimator names, the conversion of
the lavaan tables, argument checks. The tests that run R are in test_lavaan.py."""

import sys

import numpy as np
import pandas as pd
import pytest

import apatools.r.lavaan as lavaan_module
from apatools.r import bridge
from apatools.r.bridge import find_rscript, r_literal
from apatools.r.lavaan import LAVAAN_ESTIMATORS, convert_lavaan, lavaan_estimator

###########################
# unit tests (no R)       #
###########################


def test_r_literal():
    assert r_literal(None) == "NULL"
    assert (r_literal(True), r_literal(False), r_literal(np.bool_(True))) == (
        "TRUE",
        "FALSE",
        "TRUE",
    )
    assert (r_literal(5), r_literal(np.int64(-2))) == ("5L", "-2L")
    assert r_literal(0.5) == "0.5" and r_literal(np.float64(1e-9)) == "1e-09"
    assert (r_literal(float("nan")), r_literal(float("inf")), r_literal(-np.inf)) == (
        "NA_real_",
        "Inf",
        "-Inf",
    )
    assert r_literal("x1") == '"x1"'
    assert (
        r_literal('a "quoted"\nname \\ é') == '"a \\"quoted\\"\\nname \\\\ \\u00e9"'
    )  # valid R escapes
    assert (
        r_literal(["x1", "x2"]) == 'c("x1", "x2")' and r_literal((1, 2)) == "c(1L, 2L)"
    )
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
    with pytest.raises(
        RuntimeError, match=r"no-such-rscript-anywhere.*APATOOLS_RSCRIPT"
    ):
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
    assert {
        k: lavaan_estimator(k, {})[0] for k in LAVAAN_ESTIMATORS
    } == LAVAAN_ESTIMATORS
    assert lavaan_estimator("mlw", {}) == ("ML", {})  # case does not matter
    assert lavaan_estimator("FIML", {}) == ("ML", {"missing": "fiml"})
    assert lavaan_estimator("FIML", {"missing": "listwise"}) == (
        "ML",
        {},
    )  # the user's choice wins
    assert lavaan_estimator("WLSMV", {}) == (
        "WLSMV",
        {},
    )  # other lavaan estimators pass through
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
            "name": [
                "df",
                "chisq",
                "pvalue",
                "cfi",
                "rmsea",
                "srmr",
                "aic",
                "gfi",
                "npar",
            ],
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
    assert metrics.index.tolist() == [
        "N",
        "DoF",
        "chi2",
        "chi2 p-value",
        "CFI",
        "RMSEA",
        "SRMR",
        "AIC",
    ]
    assert (
        metrics.loc["N", "value"] == 120
        and metrics.loc["chi2", "value"] == 7.5
        and metrics.loc["DoF", "value"] == 5
    )

    pe["group"] = [1, 1, 1, 1, 2, 2, 2, 2]
    with pytest.raises(NotImplementedError, match="several groups"):
        convert_lavaan(pe, ss, measures, 120, ["f"])


def test_plot_arguments_are_checked_before_r_is_started(monkeypatch):
    monkeypatch.setattr(
        bridge, "run_r", lambda *a, **k: pytest.fail("R must not be started")
    )
    args = (["x", "f"], {"f"}, [("f", "x", "0.5")], [])
    with pytest.raises(ValueError, match="Unsupported figure format 'gif'"):
        lavaan_module.semplot(*args, format_fig="gif")
    with pytest.raises(ValueError, match="Unknown layout 'nope'"):
        lavaan_module.semplot(*args, layout="nope")


###########################
# plm(backend="splm")     #
###########################


def _panel(with_weights=False):
    import libpysal

    from apatools.plm import make_panel

    rows = [(e, t) for t in range(3) for e in range(4)]  # period blocks of 4 entities
    df = pd.DataFrame(rows, columns=["e", "t"])
    df["y"] = np.arange(12, dtype=float) ** 1.1
    df["x1"] = np.cos(np.arange(12))
    w = None
    if with_weights:
        w = libpysal.weights.lat2W(2, 2, rook=True)
        w.transform = "r"
    return make_panel(df, "y", ["x1"], entity="e", time="t", w=w)


def _r_tables(panel, names, values, se, loglik=np.nan, df_resid=np.nan, rsq=np.nan):
    """The tables R writes (terms, cov, obs, info), with the observations in the id-major order of R."""
    terms = pd.DataFrame({"term": names, "estimate": values, "se": se})
    cov = pd.DataFrame(np.diag(np.square(se)), columns=names)
    position = np.arange(len(panel.y))  # row of the panel layout
    e, t = position % panel.N, position // panel.N
    order = np.lexsort((t, e))  # R sorts by entity, then by period
    obs = pd.DataFrame(
        {
            "e": e[order],
            "t": t[order],
            "resid": position[order].astype(float),
            "fitted": np.nan,
        }
    )
    info = pd.Series(
        {
            "plm": "2.6",
            "splm": "1.5",
            "loglik": loglik,
            "df_resid": df_resid,
            "rsq": rsq,
            "f": np.nan,
            "f_df1": np.nan,
            "f_df2": np.nan,
        }
    )
    return terms, cov, obs, info


def test_splm_conversion_uses_the_names_and_statistics_of_apatools():
    from apatools.r.splm import _convert

    panel = _panel(with_weights=True)
    names = ["lambda", "v1"]  # splm: the spatial lag coefficient is "lambda"
    terms, cov, obs, info = _r_tables(
        panel, names, [0.3, 1.5], [0.1, 0.5], loglik=-20.0
    )
    fit = _convert("slm_fe", panel, "entity", terms, cov, obs, info, {"v1": "x1"}, [])
    assert list(fit.table.index) == ["rho", "x1"]  # apatools and spreg: "rho"
    assert fit.stat_name == "z" and fit.table.loc["x1", "stat"] == pytest.approx(3.0)
    assert fit.table.loc["x1", "p-value"] == pytest.approx(
        0.0026998, rel=1e-4
    )  # normal distribution
    assert list(fit.cov.index) == list(fit.cov.columns) == ["rho", "x1"]
    assert fit.metrics["llf"] == -20.0 and fit.metrics["aic"] == pytest.approx(
        2 * 2 + 40
    )
    # the residuals are put back in the layout of the panel (period blocks) from R's order (entity blocks)
    np.testing.assert_array_equal(fit.resid, np.arange(12.0))

    terms, cov, obs, info = _r_tables(
        panel,
        ["(Intercept)", "v1", "rho"],
        [0.1, 1.5, 0.4],
        [0.1, 0.5, 0.2],
        loglik=-20.0,
    )
    fit = _convert(
        "sem_pooled", panel, "entity", terms, cov, obs, info, {"v1": "x1"}, []
    )
    assert list(fit.table.index) == [
        "const",
        "x1",
        "lambda",
    ]  # splm: the spatial error coefficient is "rho"


def test_splm_conversion_of_a_non_spatial_model_uses_the_t_distribution():
    from apatools.r.splm import _convert

    panel = _panel()
    terms, cov, obs, info = _r_tables(panel, ["v1"], [1.5], [0.5], df_resid=8, rsq=0.5)
    fit = _convert("fe", panel, "entity", terms, cov, obs, info, {"v1": "x1"}, [])
    assert fit.stat_name == "t" and fit.table.loc["x1", "p-value"] == pytest.approx(
        0.01707, rel=1e-3
    )
    assert fit.metrics["r_sq"] == 0.5 and fit.metrics["r_sq_type"] == "within"
    # adjusted R²: 12 observations, 4 + 1 estimated entity effects and slopes: df_resid = 8 -> absorbed = 12 - 8 - 1 = 3
    assert fit.metrics["r_sq_adj"] == pytest.approx(1 - 0.5 * (12 - 3) / 8)


def test_splm_backend_reports_unsupported_requests_before_starting_r():
    from apatools.plm import plm, pred_metrics

    rows = [(e, t, e + t + (e * t) % 3, e - t) for t in range(3) for e in range(4)]
    df = pd.DataFrame(rows, columns=["e", "t", "y", "x1"]).astype(float)
    kwargs = {"entity": "e", "time": "t", "backend": "splm", "verbose": False}
    with pytest.raises(NotImplementedError, match="GMM"):
        plm(df, "y", ["x1"], model="sem_pooled_gm", **kwargs)
    with pytest.raises(NotImplementedError, match="slx_lags"):
        plm(df, "y", ["x1"], model="fe", slx_lags=1, **kwargs)
    with pytest.raises(ValueError, match="needs spatial weights"):
        plm(df, "y", ["x1"], model="slm_fe", **kwargs)
    with pytest.raises(ValueError, match="backend"):
        plm(
            df,
            "y",
            ["x1"],
            model="fe",
            entity="e",
            time="t",
            backend="nope",
            verbose=False,
        )
    with pytest.warns(UserWarning, match="backend='splm'"):
        out = pred_metrics([object()], _panel(), backend="splm")
    assert np.isnan(out[0]["pred_loo_r_sq"])
