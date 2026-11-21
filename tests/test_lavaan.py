"""The R bridge: sem(engine="lavaan") and sem_plot(engine="lavaan") (apatools/_lavaan.py).

The unit tests need no R. The integration tests need R (Rscript) with the packages lavaan (and semPlot
for the diagrams) and are skipped without them.
"""

import subprocess
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
    run_r,
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


def test_the_messages_of_r_are_in_english_by_default(monkeypatch):
    try:
        exe = find_rscript()
    except RuntimeError:
        pytest.skip("R is not installed")
    import tempfile

    code = 'message(Sys.getenv("LANGUAGE"))'
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LANGUAGE", "de")  # the system language does not matter
        monkeypatch.delenv(bridge.ENV_LANGUAGE, raising=False)
        assert run_r(code, tmp, rscript=exe).strip() == "en"
        monkeypatch.setenv(bridge.ENV_LANGUAGE, "fr")
        assert run_r(code, tmp, rscript=exe).strip() == "fr"


def test_run_r_errors_need_no_r_packages():
    pytest.importorskip("pandas")
    try:
        exe = find_rscript()
    except RuntimeError:
        pytest.skip("R is not installed")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ImportError, match=r"install.packages\('notapkg'\)"):
            run_r('stop("MISSING_R_PACKAGE:notapkg", call. = FALSE)', tmp, rscript=exe)
        with pytest.raises(RuntimeError, match="(?s)R failed:.*boom"):
            run_r('stop("boom")', tmp, rscript=exe)
        assert run_r('message("hello")', tmp, rscript=exe).strip() == "hello"  # stderr is returned
        with pytest.raises(RuntimeError, match="did not finish within 1 seconds"):
            run_r("Sys.sleep(30)", tmp, rscript=exe, timeout=1)


def test_plot_arguments_are_checked_before_r_is_started(monkeypatch):
    monkeypatch.setattr(bridge, "run_r", lambda *a, **k: pytest.fail("R must not be started"))
    args = (["x", "f"], {"f"}, [("f", "x", "0.5")], [])
    with pytest.raises(ValueError, match="Unsupported figure format 'gif'"):
        bridge.semplot(*args, format_fig="gif")
    with pytest.raises(ValueError, match="Unknown layout 'nope'"):
        bridge.semplot(*args, layout="nope")


###########################
# integration tests (R)   #
###########################


@pytest.fixture(scope="module")
def r_env():
    """The R packages that are installed: skips the test without R."""
    try:
        exe = find_rscript()
    except RuntimeError:
        pytest.skip("R is not installed")
    code = 'cat(requireNamespace("lavaan", quietly = TRUE), requireNamespace("semPlot", quietly = TRUE))'
    out = subprocess.run([exe, "-e", code], capture_output=True, text=True, check=False).stdout.split()
    return {"lavaan": out[:1] == ["TRUE"], "semPlot": out[1:] == ["TRUE"]}


@pytest.fixture(scope="module")
def need_lavaan(r_env):
    if not r_env["lavaan"]:
        pytest.skip("the R package lavaan is not installed")


@pytest.fixture(scope="module")
def need_semplot(r_env):
    if not (r_env["lavaan"] and r_env["semPlot"]):
        pytest.skip("the R packages lavaan and semPlot are not installed")


@pytest.fixture(scope="module")
def democracy():
    semopy = pytest.importorskip("semopy")
    example = semopy.examples.political_democracy
    return example.get_data(), example.get_model()


@pytest.fixture(scope="module")
def both_fits(need_lavaan, democracy):
    from apatools.sem import sem

    data, formula = democracy
    semopy_fit = sem(data, formula, bootstrap=0, se_robust=False)
    lavaan_fit = sem(data, formula, engine="lavaan", se_robust=False, return_model=True)
    return semopy_fit, lavaan_fit


def test_the_lavaan_engine_agrees_with_semopy(both_fits):
    (stats_s, metrics_s), (stats_l, metrics_l, fit) = both_fits
    assert list(stats_l.columns) == list(stats_s.columns)
    assert stats_l.attrs["latent"] == stats_s.attrs["latent"] == ["dem60", "dem65", "ind60"]
    key = ["lval", "op", "rval"]
    assert set(map(tuple, stats_l[key].values)) == set(map(tuple, stats_s[key].values))  # the same parameters
    both = stats_s.merge(stats_l, on=key, suffixes=("_s", "_l"))
    paths = both[both["op"] == "~"]
    # loadings and regressions: the same estimates, standardized estimates and standard errors of the latter
    assert np.allclose(paths["estimate_s"], paths["estimate_l"], atol=0.02)
    assert np.allclose(paths["est_std_s"], paths["est_std_l"], atol=2e-3)
    assert np.allclose(paths["se_std_s"], paths["se_std_l"], atol=2e-3)  # delta method, as lavaan's own
    assert np.allclose(paths["std_err_s"], paths["std_err_l"], atol=0.02, equal_nan=True)
    # the first loading of every latent variable is fixed to 1: no test of the raw loading, but of the standardized
    fixed = stats_l[(stats_l["op"] == "~") & stats_l["std_err"].isna()]
    assert sorted(fixed["lval"]) == ["x1", "y1", "y5"] and fixed["p-value"].isna().all()
    assert (fixed["p_std"] < 1e-6).all() and (fixed["se_std"] > 0).all()
    # fit statistics
    for name, tolerance in {"DoF": 0, "chi2": 0.01, "CFI": 2e-3, "TLI": 2e-3, "RMSEA": 2e-3, "NFI": 2e-3}.items():
        assert metrics_l.loc[name, "value"] == pytest.approx(metrics_s.loc[name, "value"], abs=tolerance)
    assert metrics_l.loc["N", "value"] == 75 and "SRMR" in metrics_l.index
    assert fit.version and fit.converged and fit.warnings == []
    assert {"lhs", "op", "rhs", "est", "est.std"} <= set(fit.parameters.columns)
    assert fit.fit_measures["cfi"] == pytest.approx(metrics_l.loc["CFI", "value"])


def test_the_lavaan_tables_feed_sem_report_and_the_other_plots(both_fits):
    from apatools.sem import sem_plot, sem_report

    _, (stats, metrics, _) = both_fits
    report = sem_report(stats, metrics, add_stars=True)
    summary = report.iloc[0]["model"]
    assert summary.startswith("χ²(35, N = 75) = 38.12") and "CFI = .99" in summary and "SRMR = " in summary
    assert report["p_std"].map(lambda v: isinstance(v, str)).all()
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    import matplotlib

    matplotlib.use("Agg")
    fig = sem_plot(stats, engine="networkx")
    labels = [t.get_text() for t in fig.axes[0].texts if t.get_text()[:1].isdigit()]
    assert len(labels) >= 20 and "0.92\np<.001" in labels
    import matplotlib.pyplot as plt

    plt.close("all")


def test_lavaan_engine_options(need_lavaan, democracy):
    from apatools.sem import sem

    data, formula = democracy
    # lavaan arguments: all loadings free and the latent variances fixed to 1 (every loading is tested)
    stats, _ = sem(data, formula, engine="lavaan", se_robust=False, lavaan_args={"std.lv": True})
    loadings = stats[(stats["op"] == "~") & stats["rval"].isin(["ind60", "dem60", "dem65"]) & stats["lval"].str.match("^[xy]")]
    assert loadings["p-value"].notna().all() and loadings["std_err"].notna().all()
    # another estimator, translated names, robust standard errors
    uls, uls_metrics = sem(data, formula, method="ULS", engine="lavaan")
    assert len(uls) == len(stats) and uls_metrics.loc["N", "value"] == 75
    robust, _ = sem(data, formula, engine="lavaan", se_robust=True)
    plain, _ = sem(data, formula, engine="lavaan", se_robust=False)
    assert not np.allclose(robust["std_err"], plain["std_err"], equal_nan=True)
    assert np.allclose(robust["estimate"], plain["estimate"])
    # FIML with missing values
    gappy = data.copy()
    gappy.iloc[::7, 3] = np.nan
    fiml, fiml_metrics = sem(gappy, formula, method="FIML", engine="lavaan")
    assert fiml_metrics.loc["N", "value"] == 75 and np.isfinite(fiml["estimate"]).all()
    # bootstrap standard errors
    boot, _ = sem(data, formula, engine="lavaan", bootstrap=20)
    assert (boot["std_err"].dropna() > 0).all() and not np.allclose(boot["std_err"], plain["std_err"], equal_nan=True)
    again, _ = sem(data, formula, engine="lavaan", bootstrap=20)
    pd.testing.assert_frame_equal(boot, again)  # the seed makes the bootstrap reproducible


def test_lavaan_engine_reports_problems(need_lavaan, democracy):
    from apatools.sem import sem

    data, formula = democracy
    with pytest.raises(NotImplementedError, match="Engine 'stata'"):
        sem(data, formula, engine="stata")
    with pytest.raises(RuntimeError, match="'no-rscript-here' was not found"):
        sem(data, formula, engine="lavaan", rscript="no-rscript-here")
    with pytest.raises(RuntimeError, match="R failed"):
        sem(data, "this is not =~ ~~ a model", engine="lavaan")
    with pytest.raises(RuntimeError, match="R failed"):
        sem(data, formula, engine="lavaan", lavaan_args={"no.such.argument": 1})
    # warnings of lavaan are Python warnings
    scaled = data.copy()
    scaled["y1"] = scaled["y1"] * 1e6
    with pytest.warns(UserWarning, match="lavaan: .*some observed variances"):
        sem(scaled, formula, engine="lavaan")


def _png_size(data):
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def test_sem_plot_with_semplot(need_semplot, both_fits, tmp_path):
    from apatools.sem import sem_plot

    (stats_s, _), (stats_l, _, _) = both_fits
    png = tmp_path / "diagram.png"
    fig = sem_plot(stats_l, engine="lavaan", save_to_file=str(png), dpi=60)
    assert fig.format == "png" and fig.path == str(png) and png.read_bytes() == fig.data
    assert fig.data.startswith(b"\x89PNG") and fig._repr_png_() == fig.data and fig._repr_svg_() is None
    assert "RFigure" in repr(fig)
    # dpi sets the resolution (same size in inches)
    big = sem_plot(stats_l, engine="lavaan", dpi=120)
    assert _png_size(big.data)[0] == pytest.approx(2 * _png_size(fig.data)[0], abs=2)
    # width / height (inches) and the other file types, by extension or by format_fig
    wide = sem_plot(stats_l, engine="lavaan", dpi=60, width=14, height=5)
    assert _png_size(wide.data) == (14 * 60, 5 * 60)
    pdf = sem_plot(stats_l, engine="lavaan", save_to_file=str(tmp_path / "diagram.pdf"))
    assert pdf.data.startswith(b"%PDF") and pdf._repr_png_() is None
    svg = sem_plot(stats_l, engine="lavaan", format_fig="svg")
    assert svg.path is None and "<svg" in svg._repr_svg_()
    other = tmp_path / "copy.png"
    assert fig.save(str(other)) == str(other) and other.read_bytes() == fig.data
    # the layouts, the options of the other engines, tables of the semopy engine, return_fig
    for layout in ("tree2", "circle", "spring"):
        assert sem_plot(stats_l, engine="lavaan", layout=layout, show_stars=True, plot_covs=False, rotation=1).data
    assert sem_plot(stats_s, engine="lavaan", std_ests=False, format_pval=False).data.startswith(b"\x89PNG")
    assert sem_plot(stats_l, engine="lavaan", return_fig=False) is None
