"""The R engines: sem(engine="lavaan") and sem_plot(engine="lavaan") (apatools/r/lavaan.py).

Every test here runs R: they need Rscript with the R package lavaan (the diagram tests also semPlot) and the whole
module is skipped without them. The tests that need no R are in test_r_bridge.py.
"""

import numpy as np
import pandas as pd
import pytest

from apatools.r import bridge
from apatools.r.bridge import run_r

from .r_reference import installed, rscript

_RSCRIPT, _PACKAGES = rscript(), installed("lavaan", "semPlot")
if not _PACKAGES.get("lavaan"):
    pytest.skip("R with the package lavaan is not available", allow_module_level=True)


def test_the_messages_of_r_are_in_english_by_default(monkeypatch):
    import tempfile

    exe = _RSCRIPT

    code = 'message(Sys.getenv("LANGUAGE"))'
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LANGUAGE", "de")  # the system language does not matter
        monkeypatch.delenv(bridge.ENV_LANGUAGE, raising=False)
        assert run_r(code, tmp, rscript=exe).strip() == "en"
        monkeypatch.setenv(bridge.ENV_LANGUAGE, "fr")
        assert run_r(code, tmp, rscript=exe).strip() == "fr"


def test_run_r_errors_need_no_r_packages():
    import tempfile

    exe = _RSCRIPT

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ImportError, match=r"install.packages\('notapkg'\)"):
            run_r('stop("MISSING_R_PACKAGE:notapkg", call. = FALSE)', tmp, rscript=exe)
        with pytest.raises(RuntimeError, match="(?s)R failed:.*boom"):
            run_r('stop("boom")', tmp, rscript=exe)
        assert (
            run_r('message("hello")', tmp, rscript=exe).strip() == "hello"
        )  # stderr is returned
        with pytest.raises(RuntimeError, match="did not finish within 1 seconds"):
            run_r("Sys.sleep(30)", tmp, rscript=exe, timeout=1)


###########################
# integration tests (R)   #
###########################


@pytest.fixture(scope="module")
def need_semplot():
    if not _PACKAGES["semPlot"]:
        pytest.skip("the R package semPlot is not installed")


@pytest.fixture(scope="module")
def democracy():
    semopy = pytest.importorskip("semopy")
    example = semopy.examples.political_democracy
    return example.get_data(), example.get_model()


@pytest.fixture(scope="module")
def both_fits(democracy):
    from apatools.sem import sem

    data, formula = democracy
    semopy_fit = sem(data, formula, bootstrap=0, se_robust=False)
    lavaan_fit = sem(data, formula, engine="lavaan", se_robust=False, return_model=True)
    return semopy_fit, lavaan_fit


def test_the_lavaan_engine_agrees_with_semopy(both_fits):
    (stats_s, metrics_s), (stats_l, metrics_l, fit) = both_fits
    assert list(stats_l.columns) == list(stats_s.columns)
    assert (
        stats_l.attrs["latent"]
        == stats_s.attrs["latent"]
        == ["dem60", "dem65", "ind60"]
    )
    key = ["lval", "op", "rval"]
    assert set(map(tuple, stats_l[key].values)) == set(
        map(tuple, stats_s[key].values)
    )  # the same parameters
    both = stats_s.merge(stats_l, on=key, suffixes=("_s", "_l"))
    paths = both[both["op"] == "~"]
    # loadings and regressions: the same estimates, standardized estimates and standard errors of the latter
    assert np.allclose(paths["estimate_s"], paths["estimate_l"], atol=0.02)
    assert np.allclose(paths["est_std_s"], paths["est_std_l"], atol=2e-3)
    assert np.allclose(
        paths["se_std_s"], paths["se_std_l"], atol=2e-3
    )  # delta method, as lavaan's own
    assert np.allclose(
        paths["std_err_s"], paths["std_err_l"], atol=0.02, equal_nan=True
    )
    # the first loading of every latent variable is fixed to 1: no test of the raw loading, but of the standardized
    fixed = stats_l[(stats_l["op"] == "~") & stats_l["std_err"].isna()]
    assert sorted(fixed["lval"]) == ["x1", "y1", "y5"] and fixed["p-value"].isna().all()
    assert (fixed["p_std"] < 1e-6).all() and (fixed["se_std"] > 0).all()
    # fit statistics
    for name, tolerance in {
        "DoF": 0,
        "chi2": 0.01,
        "CFI": 2e-3,
        "TLI": 2e-3,
        "RMSEA": 2e-3,
        "NFI": 2e-3,
    }.items():
        assert metrics_l.loc[name, "value"] == pytest.approx(
            metrics_s.loc[name, "value"], abs=tolerance
        )
    assert metrics_l.loc["N", "value"] == 75 and "SRMR" in metrics_l.index
    assert fit.version and fit.converged and fit.warnings == []
    assert {"lhs", "op", "rhs", "est", "est.std"} <= set(fit.parameters.columns)
    assert fit.fit_measures["cfi"] == pytest.approx(metrics_l.loc["CFI", "value"])


def test_the_lavaan_tables_feed_sem_report_and_the_other_plots(both_fits):
    from apatools.sem import sem_plot, sem_report

    _, (stats, metrics, _) = both_fits
    report = sem_report(stats, metrics, add_stars=True)
    summary = report.iloc[0]["model"]
    assert (
        summary.startswith("χ²(35, N = 75) = 38.12")
        and "CFI = .99" in summary
        and "SRMR = " in summary
    )
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


def test_lavaan_engine_options(democracy):
    from apatools.sem import sem

    data, formula = democracy
    # lavaan arguments: all loadings free and the latent variances fixed to 1 (every loading is tested)
    stats, _ = sem(
        data, formula, engine="lavaan", se_robust=False, lavaan_args={"std.lv": True}
    )
    loadings = stats[
        (stats["op"] == "~")
        & stats["rval"].isin(["ind60", "dem60", "dem65"])
        & stats["lval"].str.match("^[xy]")
    ]
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
    assert (boot["std_err"].dropna() > 0).all() and not np.allclose(
        boot["std_err"], plain["std_err"], equal_nan=True
    )
    again, _ = sem(data, formula, engine="lavaan", bootstrap=20)
    pd.testing.assert_frame_equal(
        boot, again
    )  # the seed makes the bootstrap reproducible


def test_lavaan_engine_reports_problems(democracy):
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
    assert (
        fig.data.startswith(b"\x89PNG")
        and fig._repr_png_() == fig.data
        and fig._repr_svg_() is None
    )
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
        assert sem_plot(
            stats_l,
            engine="lavaan",
            layout=layout,
            show_stars=True,
            plot_covs=False,
            rotation=1,
        ).data
    assert sem_plot(
        stats_s, engine="lavaan", std_ests=False, format_pval=False
    ).data.startswith(b"\x89PNG")
    assert sem_plot(stats_l, engine="lavaan", return_fig=False) is None


def test_missing_values_are_passed_to_lavaan_as_missing(democracy):
    """Listwise deletion (the default of lavaan) is reported; FIML keeps all cases; both give the right numbers."""
    from apatools.sem import sem

    data, formula = democracy
    d = data.copy()
    rng = np.random.default_rng(0)
    rows = rng.choice(len(d), 8, replace=False)
    d.loc[rows[:4], "y1"] = np.nan
    d.loc[rows[4:], "x2"] = np.nan
    complete = d.dropna()
    assert len(complete) == 67
    with pytest.warns(UserWarning, match="lavaan used 67 of 75 cases"):
        stats, metrics = sem(d, formula, engine="lavaan")
    by_hand_stats, by_hand_metrics = sem(
        complete, formula, engine="lavaan"
    )  # no warning: nothing is missing
    assert metrics.loc["N", "value"] == 67 == by_hand_metrics.loc["N", "value"]
    np.testing.assert_allclose(
        stats["estimate"].to_numpy(float),
        by_hand_stats["estimate"].to_numpy(float),
        rtol=1e-8,
    )
    # FIML uses every case, and says nothing about deleted ones
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, fiml = sem(d, formula, engine="lavaan", method="FIML")
    assert fiml.loc["N", "value"] == 75
    # a model variable without any value, and a column outside the model that has none
    nothing = data.copy()
    nothing["y1"] = np.nan
    with pytest.raises(RuntimeError, match="missing"):
        sem(nothing, formula, engine="lavaan")
    outside = data.copy()
    outside["unused"] = np.nan
    assert sem(outside, formula, engine="lavaan")[1].loc["N", "value"] == 75


def test_infinite_values_are_rejected_with_a_clear_message(democracy):
    from apatools.sem import sem

    data, formula = democracy
    d = data.copy()
    d.loc[3, "y1"] = np.inf
    with pytest.raises(ValueError, match=r"\['y1'\] have infinite values"):
        sem(d, formula, engine="lavaan")
