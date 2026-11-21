import shutil

import numpy as np
import pandas as pd
import pytest
import semopy
from scipy.stats import norm

from apatools.citation import cite
from apatools.sem import sem, sem_report, semopy_plot, sem_plot


def test_sem():
    # https://semopy.com/tutorial.html
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    assert len(data) == 75

    print(cite("sem")[0])

    stats, metrics, model = sem(
        data,
        formula,
        method="MLW",  # MLW ULS GLS FIML DWLS WLS
        solver="SLSQP",
        bootstrap=10,
        se_robust=True,
        return_model=True,
        seed=13,
    )
    assert model is not None
    # the fit itself is deterministic ...
    plain, plain_metrics = sem(data, formula, bootstrap=0, seed=13)
    assert f"{plain['estimate'].iloc[0]:.3f}" == "1.482"
    # ... the bias-corrected estimate depends on the random resamples (and thereby on the numpy/pandas
    # versions), so it is only checked loosely (1.252 with numpy 2.0-2.2 / pandas 2.3)
    assert abs(stats["estimate"].iloc[0] - plain["estimate"].iloc[0]) < 0.5

    # the reported fit of the deterministic fit is checked exactly ...
    results = sem_report(plain, plain_metrics, format_pval=True, add_stars=True)
    assert (
        results.iloc[0]["model"]
        == "χ²(35, N = 75) = 38.125, p = .329, CFI = .995, GFI = .948, AGFI = .918, NFI = .948, TLI = .993, RMSEA = 0.035, AIC = 60.983, BIC = 132.825, LogLik = 0.508"
    )
    # ... the one after the bootstrap bias correction only for its structure: AIC, BIC and LogLik are
    # computed from the corrected (random) estimates, e.g. AIC = 60.872 / 60.874 depending on the versions
    results = sem_report(stats, metrics, format_pval=True, add_stars=True)
    summary = results.iloc[0]["model"]
    assert summary.startswith(
        "χ²(35, N = 75) = 38.125, p = .329, CFI = .995, GFI = .948, AGFI = .918, NFI = .948, "
        "TLI = .993, RMSEA = 0.035, AIC = 60.8"
    )
    assert summary.count("BIC = 132.7") == 1 and "LogLik = 0.56" in summary
    print(results)


@pytest.mark.skipif(
    shutil.which("dot") is None, reason="the Graphviz executables are not installed"
)
def test_sem_plot(tmp_path):
    pytest.importorskip("graphviz")
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    stats, metrics = sem(data, formula, bootstrap=0, seed=13)

    fig = semopy_plot(
        stats,
        save_to_file=str(tmp_path / "fig1.png"),
        plot_covs=True,
        std_ests=True,
        plot_ests=True,
        layout="dot",
        latshape="circle",
        show_stars=False,
        format_pval=True,
        format_fig="png",
        decimal=3,
        seed=13,
    )
    assert fig is not None
    assert (tmp_path / "fig1.png").stat().st_size > 0

    fig = sem_plot(
        stats,
        save_to_file=str(tmp_path / "fig2.png"),
        plot_covs=True,
        std_ests=True,
        show_stars=True,
        format_pval=False,
        format_fig="png",
        engine="semopy",
        return_fig=True,
        decimal=3,
        dpi=600,
        seed=13,
    )
    assert fig is not None
    assert (tmp_path / "fig2.png").stat().st_size > 0


def test_sem_report_and_plot_details():
    import pandas as pd
    import pytest

    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    stats, metrics = sem(data, formula, bootstrap=0, seed=13)

    # latent variables are remembered for plotting and are not indicators
    assert set(stats.attrs["latent"]) == {"dem60", "dem65", "ind60"}
    # N is the first metric
    assert metrics.index[0] == "N" and metrics.loc["N", "value"] == 75

    # report is built by metric name, in APA style, and does not alter its input
    before = stats.copy()
    rep = sem_report(stats, metrics, format_pval=True, add_stars=True)
    pd.testing.assert_frame_equal(stats, before)
    summary = rep.iloc[0]["model"]
    assert summary.startswith("χ²(35, N = 75) = ")
    for name in ("CFI", "TLI", "RMSEA", "AIC", "BIC"):
        assert f"{name} = " in summary
    assert "CFI = ." in summary  # no leading zero for bounded fit indices
    assert rep["model"].iloc[1:].isna().all()

    # values outside (-1, 1) keep their sign / digits
    m2 = metrics.copy()
    m2.loc["AGFI", "value"] = -0.123
    m2.loc["GFI", "value"] = 1.0
    summary2 = sem_report(stats, m2).iloc[0]["model"]
    assert "AGFI = -.123" in summary2
    assert "GFI = 1.0" in summary2

    # unknown plot engine fails loudly instead of with UnboundLocalError
    with pytest.raises(NotImplementedError, match="engine 'unknown'"):
        sem_plot(stats, engine="unknown")
    with pytest.raises(TypeError):  # the old name of the option is gone
        sem_plot(stats, method="semopy", engine="networkx", save_to_file=None, plot_ests=True)



########################################
# the "networkx" engine of sem_plot    #
########################################


@pytest.fixture(scope="module")
def democracy_fit():
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    return sem(data, formula, bootstrap=0, seed=13)


@pytest.fixture(scope="module")
def democracy_stats(democracy_fit):
    return democracy_fit[0]


@pytest.fixture
def mpl():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_sem_plot_networkx_draws_the_diagram(mpl, democracy_stats, tmp_path):
    from matplotlib.figure import Figure
    from matplotlib.text import Annotation

    path = tmp_path / "diagram.png"
    fig = sem_plot(democracy_stats, engine="networkx", save_to_file=str(path), dpi=50)
    assert isinstance(fig, Figure) and path.stat().st_size > 0
    ax = fig.axes[0]
    arrows = [t for t in ax.texts if isinstance(t, Annotation)]
    texts = [t.get_text() for t in ax.texts if not isinstance(t, Annotation)]
    # every variable of the model is drawn once, latent variables as circles, observed ones as boxes
    variables = {"ind60", "dem60", "dem65", *[f"x{i}" for i in (1, 2, 3)], *[f"y{i}" for i in range(1, 9)]}
    assert variables <= set(texts) and all(texts.count(v) == 1 for v in variables)
    shapes = {
        t.get_text(): t.get_bbox_patch().get_boxstyle().__class__.__name__
        for t in ax.texts
        if t.get_text() in variables
    }
    assert {v for v, k in shapes.items() if k == "Circle"} == {"ind60", "dem60", "dem65"}
    assert {v for v, k in shapes.items() if k == "Square"} == variables - {"ind60", "dem60", "dem65"}
    # one arrow per regression / measurement edge and per covariance, labelled with estimate and p-value
    edges = democracy_stats[democracy_stats["op"] == "~"]
    covs = democracy_stats[(democracy_stats["op"] == "~~") & (democracy_stats["lval"] != democracy_stats["rval"])]
    assert len(arrows) == len(edges) + len(covs)
    assert len(texts) - len(variables) == len(edges) + len(covs)  # one label per edge
    assert sum("p<.001" in t for t in texts) > 0 and sum(t.startswith("0.") for t in texts) > 5
    assert not [t for t in texts if t.startswith("nan")]


def test_sem_plot_networkx_options(mpl, democracy_stats, tmp_path):
    import matplotlib.image as mpimg

    base = sem_plot(democracy_stats, engine="networkx")
    n_base = len(base.axes[0].texts)
    # stars instead of p-values, raw estimates, no covariances
    fig = sem_plot(democracy_stats, engine="networkx", show_stars=True, std_ests=False, plot_covs=False, decimal=1)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert len(texts) < n_base and any(t.endswith("***") for t in texts) and not any("p<" in t for t in texts)
    # dpi sets the resolution of the file, the file type follows the extension (format_fig if there is none)
    low, high = tmp_path / "low.png", tmp_path / "high.png"
    sem_plot(democracy_stats, engine="networkx", save_to_file=str(low), dpi=40)
    sem_plot(democracy_stats, engine="networkx", save_to_file=str(high), dpi=80)
    assert mpimg.imread(high).shape[0] == pytest.approx(2 * mpimg.imread(low).shape[0], rel=0.05)
    svg = tmp_path / "diagram.svg"
    sem_plot(democracy_stats, engine="networkx", save_to_file=str(svg))
    assert svg.read_text(encoding="utf-8").lstrip().startswith(("<?xml", "<svg"))
    noext = tmp_path / "diagram"
    sem_plot(democracy_stats, engine="networkx", save_to_file=str(noext), format_fig="pdf")
    assert noext.read_bytes().startswith(b"%PDF")
    # figsize and fontsize
    fig = sem_plot(democracy_stats, engine="networkx", figsize=(5, 4), fontsize=7)
    assert tuple(fig.get_size_inches()) == (5, 4) and fig.axes[0].texts[0].get_fontsize() == 7
    # return_fig=False closes the figure
    before = set(mpl.get_fignums())
    assert sem_plot(democracy_stats, engine="networkx", return_fig=False) is None
    assert set(mpl.get_fignums()) == before


@pytest.mark.parametrize("layout", ["layered", "spring", "kamada_kawai", "circular"])
def test_sem_plot_networkx_layouts(mpl, democracy_stats, layout):
    fig = sem_plot(democracy_stats, engine="networkx", layout=layout)
    assert len(fig.axes[0].texts) > 20
    again = sem_plot(democracy_stats, engine="networkx", layout=layout)
    positions = lambda f: [t.get_position() for t in f.axes[0].texts]  # noqa: E731
    assert positions(fig) == positions(again)  # reproducible (the spring layout is seeded)
    with pytest.raises(ValueError, match="Unknown layout"):
        sem_plot(democracy_stats, engine="networkx", layout="nope")


def test_layered_positions_put_predictors_left_of_outcomes():
    nx = pytest.importorskip("networkx")
    from apatools.sem import _layered_positions

    arrows = [("a", "m"), ("b", "m"), ("m", "y1"), ("m", "y2"), ("a", "y3")]
    pos = _layered_positions(nx, ["a", "b", "m", "y1", "y2", "y3"], arrows)
    assert all(pos[r][0] < pos[l][0] for r, l in arrows)
    # layer = longest path from a variable without predictors
    assert pos["a"][0] == pos["b"][0] == 0 and pos["y1"][0] == pos["y2"][0] == 2 * pos["m"][0]
    assert pos["m"][0] == pos["y3"][0] > 0
    assert len({p for p in pos.values()}) == len(pos)  # no two variables on the same spot
    # a variable without predictors is moved next to its outcome, not left at the far left
    pos = _layered_positions(nx, ["p", "q", "r", "s", "x"], [("p", "q"), ("q", "r"), ("r", "s"), ("x", "s")])
    assert pos["x"][0] == pos["r"][0]
    # feedback loops (non-recursive models) share a layer instead of failing
    pos = _layered_positions(nx, ["a", "b", "c"], [("a", "b"), ("b", "a"), ("b", "c")])
    assert pos["a"][0] == pos["b"][0] < pos["c"][0]


def test_sem_plot_networkx_without_latent_metadata_and_with_feedback(mpl):
    import pandas as pd
    import numpy as np

    stats = pd.DataFrame(
        {
            "lval": ["y", "z", "y", "z"],
            "op": ["~", "~", "~~", "~~"],
            "rval": ["x", "y", "z", "y"],
            "estimate": [0.5, 0.4, 0.1, 0.2],
            "est_std": [0.5, 0.4, 0.1, 0.2],
            "p-value": [0.001, 0.2, np.nan, 0.03],
        }
    )
    fig = sem_plot(stats, engine="networkx")  # no stats.attrs["latent"]: right-hand variables are latent
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert {"x", "y", "z"} <= set(texts) and "0.1" in texts  # the covariance without p-value is labelled


def test_networkx_plot_names_the_extra_when_a_package_is_missing(monkeypatch, democracy_stats):
    import sys

    from apatools.sem import networkx_plot

    for missing in ("networkx", "matplotlib.pyplot"):
        with monkeypatch.context() as m:
            m.setitem(sys.modules, missing, None)
            with pytest.raises(ImportError, match=r"apatools\[networkx\]"):
                networkx_plot(democracy_stats)


##############################################
# inference for the standardized estimates   #
##############################################


def test_standardized_estimates_have_standard_errors_and_p_values(democracy_stats):
    stats = democracy_stats
    assert {"est_std", "se_std", "z_std", "p_std"} <= set(stats.columns)
    paths = stats[stats["op"] == "~"]
    # the first loading of every latent variable is fixed to 1: no test of the raw loading ...
    fixed = paths[paths["std_err"].isna()]
    assert set(fixed["lval"]) == {"x1", "y1", "y5"} and fixed["p-value"].isna().all()
    # ... but its standardized value is estimated and has a standard error and a p-value
    assert fixed["se_std"].between(0.01, 0.1).all() and (fixed["p_std"] < 1e-6).all()
    assert np.allclose(paths["z_std"], paths["est_std"] / paths["se_std"])
    assert np.allclose(paths["p_std"], 2 * norm.sf(np.abs(paths["z_std"])))
    # regression values (expected information): x1, x2, x3 on ind60
    plain = sem(
        semopy.examples.political_democracy.get_data(),
        semopy.examples.political_democracy.get_model(),
        bootstrap=0,
        se_robust=False,
    )[0]
    loadings = plain[(plain["rval"] == "ind60") & plain["lval"].isin(["x1", "x2", "x3"])]
    assert loadings["est_std"].round(3).tolist() == [0.92, 0.973, 0.872]
    assert loadings["se_std"].round(3).tolist() == [0.023, 0.016, 0.031]


def test_standardized_standard_errors_match_theory_and_bootstrap():
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    data = pd.DataFrame({"x": x, "y": 0.5 * x + rng.normal(size=n)})
    stats = sem(data, "y ~ x", bootstrap=0, se_robust=False)[0]
    row = stats[(stats["lval"] == "y") & (stats["rval"] == "x")].iloc[0]
    r = row["est_std"]
    assert r == pytest.approx(np.corrcoef(data.x, data.y)[0, 1], abs=1e-3)
    # standard error of a correlation / standardized simple regression coefficient: (1 - r^2) / sqrt(n)
    assert row["se_std"] == pytest.approx((1 - r**2) / np.sqrt(n), rel=0.05)

    # one factor: the delta method agrees with the spread of the standardized loadings in bootstrap samples
    n = 300
    factor = rng.normal(size=n)
    items = pd.DataFrame({f"i{k}": l * factor + rng.normal(size=n) for k, l in enumerate([1.0, 0.8, 0.7, 0.9], 1)})
    formula = "f =~ i1 + i2 + i3 + i4"
    fit = sem(items, formula, bootstrap=0, se_robust=False)[0]
    loadings = fit[(fit["rval"] == "f") & (fit["op"] == "~")].reset_index(drop=True)
    draws = [
        sem(items.sample(n, replace=True, random_state=b), formula, bootstrap=0, se_robust=False)[0]
        .query("rval == 'f' and op == '~'")["est_std"]
        .to_numpy()
        for b in range(80)
    ]
    assert np.std(draws, axis=0, ddof=1) == pytest.approx(loadings["se_std"].to_numpy(), rel=0.3)


def test_standardized_inference_leaves_the_fitted_model_untouched():
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    stats, _, model = sem(data, formula, bootstrap=0, return_model=True)
    again = semopy.Model(formula)
    np.random.seed(13)
    again.fit(data)
    assert np.allclose(model.param_vals, again.param_vals, atol=1e-6)
    std = model.inspect(std_est=True, information=None)["Est. Std"].astype(float).to_numpy()
    assert np.allclose(std, stats["est_std"].to_numpy())


def test_sem_report_formats_the_standardized_p_values(democracy_fit):
    democracy_stats, metrics = democracy_fit
    before = democracy_stats.copy()
    rep = sem_report(democracy_stats, metrics)
    pd.testing.assert_frame_equal(democracy_stats, before)
    assert rep["p_std"].map(lambda v: isinstance(v, str)).all()
    assert (rep["p_std"] == "<.001").sum() > 10 and rep["p_std"].iloc[0] != ""
    raw = sem_report(democracy_stats, metrics, format_pval=False)
    assert raw["p_std"].notna().sum() > 20


def test_sem_plot_labels_use_the_p_values_of_the_shown_estimates(mpl, democracy_stats):
    from matplotlib.text import Annotation

    def labels(fig):
        return [t.get_text() for t in fig.axes[0].texts if not isinstance(t, Annotation) and t.get_text()[:1].isdigit()]

    n_paths = int((democracy_stats["op"] == "~").sum())
    std = [t for t in labels(sem_plot(democracy_stats, engine="networkx", plot_covs=False)) if "p" in t]
    assert len(std) == n_paths  # every standardized path has a p-value, also the loadings fixed to 1
    assert "0.92\np<.001" in labels(sem_plot(democracy_stats, engine="networkx", plot_covs=False))
    raw = labels(sem_plot(democracy_stats, engine="networkx", plot_covs=False, std_ests=False))
    assert len([t for t in raw if "p" in t]) == n_paths - 3  # the three loadings fixed to 1 have no test
    # tables without the standardized p-values (older versions) are still plotted
    old = democracy_stats.drop(columns=["se_std", "z_std", "p_std"])
    old.attrs.update(democracy_stats.attrs)
    assert len([t for t in labels(sem_plot(old, engine="networkx", plot_covs=False)) if "p" in t]) == n_paths - 3


if __name__ == "__main__":
    test_sem()
    print("Tests for apatools.sem are PASSED!")
