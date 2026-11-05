import semopy

from apatools.sem import sem, sem_report, semopy_plot, sem_plot, CITATION


def test_sem():
    # https://semopy.com/tutorial.html
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    assert len(data) == 75

    print(CITATION.APA)

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
    assert f"{stats['estimate'].iloc[0]:.3f}" == "1.252"

    results = sem_report(stats, metrics, format_pval=True, add_stars=True)
    assert (
        results.iloc[0]["model"]
        == "χ²(35, N = 75) = 38.125, p = .329, CFI = .995, GFI = .948, AGFI = .918, NFI = .948, TLI = .993, RMSEA = 0.035, AIC = 60.872, BIC = 132.714, LogLik = 0.564"
    )
    print(results)

    fig = semopy_plot(
        stats,
        save_to_file="fig1.png",
        plot_covs=True,
        std_ests=True,
        plot_ests=True,
        engine="dot",
        latshape="circle",
        show_stars=False,
        format_pval=True,
        format_fig="png",
        decimal=3,
        seed=13,
    )
    assert fig is not None
    
    fig = sem_plot(
        stats,
        save_to_file="fig2.png",
        plot_covs=True,
        std_ests=True,
        show_stars=True,
        format_pval=False,
        format_fig="png",
        method="semopy",
        return_fig=True,
        show_fig=False,
        decimal=3,
        dpi=600,
        seed=13,
    )
    assert fig is not None
    

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

    # unknown plot backend fails loudly instead of with UnboundLocalError
    with pytest.raises(NotImplementedError):
        sem_plot(stats, method="unknown")


if __name__ == "__main__":
    test_sem()
    print("Tests for apatools.sem are PASSED!")
