import semopy

from apatools.sem import sem, sem_report, semopy_plot, sem_plot


def test_sem():
    # https://semopy.com/tutorial.html
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    assert len(data) == 75

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
        save_to_file="fig.png",
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
        save_to_file="fig.png",
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
    

if __name__ == "__main__":
    test_sem()
    print("Tests for metatools.sem are PASSED!")
