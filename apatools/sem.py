import numpy as np
import pandas as pd
import semopy


from .format import format_p, get_stars

# https://arxiv.org/pdf/2106.01140.pdf
# https://semopy.com/cite.html
# https://gitlab.com/georgy.m/semopy
# ~ to specify structural part,
# =~ to specify measurement part,
# ~~ to specify common variance between variables.
# formula = f"""
# x1 ~ x2 + x3
# x3 ~ x2 + eta1
# eta1 =~ y1 + y2 + y3
# eta1 ~ x1
# """


def sem(
    data,
    formula,
    method="MLW",  # MLW ULS GLS FIML DWLS WLS
    solver="SLSQP",
    bootstrap=100,
    se_robust=True,  # Huber-White correction
    return_model=False,
    seed=13,
):
    np.random.seed(seed)
    model = semopy.Model(formula)
    model.fit(data, obj=method, solver=solver)
    if bootstrap:
        semopy.bias_correction(model, n=bootstrap, resample_mean=True)
    metrics = semopy.calc_stats(model).T.rename(columns={"Value": "value"})
    metrics = pd.concat([pd.Series({"N": data.shape[0]}, name="value"), metrics])
    stats = model.inspect(std_est=True, se_robust=se_robust)
    stats.columns = [i.lower().replace(". ", "_") for i in stats.columns]
    stats = stats.replace("-", np.nan).infer_objects(copy=False)
    if return_model:
        return stats, metrics, model
    return stats, metrics


def sem_report(stats, metrics, format_pval=True, add_stars=False, decimal=3):
    # https://people.ucsc.edu/~zurbrigg/psy214b/09SEM8a.pdf
    # χ2(48, N = 500) = 303.80, p < .001, TLI = .86, CFI = .90

    stats = stats.copy()
    if add_stars:
        add_stars = get_stars if not callable(add_stars) else add_stars
        stats["sig"] = stats["p-value"].fillna("").apply(add_stars)
    if format_pval:
        stats["p-value"] = (
            stats["p-value"]
            .fillna("")
            .apply(
                format_pval
                if callable(format_pval)
                else lambda x: format_p(
                    x, use_letter=False, keep_spaces=False, no_equals=True
                )
            )
        )
    r = metrics.iloc[:6].T.to_dict()
    m = metrics.iloc[6:11].T.to_dict()
    k = metrics.iloc[11:].T.to_dict()
    r = f"χ²({int(r['DoF']['value'])}, N = {int(r['N']['value'])}) = {round(r['chi2']['value'], decimal)}, {format_p(r['chi2 p-value']['value'])}"
    m = ", ".join(
        [f"{i} = " + f"{round(v['value'], decimal)}"[1:] for i, v in m.items()]
    )
    k = ", ".join([f"{i} = " + f"{round(v['value'], decimal)}" for i, v in k.items()])
    if decimal:
        stats = stats.round(decimal)
    stats.loc[stats.index[0], "model"] = r + ", " + m + ", " + k
    return stats


def sem_plot(
    stats,
    save_to_file,
    plot_covs=True,
    std_ests=True,
    show_stars=False,
    format_pval=True,
    format_fig="png",
    method="semopy",
    return_fig=True,
    show_fig=False,
    decimal=3,
    dpi=600,
    seed=13,
    **kwargs,
):
    #

    if method == "semopy":
        fig = semopy_plot(
            stats,
            save_to_file=save_to_file,
            plot_covs=plot_covs,
            std_ests=std_ests,
            plot_ests=True,
            engine="dot",
            latshape="circle",
            show_stars=show_stars,
            format_pval=format_pval,
            format_fig=format_fig,
            decimal=decimal,
            seed=seed,
            dpi=dpi,
            **kwargs,
        )

    if return_fig:
        return fig


def semopy_plot(
    stats,
    save_to_file,
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
    **kwargs,
):
    """
    Draw a SEM diagram adapted from semopy.semplot

    Parameters
    ----------
    save_to_file : str
        Name of file where to plot is saved.
    plot_covs : bool, optional
        If True, covariances are also drawn. The default is False.
    plot_exos: bool, optional
        If False, exogenous variables are not plotted. It might be useful,
        for example, in GWAS setting, where a number of exogenous variables,
        i.e. genetic markers, is oblivious. Has effect only with ModelMeans or
        ModelEffects. The default is True.
    engine : str, optional
        Graphviz engine name to use. The default is 'dot'.
    latshape : str, optional
        Graphviz-compaitable shape for latent variables. The default is
        'circle'.
    plot_ests : bool, optional
        If True, then estimates are also plotted on the graph. The default is
        True.
    std_ests : bool, optional
        If True and plot_ests is True, then standardized values are plotted
        instead. The default is False.
    Returns
    -------
    Graphviz graph.
    """
    import os
    import shutil
    import tempfile
    import graphviz
    # it uses graphviz. for windows, use https://graphviz.gitlab.io/download/

    np.random.seed(seed)
    stats = stats.copy()
    if show_stars:
        show_stars = get_stars if not callable(show_stars) else show_stars
        stats["p-value"] = stats["p-value"].fillna("").apply(show_stars)
    elif format_pval:
        stats["p-value"] = (
            stats["p-value"]
            .fillna("")
            .apply(
                format_pval
                if callable(format_pval)
                else lambda x: format_p(
                    x, use_letter=False, keep_spaces=False, no_equals=True
                )
            )
        )
    stats = stats.round(decimal)
    elements = stats[stats["op"] == "~"]
    latent = set(elements["rval"].to_list())
    observed = set(elements["lval"].to_list()) - latent
    g = graphviz.Digraph("G", format=format_fig, engine=engine)
    g.attr(overlap="scale", splines="true")
    g.attr("edge", fontsize="12")
    g.attr("node", shape=latshape, fillcolor="#cae6df", style="filled")
    for lat in latent:
        g.node(lat, label=lat)
    g.attr("node", shape="box", style="")
    for obs in observed:
        g.node(obs, label=obs)
    for _, row in elements.iterrows():
        lval, rval, pval, est = (
            row["lval"],
            row["rval"],
            row["p-value"],
            row["estimate"] if not std_ests else row["est_std"],
        )
        if rval == "1":
            continue
        label = ""
        if plot_ests:
            label = str(est)
            if pval != "" and pd.notna(pval):
                if show_stars:
                    label += pval
                else:
                    label += f"\n{pval}"
        g.edge(rval, lval, label=label)
    if plot_covs:
        elements = stats[(stats["op"] == "~~") & (stats["lval"] != stats["rval"])]
        for _, row in elements.iterrows():
            lval, rval, pval, est = (
                row["lval"],
                row["rval"],
                row["p-value"],
                row["estimate"] if not std_ests else row["est_std"],
            )
            label = ""
            if plot_ests:
                label = str(est)
                if pval != "" and pd.notna(pval):
                    if show_stars:
                        label += pval
                    else:
                        label += f"\n{pval}"
            g.edge(rval, lval, label=label, dir="both", style="dashed")
    tfile = os.path.join(tempfile.gettempdir(), "semplot_tmp_fig")
    g.render(tfile, view=False)
    if save_to_file:
        shutil.copy(f"{tfile}.{format_fig}", save_to_file)
    return g
