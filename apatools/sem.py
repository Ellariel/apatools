import numpy as np
import pandas as pd
import semopy

from .citation import Citation
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
CITATION = Citation(APA='Igolkina, A. A., & Meshcheryakov, G. (2020). \
semopy: A Python Package for Structural Equation Modeling. \
Structural Equation Modeling: A Multidisciplinary Journal, 27(6), 952-963. \
https://doi.org/10.1080/10705511.2019.1704289')


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
    """Fit a structural equation model with semopy.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed variables.
    formula : str
        Model description in semopy syntax: ``=~`` measurement part (latent
        =~ indicators), ``~`` structural part, ``~~`` (co)variances.
    method : str
        Objective function: MLW, ULS, GLS, FIML, DWLS or WLS.
    solver : str
        scipy optimizer name.
    bootstrap : int or falsy
        If truthy, ``semopy.bias_correction`` with this many resamples is applied,
        which *changes* the parameter estimates.
    se_robust : bool
        Use Huber-White robust standard errors.
    return_model : bool
        Also return the fitted ``semopy.Model``.
    seed : int
        Seed for numpy's global random generator.

    Returns
    -------
    stats : pandas.DataFrame
        Parameter table (``lval``, ``op``, ``rval``, ``estimate``, ``se``, ``z``,
        ``p-value``, ``est_std``, ...). ``stats.attrs["latent"]`` lists latent variables.
    metrics : pandas.DataFrame
        ``N`` followed by semopy's fit statistics (column ``value``).
    model : semopy.Model
        Only if ``return_model`` is True.
    """
    np.random.seed(seed)
    model = semopy.Model(formula)
    model.fit(data, obj=method, solver=solver)
    if bootstrap:
        semopy.bias_correction(model, n=bootstrap, resample_mean=True)
    metrics = semopy.calc_stats(model).T.rename(columns={"Value": "value"})
    metrics = pd.concat([pd.Series({"N": data.shape[0]}, name="value"), metrics])
    stats = model.inspect(std_est=True, se_robust=se_robust)
    stats.columns = [i.lower().replace(". ", "_") for i in stats.columns]
    stats = stats.replace("-", np.nan).infer_objects().copy()
    # semopy reports measurement paths ("=~") as "~" rows, so keep the latent
    # variable names for sem_plot() to be able to tell latent from observed nodes
    stats.attrs["latent"] = sorted(model.vars.get("latent", []))
    if return_model:
        return stats, metrics, model
    return stats, metrics


def sem_report(stats, metrics, format_pval=True, add_stars=False, decimal=3):
    """Format the output of :func:`sem` for reporting.

    Returns a copy of ``stats`` (p-values formatted, optional ``sig`` column, rounded)
    whose first row carries an APA-style fit summary in the ``model`` column, e.g.
    ``χ²(48, N = 500) = 303.80, p < .001, CFI = .90, TLI = .86, ...``.
    ``format_pval`` / ``add_stars`` may be ``True`` or a callable.
    """

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
    # select metrics by name: positional slicing would silently break if semopy reorders them
    chi = metrics["value"]
    fit_names = [i for i in ("CFI", "GFI", "AGFI", "NFI", "TLI") if i in chi.index]
    info_names = [i for i in chi.index if i not in fit_names and i not in (
        "N", "DoF", "DoF Baseline", "chi2", "chi2 p-value", "chi2 Baseline")]
    r = f"χ²({int(chi['DoF'])}, N = {int(chi['N'])}) = {round(chi['chi2'], decimal)}, {format_p(chi['chi2 p-value'])}"

    def _no_zero(v):  # APA: no leading zero for values bounded by 1, keep the sign
        s = f"{round(v, decimal)}"
        return s.replace("0.", ".", 1) if abs(v) < 1 else s

    m = ", ".join([f"{i} = " + _no_zero(chi[i]) for i in fit_names])
    k = ", ".join([f"{i} = {round(chi[i], decimal)}" for i in info_names])
    if decimal:
        stats = stats.round(decimal)
    stats.loc[stats.index[0], "model"] = r + ", " + m + ", " + k
    return stats


def sem_plot(
    stats,
    save_to_file=None,
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
    """Plot a SEM path diagram from the ``stats`` table returned by :func:`sem`.

    Parameters
    ----------
    stats : pandas.DataFrame
        Parameter table from :func:`sem`.
    save_to_file : str, optional
        Copy the rendered figure to this path.
    plot_covs : bool
        Draw covariances (dashed, double-headed arrows).
    std_ests : bool
        Show standardized instead of raw estimates.
    show_stars : bool or callable
        Show significance stars instead of formatted p-values.
    format_pval : bool or callable
        Format p-values (ignored if ``show_stars`` is set).
    format_fig : str
        Graphviz output format (``"png"``, ``"svg"``, ...).
    method : str
        Plotting backend; only ``"semopy"`` (Graphviz) is implemented.
    return_fig : bool
        Return the ``graphviz.Digraph``.
    decimal : int
        Rounding of the estimates.
    **kwargs
        Passed through (currently unused by the Graphviz backend).

    Requires the Graphviz executables (``dot``) on the system PATH.
    """
    if method != "semopy":
        raise NotImplementedError(f"Plot method '{method}' is not implemented, use 'semopy'.")
    fig = None
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
    stats : pandas.DataFrame
        Parameter table from :func:`sem`. Latent variables are read from
        ``stats.attrs["latent"]``; without it every right-hand-side variable is drawn
        as latent (circle).
    save_to_file : str or None
        Path where the rendered figure is copied to.
    plot_covs : bool, optional
        Draw covariances as dashed double-headed edges. Default True.
    std_ests : bool, optional
        Plot standardized instead of raw estimates. Default True.
    plot_ests : bool, optional
        Label edges with estimates. Default True.
    engine : str, optional
        Graphviz layout engine. Default 'dot'.
    latshape : str, optional
        Graphviz node shape for latent variables. Default 'circle'.
    show_stars : bool or callable, optional
        Append significance stars to labels instead of p-values.
    format_pval : bool or callable, optional
        Format p-values in labels (ignored when ``show_stars`` is set).
    format_fig : str, optional
        Graphviz output format. Default 'png'.
    decimal : int, optional
        Rounding of estimates. Default 3.
    seed : int, optional
        Seed for numpy's global random generator.

    Returns
    -------
    graphviz.Digraph
    """
    import os
    import shutil
    import tempfile
    import graphviz
    # It uses graphviz. For windows, use https://graphviz.gitlab.io/download/

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
                    x, keep_spaces=False, no_equals=False
                )
            )
        )
    stats = stats.round(decimal)
    elements = stats[stats["op"] == "~"]
    latent = set(stats.attrs.get("latent", []))
    if not latent:  # fallback for tables without metadata: old heuristic
        latent = set(elements["rval"].to_list())
    nodes = set(elements["lval"].to_list()) | set(elements["rval"].to_list())
    nodes.discard("1")  # semopy's intercept pseudo-variable
    observed = nodes - latent
    latent = latent & nodes
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
    tdir = tempfile.mkdtemp(prefix="semplot_")  # unique dir: no clashes between runs
    try:
        tfile = os.path.join(tdir, "semplot_tmp_fig")
        g.render(tfile, view=False)
        if save_to_file:
            shutil.copy(f"{tfile}.{format_fig}", save_to_file)
    finally:
        shutil.rmtree(tdir, ignore_errors=True)
    return g
