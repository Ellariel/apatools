"""Structural equation modeling (SEM) with APA-style reporting.

Wraps `semopy <https://semopy.com>`_: :func:`sem` fits a model given in lavaan-like
syntax and returns a parameter table plus fit statistics, :func:`sem_report` formats
them for a manuscript (p-values, stars, a one-line fit summary) and :func:`sem_plot`
draws the path diagram with Graphviz.

Requires the optional dependencies ``pandas`` and ``semopy``
(``pip install "apatools[sem]"``); plotting additionally needs the ``graphviz``
Python package (extra ``graphviz``) and the Graphviz executables (``dot``) on the system PATH.

References
----------
Igolkina, A. A., & Meshcheryakov, G. (2020). semopy: A Python package for structural
equation modeling. *Structural Equation Modeling: A Multidisciplinary Journal, 27*(6),
952-963. https://doi.org/10.1080/10705511.2019.1704289

Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance
structure analysis: Conventional criteria versus new alternatives. *Structural
Equation Modeling, 6*(1), 1-55.
"""

import numpy as np

from ._deps import missing_dependency

try:
    import pandas as pd
    import semopy
except ImportError as e:  # pragma: no cover
    missing_dependency("sem", "sem", e)

from ._common import format_significance
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
CITATION = Citation(
    APA="Igolkina, A. A., & Meshcheryakov, G. (2020). \
semopy: A Python Package for Structural Equation Modeling. \
Structural Equation Modeling: A Multidisciplinary Journal, 27(6), 952-963. \
https://doi.org/10.1080/10705511.2019.1704289"
)


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

    The model is specified in lavaan-like syntax, fitted with
    :class:`semopy.Model`, optionally bias-corrected by a bootstrap, and returned as
    a parameter table together with the fit statistics. Pass both to
    :func:`sem_report` or :func:`sem_plot`.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed variables (one row per observation, columns named as in ``formula``).
    formula : str
        Model description in semopy syntax: ``=~`` measurement part (latent
        =~ indicators), ``~`` structural part (regressions), ``~~`` (co)variances.
        Example: ``"eta1 =~ y1 + y2 + y3"``; use one line per equation for several.
    method : str, default "MLW"
        Objective function passed to ``semopy.Model.fit(obj=...)``: ``"MLW"``
        (Wishart maximum likelihood), ``"ULS"``, ``"GLS"``, ``"FIML"``, ``"DWLS"``
        or ``"WLS"``.
    solver : str, default "SLSQP"
        Name of the scipy optimizer used to minimize the objective.
    bootstrap : int or falsy, default 100
        If truthy, ``semopy.bias_correction(model, n=bootstrap, resample_mean=True)``
        is applied after fitting. This *replaces* the parameter estimates by
        bias-corrected ones; use ``0`` or ``None`` to keep the plain estimates and to
        save time.
    se_robust : bool, default True
        Report Huber-White robust standard errors (``model.inspect(se_robust=...)``).
    return_model : bool, default False
        Also return the fitted ``semopy.Model``.
    seed : int, default 13
        Seed for numpy's *global* random generator (``np.random.seed``), which makes
        the bootstrap and the optimizer's starting values reproducible. The global
        state is changed as a side effect.

    Returns
    -------
    stats : pandas.DataFrame
        One row per model parameter with semopy's ``inspect`` columns in lower case
        (``lval``, ``op``, ``rval``, ``estimate``, ``se``, ``z``, ``p-value``) and
        the standardized estimate ``est_std``. Entries semopy reports as ``"-"`` are
        converted to NaN. ``op`` is ``"~"`` for regressions *and* measurement paths
        (semopy does not distinguish them), ``"~~"`` for (co)variances.
        ``stats.attrs["latent"]`` is the sorted list of latent variables, which
        :func:`sem_plot` uses to draw latent variables as circles.
    metrics : pandas.DataFrame
        Single column ``value``, indexed by ``N`` (rows in ``data``) followed by
        semopy's fit statistics (``DoF``, ``chi2``, ``chi2 p-value``, ``CFI``,
        ``GFI``, ``AGFI``, ``NFI``, ``TLI``, ``RMSEA``, ``AIC``, ``BIC``, ``LogLik``,
        ...).
    model : semopy.Model
        The fitted model; only if ``return_model`` is True.

    Notes
    -----
    Commonly used fit indices are CFI (Bentler, 1990), TLI (Tucker & Lewis, 1973) and
    RMSEA (Steiger & Lind, 1980; Browne & Cudeck, 1993). Hu and Bentler (1999)
    suggest CFI/TLI of about .95 or higher and RMSEA of about .06 or lower as
    indicators of good fit; these are rules of thumb, not tests. Robust standard
    errors follow the sandwich estimator of White (1980).

    Requires the optional dependency ``semopy`` (``pip install "apatools[sem]"``).

    References
    ----------
    Igolkina, A. A., & Meshcheryakov, G. (2020). semopy: A Python package for
    structural equation modeling. *Structural Equation Modeling: A Multidisciplinary
    Journal, 27*(6), 952-963. https://doi.org/10.1080/10705511.2019.1704289

    Bentler, P. M. (1990). Comparative fit indexes in structural models.
    *Psychological Bulletin, 107*(2), 238-246.

    Tucker, L. R., & Lewis, C. (1973). A reliability coefficient for maximum
    likelihood factor analysis. *Psychometrika, 38*(1), 1-10.

    Steiger, J. H., & Lind, J. C. (1980, May). *Statistically based tests for the
    number of common factors* [Paper presentation]. Annual meeting of the
    Psychometric Society, Iowa City, IA.

    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model fit. In
    K. A. Bollen & J. S. Long (Eds.), *Testing structural equation models*
    (pp. 136-162). Sage.

    Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance
    structure analysis: Conventional criteria versus new alternatives. *Structural
    Equation Modeling, 6*(1), 1-55.

    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and
    a direct test for heteroskedasticity. *Econometrica, 48*(4), 817-838.

    Examples
    --------
    >>> stats, metrics = sem(df, "eta =~ y1 + y2 + y3", bootstrap=0)  # doctest: +SKIP
    >>> sem_report(stats, metrics)  # doctest: +SKIP
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

    Parameters
    ----------
    stats : pandas.DataFrame
        Parameter table returned by :func:`sem` (not modified; a copy is returned).
    metrics : pandas.DataFrame
        Fit statistics returned by :func:`sem`. Must contain ``N``, ``DoF``, ``chi2``
        and ``chi2 p-value``; the other statistics are looked up by name.
    format_pval : bool or callable, default True
        Replace the ``p-value`` column by APA-formatted strings without the ``p``
        prefix and without spaces (``".009"``, ``"<.001"``, see
        :func:`apatools.format.format_p`). A callable receives one p-value; False
        keeps the numbers. When formatting, missing p-values become empty strings.
    add_stars : bool or callable, default False
        Add a ``sig`` column with significance markers
        (:func:`apatools.format.get_stars` if True, or the callable). Stars are
        computed from the unformatted p-values.
    decimal : int, default 3
        Number of decimals for the table and the fit summary; falsy skips rounding
        of the table.

    Returns
    -------
    pandas.DataFrame
        Copy of ``stats`` with formatted p-values, the optional ``sig`` column,
        rounded numbers and a new column ``model`` that is empty except for the
        first row, which holds the fit summary, e.g.
        ``"χ²(48, N = 500) = 303.8, p < .001, CFI = .9, TLI = .86, ..., RMSEA = 0.1"``.
        In it, CFI, GFI, AGFI, NFI and TLI are written without leading zero (APA style
        for values bounded by 1); the remaining statistics (RMSEA, AIC, BIC, LogLik,
        ...) are reported as returned by semopy, rounded to ``decimal``. Trailing zeros
        are not kept, since values are rounded as floats.

    Notes
    -----
    Because the fit summary is stored in the first row of ``stats``, the table is meant
    for display or export (e.g. to a spreadsheet), not for further numerical work.
    """

    stats = stats.copy()
    format_significance(stats, add_stars, format_pval, na_rep="")
    # select metrics by name: positional slicing would silently break if semopy reorders them
    chi = metrics["value"]
    fit_names = [i for i in ("CFI", "GFI", "AGFI", "NFI", "TLI") if i in chi.index]
    info_names = [
        i
        for i in chi.index
        if i not in fit_names
        and i
        not in ("N", "DoF", "DoF Baseline", "chi2", "chi2 p-value", "chi2 Baseline")
    ]
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
    decimal=3,
    dpi=600,
    seed=13,
    **kwargs,
):
    """Plot a SEM path diagram from the ``stats`` table returned by :func:`sem`.

    Latent variables are drawn as circles, observed variables as boxes, regressions
    and measurement paths as arrows from predictor/indicator to outcome/latent and
    covariances as dashed double-headed arrows. Edges are labelled with the
    estimate and, below it, the formatted p-value (or stars). The work is done by
    :func:`semopy_plot`.

    Parameters
    ----------
    stats : pandas.DataFrame
        Parameter table from :func:`sem`.
    save_to_file : str, optional
        Copy the rendered figure to this path (the file type follows ``format_fig``).
    plot_covs : bool, default True
        Draw covariances between different variables (dashed, double-headed arrows).
    std_ests : bool, default True
        Show standardized estimates (``est_std``) instead of raw ``estimate``.
    show_stars : bool or callable, default False
        Show significance stars instead of p-values (:func:`apatools.format.get_stars`
        if True, or a callable).
    format_pval : bool or callable, default True
        Format p-values as e.g. ``"p=.009"`` / ``"p<.001"``
        (:func:`apatools.format.format_p`); a callable receives one p-value. Ignored
        if ``show_stars`` is set. False prints the raw numbers.
    format_fig : str, default "png"
        Graphviz output format (``"png"``, ``"svg"``, ``"pdf"``, ...).
    method : str, default "semopy"
        Plotting backend; only ``"semopy"`` (Graphviz) is implemented, anything else
        raises ``NotImplementedError``.
    return_fig : bool, default True
        Return the ``graphviz.Digraph``; if False the function returns None (the
        figure is still written to ``save_to_file``).
    decimal : int, default 3
        Rounding of the estimates.
    dpi : int, default 600
        Accepted for compatibility; it is passed on to :func:`semopy_plot` but
        currently has no effect on the output.
    seed : int, default 13
        Seed for numpy's global random generator.
    **kwargs
        Passed through to :func:`semopy_plot` (currently unused by the Graphviz
        backend).

    Returns
    -------
    graphviz.Digraph or None
        The graph (displays inline in Jupyter), or None if ``return_fig`` is False.

    Notes
    -----
    Requires the ``graphviz`` Python package and the Graphviz executables (``dot``) on
    the system PATH (on Windows, add the Graphviz ``bin`` folder to ``PATH``).
    """
    if method != "semopy":
        raise NotImplementedError(
            f"Plot method '{method}' is not implemented, use 'semopy'."
        )
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
    Draw a SEM diagram adapted from ``semopy.semplot``.

    Builds a Graphviz graph from the parameter table and renders it once (to a
    temporary directory that is removed afterwards) so that errors in the Graphviz
    installation surface immediately and ``save_to_file`` can be written. Edges
    with the intercept pseudo-variable ``"1"`` are skipped, and variance
    (self-covariance) rows are not drawn.

    Parameters
    ----------
    stats : pandas.DataFrame
        Parameter table from :func:`sem`. Latent variables are read from
        ``stats.attrs["latent"]``; without it every right-hand-side variable is drawn
        as latent (circle).
    save_to_file : str or None
        Path where the rendered figure is copied to; None only returns the graph.
    plot_covs : bool, optional
        Draw covariances as dashed double-headed edges. Default True.
    std_ests : bool, optional
        Plot standardized (``est_std``) instead of raw (``estimate``) estimates.
        Default True.
    plot_ests : bool, optional
        Label edges with estimates and p-values. Default True.
    engine : str, optional
        Graphviz layout engine. Default 'dot'.
    latshape : str, optional
        Graphviz node shape for latent variables. Default 'circle'.
    show_stars : bool or callable, optional
        Append significance stars to the labels instead of p-values
        (:func:`apatools.format.get_stars` if True, or the callable).
    format_pval : bool or callable, optional
        Format p-values in labels as ``"p=.009"`` (see :func:`apatools.format.format_p`;
        ignored when ``show_stars`` is set); a callable receives one p-value.
    format_fig : str, optional
        Graphviz output format. Default 'png'.
    decimal : int, optional
        Rounding of estimates. Default 3.
    seed : int, optional
        Seed for numpy's global random generator. Default 13.
    **kwargs
        Accepted and ignored (lets :func:`sem_plot` forward its extra options).

    Returns
    -------
    graphviz.Digraph
        The graph, which can be displayed or re-rendered with other options.

    Raises
    ------
    ImportError
        If the ``graphviz`` Python package is not installed.
    graphviz.ExecutableNotFound
        If the Graphviz executables are not on the PATH.
    """
    import os
    import shutil
    import tempfile

    try:
        import graphviz  # It uses graphviz. For windows, use https://graphviz.gitlab.io/download/
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "semopy_plot needs the optional dependency: pip install graphviz"
        ) from e

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
                else lambda x: format_p(x, keep_spaces=False, no_equals=False)
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
