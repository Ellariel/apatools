"""Structural equation modeling (SEM) with APA-style reporting.

Wraps `semopy <https://semopy.com>`_: :func:`sem` fits a model given in lavaan-like
syntax and returns a parameter table plus fit statistics, :func:`sem_report` formats
them for a manuscript (p-values, stars, a one-line fit summary) and :func:`sem_plot`
draws the path diagram, with Graphviz (``engine="semopy"``) or with networkx and matplotlib
(``engine="networkx"``).

Requires the optional dependencies ``pandas`` and ``semopy``
(``pip install "apatools[sem]"``). Plotting with the ``"semopy"`` engine additionally needs the
``graphviz`` Python package (extra ``graphviz``) and the Graphviz executables (``dot``) on the
system PATH; the ``"networkx"`` engine needs only networkx and matplotlib (extra ``networkx``).

References
----------
Igolkina, A. A., & Meshcheryakov, G. (2020). semopy: A Python package for structural
equation modeling. *Structural Equation Modeling: A Multidisciplinary Journal, 27*(6),
952-963. https://doi.org/10.1080/10705511.2019.1704289

Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis:
Conventional criteria versus new alternatives. *Structural Equation Modeling: A Multidisciplinary
Journal, 6*(1), 1-55. https://doi.org/10.1080/10705519909540118
"""

import os

import numpy as np
from scipy.stats import norm

from ._deps import missing_dependency

try:
    import pandas as pd
    import semopy
except ImportError as e:  # pragma: no cover
    missing_dependency("sem", "sem", e)

from ._common import format_significance
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
def _standardized_inference(model, robust):
    """
    Standard errors, z-values and p-values of the standardized estimates (delta method).

    semopy reports a standardized estimate (``Est. Std``) but no standard error for it. The
    standardized estimates are functions of the free parameters, so their covariance matrix is
    ``J S J'`` with ``S`` the covariance matrix of the free parameters (the one behind semopy's
    standard errors: expected information, with the Huber-White sandwich if ``robust``) and ``J``
    the numerical Jacobian of the standardized estimates (central differences). This is the
    approach of lavaan's ``standardizedSolution()``. Also parameters that are fixed in the
    unstandardized solution (the first loading of a latent variable, fixed to 1) get a standard
    error, because their standardized value is estimated.

    Returns three arrays aligned with the rows of ``model.inspect(std_est=True)``; NaN where the
    standardized value does not depend on any free parameter or the covariance matrix is not
    usable.
    """
    theta = np.array(model.param_vals, dtype=float)

    def standardized(values):
        model.param_vals = values
        model.update_matrices(values)
        table = model.inspect(std_est=True, information=None)
        return pd.to_numeric(table["Est. Std"], errors="coerce").to_numpy(dtype=float)

    cov = model.calc_fim(inverse=True)[1]
    if robust:
        g = sum(np.outer(g, g) for g in model.grad_se_g(model.param_vals))
        cov = cov @ g @ cov
    try:
        estimate = standardized(theta)
        jacobian = np.empty((len(estimate), len(theta)))
        for i in range(len(theta)):
            step = 1e-6 * max(1.0, abs(theta[i]))
            up, down = theta.copy(), theta.copy()
            up[i] += step
            down[i] -= step
            jacobian[:, i] = (standardized(up) - standardized(down)) / (2 * step)
    finally:  # leave the fitted model as it was
        model.param_vals = theta
        model.update_matrices(theta)
    variance = np.einsum("ij,jk,ik->i", jacobian, cov, jacobian)
    variance[(variance < 0) & (variance > -1e-10)] = 0.0
    se = np.sqrt(np.where(variance >= 0, variance, np.nan))
    se[se < 1e-10] = np.nan  # a constant has no standard error
    z = estimate / se
    return se, z, 2 * norm.sf(np.abs(z))


SEM_ENGINES = ("semopy", "lavaan")


def sem(
    data,
    formula,
    method=None,  # semopy: MLW (default) ULS GLS FIML DWLS WLS; lavaan: ML (default) ULS GLS ...
    solver="SLSQP",
    bootstrap=None,  # semopy: 100 (default); lavaan: 0 (default)
    se_robust=True,  # Huber-White correction
    return_model=False,
    seed=13,
    engine="semopy",
    lavaan_args=None,
    rscript=None,
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
    method : str, optional
        Estimator. ``engine="semopy"`` (objective function passed to
        ``semopy.Model.fit(obj=...)``): ``"MLW"`` (Wishart maximum likelihood, the default),
        ``"ULS"``, ``"GLS"``, ``"FIML"``, ``"DWLS"`` or ``"WLS"``. ``engine="lavaan"``: the same
        names are translated (``"MLW"`` and ``"FIML"`` to ML, the latter with
        ``missing="fiml"``) and any lavaan estimator can be given (``"ML"``, the default,
        ``"MLR"``, ``"WLSMV"``, ...).
    solver : str, default "SLSQP"
        Name of the scipy optimizer used to minimize the objective (``engine="semopy"`` only).
    bootstrap : int or falsy, optional
        ``engine="semopy"`` (default 100): if truthy,
        ``semopy.bias_correction(model, n=bootstrap, resample_mean=True)`` is applied after
        fitting. This *replaces* the parameter estimates by bias-corrected ones; use ``0`` to
        keep the plain estimates and to save time. ``engine="lavaan"`` (default 0): if truthy,
        lavaan computes bootstrap standard errors and p-values from that many resamples
        (the estimates themselves are not changed).
    se_robust : bool, default True
        Report Huber-White robust standard errors (``model.inspect(se_robust=...)``; with
        lavaan ``se="robust.huber.white"``, only for the ML estimator).
    return_model : bool, default False
        Also return the fitted model: the ``semopy.Model``, or for ``engine="lavaan"`` a
        ``LavaanFit`` (lavaan's own parameter tables and fit measures, its version and warnings).
    seed : int, default 13
        Seed for numpy's *global* random generator (``np.random.seed``), which makes
        the bootstrap and the optimizer's starting values reproducible. The global
        state is changed as a side effect. With lavaan it seeds R (``set.seed``).
    engine : {"semopy", "lavaan"}, default "semopy"
        ``"semopy"`` fits the model in Python. ``"lavaan"`` fits it with the R package lavaan
        (Rosseel, 2012) in an ``Rscript`` subprocess and returns the same tables, so that
        :func:`sem_report` and :func:`sem_plot` work with both. It needs R with the package
        lavaan (``install.packages("lavaan")``) and no further Python package.
    lavaan_args : dict, optional
        Further arguments of ``lavaan::sem`` for ``engine="lavaan"``, for example
        ``{"std.lv": True, "ordered": ["x1", "x2"], "missing": "fiml"}`` (they override the
        arguments derived from ``method``, ``se_robust`` and ``bootstrap``).
    rscript : str, optional
        Path of ``Rscript`` (``engine="lavaan"``); by default the environment variable
        ``APATOOLS_RSCRIPT``, the ``PATH`` and the usual install locations are searched.

    Returns
    -------
    stats : pandas.DataFrame
        (The columns are the same for both engines.) One row per model parameter with semopy's ``inspect`` columns in lower case
        (``lval``, ``op``, ``rval``, ``estimate``, ``std_err``, ``z-value``, ``p-value``), the
        standardized estimate ``est_std`` and its standard error ``se_std``, z-value
        ``z_std`` and p-value ``p_std`` (delta method, as in lavaan's
        ``standardizedSolution()``; the loadings that are fixed to 1 in the unstandardized
        solution have no ``p-value`` but do have a ``p_std``). Entries semopy reports as
        ``"-"`` are converted to NaN. ``op`` is ``"~"`` for regressions *and* measurement paths
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

    Bentler, P. M. (1990). Comparative fit indexes in structural models. *Psychological Bulletin,
    107*(2), 238-246. https://doi.org/10.1037/0033-2909.107.2.238

    Tucker, L. R., & Lewis, C. (1973). A reliability coefficient for maximum likelihood factor
    analysis. *Psychometrika, 38*(1), 1-10. https://doi.org/10.1007/bf02291170

    Steiger, J. H., & Lind, J. C. (1980, May). *Statistically based tests for the
    number of common factors* [Paper presentation]. Annual meeting of the
    Psychometric Society, Iowa City, IA.

    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model fit. In K. A. Bollen &
    J. S. Long (Eds.), *Testing structural equation models*. Sage.

    Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure
    analysis: Conventional criteria versus new alternatives. *Structural Equation Modeling: A
    Multidisciplinary Journal, 6*(1), 1-55. https://doi.org/10.1080/10705519909540118

    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test
    for heteroskedasticity. *Econometrica, 48*(4), 817-838. https://doi.org/10.2307/1912934

    Examples
    --------
    >>> stats, metrics = sem(df, "eta =~ y1 + y2 + y3", bootstrap=0)  # doctest: +SKIP
    >>> sem_report(stats, metrics)  # doctest: +SKIP
    """
    if engine not in SEM_ENGINES:
        raise NotImplementedError(f"Engine '{engine}' is not implemented, use one of {SEM_ENGINES}.")
    if engine == "lavaan":
        from .r.lavaan import fit_lavaan, lavaan_estimator

        lavaan_args = dict(lavaan_args or {})
        estimator, implied = lavaan_estimator("ML" if method is None else method, lavaan_args)
        stats, metrics, fit = fit_lavaan(
            data,
            formula,
            estimator,
            se="robust.huber.white" if se_robust and estimator == "ML" else None,
            bootstrap=bootstrap or 0,
            lavaan_args={**implied, **lavaan_args},
            seed=seed,
            rscript=rscript,
        )
        return (stats, metrics, fit) if return_model else (stats, metrics)
    method = "MLW" if method is None else method
    bootstrap = 100 if bootstrap is None else bootstrap
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
    se_std, z_std, p_std = _standardized_inference(model, se_robust)
    stats["se_std"], stats["z_std"], stats["p_std"] = se_std, z_std, p_std
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
    if "p_std" in stats:  # p-values of the standardized estimates, formatted like the others
        format_significance(stats, False, format_pval, p_column="p_std", na_rep="")
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


PLOT_ENGINES = ("semopy", "networkx", "lavaan")


def sem_plot(
    stats,
    save_to_file=None,
    plot_covs=True,
    std_ests=True,
    show_stars=False,
    format_pval=True,
    format_fig="png",
    engine="semopy",
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
    estimate and, below it, the formatted p-value (or stars). The drawing is done by one
    of two engines that show the same diagram:

    * ``"semopy"`` (default): the diagram of ``semopy.semplot`` drawn by Graphviz
      (:func:`semopy_plot`); needs the Graphviz executables.
    * ``"networkx"``: a pure Python drawing with networkx and matplotlib
      (:func:`networkx_plot`); needs no external program and also supports ``dpi``.
    * ``"lavaan"``: drawn by the R package semPlot (``semPlot::semPaths``), the usual plotter of
      lavaan models, in an ``Rscript`` subprocess; works with the tables of both ``sem`` engines.
      Needs R with the packages lavaan and semPlot; returns an ``RFigure`` (displays in Jupyter,
      ``.save(path)``).

    Parameters
    ----------
    stats : pandas.DataFrame
        Parameter table from :func:`sem`.
    save_to_file : str, optional
        Write the figure to this path. The file type follows ``format_fig`` for the ``"semopy"``
        engine; for ``"networkx"`` it follows the extension of the path (``format_fig`` if the
        path has none).
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
        Output format (``"png"``, ``"svg"``, ``"pdf"``, ...), see ``save_to_file``.
    engine : {"semopy", "networkx", "lavaan"}, default "semopy"
        Drawing engine. Any other value raises ``NotImplementedError``.
    return_fig : bool, default True
        Return the figure object; if False the function returns None (the figure is still
        written to ``save_to_file``, and a matplotlib figure is closed).
    decimal : int, default 3
        Rounding of the estimates.
    dpi : int, default 600
        Resolution of the saved figure; used by the ``"networkx"`` engine only (the
        ``"semopy"`` engine accepts it but ignores it).
    seed : int, default 13
        Seed for numpy's global random generator.
    **kwargs
        Passed through to the engine function: ``layout`` for both engines (a Graphviz layout
        program such as ``"dot"`` / ``"neato"`` for ``"semopy"``, see :func:`semopy_plot`; ``"layered"``,
        ``"spring"``, ``"kamada_kawai"``, ``"circular"`` for ``"networkx"``; ``"tree"``, ``"tree2"``,
        ``"circle"``, ``"circle2"``, ``"spring"`` for ``"lavaan"``), ``figsize`` and ``fontsize`` for
        ``"networkx"``, see :func:`networkx_plot`, and ``rotation``, ``width``, ``height``, ``fontsize``
        and ``rscript`` for ``"lavaan"``.

    Returns
    -------
    graphviz.Digraph, matplotlib.figure.Figure, RFigure or None
        ``"semopy"``: the Graphviz graph (displays inline in Jupyter). ``"networkx"``: the
        matplotlib figure. ``"lavaan"``: the figure drawn by R. None if ``return_fig`` is False.

    Notes
    -----
    The ``"semopy"`` engine needs the ``graphviz`` Python package and the Graphviz executables
    (``dot``) on the system PATH (on Windows, add the Graphviz ``bin`` folder to ``PATH``):
    ``pip install "apatools[graphviz]"``. The ``"networkx"`` engine needs networkx and
    matplotlib: ``pip install "apatools[networkx]"``. The ``"lavaan"`` engine needs R with the packages
    lavaan and semPlot (``install.packages(c("lavaan", "semPlot"))``).
    """
    if engine not in PLOT_ENGINES:
        raise NotImplementedError(
            f"Plot engine '{engine}' is not implemented, use one of {PLOT_ENGINES}."
        )
    common = {
        "save_to_file": save_to_file,
        "plot_covs": plot_covs,
        "std_ests": std_ests,
        "plot_ests": True,
        "show_stars": show_stars,
        "format_pval": format_pval,
        "format_fig": format_fig,
        "decimal": decimal,
        "seed": seed,
    }
    if engine == "semopy":
        fig = semopy_plot(stats, latshape="circle", dpi=dpi, **common, **kwargs)
    elif engine == "lavaan":
        from .r.lavaan import semplot

        np.random.seed(seed)
        latent, _, edges, covs, order = _path_diagram(
            stats, plot_covs, std_ests, True, show_stars, format_pval, decimal
        )
        fig = semplot(
            order, latent, edges, covs, save_to_file=save_to_file, format_fig=format_fig, dpi=dpi, **kwargs
        )
    else:
        fig = networkx_plot(stats, dpi=dpi, **common, **kwargs)
        if not return_fig:
            import matplotlib.pyplot as plt

            plt.close(fig)

    if return_fig:
        return fig


def _path_diagram(stats, plot_covs, std_ests, plot_ests, show_stars, format_pval, decimal):
    """
    Nodes and labelled edges of the path diagram, shared by the plotting engines.

    Returns ``(latent, observed, edges, covs, order)``: the sets of latent and observed
    variables, the regression / measurement edges ``(predictor, outcome, label)``, the
    covariance edges (empty unless ``plot_covs``) and the variables in order of appearance.
    Edges with the intercept pseudo-variable ``"1"`` and variances are left out.
    """
    stats = stats.copy()
    # the p-values must belong to the estimates that are shown (tables from older versions of
    # sem() have no p_std: then the p-values of the unstandardized estimates are used)
    pcol = "p_std" if std_ests and "p_std" in stats else "p-value"
    if show_stars:
        show_stars = get_stars if not callable(show_stars) else show_stars
        stats[pcol] = stats[pcol].fillna("").apply(show_stars)
    elif format_pval:
        stats[pcol] = (
            stats[pcol]
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

    def label(row):
        est = row["estimate"] if not std_ests else row["est_std"]
        pval = row[pcol]
        text = ""
        if plot_ests:
            text = str(est)
            if pval != "" and pd.notna(pval):
                text += pval if show_stars else f"\n{pval}"
        return text

    edges = [
        (row["rval"], row["lval"], label(row))
        for _, row in elements.iterrows()
        if row["rval"] != "1"
    ]
    covs = []
    if plot_covs:
        elements = stats[(stats["op"] == "~~") & (stats["lval"] != stats["rval"])]
        covs = [
            (row["rval"], row["lval"], label(row)) for _, row in elements.iterrows()
        ]
    order = list(dict.fromkeys(n for rval, lval, _ in edges for n in (rval, lval)))
    order += sorted(nodes - set(order))
    return latent, observed, edges, covs, order


def semopy_plot(
    stats,
    save_to_file,
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
    **kwargs,
):
    """
    Draw a SEM diagram adapted from ``semopy.semplot`` (the ``"semopy"`` engine of :func:`sem_plot`).

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
    layout : str, optional
        Graphviz layout program (``"dot"``, ``"neato"``, ``"fdp"``, ...). Default 'dot'.
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
    import shutil
    import tempfile

    try:
        import graphviz  # It uses graphviz. For windows, use https://graphviz.gitlab.io/download/
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "semopy_plot needs the optional dependency: pip install graphviz"
        ) from e

    np.random.seed(seed)
    latent, observed, edges, covs, _ = _path_diagram(
        stats, plot_covs, std_ests, plot_ests, show_stars, format_pval, decimal
    )
    g = graphviz.Digraph("G", format=format_fig, engine=layout)
    g.attr(overlap="scale", splines="true")
    g.attr("edge", fontsize="12")
    g.attr("node", shape=latshape, fillcolor="#cae6df", style="filled")
    for lat in latent:
        g.node(lat, label=lat)
    g.attr("node", shape="box", style="")
    for obs in observed:
        g.node(obs, label=obs)
    for rval, lval, label in edges:
        g.edge(rval, lval, label=label)
    for rval, lval, label in covs:
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


def _layered_positions(nx, order, arrows, dx=3.6, dy=1.7):
    """
    Left-to-right layers for a path diagram: predictors on the left, outcomes on the right.

    The layer of a variable is the length of the longest path that leads to it (variables
    in a feedback loop share a layer; variables without predictors are moved next to their
    first outcome). Within a layer the order is improved by a few barycenter sweeps to
    reduce crossing edges. Deterministic. Returns ``{variable: (x, y)}``.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    graph.add_edges_from(arrows)
    condensed = nx.condensation(graph)  # feedback loops collapse into one node
    mapping = condensed.graph["mapping"]
    layer_of = {}
    for k, generation in enumerate(nx.topological_generations(condensed)):
        for component in generation:
            layer_of[component] = k
    layer = {n: layer_of[mapping[n]] for n in graph}
    for n in order:
        if graph.in_degree(n) == 0 and graph.out_degree(n) > 0:
            layer[n] = min(layer[s] for s in graph.successors(n)) - 1

    layers = {}
    for n in order:
        layers.setdefault(layer[n], []).append(n)
    ks = sorted(layers)
    y = {}

    def recenter(k):
        for i, n in enumerate(layers[k]):
            y[n] = (len(layers[k]) - 1) / 2 - i

    for k in ks:
        recenter(k)
    undirected = graph.to_undirected()
    for sweep in range(8):
        for k in ks if sweep % 2 == 0 else ks[::-1]:

            def key(n, k=k):
                above = [y[v] for v in undirected[n] if layer[v] != k]
                return -(sum(above) / len(above)) if above else -y[n]

            layers[k].sort(key=key)  # stable: ties keep the order of appearance
            recenter(k)
    return {n: (layer[n] * dx, y[n] * dy) for n in graph}


def networkx_plot(
    stats,
    save_to_file=None,
    plot_covs=True,
    std_ests=True,
    plot_ests=True,
    layout="layered",
    show_stars=False,
    format_pval=True,
    format_fig="png",
    decimal=3,
    dpi=600,
    seed=13,
    figsize=None,
    fontsize=10,
    **kwargs,
):
    """
    Draw a SEM path diagram with networkx and matplotlib (the ``"networkx"`` engine of :func:`sem_plot`).

    The same diagram as :func:`semopy_plot` (circles for latent and boxes for observed variables,
    arrows from predictor / indicator to outcome / latent, dashed double-headed arrows for
    covariances, edge labels with the estimate and the p-value or stars) but drawn in Python:
    no Graphviz executable is needed, the output is a matplotlib figure and the resolution can be
    set with ``dpi``. Edges with the intercept pseudo-variable ``"1"`` and variances are not
    drawn.

    Parameters
    ----------
    stats : pandas.DataFrame
        Parameter table from :func:`sem` (latent variables are read from
        ``stats.attrs["latent"]``, as in :func:`semopy_plot`).
    save_to_file : str or None
        Write the figure to this path (``dpi`` resolution, cropped to the diagram). The file type
        follows the extension; ``format_fig`` is used if the path has none.
    plot_covs, std_ests, plot_ests, show_stars, format_pval, decimal
        As in :func:`sem_plot` / :func:`semopy_plot`.
    layout : {"layered", "spring", "kamada_kawai", "circular"}, default "layered"
        ``"layered"``: predictors on the left, outcomes on the right (latent variables between
        their causes and their indicators), ordered to reduce crossing edges; deterministic.
        The others are generic networkx layouts (``"spring"`` is seeded with ``seed``).
    format_fig : str, default "png"
        File type if ``save_to_file`` has no extension.
    dpi : int, default 600
        Resolution of the saved file.
    seed : int, default 13
        Seed for numpy's global random generator and for the ``"spring"`` layout.
    figsize : tuple of float, optional
        Figure size in inches; by default it follows the size of the diagram.
    fontsize : float, default 10
        Font size of the variable names (edge labels are one point smaller).
    **kwargs
        Accepted and ignored (lets :func:`sem_plot` forward its extra options).

    Returns
    -------
    matplotlib.figure.Figure
        The figure (created with ``pyplot``: show it with ``plt.show()`` or close it with
        ``plt.close(fig)``).

    Raises
    ------
    ImportError
        If networkx or matplotlib is not installed (``pip install "apatools[networkx]"``).
    ValueError
        For an unknown ``layout``.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'networkx_plot needs the optional dependencies networkx and matplotlib: pip install "apatools[networkx]"'
        ) from e

    layouts = ("layered", "spring", "kamada_kawai", "circular")
    if layout not in layouts:
        raise ValueError(f"Unknown layout '{layout}', use one of {layouts}.")
    np.random.seed(seed)
    latent, _observed, edges, covs, order = _path_diagram(
        stats, plot_covs, std_ests, plot_ests, show_stars, format_pval, decimal
    )
    arrows = [(rval, lval) for rval, lval, _ in edges]
    if layout == "layered":
        pos = _layered_positions(nx, order, arrows)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(order)
        graph.add_edges_from(arrows + [(rval, lval) for rval, lval, _ in covs])
        raw = {
            "spring": lambda: nx.spring_layout(graph, seed=seed),
            "kamada_kawai": lambda: nx.kamada_kawai_layout(graph),
            "circular": lambda: nx.circular_layout(graph),
        }[layout]()
        scale = 1.7 * np.sqrt(len(order)) + 1  # same units as the layered layout
        pos = {n: (float(x) * scale, float(y) * scale) for n, (x, y) in raw.items()}

    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    pad = 1.9  # room for the nodes and for curved edges
    xlim, ylim = (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)
    if figsize is None:  # about 0.7 inch per unit of the layout
        figsize = (
            max(4.0, 0.7 * (xlim[1] - xlim[0])),
            max(3.0, 0.7 * (ylim[1] - ylim[0])),
        )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")  # circles stay circles, curved edges keep their shape
    ax.set_axis_off()

    texts = {}
    for n in order:
        is_latent = n in latent
        texts[n] = ax.text(
            *pos[n],
            n,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
            bbox={
                "boxstyle": "circle,pad=0.5" if is_latent else "square,pad=0.5",
                "fc": "#cae6df" if is_latent else "white",
                "ec": "black",
                "lw": 1,
            },
        )

    def draw_edge(rval, lval, label, rad, arrowstyle, linestyle, t=0.5):
        (x1, y1), (x2, y2) = pos[rval], pos[lval]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            zorder=3,  # after the nodes: their boxes must be placed to clip the arrows
            arrowprops={
                "arrowstyle": arrowstyle,
                "color": "black",
                "lw": 1,
                "linestyle": linestyle,
                "patchA": texts[rval].get_bbox_patch(),  # the arrow starts / ends at the node borders
                "patchB": texts[lval].get_bbox_patch(),
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 12,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )
        if label:
            # point at fraction t of the curve (matplotlib's arc3 is a quadratic bezier curve
            # with the control point at the midpoint shifted by rad)
            cx = (x1 + x2) / 2 + rad * (y2 - y1)
            cy = (y1 + y2) / 2 - rad * (x2 - x1)
            mx = (1 - t) ** 2 * x1 + 2 * t * (1 - t) * cx + t**2 * x2
            my = (1 - t) ** 2 * y1 + 2 * t * (1 - t) * cy + t**2 * y2
            ax.text(
                mx,
                my,
                label,
                ha="center",
                va="center",
                fontsize=fontsize - 1,
                zorder=3.5,  # above the arrows, so that lines do not cross the text
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
            )

    both_ways = {(a, b) for a, b in arrows if (b, a) in set(arrows)}
    out_degree = {n: sum(a == n for a, _ in arrows) for n in order}
    in_degree = {n: sum(b == n for _, b in arrows) for n in order}
    for rval, lval, label in edges:
        rad = 0.2 if (rval, lval) in both_ways else 0.0  # feedback pairs do not overlap
        # the label goes to the end of the edge where the edges of the crowded node have fanned out
        t = 0.7 if out_degree[rval] >= in_degree[lval] else 0.3
        draw_edge(rval, lval, label, rad, "-|>", "-", t)
    for rval, lval, label in covs:
        draw_edge(rval, lval, label, 0.35, "<|-|>", "--")

    if save_to_file:
        suffix = os.path.splitext(str(save_to_file))[1]
        fig.savefig(
            save_to_file,
            dpi=dpi,
            bbox_inches="tight",
            format=None if suffix else format_fig,
        )
    return fig
