"""What to cite: the literature behind the software that apatools uses.

``cite("sem", engine=["lavaan", "networkx"])`` returns the references (APA 7th edition) of the software that
did the work for a module, so that they can be cited when results are reported: name the engines you used,
for fitting and for drawing alike. Every reference is written once, in :data:`REFERENCES`; :data:`ENGINES` says
which of them an engine needs. This module needs no optional dependency.

Examples
--------
>>> from apatools.citation import cite
>>> cite("psm")[0].startswith("Kline, A., & Luo, Y. (2022)")
True
>>> [r[:6] for r in cite("sem", engine=["lavaan", "semPlot"])]
['Epskam', 'R Core', 'Rossee']
"""

REFERENCES = {
    "statsmodels": (
        "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. "
        "In S. van der Walt & J. Millman (Eds.), Proceedings of the 9th Python in Science Conference "
        "(pp. 92\u201396). https://doi.org/10.25080/Majora-92bf1922-011"
    ),
    "pysal": (
        "Rey, S. J., & Anselin, L. (2007). PySAL: A Python library of spatial analytical methods. "
        "The Review of Regional Studies, 37(1), 5\u201327. https://doi.org/10.52324/001c.8285"
    ),
    "linearmodels": (  # the DOI of all versions; every release also has a DOI of its own on Zenodo
        "Sheppard, K., & contributors. (2026). linearmodels: Linear models for panel data, instrumental "
        "variables and system regression [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.2591973"
    ),
    "semopy": (
        "Igolkina, A. A., & Meshcheryakov, G. (2020). semopy: A Python package for structural equation "
        "modeling. Structural Equation Modeling: A Multidisciplinary Journal, 27(6), 952\u2013963. "
        "https://doi.org/10.1080/10705511.2019.1704289"
    ),
    "lavaan": (
        "Rosseel, Y. (2012). lavaan: An R package for structural equation modeling. "
        "Journal of Statistical Software, 48(2), 1\u201336. https://doi.org/10.18637/jss.v048.i02"
    ),
    "semPlot": (
        "Epskamp, S. (2015). semPlot: Unified visualizations of structural equation models. "
        "Structural Equation Modeling: A Multidisciplinary Journal, 22(3), 474\u2013483. "
        "https://doi.org/10.1080/10705511.2014.937847"
    ),
    "plm": (
        "Croissant, Y., & Millo, G. (2008). Panel data econometrics in R: The plm package. "
        "Journal of Statistical Software, 27(2), 1–43. https://doi.org/10.18637/jss.v027.i02"
    ),
    "splm": (
        "Millo, G., & Piras, G. (2012). splm: Spatial panel data models in R. "
        "Journal of Statistical Software, 47(1), 1–38. https://doi.org/10.18637/jss.v047.i01"
    ),
    "spdep": (
        "Bivand, R. S., & Wong, D. W. S. (2018). Comparing implementations of global and local indicators of "
        "spatial association. TEST, 27(3), 716–748. https://doi.org/10.1007/s11749-018-0599-x"
    ),
    "R": (
        "R Core Team. (2025). R: A language and environment for statistical computing [Computer software]. "
        "R Foundation for Statistical Computing. https://www.R-project.org/"
    ),
    "psmpy": (
        "Kline, A., & Luo, Y. (2022). PsmPy: A package for retrospective cohort matching in Python. "
        "In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology "
        "Society (EMBC) (pp. 1354\u20131357). https://doi.org/10.1109/EMBC48229.2022.9871333"
    ),
    "networkx": (
        "Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and "
        "function using NetworkX. In G. Varoquaux, T. Vaught, & J. Millman (Eds.), Proceedings of the 7th "
        "Python in Science Conference (pp. 11\u201315). https://doi.org/10.25080/TCWV9851"
    ),
    "matplotlib": (
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, "
        "9(3), 90\u201395. https://doi.org/10.1109/MCSE.2007.55"
    ),
    "graphviz": (
        "Gansner, E. R., & North, S. C. (2000). An open graph visualization system and its applications "
        "to software engineering. Software: Practice and Experience, 30(11), 1203\u20131233. "
        "https://doi.org/10.1002/1097-024X(200009)30:11<1203::AID-SPE338>3.0.CO;2-N"
    ),
}

# module -> engine -> names in REFERENCES; the first engine of a module is its default
ENGINES = {
    "lm": {"statsmodels": ["statsmodels"]},
    "slm": {"spreg": ["pysal"]},
    "plm": {
        "spreg": ["pysal"],
        "linearmodels": ["pysal", "linearmodels"],
        "splm": [
            "R",
            "plm",
            "splm",
            "spdep",
        ],  # R packages plm (non-spatial models), splm (spatial) and spdep (weights)
    },
    "sem": {
        "semopy": [
            "semopy"
        ],  # fitting (and the Graphviz drawing of semopy_plot: add "graphviz")
        "lavaan": ["R", "lavaan"],
        "semPlot": ["R", "semPlot"],  # drawing with R
        "networkx": ["networkx", "matplotlib"],  # drawing in Python
        "graphviz": ["graphviz"],
    },
    "psm": {"psmpy": ["psmpy"]},
}


def cite(module, engine=None):
    """
    APA references of the software behind ``module``, without duplicates, in alphabetical order.

    Parameters
    ----------
    module : {"lm", "slm", "plm", "sem", "psm"}
        The apatools module whose results are reported.
    engine : str or list of str, optional
        The engines (software) that were used; list all that should be cited. By default the first one:
        ``"statsmodels"`` (``lm``), ``"spreg"`` (``slm``, ``plm``; also ``"linearmodels"`` and ``"splm"``, the R packages), ``"semopy"``
        (``sem``; also ``"lavaan"`` for fitting and ``"semPlot"``, ``"networkx"``, ``"graphviz"`` for the
        path diagram) and ``"psmpy"`` (``psm``).

    Returns
    -------
    list of str
        The references, as APA 7th edition strings. Check the year / version of the software you used.

    Raises
    ------
    ValueError
        For an unknown module or engine (the message lists the valid ones).
    """
    if module not in ENGINES:
        raise ValueError(f"Unknown module '{module}', use one of {sorted(ENGINES)}.")
    available = ENGINES[module]
    wanted = (
        [next(iter(available))]
        if engine is None
        else [engine]
        if isinstance(engine, str)
        else list(engine)
    )
    names = set()
    for name in wanted:
        if name not in available:
            raise ValueError(
                f"Unknown engine '{name}' for '{module}', use one of {sorted(available)}."
            )
        names.update(available[name])
    return sorted(REFERENCES[name] for name in names)
