"""Propensity score matching (PSM) with `psmpy <https://github.com/adriennekline/psmpy>`_.

:func:`psm` estimates propensity scores by logistic regression and matches every treated
observation to its nearest control on the logit of the score; :func:`psm_effect_size`
reports covariate balance (standardized mean differences) before and after matching.

Requires the optional dependencies ``pandas`` and ``psmpy`` (``pip install "apatools[psm]"``).

References
----------
Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational
studies for causal effects. *Biometrika, 70*(1), 41-55. https://doi.org/10.1093/biomet/70.1.41

Kline, A., & Luo, Y. (2022). PsmPy: A package for retrospective cohort matching in Python.
*2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology
Society (EMBC)* (pp. 1354-1357). https://doi.org/10.1109/EMBC48229.2022.9871333
"""

import inspect

import numpy as np

from ._deps import missing_dependency

try:
    import pandas as pd
    from psmpy import PsmPy
    from psmpy.functions import cohenD
except ImportError as e:  # pragma: no cover
    missing_dependency("psm", "psm", e)


# https://pypi.org/project/psmpy/
# https://github.com/adriennekline/psmpy
def psm_effect_size(model):
    """Standardized mean differences (Cohen's d) of the covariates before and after matching.

    For every covariate the absolute standardized mean difference between treated
    (``1``) and control (``0``) observations is computed with :func:`psmpy.functions.cohenD`
    on the full data (before matching) and on the matched sample (after matching).

    Parameters
    ----------
    model : psmpy.PsmPy
        Fitted and matched ``PsmPy`` object, i.e. the second return value of
        ``psm(..., return_model=True)``.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per covariate and stage (two rows per covariate,
        ``before`` first) and columns ``variable``, ``matching`` (``"before"`` or
        ``"after"``) and ``effect_size`` (float, never negative).

    Notes
    -----
    The statistic is Cohen's d with the pooled standard deviation,
    ``|mean_treated - mean_control| / sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))``
    (Cohen, 1988). If the pooled standard deviation is 0, psmpy divides by ``1e-6``
    instead, which yields a very large value for any difference. Covariates are cast to
    float, so they must be numeric (dummy-code categorical variables first). Values below
    about 0.1 after matching are commonly taken to indicate adequate balance
    (Austin, 2011); this is a convention, not a test.

    References
    ----------
    Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence
    Erlbaum Associates. (Reprinted 2013 by Routledge, https://doi.org/10.4324/9780203771587)

    Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of
    confounding in observational studies. *Multivariate Behavioral Research, 46*(3), 399-424.
    https://doi.org/10.1080/00273171.2011.568786
    """
    df_preds_after = model.df_matched[[model.treatment] + model.xvars]
    df_preds_b4 = model.data[[model.treatment] + model.xvars]
    df_preds_after_float = df_preds_after.astype(float)
    df_preds_b4_float = df_preds_b4.astype(float)

    data = []
    for cl in model.xvars:
        data.append([cl, "before", cohenD(df_preds_b4_float, model.treatment, cl)])
        data.append([cl, "after", cohenD(df_preds_after_float, model.treatment, cl)])
    return pd.DataFrame(data, columns=["variable", "matching", "effect_size"])


def _psmpy_seed(seed):
    """``{"seed": seed}`` if this psmpy version accepts a seed (older ones do not), else ``{}``."""
    if "seed" in inspect.signature(PsmPy.__init__).parameters:
        return {"seed": seed}
    return {}


def psm(
    data,
    treatment_variable,
    include_vars=None,
    exclude_vars=None,
    index_variable=None,
    balance=True,
    replacement=True,
    drop_unmatched=True,
    return_coupled=False,
    coupled_suffix="_",
    return_model=False,
    seed=13,
):
    """Propensity score matching (logistic PS, nearest neighbour on logit) with psmpy.

    The propensity score (probability of treatment given the covariates) is estimated
    with a logistic regression; each treated observation is then matched to the control
    with the nearest *logit* of the propensity score (``matcher="propensity_logit"``,
    no caliper). Matching is one-to-one; with ``replacement=True`` a control may be used
    for several treated observations and then appears several times in the output.

    Parameters
    ----------
    data : pandas.DataFrame
        Data with the treatment indicator and the covariates.
    treatment_variable : str
        Binary column (1 = treated, 0 = control).
    include_vars : str or list, optional
        Covariates used for the propensity model. All other columns (except the
        treatment and index columns) are excluded.
    exclude_vars : str or list, optional
        Columns to leave out of the propensity model; all remaining columns are used as
        covariates. Used only if ``include_vars`` is empty; if both are given,
        ``include_vars`` wins. If neither is given, all columns are covariates.
    index_variable : str, optional
        Unique id column; if None, the row position is used (new column ``index``
        added in front of the other columns of the output).
    balance : bool, default True
        Fit the logistic model(s) on balanced data as psmpy does: the larger group is
        split into chunks of the size of the smaller group and one model is fitted per
        chunk (``n_control == n_treated`` in each fit); useful when the groups differ
        strongly in size. False fits one model on all rows.
    replacement : bool, default True
        Match controls with replacement.
    drop_unmatched : bool, default True
        Drop observations that could not be matched.
    return_coupled : bool, default False
        True: return treated and matched control rows side by side (columns of the
        control rows get ``coupled_suffix`` appended); False: stack them vertically
        (treated rows first, then matched controls).
    coupled_suffix : str, default "_"
        Suffix for the columns of the matched controls if ``return_coupled`` is True.
    return_model : bool, default False
        Also return the fitted ``PsmPy`` object (needed for :func:`psm_effect_size`).
    seed : int, default 13
        Seed of psmpy's own sampling (the random splitting of the larger group when
        ``balance=True``) and of numpy's *global* random generator (changed as a side
        effect). The same ``seed`` gives the same matched sample.

    Returns
    -------
    pandas.DataFrame
        The matched sample (treated rows and matched control rows, see
        ``return_coupled``) with all original columns plus the ``index`` column if
        ``index_variable`` was None.
    psmpy.PsmPy
        The fitted model; only if ``return_model`` is True (returned as a tuple
        ``(df, model)``).

    Notes
    -----
    Propensity scores were introduced by Rosenbaum and Rubin (1983). Matching without a
    caliper can pair dissimilar observations; check balance with :func:`psm_effect_size`.
    Austin (2011) recommends a caliper of 0.2 standard deviations of the logit of the
    propensity score, which this function does not apply.

    Requires the optional dependency ``psmpy`` (``pip install "apatools[psm]"``).

    References
    ----------
    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in
    observational studies for causal effects. *Biometrika, 70*(1), 41-55.
    https://doi.org/10.1093/biomet/70.1.41

    Austin, P. C. (2011). Optimal caliper widths for propensity-score matching when estimating
    differences in means and differences in proportions in observational studies. *Pharmaceutical
    Statistics, 10*(2), 150-161. https://doi.org/10.1002/pst.433

    Kline, A., & Luo, Y. (2022). PsmPy: A package for retrospective cohort matching in
    Python. *2022 44th Annual International Conference of the IEEE Engineering in
    Medicine & Biology Society (EMBC)* (pp. 1354-1357).
    https://doi.org/10.1109/EMBC48229.2022.9871333
    """
    np.random.seed(seed)
    df = data.copy()
    if index_variable is None:
        index_variable = "index"
        df = df.reset_index(drop=True).reset_index()
    all_vars = []
    include_vars = [] if include_vars is None else include_vars
    exclude_vars = [] if exclude_vars is None else exclude_vars
    if not isinstance(include_vars, list):
        include_vars = [include_vars]
    if not isinstance(exclude_vars, list):
        exclude_vars = [exclude_vars]
    if len(include_vars):
        all_vars = (
            set(data.columns) - set(include_vars) - {index_variable, treatment_variable}
        )
    else:
        if len(exclude_vars):
            all_vars = set(exclude_vars)

    model = PsmPy(
        df,
        treatment=treatment_variable,
        indx=index_variable,
        exclude=list(all_vars),
        **_psmpy_seed(seed),
    )
    model.logistic_ps(balance=balance)

    if hasattr(model, "knn_matched"):
        matched_method = model.knn_matched  # <= v0.3.15
    else:
        matched_method = model.kdtree_matched  # >= v0.3.16

    matched_method(
        matcher="propensity_logit",
        caliper=None,
        replacement=replacement,
        drop_unmatched=drop_unmatched,
    )
    a = pd.merge(model.matched_ids[[index_variable]], df, how="left", on=index_variable)
    b = pd.merge(
        model.matched_ids[["matched_ID"]].rename(
            columns={"matched_ID": index_variable}
        ),
        df,
        how="left",
        on=index_variable,
    )
    if return_coupled:
        b.columns = [i + coupled_suffix if i in a else i for i in b.columns]
        df = pd.concat([a, b], axis=1)
    else:
        df = pd.concat([a, b], axis=0)
    if return_model:
        return df, model
    return df
