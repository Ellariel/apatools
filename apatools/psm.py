import numpy as np
import pandas as pd
from psmpy import PsmPy
from psmpy.functions import cohenD

from .citation import Citation


# https://pypi.org/project/psmpy/
# https://github.com/adriennekline/psmpy
CITATION = Citation(APA='Kline, A., & Luo, Y. (2022). \
PsmPy: A Package for Retrospective Cohort Matching in Python. \
2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 1354-1357), \
Glasgow, Scotland, United Kingdom. \
https://doi.org/10.1109/EMBC48229.2022.9871333')


def psm_effect_size(model):
    df_preds_after = model.df_matched[[model.treatment] + model.xvars]
    df_preds_b4 = model.data[[model.treatment] + model.xvars]
    df_preds_after_float = df_preds_after.astype(float)
    df_preds_b4_float = df_preds_b4.astype(float)

    data = []
    for cl in model.xvars:
        data.append([cl, 'before', cohenD(
                df_preds_b4_float, model.treatment, cl)])
        data.append([cl, 'after', cohenD(
                df_preds_after_float, model.treatment, cl)])
    return pd.DataFrame(
            data, columns=['variable', 'matching', 'effect_size'])


def psm(
    data,
    treatment_variable,
    include_vars=[],
    exclude_vars=[],
    index_variable=None,
    balance=True,
    replacement=True,
    return_model=False,
    seed=13,
):
    np.random.seed(seed)
    df = data.copy()
    if index_variable is None:
        index_variable = 'index'
        df = df.reset_index(drop=True).reset_index()
    all_vars = []
    if not isinstance(include_vars, list):
        include_vars = [include_vars]
    if not isinstance(exclude_vars, list):
        exclude_vars = [exclude_vars]
    if len(include_vars):
        all_vars = set(data.columns) - set(include_vars) - {index_variable, treatment_variable}
    else:
        if len(exclude_vars):
            all_vars = set(exclude_vars)

    model = PsmPy(df, treatment=treatment_variable, 
                indx=index_variable, 
                exclude=list(all_vars))
    model.logistic_ps(balance=balance)
    model.knn_matched(matcher='propensity_logit', 
                    replacement=replacement, 
                    caliper=None, 
                    drop_unmatched=True)
    df = pd.concat([pd.merge(model.matched_ids[[index_variable]], df, 
                    how='left', on=index_variable),
        pd.merge(model.matched_ids[['matched_ID']]\
                    .rename(columns={'matched_ID': index_variable}), df, 
                        how='left', on=index_variable)])
    if return_model:
        return df, model
    return df
