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
    drop_unmatched=True,
    return_coupled=False,
    coupled_suffix='_',
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
    
    if hasattr(model, "knn_matched"):
        matched_method = getattr(model, "knn_matched") # <= v0.3.15
    else:
        matched_method = getattr(model, "kdtree_matched") # >= v0.3.16
            
    matched_method(matcher='propensity_logit', 
                    caliper=None, 
                    replacement=replacement, 
                    drop_unmatched=drop_unmatched)
    a = pd.merge(model.matched_ids[[index_variable]], df, 
                    how='left', on=index_variable)
    b = pd.merge(model.matched_ids[['matched_ID']]\
                    .rename(columns={'matched_ID': index_variable}), df, 
                        how='left', on=index_variable)
    if return_coupled:
        b.columns = [i + coupled_suffix if i in a else i for i in b.columns]
        df = pd.concat([a, b], axis=1)
    else:
        df = pd.concat([a, b], axis=0)
    if return_model:
        return df, model
    return df
