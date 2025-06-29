import numpy as np
import pandas as pd
from psmpy import PsmPy



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
    if not isinstance(include_vars, list):
        include_vars = [include_vars]
    if not isinstance(exclude_vars, list):
        exclude_vars = [exclude_vars]
    if len(include_vars):
        var_list = set(data.columns) - set(include_vars)
    else:
        if len(exclude_vars):
            var_list = set(exclude_vars)
    df = data.copy()
    if index_variable is None:
        index_variable = 'index'
        df = df.reset_index(drop=True).reset_index()
    psm = PsmPy(df, treatment=treatment_variable, 
                indx=index_variable, 
                exclude=var_list)
    psm.logistic_ps(balance=balance)
    psm.knn_matched(matcher='propensity_logit', 
                    replacement=replacement, 
                    caliper=None, 
                    drop_unmatched=True)
    df = pd.concat([pd.merge(psm.matched_ids[[index_variable]], df, 
                    how='left', on=index_variable)\
                        .drop(index_variable, axis=1),
        pd.merge(psm.matched_ids[['matched_ID']]\
                    .rename(columns={'matched_ID': index_variable}), df, 
                        how='left', on=index_variable)\
                            .drop(index_variable, axis=1)])
    if return_model:
        return df, psm
    return df
