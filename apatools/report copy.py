# !pip install factor-analyzer
# !pip install semopy
# !pip install lxml

import math, random, scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import decomposition, preprocessing
from itertools import zip_longest

from factor_analyzer import FactorAnalyzer
from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
import semopy as sem


### FACTOR ANALYSIS
def factor_analysis(data, n_factors=1, rotation=None, sort=False, rename_dict=None):
    d, cols = data.dropna(), data.columns.to_list()
    d = preprocessing.scale(d)
    fa = FactorAnalyzer(rotation=rotation, n_factors=1)
    fa.fit(d)
    t0 = pd.concat([pd.Series([i if not rename_dict else rename_dict[i] for i in cols]),
                    pd.DataFrame(fa.loadings_)], axis=1)
    t0.columns = ['index', 'f0']
    t0.set_index('index', inplace=True)
    t0 = t0.apply(lambda x: x.apply(lambda y: f'{y:.2f}'))
    #
    fa = FactorAnalyzer(rotation=None, n_factors=n_factors)
    fa.fit(d)
    t1 = pd.concat([pd.Series([i if not rename_dict else rename_dict[i] for i in cols]),
                    pd.DataFrame(fa.loadings_)], axis=1)
    t1.columns = ['index'] + [f'f{i+1}' for i in range(fa.n_factors)]
    t1.set_index('index', inplace=True)
    t1 = t1.apply(lambda x: x.apply(lambda y: f'{y:.2f}'))
    #
    f = pd.concat([t0, t1], ignore_index=False, axis=1)
    if sort:
        f.sort_values(by=sort, ascending=False, inplace=True)
    return f

### PLOTS
def errorbars(data, method='std'):
    if method == 'std':
        y = np.std(data)
    return y

def plot_ecdf(data, xlabel=None, ylabel=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.ecdfplot(data = data, legend = data)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel("ECDF")
    else:
        plt.ylabel(ylabel)
    #plt.show()


### SEM

# https://arxiv.org/pdf/2106.01140.pdf
# formula = f"""
# ПП =~ {' + '.join(p)}
# НЭ =~ {' + '.join(e)}
# ПП ~ НЭ + {' + '.join(x)} + Группа
# """

def run_sem(data, formula, obj='MLW', solver='SLSQP', bootstrap=False, plot_covs=True, standardized=True, save_to_file='sem.pdf', seed=48):
    random.seed(seed)
    np.random.seed(seed)
    model = sem.Model(formula)
    model.fit(data, obj=obj, solver=solver) #MLW ULS GLS FIML DWLS WLS
    if bootstrap:
        sem.bias_correction(model, n=bootstrap, resample_mean=True)
    g = sem.semplot(model, save_to_file, plot_covs=plot_covs, std_ests=standardized, show=True)
    return g, sem.calc_stats(model).T, model.inspect()
