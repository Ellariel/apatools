# A Toolbox for Meta-Analysis and APA-Compliant Reporting

## About
This toolbox is a curated collection of utilities I frequently use for statistical analysis and nicely reporting research findings in accordance with APA standards. While some of the included libraries may be slightly outdated, they remain functional and reliable, making them my go-to tools.

## Install
```shell
pip install git+https://github.com/ellariel/apatools.git
```

## Content and examples
* ***apatools.calc*** - *a basic module for statistical conversions and bootstrapping*
* ***apatools.lm*** - *a module offering wrapper functions for statsmodels, streamlining the fitting of OLS, RLM, and GLM models, as well as generating detailed tables*
* ***apatools.slm*** - *a module offering wrapper functions for spreg, streamlining the fitting of OLS, spatial lag (SLM) and spatial error (SEM) models, as well as generating detailed tables*
* ***apatools.sem*** - *a module for performing structural equation modeling, complete with result reporting and path diagram visualization*
* ***apatools.psm*** - *a module for performing propensity score matching, complete with effect sizes reporting*
* ***apatools.format*** - *simple APA-compliant formatting functions for numerical results*

### *apatools.calc*

This module provides essential conversion functions alongside tools for bootstrapping means, confidence intervals, and related statistics. It includes a comprehensive set of approximate conversion functions for bidirectional transformations between t-, z-, and F-statistics, as well as conversions involving Cohen's r, Cohen's d, and p-values.

#### Conversion examples
```python
from apatools.calc import cohen_r_from_z, cohen_d_from_z
# calculation Cohen's r and d from z-score:
# there is also a shorten aliases r_from_z and d_from_z
z = -1.472
r = cohen_r_from_z(z) # r = -0.9
d = cohen_d_from_z(z) # d = -4.1
```

```python
from apatools.calc import p_from_r
# calculation p-value for Cohen's r  or correlation coefficient:
r = -0.79
n = 100
p = p_from_r(r, n) # p = 0.0
```

#### Bootstrap examples
```python
import numpy as np
from apatools.calc import bootstrap
# bootstraping confidence intervals for a mean value:
cil, m, cir = bootstrap([1, 2, 3, 4, 5, 6, 7, 8, 9],
        func=np.mean,
        n_rep=1000) # cil = 3.3, m = 5.0, cir = 6.7
```

```python
from apatools.calc import bootstrap
from scipy.stats import spearmanr
# bootstraping value for Spearman's correlation:
def func(*args):
    return spearmanr(*args).statistic

_, m, _ = bootstrap(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 8, 4, 5, 6, 7, 8, 9],
    func=func,
    n_rep=1000) # m = 0.75
```


### *apatools.lm*
Fitting of OLS, RLM, GLM, QuantReg and MixedLM models using *statsmodels*. Arguments for the model constructor and for `.fit()` are passed as `<model>_<stage>_<argument>`: the model prefix (`ols_`, `rlm_`, `glm_`, `qlm_`, `mlm_`) followed by `model_` (constructor) or `fit_` (the `.fit()` call), e.g. `ols_fit_cov_type`, `glm_model_family`, `rlm_model_M`.

Specific parameters:
* `qlm_fit_q` – adds a quantile for QuantReg, e.g. 0.5
* `mlm_model_groups` – adds a grouping variable for MixedLM, e.g. `country`
* `vif` - adds a VIF column to the metrics dictionary and reported table
* `formula` - a formula string as in *statsmodels*
* `pred_r_sq` - calculates [predicted R²](https://blog.minitab.com/en/adventures-in-statistics-2/multiple-regession-analysis-use-adjusted-r-squared-and-predicted-r-squared-to-include-the-correct-number-of-variables#:~:text=What%20Is%20the%20Predicted%20R,valid%20predictions%20for%20new%20observations.)

```python
from apatools.lm import lm, lm_report
results, metrics = lm(
        data,
        y, x,
        model=["ols", "rlm", "glm"],
        verbose=True,
        vif=True,
        pred_r_sq=True,
        ols_fit_cov_type="HC1",
        rlm_model_M=sm.robust.norms.RamsayE(),
        glm_model_family=sm.families.Gaussian(),
    )
results_rep = lm_report(results, metrics)
print(results_rep, metrics)
```


### *apatools.slm*
Fitting of OLS, SLM, and SEM spatial models using *spreg*. Arguments for the spreg model constructor are passed as `<model>_model_<argument>` with the model prefix `ols_`, `slm_` or `sem_`, e.g. `ols_model_spat_diag`.

Specific parameters:
* `w` - a spatial matrix, created with *libpysal*; its rows must correspond to the rows of `data` (rows dropped because of NAs are removed from `w` automatically)
* `formula` - a formula string as in *statsmodels*
* `pred_metrics` - adds leave-one-out prediction metrics (`R²pred`, `MAEpred`, `MADpred`) to the metrics dictionary and the reported table. Every observation is held out in turn, the model and the weights are refitted without it, and the held-out value is predicted: `x·β` for OLS and SEM, and `ρ·(Wy)ᵢ + x·β` for SLM (observed neighbour values enter through the weights, so SLM is not strictly comparable with the other two). It needs one fit per observation and model; for large samples pass e.g. `slm_model_method="LU"`.
* `pred_exclude_neighbors` - with `pred_metrics`: also remove the neighbours of the held-out observation from the training sample

> **Caveat:** plain leave-one-out is *optimistic* for spatial data. The neighbours of the held-out observation stay in the training sample and are correlated with it, so `R²pred` overestimates how well the model predicts a new, unseen region. `pred_exclude_neighbors=True` gives a more honest (slightly pessimistic) estimate; it may create islands in the training weights, which is reported with a warning.


```python
from apatools.slm import slm, slm_report
results, metrics = slm(
        data,
        y, x, w,
        model=["ols", "slm", "sem"],
        verbose=True,
        ols_model_spat_diag=True,
        ols_model_nonspat_diag=True,
        pred_metrics=True,  # leave-one-out R²pred etc. (optimistic, see the caveat)
        pred_exclude_neighbors=True,  # leave the neighbourhood out as well
    )
results_rep = slm_report(results, metrics)
print(results_rep, metrics)
```


### *apatools.sem*
Structural equation modeling implemented with *semopy*. The module provides modeling function `sem()`, formatting `sem_report()` and plotting `sem_plot()` functions.

```python
from apatools.sem import sem, sem_report, sem_plot
stats, metrics, model = sem(
        data,
        formula,
        method="MLW",  # MLW ULS GLS FIML DWLS WLS
        return_model=True,
    )
results_rep = sem_report(stats, metrics)
print(results_rep)
fig = sem_plot(
        stats,
        save_to_file="fig.png",
        return_fig=True,
    )
```


### *apatools.psm*
Propensity score matching implemented with *psmpy*. The module provides modeling function `psm()` and `psm_effect_size()` to report differences.

```python
from apatools.psm import psm, psm_effect_size
results, model = psm(
        data,
        treatment_variable='vs',
        include_vars=['age', 'gender'],
        balance=True,
        replacement=True,
        return_model=True,
    )
print(psm_effect_size(model))
```


### *apatools.format*
Simple APA-compliant functions for formatting numeric results, such as formatting R and r values, p-values, etc.


#### Formatting examples
```python
from apatools.format import get_stars, format_p, format_r
# formatting numeric results
r = format_r(0.999, keep_spaces=True, use_letter="r", no_equals=False) # r = 'r = 1.'
r = format_r(0.011, keep_spaces=True, use_letter="R", no_equals=False) # r = 'R = .01'
p = format_p(0.0004, use_letter="p", keep_spaces=True, no_equals=False) # p = 'p < .001'
s = get_stars(0.04, p001="***", p01="**", p05="*", p10="⁺", p_="") # s = '*'
```


## Sources and references
* [factor-analyzer](https://pypi.org/project/factor-analyzer/) (GPL-2.0 License)
* [statsmodels](https://www.statsmodels.org/stable/) (BSD License)
* [semopy](https://pypi.org/project/semopy/) (MIT License)
* [spreg](https://pypi.org/project/spreg/) (BSD License)
* [psmpy](https://pypi.org/project/psmpy/) (MIT License)
* [scipy](https://pypi.org/project/scipy/) (BSD License)

Note that APA citations of the original packages can be imported from each specific implementation as follows:
```python
# e.g., for psmpy
from apatools.psm import CITATION
print(CITATION.APA) 
```

