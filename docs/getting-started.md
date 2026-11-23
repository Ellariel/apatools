# Getting started

## Install

```shell
pip install apatools
```

Python 3.10 or newer. The development version: `pip install git+https://github.com/ellariel/apatools.git`.

The base installation is light: it contains `apatools.calc` and `apatools.format` and needs only numpy and scipy. Every
other module has its own, heavier dependencies and is installed through an extra of the same name; importing a module
without its extra gives an `ImportError` that names the extra to install. Extras can be combined, e.g.
`pip install "apatools[lm,plm]"`.

| install | module | brings in |
|---|---|---|
| `pip install apatools` | `calc`, `format` | numpy, scipy |
| `pip install "apatools[lm]"` | `lm` | pandas, statsmodels |
| `pip install "apatools[slm]"` | `slm` | pandas, statsmodels, spreg, libpysal |
| `pip install "apatools[plm]"` | `plm` | pandas, spreg, libpysal, patsy |
| `pip install "apatools[sem]"` | `sem` | pandas, semopy |
| `pip install "apatools[psm]"` | `psm` | pandas, psmpy |
| `pip install "apatools[lib]"` | `lib` | pandas, scipy, statsmodels |
| `pip install "apatools[linearmodels]"` | `plm` with the *linearmodels* backend | the `plm` extra and linearmodels |
| `pip install "apatools[graphviz]"` | `sem_plot` path diagrams, `engine="semopy"` | the `sem` extra and the Python package *graphviz*; the [Graphviz executables](https://graphviz.org/download/) must also be installed and on the `PATH` |
| `pip install "apatools[networkx]"` | `sem_plot` path diagrams, `engine="networkx"` | the `sem` extra, *networkx* and *matplotlib*; no external program needed |
| `pip install "apatools[all]"` | everything | all of the above |

## R engines

Three engines run in R, started as a subprocess (`Rscript`); no Python package is needed besides the extra of the
module (see [engines](engines.md) for what they are good for and how R is found):

| used by | R packages |
|---|---|
| `sem(engine="lavaan")` | `lavaan` |
| `sem_plot(engine="lavaan")` | `lavaan`, `semPlot` |
| `plm(backend="splm")` | `plm`, `splm`, `spdep` |

Install R, then in R: `install.packages(c("lavaan", "semPlot", "plm", "splm", "spdep"))`.

## The shape of the API

Every modelling module has the same four parts, so that what you learn for one carries over:

1. a **fitting function** (`lm`, `slm`, `plm`, `sem`, `psm`) that takes a data frame and returns `(results, metrics)`
   (`sem` and `psm` return their own tables), where `results` are the engine's fitted objects and `metrics` a list of
   dictionaries with fit statistics. Arguments of the engine are passed as `<model>_<stage>_<argument>`, e.g.
   `ols_fit_cov_type="HC1"` or `slm_fe_model_method="LU"`.
2. a **report function** (`lm_report`, `slm_report`, `plm_report`, `sem_report`) that turns them into APA-style
   tables: coefficient, standard error, test statistic, p-value (formatted without a leading zero, `<.001`),
   significance stars, confidence limits and a one-line fit summary in the first row.
3. a **tests function** (`lm_tests`, `slm_tests`, `plm_tests`) that collects the specification tests in one tidy
   table (columns `test`, `model`, `statistic`, `df`, `p-value`, `report`).
4. for spatial lag models, an **effects function** (`slm_effects`, `plm_effects`) with the direct, indirect and total
   effects.

Pass `format_pval=True` and `add_stars=True` to get strings for the manuscript; the default of the tests and effects
functions is raw numbers, from which the stars are always computed.
