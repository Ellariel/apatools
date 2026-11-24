# apatools: a toolbox for statistical analysis and APA-compliant reporting

One convenient interface to the most common tools: apatools wraps statsmodels, spreg, linearmodels, semopy and psmpy
and, optionally, the R packages lavaan, plm, splm and spdep, and returns tidy tables with the statistics a manuscript
needs, together with the specification tests, effect conversions and reliability coefficients that go with them.

**What sets it apart:** no method is just linked to an engine and trusted. Methods are benchmarked against independent
references (mostly R packages, otherwise the definition itself), and the documentation discloses the design decisions
behind every module, its caveats, and where and why the numbers differ from R; the few methods without a benchmark are
marked as such.

## Install

```shell
pip install apatools                # calc and format: numpy and scipy only
pip install "apatools[lm,plm]"      # extras per module; "apatools[all]" installs everything
```

Python 3.10 or newer; the development version: `pip install git+https://github.com/ellariel/apatools.git`.
Importing a module without its extra gives an `ImportError` that names the extra to install.

| extra | module | brings in |
|---|---|---|
| (none) | `calc`, `format` | numpy, scipy |
| `lm` | `lm` | pandas, statsmodels |
| `slm` | `slm` | pandas, statsmodels, spreg, libpysal |
| `plm` | `plm` | pandas, spreg, libpysal, patsy |
| `sem` | `sem` | pandas, semopy |
| `psm` | `psm` | pandas, psmpy |
| `lib` | `lib` | pandas, scipy, statsmodels |
| `linearmodels` | `plm` with the linearmodels backend | the `plm` extra and linearmodels |
| `graphviz` / `networkx` | `sem_plot` diagrams with Graphviz / with networkx and matplotlib | the `sem` extra and the drawing packages |
| `docs` | building the documentation | sphinx, furo, myst-parser |

Three engines run in R (`sem(engine="lavaan")`, `sem_plot(engine="lavaan")`, `plm(backend="splm")`): install R and,
in R, `install.packages(c("lavaan", "semPlot", "plm", "splm", "spdep"))`. `Rscript` is found on the `PATH`, in the usual
install locations, through the environment variable `APATOOLS_RSCRIPT` or the argument `rscript`.

## A first look

```python
from apatools.lm import lm, lm_report, lm_tests
from apatools.plm import plm, plm_report, plm_tests, plm_effects

results, metrics = lm(data, "y", ["x1", "x2"], model=["ols", "rlm"])
print(lm_report(results, metrics))                              # APA table: b, SE, t, p, stars, CI, fit line
print(lm_tests(results, format_pval=True, decimal=3))           # diagnostics that suit each kind of model

results, metrics = plm(panel, "y", ["x1", "x2"], w=w, entity="county", time="year",
                       model=["pooled", "fe", "re", "slm_fe"], effects="twoway")
print(plm_report(results, metrics))
print(plm_tests(results))                                       # Hausman, Mundlak, F and LM tests
print(plm_effects(results))                                     # direct, indirect and total effects
```

## What is in it

| module | does |
|---|---|
| `apatools.calc` | conversions between r, d, t, z, F, p, odds ratios; confidence intervals |
| `apatools.format` | APA formatting of p-values, correlations and stars |
| `apatools.lm` | OLS, RLM, GLM, QuantReg, MixedLM (statsmodels): tables and diagnostic tests |
| `apatools.slm` | OLS, spatial lag and error models (spreg): tables, spatial diagnostics, direct / indirect / total effects |
| `apatools.plm` | pooled, FE, RE panel models, optionally spatial (spreg, linearmodels, R's plm / splm): tables, specification tests, effects |
| `apatools.sem` | structural equation models (semopy, lavaan), reports, path diagrams (Graphviz, networkx, semPlot) |
| `apatools.psm` | propensity score matching (psmpy) and covariate balance |
| `apatools.lib` | bootstrap, Cronbach's alpha, McDonald's omega (robust, bifactor), Fleiss' kappa, Krippendorff's alpha, correlations, HTMT, Gini |

## Documentation

The documentation is at **[ellariel.github.io/apatools](https://ellariel.github.io/apatools/)** (GitHub Pages, built from
the main branch). Its source is in [`docs/`](docs/) (Sphinx; build it locally with `pip install -e ".[all,docs]"` and
`sphinx-build -W -b html docs docs/_build/html`):

* a user-guide page per module, each with the **design choices and caveats** of the module and **where it differs from
  R**;
* the [validation overview](docs/validation.md): what was compared with what, and where apatools intentionally does not
  mirror R;
* the [software](docs/software.md) page (what to cite for each engine), and the literature at the end of each module page;
* the API reference, generated from the docstrings, and the [changelog](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE).
