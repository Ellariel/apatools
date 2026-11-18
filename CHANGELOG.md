# Changelog

## 0.1.0

First release on PyPI.

The base installation (`pip install apatools`) contains `apatools.calc` and `apatools.format` and needs
only numpy and scipy. The other modules are installed through extras named after the module
(`apatools[lm]`, `[slm]`, `[plm]`, `[sem]`, `[psm]`, `[lib]`; `[linearmodels]`, `[graphviz]`,
`[krippendorff]`; `[all]`). Importing a module without its extra raises an `ImportError` that names the extra.

- `apatools.calc`: conversions between r, d, t, z, F, p and odds ratios, confidence intervals, `bootstrap`.
- `apatools.format`: APA formatting of p-values, correlations and significance stars.
- `apatools.lm`: OLS, RLM, GLM, QuantReg and MixedLM with *statsmodels*, APA tables, leave-one-out metrics, VIF.
- `apatools.slm`: OLS, spatial lag and spatial error models with *spreg*, APA tables, leave-one-out
  prediction metrics (also leaving out the neighbours of the held-out observation).
- `apatools.plm`: panel regressions (pooled, fixed and random effects, optionally with a spatial lag or
  spatial error) with *spreg* or, optionally, *linearmodels*; APA tables, leave-one-group-out prediction
  metrics and specification tests (Hausman, Mundlak, F test of entity effects, Baltagi-Song-Koh and
  Anselin LM tests).
- `apatools.sem`: structural equation models with *semopy*, reporting and path diagrams.
- `apatools.psm`: propensity score matching with *psmpy* and effect sizes.
- `apatools.lib`: inter-rater agreement, pooling and comparing correlations, HTMT ratio, Gini coefficient.
