# Changelog

## 0.2.0

Only the changes to `apatools.lib` are listed here.

### Fixed
- `krippendorffs_alpha` returned wrong values except for nominal data without missing ratings.

### Changed
- `apatools.lib` is a package now (`misc`, `fleiss`, `krippendorff`, `correlations`, `omega`); everything is
  still imported from `apatools.lib`, only the private helpers moved into the submodules.
- **Breaking:** `bootstrap` moved from `apatools.calc` to `apatools.lib` (`from apatools.lib import bootstrap`),
  so it now needs the `lib` extra; `apatools.calc` no longer has it.

### Added
- `mcdonald_omega`: McDonald's omega, always returned with its confidence interval as `(omega, lower, upper)`:
  a delta-method interval (normal-theory maximum likelihood, computed on the logit scale) by default, or a
  percentile bootstrap interval with `bootstrap_iter`. The one-factor model is fitted by maximum likelihood
  with numpy/scipy (no scikit-learn or factor-analyzer).

### Removed
- The `krippendorff` extra: the package (GPL-3.0-or-later) is no longer used.
- `average_morans_i`.

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
