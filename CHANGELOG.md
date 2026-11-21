# Changelog

## 0.2.0

This list is not complete yet: it covers `apatools.lib` and the changes listed below.

### Fixed
- `sem_plot` labelled the arrows with the p-values of the raw estimates although it shows standardized estimates by
  default (and none for the loadings fixed to 1): the labels use `p_std` now when `std_ests=True`.
- `lm(formula=..., model="mlm", mlm_model_groups=...)` failed (IndexError) or used the wrong groups when the data
  had missing values: the groups are now aligned with the rows that patsy kept.
- `lm(..., vif=True, base_metrics=False)` silently dropped the VIFs.
- `mlm_icc` ignored the variance components of `vc_formula` while `mlm_r_sq` included them; both use the same
  random-effects variance now (the ICC of such models changes).
- `psm(seed=...)` did not seed psmpy's own sampling (`balance=True` with unequal groups); the seed is passed on now
  (psmpy versions without a seed argument are still supported).
- `d_from_r`, `d_from_z`, `d_from_p`, `p_from_d`, `z_from_d` (and the `cohen_*` aliases) assumed two groups while
  `r_from_d` had `n_groups`: they take `n_groups` too (default 2, unchanged results), so the conversions round-trip.
- `independent_correlations(..., method="zou")` failed when `n2` was omitted.
- `krippendorffs_alpha` returned wrong values except for nominal data without missing ratings.
- `fleiss_kappa`: the standard error was 2-3 times too small (about 30 % false positives at the 5 % level), now the
  null variance of Fleiss, Nee & Landis (1979) as in R's `irr`; missing ratings no longer count as a category, items
  with a missing rating are left out (NaN in the per-item agreement); the one-tailed p-value used |z|; the
  interpretation bands in the docstring were wrong.
- `lm_report` replaced the coefficient names by NaN for a model without intercept, and `lm_report` and `slm_report`
  renamed every label that merely contains `const` (e.g. a variable `const_income`). Only the intercept is
  renamed now, in `lm_report`, `slm_report` and `plm_report` alike.

### Changed
- **Breaking:** the one-field `Citation` class and the `CITATION` constants of `lm`, `slm`, `plm`, `sem` and `psm` are
  replaced by `apatools.citation.cite(module, engine=None)`: name the engines you used (a string or a list, for
  fitting and for drawing alike, e.g. `cite("sem", engine=["lavaan", "semPlot"])`) and get their APA references
  without duplicates. The references are written once in `apatools.citation.REFERENCES`; they were checked against
  Crossref and other sources: the statsmodels page range is corrected (92-96), DOIs are added (PySAL, NetworkX,
  Graphviz, linearmodels), Python is capitalized and the editors of the proceedings are given. New references:
  lavaan, semPlot, R, NetworkX, matplotlib and Graphviz.
- **Breaking:** `sem_plot(method=...)` is now `sem_plot(engine=...)` (`"semopy"`, as before, or the new `"networkx"`), and the
  Graphviz layout program of `semopy_plot` is `layout=` (was `engine=`), so that `engine` always names the drawing
  engine. The diagram of the `"semopy"` engine is unchanged.
- **Breaking:** one name for the confidence level everywhere: `conf_level` (e.g. 0.95). It replaces `alpha` in
  `dependent_correlations`, `independent_correlations`, `bootstrap`, `calc.p_from_ci`, `calc.ci_from_r` and
  `calc.ci_from_p` (where `alpha` was already a confidence level, despite the name) and in `average_correlations`
  (where `alpha=0.05` was a significance level: `conf_level=0.95` is the equivalent).
- Docstrings of all public functions completed (parameters, return values, notes) with literature references for the
  non-trivial methods.
- `z_transform` returns NaN (instead of dividing by zero and warning) for constant or all-NaN input with
  `ignore_errors=True`, and raises for all-NaN input with `ignore_errors=False`.
- `apatools.lib` is a package now (`misc`, `fleiss`, `krippendorff`, `correlations`, `cronbach`, `mcdonald`); everything is
  still imported from `apatools.lib`, only the private helpers moved into the submodules.
- **Breaking:** `bootstrap` moved from `apatools.calc` to `apatools.lib` (`from apatools.lib import bootstrap`),
  so it now needs the `lib` extra; `apatools.calc` no longer has it.
- `lm_report`, `slm_report`, `plm_report`, `plm_tests` and `sem_report` share one implementation of the rounding,
  significance stars and p-value formatting (private, in `apatools._common`); their output is unchanged.

### Added
- `lm_tests`: diagnostic tests of the OLS models fitted with `lm` in one table, in the format of `plm_tests`: Jarque-Bera,
  omnibus and Shapiro-Wilk (normality), Breusch-Pagan and White (heteroskedasticity), RESET (functional form),
  Breusch-Godfrey and Durbin-Watson (autocorrelation, with `lags`), the Bonferroni outlier test and Cook's distance,
  and the largest VIF and the condition number.
- `slm_tests`: diagnostic tests of the fitted `slm` models in one table, in the format of `plm_tests`: for OLS Moran's I
  of the residuals, LM-lag, LM-error, their robust versions, LM-SARMA, Jarque-Bera, Breusch-Pagan, Koenker-Bassett and
  White; for the spatial lag and error models the z and likelihood-ratio tests of the spatial parameter,
  Jarque-Bera and Moran's I (permutation test) of the residuals.
- `sem(engine="lavaan")`: fits the model with the R package lavaan (through `Rscript`, no further Python package) and
  returns the same tables as the semopy engine (`method` accepts semopy's and lavaan's estimator names, `lavaan_args`
  passes further arguments to `lavaan::sem`, `bootstrap` gives bootstrap standard errors, SRMR is added to the fit
  statistics). `sem(method=None, bootstrap=None)` now means "the default of the engine" (`"MLW"` / 100 for semopy,
  `"ML"` / 0 for lavaan).
- `sem_plot(engine="lavaan")`: the path diagram drawn by the R package semPlot, from the table of either engine;
  returns an `RFigure` (PNG, PDF, SVG, ...). R messages are in English (`APATOOLS_R_LANGUAGE` changes that).
- `sem`: standard error, z-value and p-value of the standardized estimates (`se_std`, `z_std`, `p_std`; delta method
  as in lavaan's `standardizedSolution()`). They also exist for the loadings that are fixed to 1 in the raw solution.
  `sem_report` formats `p_std` like `p-value`.
- `sem_plot(engine="networkx")` (new extra `networkx`): the same path diagram drawn with networkx and matplotlib, without
  the Graphviz executables; returns a matplotlib figure, honours `dpi`, and offers the layouts `layered` (default),
  `spring`, `kamada_kawai` and `circular` (`networkx_plot`).
- `mcdonald_omega`: McDonald's omega, always returned with its confidence interval as `(omega, lower, upper)`:
  a delta-method interval (normal-theory maximum likelihood, computed on the logit scale) by default, or a
  percentile bootstrap interval with `bootstrap_iter`. The one-factor model is fitted by maximum likelihood
  with numpy/scipy (no scikit-learn or factor-analyzer).
- `cronbachs_alpha`: Cronbach's alpha, returned like `mcdonald_omega` as `(alpha, lower, upper)`: an analytic
  interval (asymptotic standard error under normal theory, van Zyl et al., 2000; Duhachek & Iacobucci, 2004)
  by default, or a percentile bootstrap interval with `bootstrap_iter`.

### Removed
- The `krippendorff` extra: the package (GPL-3.0-or-later) is no longer used.
- `average_morans_i`.
- The module `apatools.utils` with `read_zip` and `take_until_timeout`; its remaining helpers live in the private
  module `apatools._common`.

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
