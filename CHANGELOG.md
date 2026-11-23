# Changelog

## 0.3.0

### Added
- The documentation (`docs/`, Sphinx; the extra `docs`): a user-guide page per module that states the **design choices
  and caveats** of the module and **where it differs from R**, an overview of the validation against R, the software
  page (what to cite for each engine), and an API reference and a literature page generated from the docstrings at
  build time. The README is a short landing page now.
- `slm_effects` and `plm_effects`: the direct, indirect (spillover) and total effects of the regressors of the spatial
  lag models (`slm`; `slm_fe` / `slm_re` of spreg and splm; with spreg also the Durbin model of `slx_lags`) after LeSage &
  Pace (2009), with standard errors, z values, p-values and confidence intervals from LeSage & Pace's simulation or
  the delta method (`inference="simulation"` / `"analytic"`). The averages are series in the traces of the powers of
  `W` (exact from the eigenvalues, or Hutchinson's estimator for large `n`), no inverse is needed. The point
  estimates equal the dense inverse, spreg's multipliers and R's `spatialreg::impacts` / `splm::impacts`.
- `plm(backend="splm")`: the R packages *plm* (`pooled`, `fe`, `re`) and *splm* (`slm_fe`, `slm_re`, `sem_pooled`,
  `sem_fe`, `sem_re`; ML) as the engine of the panel models, the way lavaan is the engine of `sem`; *spdep* turns the
  weights into R's. The results (`SplmFit`) feed `plm_report`, `base_metrics`, `plm_tests`, `hausman` and `mundlak`
  unchanged; `effects="time"` works for `slm_fe` as well. Arguments of the R functions go in `<model>_model_<arg>`,
  `rscript=` and `r_timeout=` control R. Not available with splm: the GMM models, `slx_lags` and `pred_metrics` (NaN
  with a warning).
- `plm_tests`: the Baltagi-Song-Koh LM tests are computed by apatools (they equal spreg's `BSK_tests`, R's `plmtest`
  and `slmtest`; the p-values of LM1 and LMJ no longer underflow to 0), so they also work for splm results.

### Removed
- `apatools.citation` and `cite()`: the literature behind every method and engine is collected in the
  documentation (references page and the page of each module) instead.

### Changed
- The R bridge lives in the subpackage `apatools.r`: `bridge` (finding and calling `Rscript`, shared by all R engines),
  `lavaan` (formerly `apatools._lavaan`) and `splm`.

## 0.2.1

### Added
- `plm(effects="entity" | "time" | "twoway")`: the effects of the fixed-effects models (entity effects as before,
  period effects, or two-way fixed effects). With `backend="linearmodels"` it sets `entity_effects` /
  `time_effects` (still available as `fe_model_*` arguments); with spreg the period effects are period dummies
  (`fe`, `slm_fe`, `sem_fe`; `effects="time"` not for `slm_fe`), hidden in `plm_report`. Checked against OLS with
  dummies, linearmodels (coefficients, R², adjusted R², F, df: identical) and a maximum-likelihood fit written out
  without spreg for the spatial models (estimates equal to 5 decimals). Models with period effects have no `pred_*`
  metrics.
- `plm_tests` for models with period effects: F tests of time effects (`twoway` vs `fe`, `time` vs `pooled`), of
  entity effects given time effects (`twoway` vs `time`) and of both (`twoway` vs `pooled`), equal to `anova_lm` of the
  dummy regressions; likelihood-ratio test of time effects for `slm_fe` / `sem_fe`, scaled by the residual degrees of
  freedom over N·T because the plain within-likelihood statistic rejected 12-16 % of true nulls at the 5 % level
  (6-7 % scaled). Hausman, Mundlak and the spatial LM tests stay without period effects.

### Fixed
- Found by comparing with R (`plm`, `splm`, `spdep`, `spatialreg`, `lmtest`; see the tests `tests/test_r_*.py`):
  - `plm_tests`: the Anselin LM tests (LM lag, LM error and the robust versions) of the pooled model were wrong:
    spreg's `panel_LMlag` divides by the squared trace of `W'W + W²` instead of `T` times it (LM lag 11.8 instead of
    34.7 in an example) and its panel tests use `e'e / (N·T - k)`. They are computed here now, equal to `splm::slmtest`.
  - `slm_tests`: Breusch-Pagan and Koenker-Bassett regressed the squared residuals on the *squares* of the
    regressors (spreg's default); they use the regressors, as R's `bptest` and statsmodels (and `lm_tests`) do.
  - `slm` metrics: AIC and BIC of the spatial error model did not count `lambda` (spreg's `k`), those of the lag
    model count `rho`; the error model counts `lambda` now, so the three models are comparable.
- `lm_report` no longer needs `lxml`: the coefficient table was read from the HTML of the statsmodels summary and so
  carried only the rounded display values (a small coefficient 0.000254 became 0.0003 and its interval 0.000 to
  0.000; p-values below the display rounding became 0). It is built from the results at full precision now; the
  columns and the fit summary are unchanged. The `lm` extra no longer installs `lxml`.
- `plm_tests` treated a linearmodels two-way (or time-effects) model as an entity-effects model in the Hausman test
  and the F test of entity effects.
- `plm_report` / `base_metrics` of a fixed-effects model with time effects (`backend="linearmodels"`): R² is now the
  within R² after removing all effects and the adjusted R² accounts for the time dummies as well (before, only the
  entity effects were counted). Coefficients, standard errors and degrees of freedom were correct.

## 0.2.0

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
- `mcdonald_omega_bifactor`: omega hierarchical, omega total, the explained common variance and the subscale omegas
  (`omega_s`, `omega_hs`) of a scale with subscales, from a confirmatory bifactor model given by item groups, with
  delta-method (sandwich) or bootstrap intervals. Estimates and standard errors agree with lavaan.
- `mcdonald_omega(robust=True, varphi=0.1)`: robust omega after Zhang & Yuan (2016): the covariance matrix of the items
  is a Huber-type M-estimate (cases far from the center are downweighted) and the interval uses the sandwich covariance
  matrix of the estimator; `varphi=0` keeps the sample covariance matrix with the distribution-free standard error. It
  agrees with the R package coefficientalpha. `mcdonald_omega(standardized=False)` gives the reliability of the raw sum
  score (the default, omega of the standardized items, is unchanged). `percentile_bootstrap` accepts statistics that
  return several values.
- `lm_tests`: diagnostic tests of the models fitted with `lm` in one table, in the format of `plm_tests`, with the diagnostics
  that are meaningful for each kind of model. OLS: Jarque-Bera, omnibus and Shapiro-Wilk (normality), Breusch-Pagan and White
  (heteroskedasticity), RESET, Breusch-Godfrey and Durbin-Watson (with `lags`), the Bonferroni outlier test, Cook's distance,
  VIF and condition number. GLM: dispersion, Cameron-Trivedi overdispersion, Pearson / deviance goodness of fit,
  Hosmer-Lemeshow, Pregibon's link test, normality of randomized quantile residuals (Dunn & Smyth), leverage and influence.
  RLM: robust weights, scale ratio, RLM-OLS coefficient change and leverage (descriptive), and with `approximate=True` the
  classical tests on the robust residuals, labelled "approx.". MixedLM: ICC, boundary-corrected likelihood-ratio test of the
  random effect, normality of the random effects and conditional residuals ("approx."), and the influence of whole groups.
  QuantReg models are skipped with a warning.
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
