# plm: panel regression

Pooled, fixed-effects and random-effects panel models, optionally with a spatial lag or spatial error term, in the same
format as `lm` and `slm`, with the specification tests that belong to panels and the direct, indirect and total effects
of the spatial lag models. Three backends: [spreg](https://pysal.org/spreg/) (default; all models),
[linearmodels](https://bashtage.github.io/linearmodels/) (optional; non-spatial models, clustered covariances) and the R
packages *plm* and *splm* (`backend="splm"`).

```python
from apatools.plm import plm, plm_report, plm_tests, plm_effects

results, metrics = plm(
    data, "y", ["x1", "x2"],
    w=w, entity="county", time="year",          # a long data frame; w: libpysal weights of the entities
    model=["pooled", "fe", "re", "slm_fe"],
    effects="twoway",                           # entity (default), time or twoway: the fixed-effects models
    backend="spreg",                            # or "linearmodels", or "splm" (R)
    pred_metrics=True, pred_group="time",
)
print(plm_report(results, metrics))
print(plm_tests(results))                       # Hausman, Mundlak, F and LM tests ...
print(plm_effects(results))                     # direct, indirect, total effects of the spatial lag models
```

## Models, effects and backends

| `model=` | model | spreg | linearmodels | splm (R) |
|---|---|---|---|---|
| `pooled` | pooled OLS | ✓ | ✓ | ✓ |
| `fe` | fixed effects (within) | ✓ | ✓ | ✓ |
| `re` | random effects | ✓ | ✓ | ✓ |
| `slm_fe`, `slm_re` | spatial lag, fixed / random effects (ML) | ✓ | – | ✓ |
| `sem_pooled`, `sem_fe`, `sem_re` | spatial error (ML) | ✓ | – | ✓ |
| `sem_pooled_gm`, `sem_re_gm` | spatial error (GMM; the second after Kapoor, Kelejian and Prucha) | ✓ | – | – |

Where `effects=` (entity, time, twoway) applies, and which backend offers which:

| model | spreg | linearmodels | splm (R) |
|---|---|---|---|
| `fe` | entity, time, twoway | entity, time, twoway | entity, time, twoway |
| `slm_fe` | entity, twoway | not available | entity, time, twoway |
| `sem_fe` | entity, time, twoway | not available | entity, time, twoway |
| all other models | ignored | ignored | ignored |

`slx_lags` (spatially lagged regressors, which turn the lag model into a Durbin model) is a spreg option; `pred_metrics`
is not available with splm.

## Design choices and caveats

**Data layout.** Pass an ordinary long data frame with an entity and a time column (or an `(entity, time)` index).
Panels must be *balanced*: with `balance="drop"` entities that are not observed in every period are dropped (with a
warning), otherwise an error is raised. The entities are aligned with the rows of the weights by `w.id_order` (or
`w_ids`), and the rows are sorted into period blocks of entities, so the row order of your data does not matter.

**Missing data.** Rows with a missing value in the dependent variable, a predictor, the entity or the period are
removed (`dropna=True`, the default; with a formula, patsy always removes them). A panel must be balanced, so entities
that then lack a period raise an error or, with `balance="drop"`, are dropped with a warning; this is the same for every
backend and the rows never reach R. `dropna=False` with missing values, and infinite values in any case, raise a
`ValueError` that names the variables (before this was checked, spreg failed with a generic exception and R dropped the
rows itself, so that its result no longer matched the panel). Columns of the data frame that the model does not use
may contain anything.

**Fit statistics.**

* R² is *within* R² for fixed effects (after removing all the effects the model has), *overall* R² for the random
  effects of linearmodels, a *pseudo* R² (the squared correlation of observed and predicted y) for spatial models and
  for the random-effects models of spreg and splm, and the plain R² for pooled OLS. The type is part of the fit
  summary line.
* spreg's `PanelFE` reports residuals and a pseudo R² that ignore the fixed effects, so apatools recomputes the within
  R², the residuals and the F test from the within model; they agree with linearmodels and R.
* **spreg's fixed-effects standard errors use `N·T − N` degrees of freedom** instead of `N·T − N − k`, so they are
  slightly too small (about 2 % in an example with 36 entities and 6 periods; with two-way effects the period dummies
  make the gap a little larger). This is spreg's convention, kept and documented; linearmodels and R use the correct
  value.
* **The adjusted R² is `1 − (1 − R²)(N·T − A)/(N·T − A − k)`** with `A` the number of effects estimated away (entity,
  period, the constant), like linearmodels. R's `plm` uses `(N·T − 1)` in the numerator.
* AIC and BIC of spreg do not count the fixed effects.

**Period effects (`effects="time"` / `"twoway"`).** linearmodels and R have them natively. spreg has no such option, so
apatools adds period dummies to the regressors (`time[<period>]` in the summary, hidden by `plm_report`), which is the
model with entity and period effects for *any* spatial weights; `"time"` is then the pooled model with the dummies
(not available for `slm_fe`, as there is no pooled spatial lag model in spreg). With period effects `slx_lags` is not
available, `plm_tests` compares models by F and likelihood-ratio tests instead of the Hausman test, and there are no
`pred_*` metrics (the effect of a held-out period cannot be estimated). The standard errors of the spatial parameter
come from spreg's analytic information matrix and understate the sampling spread somewhat in small panels (the within
ML estimator is biased towards zero: in simulations with 36 entities, ρ̂ averaged 0.36 for a true 0.4).

**Specification tests (`plm_tests`).** One tidy table; a test appears when the models it needs are among the results.

* **Hausman** (FE vs RE; undefined, and then omitted with a warning, when `V_fe − V_re` is not positive definite,
  which is typical when the effects *are* correlated with the predictors) and the regression-based **Mundlak** test,
  which stays defined then. Neither is offered for models with period effects, which would need random-effects models
  with period effects.
* **F test of entity effects** (`fe` vs `pooled`) and, with period effects, the **F tests** of time effects (`twoway`
  vs `fe`, `time` vs `pooled`), of entity effects given time effects (`twoway` vs `time`) and of both. The F test of
  time effects against `pooled` is *conservative* when the data have strong entity effects, because they inflate the
  pooled residuals (in simulations with 36 entities and 6 periods: 0 % rejections of true nulls and 87 % power for time
  effects with half the error SD, against 7 % and 97 % for `twoway` vs `fe`); read the entity-conditional test then.
* **Likelihood-ratio test of time effects** for `slm_fe` / `sem_fe`. The within likelihood counts `N·T` observations
  although the effects use up `N + T − 1` of them, so the plain statistic is too large (12–16 % rejections of true nulls
  at the 5 % level in simulations with 36 to 100 entities and 6 periods); it is multiplied by the residual degrees of
  freedom over `N·T` (6–7 % rejections).
* **Baltagi–Song–Koh LM tests** (random effects, `LM1`, in Honda's one-sided form; spatial error, `LM2`; joint, `LMJ`)
  and **Anselin's LM tests** for a spatial lag or error and their robust versions, from the pooled model with weights.
  Both are computed by apatools. The Anselin tests of spreg's `diagnostics_panel` are *not* used: its `panel_LMlag`
  divides by the squared trace of `W'W + W²` instead of `T` times it (LM lag 11.8 instead of 34.7 in an example), and its
  panel tests use `e'e/(N·T − k)`; the textbook statistics equal R's `splm::slmtest`.

**Random effects with spatial errors.** `sem_re` (ML) of spreg 1.9.0 failed with older numpy/scipy versions (complex
numbers; numpy 2.0–2.2, scipy 1.13–1.15) and worked with recent ones; `sem_re_gm` worked everywhere, and so does
`sem_re` with the splm backend.

**Prediction metrics (`pred_metrics`).** Holding single observations out is not meaningful in panels (the other periods
of the same entity leak), so whole entities (`pred_group="entity"`: predicts a new entity, impossible for fixed-effects
models) or whole periods (`pred_group="time"`) are held out, the model and the weights are refitted, and the held-out
rows are predicted (`x·β`, plus `ρ·Wy` for the spatial lag and the estimated entity effect for fixed effects). With
spatial data the neighbours of a held-out entity stay in the training sample, which is optimistic;
`pred_exclude_neighbors=True` removes them as well (see the caveat in [slm](slm.md)).

**`plm_effects`: direct, indirect and total effects** of the spatial lag models (`slm_fe`, `slm_re`; with spreg also the
Durbin model, with the effect matrix $(I - \rho W)^{-1}(\beta_k I + \theta_k W)$), after LeSage and Pace (2009) and
Elhorst (2014); the weights are those of the panel. The options and the caveats about the delta method and the
simulation are those of `slm_effects` (see [slm](slm.md)). The covariance matrix of the backend is used: spreg's contains
the covariance between the coefficients and `ρ`, splm's does not, so the standard errors of the indirect and total
effects differ somewhat between the backends.

## The splm backend (R)

`backend="splm"` fits `pooled`, `fe` and `re` with R's *plm* (Croissant and Millo, 2008) and the spatial ML models with
*splm* (Millo and Piras, 2012); *spdep* turns `w` into R's weights (see [engines](../engines.md) for how R is called
and `r_timeout`). The results are `SplmFit` objects and feed `plm_report`, `base_metrics`, `plm_tests` and `mundlak`
unchanged. Arguments of the R functions go in `<model>_model_<arg>` (for example `slm_fe_model_zero.policy=True`).
Not available: the GMM models, `slx_lags` and the prediction metrics.

Conventions of splm that apatools adapts or keeps:

* **splm calls the spatial lag coefficient `lambda` and the spatial error coefficient `rho`**; apatools (and spreg) use
  the other way round, so the results are renamed.
* **R² of random effects** is the pseudo R², not the R² of the transformed model that R reports.
* **AIC/BIC** are `2k − 2·logLik` (and `k·log n − 2·logLik`) with `k` the coefficients, the spatial parameter and, for
  random effects, `φ`; the error variance is not counted.
* **splm's covariance matrix treats the spatial parameter and `φ` as uncorrelated with the coefficients.**
* In the spatial error model with random effects, splm's `φ` is the ratio of the variance of the entity effects to that
  of the remainder, `σ²_μ/σ²_ε` (the parameterization of Baltagi, Song and Koh, 2003).
* The ML optimizer of splm is less tight than spreg's: the spatial parameter agrees within 0.001, the slopes within
  0.0002 (random effects: 0.002) and the standard errors within 3 %.

## Where it differs from R

| what | apatools | R (`plm`, `splm`) | difference |
|---|---|---|---|
| `fe` coefficients | all backends | `plm::plm` | agree (1e-8) |
| `fe` standard errors | linearmodels, splm | `plm::plm` | agree (1e-7) |
| `fe` standard errors | spreg | `plm::plm` | `se_R·√(df_resid/(N·T−N))`, with `df_resid` that of R: spreg's degrees of freedom |
| R² (within), F, residual df | all | `summary(plm)` | agree (1e-8) |
| adjusted R² | `(N·T−A)/(N·T−A−k)` | `(N·T−1)/(N·T−A−k)` | differs by convention |
| `re` coefficients | all | `plm(model="random")` | agree (1e-6); standard errors of spreg are up to 3 % smaller |
| Hausman test | linearmodels, splm | `phtest` | agree (1e-6); with spreg's random effects the value differs (0.92 against 0.95 in an example) because its standard errors differ |
| F tests of effects | all | `pFtest` | agree exactly (five tests) |
| LM1 (Honda), LM lag / error, robust | apatools | `plmtest`, `slmtest` | agree (1e-7) |
| LM2, LMJ | apatools | `splm::bsktest` failed in our environment (with the installed Matrix package) | checked through their relations: `LM2 = √LM-error`, `LMJ = LM1² + LM2²` |
| spatial models | spreg, splm | `splm::spml` | see the splm tolerances above |
| spatial lag effects | apatools | `splm::impacts` | point estimates agree (1e-6); standard errors of the simulation within its noise |

## Checked against

`tests/test_r_plm.py` compares with R (`plm`, `splm`, `spdep`) function by function; `tests/test_splm.py` compares the
R backend with linearmodels, spreg, OLS with dummies and two maximum-likelihood fits written out in numpy (the spatial
lag model with all effects as explicit parameters, and the spatial error random-effects model of Baltagi, Song and
Koh, 2003, which spreg cannot fit in some environments); `tests/test_plm.py` checks the period effects against OLS with
dummies, linearmodels and the same numpy likelihoods; `tests/test_effects.py` the effects.

## Literature

The references cited in the docstrings of the functions of this module (they are collected from the docstrings when the
documentation is built; the software that does the work is on the [software page](../software.md)).

```{include} ../_generated/references_plm.md
```
