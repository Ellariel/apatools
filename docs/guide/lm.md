# lm: linear models

OLS, RLM (robust), GLM, QuantReg and MixedLM models with [statsmodels](https://www.statsmodels.org), reported as APA
tables, with the specification tests that suit each kind of model collected in one table.

```python
import statsmodels.api as sm
from apatools.lm import lm, lm_report, lm_tests

results, metrics = lm(
    data, y, x,
    model=["ols", "rlm", "glm"],
    vif=True,
    pred_r_sq=True,
    ols_fit_cov_type="HC1",                      # <model>_<stage>_<argument>
    rlm_model_M=sm.robust.norms.RamsayE(),
    glm_model_family=sm.families.Gaussian(),
)
print(lm_report(results, metrics))
print(lm_tests(results, format_pval=True, add_stars=True, decimal=3))
```

Arguments of the model constructor and of `.fit()` are passed as `<model>_<stage>_<argument>`: the model prefix
(`ols_`, `rlm_`, `glm_`, `qlm_`, `mlm_`), then `model_` (constructor) or `fit_` (the `.fit()` call). Extra parameters:
`qlm_fit_q` (the quantile), `mlm_model_groups` (the grouping variable), `vif`, `formula` and `pred_r_sq`.

## Design choices and caveats

**The coefficient table.**

* It is built from the fitted results at full precision. (Before 0.3.0 it was read from the rounded text of the
  statsmodels summary: a coefficient of 0.000254 showed as 0.0003 with an interval of 0.000 to 0.000, and p-values below
  the display rounding became 0.)
* The confidence level is 95 % (the statsmodels default), and the statistic column is called `t` or `z` as in the
  statsmodels summary (t for OLS, z for RLM and GLM). For mixed models the table also contains the
  variance parameters as rows.
* Only the intercept is renamed to `Intercept` (`intercept_name=`); a regressor that merely contains `const` in its
  name is left alone.

**Prediction metrics.** `pred_metrics` and `pred_r_sq` hold every observation out in turn and refit the model
(`n` fits): predicted R² is `1 - PRESS/SST`, clipped to [-1, 1]. It is a cross-validation statistic, not a test.

**`lm_tests`: what each kind of model gets.** The classical OLS tests are not valid for the other models, so every
model gets the diagnostics that are meaningful for it; QuantReg models are skipped with a warning. The tests check
assumptions, specification and influence, not the quality of the fit. With many observations nearly every test
rejects: look at the size of the departure, not only at the p-value.

* **OLS.** Normality (Jarque–Bera, omnibus, Shapiro–Wilk); heteroskedasticity (Breusch–Pagan in Koenker's studentized
  version, which is what R's `bptest` does by default, and White's test, whose degrees of freedom are the *rank* of the
  auxiliary regression, so dummy variables do not inflate them); functional form (Ramsey's RESET with powers 2 to
  `reset_power`, default 3, as `resettest` in R); autocorrelation (Breusch–Godfrey and Durbin–Watson, only with
  `lags`, for data ordered in time); the Bonferroni outlier test (computed in closed form from the studentized
  residuals, without refitting the model *n* times) and the largest Cook's distance; the largest VIF and the condition
  number of the design matrix (without p-values; the condition number is that of the raw design, not of a scaled one).
  The tests use the residuals, so they do not depend on the covariance estimator chosen for the standard errors.
* **GLM.** Dispersion (Pearson χ²/df), the Cameron–Trivedi overdispersion test (Poisson), Pearson and deviance
  goodness of fit (Poisson, grouped binomial; not for ungrouped binary data, where they are meaningless), the
  Hosmer–Lemeshow test (binary), Pregibon's link test, and Shapiro–Wilk of the *randomized quantile residuals* of Dunn
  and Smyth (normal under the fitted model for the Gaussian, Gamma, inverse Gaussian, Poisson, negative binomial and
  binary binomial families; `seed` makes the randomization reproducible), plus leverage, studentized residuals, Cook's
  distance and the multicollinearity diagnostics.
* **RLM.** Descriptive diagnostics *without* p-values: the smallest robust weight, the ratio of the robust scale to the
  OLS residual SD, the largest RLM–OLS coefficient difference in OLS standard errors, the leverage of the design and
  the multicollinearity diagnostics. A Hausman-type test against OLS is not offered: in simulations it was unreliable
  (its covariance difference is not positive semidefinite and the test rejected 16 % of correct models at the 5 %
  level). With `approximate=True` the classical tests are added on the robust residuals (Breusch–Pagan, White, RESET as
  a robust Wald test and, with `lags`, Breusch–Godfrey) and labelled *approx.*: their theory is for OLS, and in
  simulations they rejected about 4–8 % of correct models at the 5 % level.
* **MixedLM.** The ICC and the likelihood-ratio test of the random effect for models with one random effect (with the
  boundary-corrected p-value, a 50:50 mixture of χ²(0) and χ²(1)), Shapiro–Wilk tests of the predicted random effects and
  of the conditional residuals (labelled *approx.*: both are confounded or shrunken), the Cook's distance of deleting
  whole groups (one refit per group, up to `max_refits` groups) and the multicollinearity diagnostics. The ICC and
  `mlm_r_sq` use the same random-effects variance, including the variance components of `vc_formula`.

**Formulas and groups.** `lm(formula=..., model="mlm", mlm_model_groups=...)` aligns the groups with the rows that
patsy kept when the data have missing values (it used to fail or use the wrong groups).

## Where it differs from R

| what | apatools | R | difference |
|---|---|---|---|
| Breusch–Pagan | Koenker's studentized version | `lmtest::bptest` (studentized by default) | none (statistics agree to 1e-6) |
| RLM | statsmodels' `RLM` (Huber) | `MASS::rlm` | coefficients within 0.2 %, standard errors within 2 %: different convergence rules and scale updates |
| MixedLM | statsmodels' REML optimizer | `lme4::lmer` | fixed effects within about 0.5 %, standard errors and ICC within 2 %, the likelihood-ratio test within 1 %: statsmodels stops at a looser optimizer tolerance |
| GLM leverage, Cook's distance | from statsmodels' IRLS fit | `hatvalues`, `cooks.distance` | within 1e-4: the IRLS stops at a looser tolerance |
| Pregibon's link test | refits with the squared linear predictor | the same, by hand | within 1e-3 (a refit) |
| Jarque–Bera | from the moments of the residuals | no function in base R | the same formula as the R packages that have one |

## Checked against

`tests/test_r_lm.py` compares with R (base R, `lmtest`, `MASS`, `lme4`), the numbers in the table above being the
tolerances used. The test statistics, degrees of freedom and p-values of the OLS diagnostics, the dispersion and the
Pearson and deviance goodness of fit of a Poisson GLM agree with R to 1e-5 or better. **Not compared with R** (there
is no standard implementation to compare with): the Hosmer–Lemeshow and Cameron–Trivedi tests, the quantile-residual
test, the group Cook's distance of mixed models and the approximate tests of RLM; they are validated by their
definitions and by simulations of their size and power.
