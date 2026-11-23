# slm: spatial regression

OLS, spatial lag (SLM) and spatial error (SEM) models by maximum likelihood with [spreg](https://pysal.org/spreg/),
reported as APA tables, with the diagnostics for spatial dependence and the direct, indirect and total effects of the
spatial lag model.

```python
from apatools.slm import slm, slm_report, slm_tests, slm_effects

results, metrics = slm(
    data, "y", ["x1", "x2"], w,                 # w: libpysal weights, rows in the order of the data
    model=["ols", "slm", "sem"],
    ols_model_spat_diag=True,
    pred_metrics=True, pred_exclude_neighbors=True,
)
print(slm_report(results, metrics))
print(slm_tests(results, w, format_pval=True, add_stars=True, decimal=3))
print(slm_effects(results, w, decimal=3, add_stars=True, format_pval=True))
```

Arguments of the spreg model constructor are passed as `<model>_model_<argument>` with the prefix `ols_`, `slm_` or
`sem_` (for example `ols_model_spat_diag`, or `slm_model_method="LU"` for large samples). Rows dropped because of
missing values are removed from `w` automatically.

## Design choices and caveats

**Fit statistics.**

* R² of the ML models is spreg's *pseudo R²* (the squared correlation of observed and predicted y); OLS has its usual
  R².
* **AIC and BIC count the spatial parameter in both spatial models.** spreg's own criteria of the spatial lag model
  count `rho`, those of the spatial error model leave `lambda` out of the parameter count, which flatters the error
  model in comparisons. apatools adds it (2 to the AIC and `log n` to the BIC), so that OLS, SLM and SEM are
  comparable. Like spreg, none of them counts the error variance (see the table below for R).

**Leave-one-out prediction metrics** (`pred_metrics`): every observation is held out in turn, the model and the weights
are refitted without it and the held-out value is predicted: `x·β` for OLS and SEM, `ρ·(Wy)ᵢ + x·β` for SLM (the
observed neighbour values enter through the weights, so SLM is not strictly comparable with the other two). One fit per
observation and model.

> **Caveat: plain leave-one-out is optimistic for spatial data.** The neighbours of the held-out observation stay in
> the training sample and are correlated with it, so `R²pred` overestimates how well the model predicts a new, unseen
> region. `pred_exclude_neighbors=True` gives a more honest (slightly pessimistic) estimate; it may create islands in
> the training weights, which is reported with a warning.

**`slm_tests`.** One table of diagnostics, to check assumptions and the need for a spatial term, not the quality of the
fit.

* *OLS:* Moran's I of the residuals (a z test), the Lagrange multiplier tests for a missing spatial lag (`LM-lag`) and
  for spatial error dependence (`LM-error`), their robust versions and `LM-SARMA` (Anselin et al., 1996), and
  Jarque–Bera, Breusch–Pagan, Koenker–Bassett and White tests of the residuals. The spatial tests need `w`. The
  customary reading (Anselin and Rey, 2014): if both LM tests are insignificant, OLS is fine; otherwise fit the model of
  the more significant one, and if both are significant, decide by the robust versions. The function does not make
  that decision.
* **Breusch–Pagan and Koenker–Bassett regress on the regressors of the model**, as R's `bptest` and statsmodels do.
  spreg's default is to use their *squares*; apatools overrides that default (and says so here), so that the numbers
  equal the ones of the other software. White's test is spreg's.
* *SLM and SEM:* the z test and the likelihood-ratio test (against the OLS model among the results) of `rho` /
  `lambda`, Jarque–Bera of the residuals, and Moran's I of the residuals with a *permutation* p-value (`n_perm`,
  `seed`; a significant value means that spatial dependence is left in the model). The residuals are those of the
  model: `y - ρWy - Xb` for the lag model and the filtered ones, `(I - λW)(y - Xb)`, for the error model. The
  permutation test treats the residuals as exchangeable, an approximation for model residuals.

**`slm_effects`: direct, indirect and total effects.** In the spatial lag model the coefficient of a regressor is not
its effect: a change in one location spills over to its neighbours through `ρWy`. LeSage and Pace (2009) summarize the
effect matrix $(I - \rho W)^{-1}\beta_k$ by three averages: the **direct** effect (the diagonal: the effect on the
location itself, including the feedback through its neighbours), the **total** effect (the row sums: a unit change of the
regressor in *all* locations) and the **indirect** effect, the spillover (total minus direct).

* The averages are series in the traces of the powers of `W`, so no inverse of `I - ρW` is needed; `traces="exact"`
  takes them from the eigenvalues (cost n³, the default up to 2000 units), `traces="mc"` estimates them with random
  vectors (Hutchinson) for large `n`.
* **Two ways to the standard errors.** `inference="simulation"` (default) draws `n_draws` times from the estimated joint
  normal distribution of the coefficients and `ρ` (LeSage and Pace) and takes the standard deviation and the
  percentiles of the effects of the draws (the interval can be asymmetric); `inference="analytic"` is the delta method
  with the gradient of the same series (the interval is estimate ± z · se).
* **When the two disagree.** The delta method treats the effects as linear in the estimates. That holds when `ρ` is
  estimated precisely; with few units and a large `ρ` it can understate the standard errors of the indirect and total
  effects badly (in a 64-unit example with ρ = 0.6: 0.44 against 1.48 by simulation with 100,000 draws and 1.43 in
  R's `impacts`). The direct effect is much less sensitive. Prefer the simulation then.
* **The simulation is noisy in the same situation**, because a few draws of `ρ` near its bound give huge effects: the
  standard error of that indirect effect was 1.03 with 5000 draws and 1.48 with 100,000 (R's own simulation behaves the
  same way: 1.18 and 1.43). Raise `n_draws` and check that the result is stable with another `seed`. Draws with `|ρ|`
  at or above the bound given by the spectral radius of `W` are replaced.
* The covariance matrix is spreg's, which includes the covariance between the coefficients and `ρ`.
* OLS has no spillovers, and in the spatial error model the coefficients already are the direct and total effects:
  those models are skipped with a warning. The effects are *averages*; they do not say how the effect differs between
  locations.

## Where it differs from R

| what | apatools / spreg | R (`spatialreg`, `spdep`, `lmtest`) | difference |
|---|---|---|---|
| ML estimates, standard errors, log-likelihood | spreg | `lagsarlm`, `errorsarlm` | agree: coefficients and standard errors within 1e-4, log-likelihood within 1e-8 |
| AIC, BIC | do not count the error variance (spatial parameter counted) | count it | R's AIC is 2 higher (BIC: `log n` higher) for every model, a constant |
| Moran's I of the OLS residuals | z test, two-sided p-value | `lm.morantest`: one-sided (greater) | our p-value is twice R's when I is above its expectation |
| LM tests, SARMA | spreg | `lm.RStests` | agree (1e-7) |
| Breusch–Pagan, Koenker–Bassett | regress on the regressors | `bptest` | agree (1e-7); spreg's own default differs (squares) |
| White | spreg | `bptest` with squares and cross products | agree (1e-7) |
| direct, indirect, total effects | series in the traces | `impacts` | point estimates agree (1e-6); standard errors of the simulation within the noise of the simulations |

## Checked against

`tests/test_r_slm.py` compares with `spatialreg`, `spdep` and `lmtest` on two simulated data sets (rook and queen
lattices), `tests/test_effects.py` with the dense matrix inverse, spreg's multipliers, finite differences of the effects
(the gradients of the delta method) and the simulation (the standard errors), and the point estimates of the effects
with `spatialreg::impacts`.
