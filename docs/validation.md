# Validation

What the numbers of apatools were compared with, and where they intentionally differ from the benchmark. The same
information, in the context of each method, is on the [module pages](guide/index.md); this page is the overview.

## Principles

1. **An independent reference wherever there is one.** R is the reference implementation of much of what apatools
   reports (`plm`, `splm`, `spatialreg`, `spdep`, `lmtest`, `MASS`, `lme4`, `lavaan`, `coefficientalpha`, `irr`).
   Where there is none, the reference is the definition: a dense matrix inverse for a trace series, OLS with dummies
   for a within estimator, a likelihood written out in numpy for an ML fit, finite differences for a gradient.
2. **Disagreements are findings.** Comparing with R and with the literature found four errors in what the engines or
   earlier versions of apatools reported: the Anselin LM tests of spreg's panel module (squared trace, wrong variance), the
   Breusch–Pagan and Koenker–Bassett tests of `slm_tests` (squares of the regressors instead of the regressors), the AIC
   and BIC of the spatial error model (`lambda` not counted) and the standard error of Fleiss' kappa (2–3 times too small).
   All are fixed and listed in the [changelog](changelog.md).
3. **Conventions are documented, not hidden.** Where an engine's convention differs from R's (degrees of freedom, the
   count of parameters in the AIC, the sidedness of a p-value) apatools keeps one and says which; the difference is
   pinned in the tests so that a change is noticed.
4. **Tests are checked by breaking the code.** A new test is accepted when deliberately wrong code makes it fail.

## What was compared with what

| module | compared with | what |
|---|---|---|
| `lm` | R: base `lm`/`glm`, `lmtest`, `MASS::rlm`, `lme4` | OLS diagnostics (1e-6), GLM dispersion and goodness of fit (1e-8), RLM coefficients (0.2 %), mixed models (0.5 %) |
| `slm` | R: `spatialreg`, `spdep`, `lmtest` | ML estimates, standard errors, log-likelihood, LM tests, Moran's I, heteroskedasticity tests |
| `slm_effects`, `plm_effects` | dense inverse of `I − ρW`, spreg's multipliers, finite differences, simulation; R: `spatialreg::impacts`, `splm::impacts` | point estimates (1e-6 to 1e-8), gradients of the delta method, standard errors (within the noise of the simulations) |
| `plm` | R: `plm`, `splm`, `spdep` (`pFtest`, `phtest`, `plmtest`, `slmtest`, `spml`); linearmodels; OLS with dummies; numpy likelihoods | coefficients, standard errors, R², F, tests of effects, LM tests, spatial ML models |
| `plm` (splm backend) | linearmodels, spreg, OLS with dummies, two numpy likelihoods | the R engine's conversion into apatools' objects |
| `sem` | lavaan through the lavaan engine | loadings, regressions, standardized estimates and their standard errors, fit indices |
| `lib` omega | lavaan (bifactor, delta method), `coefficientalpha` (robust omega) | estimates (1e-5), standard errors (1–2 %) |
| `lib` kappa, alpha | the formulas of `irr`, published reference values and examples | |
| `calc`, `format` | the textbook formulas and the APA rules | round trips of the conversions |
| `psm` | the example of psmpy | balance improves, reproducible with a seed |

## Where apatools does not mirror the R benchmark

These are the differences that change a printed number, by module; the reasons are on the module pages.

| module | quantity | apatools | R | tested in |
|---|---|---|---|---|
| `slm`, `plm` (spatial) | AIC, BIC | do not count the error variance | count it (+2, +log n) | `test_r_slm.py` |
| `slm` | Moran's I of OLS residuals, p-value | two-sided | one-sided (`lm.morantest`) | `test_r_slm.py` |
| `plm` (spreg) | fixed-effects standard errors | `N·T − N` degrees of freedom | `N·T − N − k` | `test_r_plm.py` |
| `plm` | adjusted R² | `(N·T − A)` in the numerator | `(N·T − 1)` | `test_plm.py` |
| `plm` (spreg) | random-effects standard errors, Hausman test | standard errors slightly smaller (up to 3 %) | `plm` (swar) | `test_r_plm.py` (standard errors) |
| `plm` (splm backend) | spatial lag / error coefficient names | `rho` / `lambda` | `lambda` / `rho` | `test_r_bridge.py` |
| `plm` (splm backend) | R² of random effects | pseudo R² | R² of the transformed model | — |
| `plm` (splm backend) | covariance of spatial parameter and coefficients | spreg: full matrix; splm: block-diagonal | as splm | — |
| `lm` | mixed models, RLM, GLM influence | statsmodels' optimizers and tolerances | lme4, MASS, base R | `test_r_lm.py` |
| `sem` | GFI, AIC, BIC, log-likelihood, standardized residual covariances | semopy: its own definitions | lavaan: the usual ML values | `test_lavaan.py` (the measures that agree) |
| `lib` | omega of items with unequal variances | standardized items by default | raw sum score | `test_mcdonald.py` |

## What was not compared with R

There is no standard R function for: the Hosmer–Lemeshow and Cameron–Trivedi tests as implemented, the randomized
quantile residual test, the group Cook's distance of mixed models, the approximate tests of robust models, the
Baltagi–Song–Koh joint and spatial tests (`splm::bsktest` failed with the Matrix package installed here; they are
checked through their relations to `plmtest` and `slmtest`), the LR test of time effects with its degrees-of-freedom
scaling, and the permutation Moran test. These are validated by their definitions and by simulations of their size and
power, which are described on the module pages.

## Running the checks

```shell
pytest tests                                # everything possible on this machine
APATOOLS_RSCRIPT=no-such-rscript pytest     # the suite without R: the R tests are skipped
```

The R tests need `Rscript` and the packages `plm`, `splm`, `spdep`, `spatialreg`, `lmtest`, `MASS`, `lme4`, `lavaan`,
`semPlot` and `coefficientalpha`; `tests/r_reference.py` finds them and skips the test when one is missing. The
continuous-integration job with R runs the lavaan tests.
