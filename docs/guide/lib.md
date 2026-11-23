# lib: reliability, agreement, correlations and bootstrap

Assorted helpers (extra `lib`): percentile bootstrap, scale reliability (Cronbach's alpha, McDonald's omega with
variants), inter-rater agreement (Fleiss' kappa, Krippendorff's alpha), pooling and comparing correlations, the HTMT
ratio and the Gini coefficient. Everything is imported from `apatools.lib`.

```python
import numpy as np
from apatools.lib import bootstrap, cronbachs_alpha, mcdonald_omega, mcdonald_omega_bifactor, fleiss_kappa

cil, m, cir = bootstrap([1, 2, 3, 4, 5, 6, 7, 8, 9], func=np.mean, n_rep=1000, conf_level=0.95)

alpha, lower, upper = cronbachs_alpha(items)                       # analytic interval
omega, lower, upper = mcdonald_omega(items)                        # delta method, on the logit scale
omega, lower, upper = mcdonald_omega(items, robust=True)           # robust to outlying respondents
table = mcdonald_omega_bifactor(items, {"A": ["x1", "x2", "x3"], "B": ["x4", "x5", "x6"], "C": ["x7", "x8", "x9"]})
```

## Design choices and caveats

**Bootstrap.** A percentile bootstrap: simple, but it can be biased and too narrow for small samples or skewed
statistics; the interval is taken from the replications that are not missing.

**Cronbach's alpha.** `alpha = k/(k−1)·(1 − Σ item variances / variance of the sum score)` from the sample covariance
matrix. It equals the reliability only for tau-equivalent items (equal loadings, uncorrelated errors) and otherwise
underestimates it: report it with omega (Kelley and Pornprasertmanit, 2016; Sijtsma, 2009). It changes when single
items are rescaled. The default interval is analytic, `alpha ± z·ASE` with the asymptotic standard error under normal
theory (van Zyl et al., 2000; Duhachek and Iacobucci, 2004), the upper limit capped at 1; it relies on normally
distributed items, so for skewed items, few categories or small samples use `bootstrap_iter` (Padilla et al., 2012).
Items must be keyed in the same direction (a warning is given otherwise).

**McDonald's omega.**

* *Model.* A one-factor model with unequal loadings (congeneric items), estimated by maximum likelihood *by apatools*
  with numpy and scipy (no scikit-learn or factor-analyzer needed).
* *Standardized by default.* Omega of the standardized items does not change when items are rescaled or shifted.
  `standardized=False` gives the reliability of the raw sum score, the definition of lavaan/semTools and of Zhang and
  Yuan (2016); the two differ when the items have different variances.
* *Interval.* The default is a delta-method interval (normal-theory ML, computed on the logit scale so that it stays
  inside (0, 1)); `bootstrap_iter` gives a percentile bootstrap interval.
* *Robust omega* (`robust=True`, `varphi=0.1`; Zhang and Yuan, 2016). The covariance matrix of the items is a Huber-type
  M-estimate that gives cases far from the center weights below one (a share `varphi` of the cases is downweighted
  under normality), and the interval uses the sandwich covariance matrix of the estimator; `varphi=0` keeps the sample
  covariance matrix with the distribution-free standard error. In simulations with 8 % outlying respondents the
  ordinary omega fell from .82 to .57 while the robust omega stayed at .79; the robust interval is then still somewhat
  too narrow (coverage about 70 %), so use `bootstrap_iter` when outliers are many.
* *Bifactor omega* (`mcdonald_omega_bifactor`). A confirmatory bifactor model (a general factor and one uncorrelated
  group factor per subscale; at least two groups of at least three items; items outside the groups load on the general
  factor only) gives omega hierarchical (`omega_h`), omega total (`omega_t`), the explained common variance (`ecv`) and
  the subscale omegas (`omega_s`, `omega_hs`), with delta-method (sandwich) or bootstrap intervals and the same
  `robust`, `varphi` and `standardized` options. **Bifactor solutions can be improper in small samples** (a uniqueness at
  zero, a Heywood case; in about 1–2 % of simulated samples of 500): the analytic interval is then not available and the
  bootstrap is the alternative.

**Fleiss' kappa.** The agreement of several raters, with a z test against chance agreement. The standard error is the
null variance of Fleiss, Nee and Landis (1979), as in R's `irr::kappam.fleiss`. A missing rating is not a category: items
with a missing rating are left out (NaN in the per-item agreement). The one-tailed p-value uses the signed z. The
interpretation bands of Landis and Koch (1977) are conventions; the test tells whether the agreement exceeds chance,
not how large it is. (Before 0.2.0 the standard error was 2–3 times too small, with about 30 % false positives at the
5 % level.)

**Krippendorff's alpha.** `1 − D_o/D_e` for any number of raters, with missing ratings (excluded, not counted as a
category; units with fewer than two ratings are ignored) and four levels of measurement (nominal, ordinal, interval,
ratio: the *values* are used, not the position of a category). A permutation z-score and p-value are added. It is
implemented in apatools (the `krippendorff` package, GPL, is no longer a dependency); **the layout is one row per unit
and one column per rater**, the transpose of that package. Krippendorff recommends alpha ≥ .80 for firm conclusions
and ≥ .667 for tentative ones.

**Comparing and pooling correlations.**

* `dependent_correlations` (two correlations that share a variable): Williams' t test as given by Steiger (1980), or
  Zou's (2007) confidence interval of the difference; `independent_correlations`: a z test on the Fisher-transformed
  correlations, or Zou's interval. `rz_ci` and `rho_rxy_rxz` are the building blocks.
* `average_correlations` pools correlations on Fisher's z scale with the weights `n − 3`: a *fixed-effect* model that
  assumes one common population correlation. Heterogeneity is neither tested nor modelled; use a random-effects model
  for that.
* The confidence level is `conf_level` (0.95); with `method="one-tailed"` both limits use the one-sided critical value.

**HTMT ratio.** The similarity of items of different constructs relative to items of the same construct; values near 1
mean that the constructs are not empirically distinct. Thresholds .85 (conservative) or .90 (liberal; Henseler et al.,
2015) are conventions, and the bootstrap interval that would show an HTMT *significantly* below them is not provided.
At least two items per construct are needed.

**Gini coefficient.** The sample Gini without the `n/(n−1)` correction, identical to `ineq::Gini(x, corr=FALSE)` in R.
Negative values are not allowed: if any value is below `tolerance`, all values are shifted by the absolute minimum
(which changes the result), and unless `drop_zeros` a tolerance is added so that no value is exactly zero.

**Transformations.** `z_transform` standardizes with the mean and the *population* standard deviation (constant or
all-NaN input gives NaN with `ignore_errors=True`, instead of dividing by zero). `log_transform` is the signed
pseudo-logarithm `sign(x)·log10(1 + |x|)`.

## Where it differs from R

| what | apatools | R | difference |
|---|---|---|---|
| Fleiss' kappa standard error | Fleiss–Nee–Landis null variance | `irr::kappam.fleiss` | the same formula |
| McDonald's omega, bifactor | one-factor / bifactor ML in numpy | `lavaan` (delta method) | estimates within 2e-5, standard errors within 1 % |
| robust omega | Huber M-estimate with sandwich | `coefficientalpha::omega` | estimates within 1e-5, standard errors within 2 % (0.3 % in the check against the package) |
| omega definition | standardized items by default | lavaan, semTools: raw sum score | use `standardized=False` for theirs |
| Gini | sample Gini | `ineq::Gini(corr=FALSE)` | the same |
| Krippendorff's alpha | rows are units | Python `krippendorff` package: columns are units | transposed input |

## Checked against

`tests/test_mcdonald.py` compares the bifactor omega with lavaan and the robust omega with the R package
*coefficientalpha* (skipped without R), and the delta-method intervals with simulations; `tests/test_lib.py` the Fleiss
kappa with published reference values and the formula of `irr`, Krippendorff's alpha with the published example, the
definition and invariances, and the correlation tests with their definitions.

## Literature

The references cited in the docstrings of the functions of this module (they are collected from the docstrings when the
documentation is built; the software that does the work is on the [software page](../software.md)).

```{include} ../_generated/references_lib.md
```
