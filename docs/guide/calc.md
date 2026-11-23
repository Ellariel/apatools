# calc: conversions and confidence intervals

Approximate conversions between the statistics a meta-analyst meets in published papers: r, Cohen's d, t, z, F,
p-values, odds ratios and logit coefficients, and confidence intervals of correlations. Needs only numpy and scipy.

```python
from apatools.calc import cohen_r_from_z, cohen_d_from_z, p_from_r, ci_from_r

z = -1.472
r = cohen_r_from_z(z)        # r = -0.9      (short aliases: r_from_z, d_from_z, ...)
d = cohen_d_from_z(z)        # d = -4.1
p = p_from_r(-0.79, 100)     # p = 0.0
low, high = ci_from_r(0.45, 100, conf_level=0.95)
```

## Design choices and caveats

* **All conversions are approximations.** Most of them need an assumption that is rarely stated in the source paper;
  the docstring of each function names it.
* **`n` is the total sample size** and the degrees of freedom of a t statistic are `n - 2`.
* **Cohen's d conversions assume two groups of equal size** (`n_groups=2`, with the number of groups as an argument
  where it makes sense). With unequal groups the result is biased; pass the group sizes to a dedicated meta-analysis
  tool instead.
* **`d_from_f` is only meaningful for a two-group comparison** (one numerator degree of freedom, where `F = t²`).
* **Two-tailed by default.** `method="two-tailed"`; any other string means one-tailed. The sign of an effect is lost
  when it is recovered from a p-value (`r_from_p`, `d_from_p`, `ci_from_p`): the result is `|r|` or `|d|`.
* **`z` is a standard-normal deviate** unless the name says "Fisher's z" (`z_from_r`, `r_from_z`, `unbiased_z`).
  `z_from_t` has two interpretations that measure different things and are not interchangeable: without `method` it
  returns Fisher's z of the corresponding correlation (sign kept); with a `method` it returns the standard-normal
  deviate that has the same p-value as the t statistic (a normal approximation of the t distribution; always
  non-negative, the sign is lost).
* **`ci_from_r`** uses Fisher's z transformation and so assumes bivariate normality; the interval is asymmetric around
  `r`. `unbiased_z` applies the small-sample correction (Olkin and Pratt) and is recommended for `n < 20`.
* **`p_from_ci`** assumes a symmetric interval on the scale of the estimate; for ratios (OR, RR, HR) pass the
  log-transformed estimate and limits.
* **`logit_from_probit`** uses the rule-of-thumb factor 1.7 (the theoretical ratio of the logistic and normal scales is
  about 1.81). The d ↔ log odds ratio conversions assume a logistic distribution of the underlying continuous
  variable (Hasselblad and Hedges, 1995; Chinn, 2000).
* **`conf_level`** is the confidence level (0.95), in every function. Before version 0.2.0 some functions called it
  `alpha`, and one of them meant a significance level by it; see the [changelog](../changelog.md).

## Where it differs from R

There is no single R function to compare with: the conversions are textbook formulas (Cohen, 1988; Fritz et al.,
2012; Borenstein et al., 2009). Where R packages such as `esc` or `compute.es` give a conversion, expect agreement only
when the same assumptions (equal groups, the same degrees of freedom) are made.

## Checked against

The formulas are covered by unit tests, including the property that conversions are inverses of each other where they
are defined (a round trip d → r → d returns d), which is the reason `n_groups` exists in all of them.
