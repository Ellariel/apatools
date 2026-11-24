# psm: propensity score matching

Propensity score matching with [psmpy](https://github.com/adriennekline/psmpy) and a report of the covariate balance.
`psm` estimates propensity scores by logistic regression and matches every treated observation to its nearest control
on the logit of the score; `psm_effect_size` reports the standardized mean differences before and after matching.

```python
from apatools.psm import psm, psm_effect_size

results, model = psm(
    data,
    treatment_variable="vs",
    include_vars=["age", "gender"],
    balance=True,
    replacement=True,
    seed=1,
    return_model=True,
)
print(psm_effect_size(model))
```

## Design choices and caveats

* **No caliper.** Matching without a caliper can pair dissimilar observations. Austin (2011) recommends a caliper of 0.2
  standard deviations of the logit of the propensity score; `psm` does not apply one. Check the balance with
  `psm_effect_size` and, if it is poor, restrict the sample yourself.
* **Balance is measured as Cohen's d** with the pooled standard deviation, `|mean_treated − mean_control| /
  √(((n₁−1)s₁² + (n₀−1)s₀²)/(n₁+n₀−2))`. Values below about 0.1 after matching are commonly taken to indicate adequate
  balance (Austin, 2011); that is a convention, not a test.
* **A zero pooled standard deviation** (a constant covariate) makes psmpy divide by `1e-6`, which gives a very large
  value for any difference.
* **Covariates must be numeric**: they are cast to float, so dummy-code categorical variables first.
* **Reproducibility.** `seed` is passed on to psmpy's own sampling (used with `balance=True` and unequal groups).
* **What it is not.** A propensity score method removes the imbalance of the *observed* covariates in the matched
  sample; it says nothing about unobserved confounding.

## Where it differs from R

R's `MatchIt` offers the same family of methods with many more options (calipers, other distance measures, exact
matching). apatools does not reproduce `MatchIt`; psmpy's defaults (logistic scores, nearest neighbour on the logit)
are what you get.

## Checked against

`tests/test_psm.py` runs the example of psmpy (the `mtcars` data) and checks that matching reduces the standardized
differences and that a fixed seed reproduces the result. There is no comparison with R.

## Literature

The references cited in the docstrings of the functions of this module (they are collected from the docstrings when the
documentation is built; the software that does the work is on the [software page](../software.md)).

```{include} ../_generated/references_psm.md
```
