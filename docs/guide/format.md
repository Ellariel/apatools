# format: APA formatting

Three functions that turn numbers into the strings an APA manuscript uses. They need only numpy and are applied by all
report functions of the package.

```python
from apatools.format import get_stars, format_p, format_r

format_p(0.0004)              # 'p < .001'
format_p(0.009)               # 'p = .009'
format_r(0.011, use_letter="R")   # 'R = .01'
get_stars(0.04)               # '*'
```

## Design choices and caveats

* **Strings are for the final report.** `format_p` and `format_r` return text, so apply them last: do not compute with
  their output. All three functions pass non-numeric input (strings, `None`) through unchanged, so they can be applied
  to mixed table columns.
* **p-values follow APA 7th edition.** No leading zero (a p-value cannot exceed 1), three decimals (`.009`, `.050`),
  values below .001 are `< .001` (never `p = .000`), values that would round to 1.000 are `> .999`. NaN, infinite and
  negative values are reported as `= NaN`.
* **Correlations and R² have two decimals and no leading zero.** `|r| < .01` shows as `0.` and `|r| > .99` as `1.`
  (or `-1.`), not as `.00` / `1.00`. Values outside [-1, 1] are *not* flagged: validate before you format.
* **Stars use strict upper bounds**: `.001`, `.01`, `.05`, `.10`, so `p = .05` gets the marker of the .10 level, not of
  the .05 level. The marker for p > .10 is empty, and a table note must state the thresholds you used. The default
  marker for the .10 level is `⁺`; all markers can be changed.
* **The tables of the other modules compute the stars from the numeric p-value** even when `format_pval` has turned the
  p-values into strings, so the two columns always agree.

## Where it differs from R

Base R has no APA formatter; R packages for reporting (for example papaja) do the same job. The rules above are those of
the Publication Manual (7th edition); where a journal asks for something else, follow the journal.
