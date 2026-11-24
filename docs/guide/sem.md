# sem: structural equation models

Structural equation modeling with [semopy](https://semopy.com) (default) or, through R, [lavaan](https://lavaan.ugent.be),
with a table of estimates and fit statistics, an APA-style report and path diagrams drawn by three renderers.

```python
from apatools.sem import sem, sem_report, sem_plot

stats, metrics, model = sem(
    data, formula,                      # formula in lavaan syntax
    method="MLW",                       # MLW ULS GLS FIML DWLS WLS (semopy); engine="lavaan" for R
    return_model=True,
)
print(sem_report(stats, metrics))
fig = sem_plot(stats, save_to_file="fig.png", return_fig=True)           # semopy + Graphviz
fig = sem_plot(stats, engine="networkx", save_to_file="fig.svg")         # Python only
```

## Design choices and caveats

**What `stats` contains.** semopy's parameter table plus the standardized estimate `est_std` with its standard error
`se_std`, z value `z_std` and p-value `p_std`. The standard errors of the standardized estimates are computed by
the *delta method*, like lavaan's `standardizedSolution()` (semopy reports none). The first loading of each latent
variable is fixed to 1 for identification, so it has no standard error or p-value in the raw solution (`p-value` is
NaN, in lavaan as well), but its standardized loading is estimated and does have a `p_std`.

**Fit indices** are semopy's (or lavaan's), reported as they come: CFI (Bentler, 1990), TLI (Tucker and Lewis, 1973),
RMSEA (Steiger and Lind, 1980; Browne and Cudeck, 1993). Hu and Bentler's (1999) cut-offs (CFI/TLI of about .95 or
higher, RMSEA of about .06 or lower) are rules of thumb, not tests. Robust standard errors follow the sandwich
estimator of White (1980).

**Missing data: the engines do not treat NaN alike.**

* *lavaan* (R): `NaN` is passed on as missing (`NA`). By default lavaan deletes the cases that have a missing value in a
  variable *of the model* (listwise deletion); `N` in the fit summary is the number of cases used and a warning says how
  many were deleted. `method="FIML"` (full-information maximum likelihood) keeps all cases and adds the intercepts
  (`~ 1` rows) to the table. Columns that the model does not use are ignored, whatever they contain. Infinite values in
  a model variable raise a `ValueError` that names the variables (lavaan's own message talks about variances); a model
  variable without any value raises lavaan's error.
* *semopy*: this wrapper does not delete incomplete cases, and `N` in the fit summary is the number of rows of the data
  whatever is missing. In an example with 8 of 75 rows incomplete the default estimator reported N = 75 and estimates
  that differ from those of listwise deletion (1.28 instead of 1.34 for one loading), and semopy's FIML gave an
  implausible CFI (−0.26). We have not traced how semopy uses incomplete rows. For results you can rely on, drop the
  incomplete rows yourself (and report it) or use the lavaan engine with FIML.

**`method=None` means the default of the engine** (`"MLW"` and 100 bootstrap resamples for semopy, `"ML"` and none for
lavaan). `sem_report` formats `p_std` like `p-value`; because the fit summary sits in the first row of `stats`, the
reported table is for display or export, not for further numerical work.

**`sem_plot`** draws the same diagram with three renderers (circles for latent and boxes for observed variables,
arrows labelled with the estimate and the p-value or stars, dashed arrows for covariances), and labels the arrows with
the p-values that belong to the estimates it shows: `p_std` for the standardized estimates (the default) and `p-value`
for the raw ones (`std_ests=False`).

| `engine=` | drawn by | needs | returns |
|---|---|---|---|
| `"semopy"` (default) | `semopy.semplot`, Graphviz | extra `graphviz` and the Graphviz executables on the `PATH` | `graphviz.Digraph`; `layout="dot"`, `"neato"`, ... |
| `"networkx"` | networkx and matplotlib | extra `networkx`; no external program | a matplotlib figure, honours `dpi`; `layout="layered"` (default: predictors left, outcomes right), `"spring"`, `"kamada_kawai"`, `"circular"` |
| `"lavaan"` | R package *semPlot* in an `Rscript` subprocess, from the table of either `sem` engine | R with `lavaan` and `semPlot` | an `RFigure` (shows in Jupyter, `.save(path)`: PNG, PDF, SVG, JPEG, TIFF, BMP); `layout="tree"` (default), `"tree2"`, `"circle"`, `"circle2"`, `"spring"`, plus `rotation`, `width`, `height`, `fontsize` |

## The lavaan engine

`sem(data, formula, engine="lavaan")` fits the same model with lavaan and returns the same two tables, so that
`sem_report` and `sem_plot` work unchanged (`return_model=True` gives a `LavaanFit` with lavaan's own tables, its
version and warnings). `method` accepts the names of semopy (`"MLW"` and `"FIML"` become ML, the latter with
`missing="fiml"`) and every lavaan estimator (`"MLR"`, `"WLSMV"`, ...); `se_robust=True` means
`se="robust.huber.white"` for ML; `bootstrap=n` gives lavaan's bootstrap standard errors; any other argument of
`lavaan::sem` goes into `lavaan_args`:

```python
stats, metrics = sem(data, formula, engine="lavaan", lavaan_args={"std.lv": True, "ordered": ["x1", "x2"]})
fig = sem_plot(stats, engine="lavaan", save_to_file="fig.png", dpi=300)
```

Limits: only single-group models are supported, and thresholds and defined parameters (`:=`) are not part of the table.
The lavaan engine adds SRMR to the fit statistics.

## Where the engines differ

On the political-democracy data (the example of the semopy tutorial) semopy and lavaan agree on:

* the loadings and regressions (estimates within 0.02), the standardized estimates (within 0.002) and the standard
  errors of the standardized estimates (within 0.002);
* chi-square, CFI, TLI, RMSEA and NFI.

They differ in:

* **what they call GFI, AIC, BIC and the log-likelihood**: lavaan reports the usual ML values;
* **the standardized residual covariances**: lavaan reports the correlation of the residuals.

## Checked against

`tests/test_lavaan.py` fits the political-democracy model with both engines and compares the tables and fit measures,
`tests/test_sem.py` covers the semopy engine and the diagrams. The delta-method standard errors of the standardized
estimates of the semopy engine are compared with the ones lavaan itself reports.

## Literature

The references cited in the docstrings of the functions of this module (they are collected from the docstrings when the
documentation is built; the software that does the work is on the [software page](../software.md)).

```{include} ../_generated/references_sem.md
```
