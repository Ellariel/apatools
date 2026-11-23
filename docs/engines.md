# Engines

A function of apatools does not estimate anything itself (with a few exceptions named on the module pages): it hands
the model to an *engine*, checks the result and reports it. This page says which engines exist, what they can do and
how R is used.

## Which engine fits what

| module | engine | what it fits | notes |
|---|---|---|---|
| `lm` | statsmodels | OLS, RLM, GLM, QuantReg, MixedLM | the only engine |
| `slm` | spreg | OLS, spatial lag, spatial error (ML) | the only engine |
| `plm` | spreg (default) | pooled, FE, RE; spatial lag / error with ML; GMM error models; Durbin (`slx_lags`) | period effects as dummies; its random-effects error model fails with some numpy/scipy versions |
| `plm` | linearmodels | pooled, FE, RE (no spatial models) | clustered covariances, entity / time / two-way effects natively |
| `plm` | splm (R) | pooled, FE, RE with R's plm; spatial lag / error ML with R's splm | entity / time / two-way effects for all fixed-effects models; no GMM, no `slx_lags`, no `pred_metrics` |
| `sem` | semopy (default) | structural equation models | |
| `sem` | lavaan (R) | the same models | adds SRMR; many more estimators |
| `sem_plot` | semopy / Graphviz, networkx, semPlot (R) | path diagrams | the same diagram in three renderers |
| `psm` | psmpy | propensity score matching | |

## The R engines

R is started as a subprocess (`Rscript`) and exchanges data with Python through temporary CSV files. This keeps the
Python side free of `rpy2` and its build problems: all that is needed is a working `Rscript` and the R packages.

**Finding Rscript.** In this order: the `rscript=` argument, the environment variable `APATOOLS_RSCRIPT`, the `PATH`,
`R_HOME`, and the usual install locations of Windows, macOS and Linux (the newest R version on Windows). If R is not
found, a `RuntimeError` says where it looked.

**Language.** R's messages are set to English (`LANGUAGE=en`) whatever the system language, so that warnings that are
passed on are readable and testable. Set the environment variable `APATOOLS_R_LANGUAGE` (for example `de`) to change
that.

**Missing R packages** raise an `ImportError` that names the package and the `install.packages()` call.

**Time.** Every model starts R again: about 1–2 seconds per model for a panel of 36 entities and 6 periods, more for
larger panels (not timed). `plm(backend="splm", r_timeout=...)` limits the time of *each* R run in seconds; when it is
exceeded the process is stopped and a `RuntimeError` is raised. The default is no limit. Leave it unset while
exploring: a slow fit is usually just a big panel (the spatial ML models are the slow ones, presumably because of the
determinant of `I - rho W` / `I - lambda W`, which grows quickly with the number of entities). Set it for unattended
runs, scripts and CI, where a hanging R process would block everything, to a generous multiple of a typical fit
(for example 300 seconds): the first run is slower (R starts and loads the packages), and a limit that is too short
fails a fit that would have worked. After a timeout the models fitted before the failing one are lost, as with any
other error: fit them one by one or reduce the panel.

**What comes back.** R results are converted to Python objects with the same interface as the other engines' results,
so that reports, tests and effects work unchanged: `LavaanFit` for lavaan and `SplmFit` for plm / splm. They keep the
coefficient table, the covariance matrix, the residuals, the version of the R packages and the warnings R raised.

## Why have several engines at all

Two reasons. First, **cross-checking**: an estimate that two independent implementations agree on is more
trustworthy, and where they disagree the [module pages](guide/index.md) say how and why. Second, **coverage**: each
engine has things the others lack (linearmodels' clustered covariances, splm's period effects for the spatial lag
model, lavaan's estimators). The price is that the engines differ in conventions; apatools never hides a difference
that changes a number, see [validation](validation.md).
