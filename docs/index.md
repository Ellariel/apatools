# apatools

A toolbox for statistical analysis and for reporting results the way APA journals ask for them: wrappers around
established engines (statsmodels, spreg, linearmodels, semopy, psmpy and, optionally, R packages such as lavaan, plm
and splm) that return tidy tables with the statistics a manuscript needs, plus the specification tests, effect
conversions and reliability coefficients that go with them.

Two things set the documentation apart:

* **Every module page says what the package decided for you.** Where a number depends on a convention (degrees of
  freedom, which variance is used, which test variant), the page says which one and where it differs from the
  benchmark in R. Read the *Design choices and caveats* section of a module before you cite its numbers.
* **The literature is part of the documentation.** The [literature page](literature.md) lists every reference cited in
  the docstrings, and the [software page](software.md) says what to cite for each engine.

```{toctree}
:maxdepth: 2
:caption: Using apatools

getting-started
guide/index
engines
validation
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
literature
software
changelog
contributing
```
