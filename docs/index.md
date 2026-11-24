# apatools

A toolbox for statistical analysis and for reporting results the way APA journals ask for them. It conveniently wraps
the most common tools (statsmodels, spreg, linearmodels, semopy, psmpy and, optionally, the R packages lavaan, plm, splm
and spdep) behind one interface and returns tidy tables with the statistics a manuscript needs, plus the specification
tests, effect conversions and reliability coefficients that go with them.

**No method is just linked to an engine and trusted.** Methods are benchmarked against independent references (mostly
R packages, otherwise the definition itself), and the documentation discloses the design decisions, the caveats and
where and why the numbers differ from R; the few methods without a benchmark are marked as such (see
[validation](validation.md)).

Two things set the documentation apart:

* **Every module page says what the package decided for you.** Where a number depends on a convention (degrees of
  freedom, which variance is used, which test variant), the page says which one and where it differs from the
  benchmark in R. Read the *Design choices and caveats* section of a module before you cite its numbers.
* **The literature is part of the documentation.** Every module page ends with the references cited in the docstrings
  of its functions, and the [software page](software.md) says what to cite for each engine.

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
software
changelog
contributing
```
