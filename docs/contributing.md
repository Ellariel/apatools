# Contributing and development

## Tests

```shell
pip install -e ".[all]" pytest ruff
pytest tests                 # everything that can run on this machine
ruff check apatools tests
```

The test files are skipped, not failed, when their dependencies are missing (`tests/conftest.py` lists what each
file needs), so `pip install apatools pytest && pytest` tests the core only.

**Tests against R.** `tests/test_r_*.py`, `tests/test_lavaan.py`, `tests/test_splm.py` and parts of
`tests/test_mcdonald.py` compare apatools with R (see [validation](validation.md)). They run when `Rscript` and the
packages are found and are skipped otherwise; set `APATOOLS_RSCRIPT=no-such-rscript` to see the suite without R. The
shared plumbing is `tests/r_reference.py`: write the R code in the test, pass data frames and matrices, get data
frames back.

**What a good test looks like here.** Prefer a *reference that is independent of the code under test*: a dense
matrix inverse for a trace series, OLS with dummies for a within estimator, a likelihood written out in numpy for an
ML fit, R for everything R has. Check a new test by breaking the code on purpose and confirming that it fails.

## Documentation

```shell
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The API reference and the *Literature* section at the end of each module page are generated from the docstrings at
build time
(`docs/_ext/apatools_docs.py`), so they cannot drift from the code: write complete NumPy-style docstrings, with a
`References` section for every non-trivial method (author, year, title, journal, DOI). The module pages in
`docs/guide/` are written by hand; when a change alters a convention, a tolerance or a caveat, update the *Design
choices and caveats* section of the module and the [changelog](changelog.md).

## Conventions of the code

* One name for the confidence level everywhere: `conf_level` (0.95), never `alpha`.
* Tables are tidy `pandas.DataFrame`s; strings (APA formatting) only at the end, behind `format_pval`, `add_stars`
  and `decimal`.
* Private helpers shared by modules live in `apatools/_common.py`; the R bridge and the adapters of the R packages
  in `apatools/r/`.
