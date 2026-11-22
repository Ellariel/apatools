"""Shared plumbing of the tests that compare apatools with R packages (the "R references").

The R code of a comparison lives in the test file that needs it; this module only finds R, checks that the
R packages are installed (the test is skipped otherwise), moves data frames and matrices to R as CSV files
and reads the results back. It uses the R bridge of the package (``apatools._lavaan``), so
``APATOOLS_RSCRIPT`` selects the Rscript here as well.

Usage::

    out = run_r(
        '''
        m <- lm(y ~ x, data = d)
        res <- data.frame(term = names(coef(m)), estimate = coef(m))
        ''',
        frames={"d": df},
        outputs=["res"],
        packages=["MASS"],
    )
    out["res"]  # a DataFrame; matrices go in ``matrices`` (read as ``M <- as.matrix(read.csv(..., header = FALSE))``)
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apatools._lavaan import find_rscript, run_r as _run_script

_PROBED = {}  # R package -> installed (None: no R)
_EXE = []


def rscript():
    """Path of Rscript, or None without R."""
    if not _EXE:
        try:
            _EXE.append(find_rscript())
        except RuntimeError:
            _EXE.append(None)
    return _EXE[0]


def _probe(packages):
    exe = rscript()
    todo = [p for p in packages if p not in _PROBED]
    if exe and todo:
        code = " ".join(
            f'cat(requireNamespace("{p}", quietly = TRUE), "\\n");' for p in todo
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.R"
            path.write_text(code, encoding="utf-8")
            done = subprocess.run(
                [exe, str(path)], capture_output=True, text=True, check=False
            )
        flags = done.stdout.split()
        for p, flag in zip(todo, flags):
            _PROBED[p] = flag == "TRUE"
    return exe


def installed(*packages):
    """{package: is it installed in R} (all False without R)."""
    _probe(packages)
    return {p: bool(_PROBED.get(p)) for p in packages}


def require_r(*packages):
    """Skip the test (or, at module level with ``allow_module_level``-style use, the module) without R and the packages."""
    exe = _probe(packages)
    if exe is None:
        pytest.skip("R is not installed", allow_module_level=True)
    missing = [p for p in packages if not _PROBED.get(p)]
    if missing:
        pytest.skip(
            f"the R packages {missing} are not installed", allow_module_level=True
        )
    return exe


def run_r(script, frames=None, matrices=None, outputs=(), packages=()):
    """
    Run R code and return the data frames ``outputs`` it wrote.

    ``frames`` (name -> DataFrame) are available in R as data frames of that name; ``matrices`` (name -> array)
    as numeric matrices. The names listed in ``outputs`` must be data frames in the script; they come back as
    DataFrames. The R packages are attached first.
    """
    require_r(*packages)
    with tempfile.TemporaryDirectory(prefix="apatools_ref_") as tmp:
        tmp = Path(tmp)
        prelude = [f"suppressMessages(library({p}))" for p in packages]
        for name, frame in (frames or {}).items():
            frame.to_csv(tmp / f"{name}.csv", index=False)
            prelude.append(f'{name} <- read.csv("{name}.csv")')
        for name, matrix in (matrices or {}).items():
            pd.DataFrame(np.asarray(matrix, dtype=float)).to_csv(
                tmp / f"{name}.csv", index=False, header=False
            )
            prelude.append(
                f'{name} <- as.matrix(read.csv("{name}.csv", header = FALSE))'
            )
        epilogue = [
            f'write.csv({name}, "out_{name}.csv", row.names = FALSE)'
            for name in outputs
        ]
        _run_script("\n".join(prelude + [script] + epilogue), tmp)
        return {name: pd.read_csv(tmp / f"out_{name}.csv") for name in outputs}
