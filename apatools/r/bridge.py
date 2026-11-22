"""Bridge to R shared by the R engines of apatools (:mod:`apatools.r.lavaan`, :mod:`apatools.r.splm`).

R is started as a subprocess (``Rscript``) and exchanges data with Python through temporary CSV files, so no Python
package beyond pandas is needed. ``Rscript`` is looked up in this order: the ``rscript`` argument, the environment
variable ``APATOOLS_RSCRIPT``, the ``PATH``, ``R_HOME`` and the usual install locations. The messages of R are in
English (``LANGUAGE=en``) whatever the system language; set the environment variable ``APATOOLS_R_LANGUAGE``
(e.g. ``de``) to get them translated.

Private: the R engines are used through :func:`apatools.sem.sem`, :func:`apatools.sem.sem_plot` and
:func:`apatools.plm.plm`.
"""

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ENV_RSCRIPT = "APATOOLS_RSCRIPT"
ENV_LANGUAGE = (
    "APATOOLS_R_LANGUAGE"  # language of the messages of R; English by default
)
_MISSING_PACKAGE = "MISSING_R_PACKAGE:"


def find_rscript(rscript=None):
    """
    Path of the ``Rscript`` executable.

    Looks at ``rscript`` (a path or a command name), the environment variable ``APATOOLS_RSCRIPT``,
    the ``PATH``, ``R_HOME`` and the usual install locations of Windows, macOS and Linux (the newest
    R version on Windows). Raises ``RuntimeError`` with installation hints if R is not found.
    """
    candidates = []
    for explicit in (rscript, os.environ.get(ENV_RSCRIPT)):
        if explicit:
            found = shutil.which(str(explicit))
            if found:
                return found
            raise RuntimeError(
                f"Rscript '{explicit}' was not found: check the path "
                f"(argument 'rscript' or environment variable {ENV_RSCRIPT})."
            )
    found = shutil.which("Rscript")
    if found:
        return found
    exe = "Rscript.exe" if os.name == "nt" else "Rscript"
    if os.environ.get("R_HOME"):
        candidates.append(os.path.join(os.environ["R_HOME"], "bin", exe))
    candidates += sorted(
        glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True
    )
    candidates += sorted(
        glob.glob(r"C:\Program Files\R\R-*\bin\x64\Rscript.exe"), reverse=True
    )
    candidates += [
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/bin/Rscript",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        "R was not found (needed for engine='lavaan'). Install R from https://cran.r-project.org and the R "
        "packages with install.packages(c('lavaan', 'semPlot')); if Rscript is not on the PATH, pass its "
        f"location as 'rscript' or set the environment variable {ENV_RSCRIPT}."
    )


def r_literal(value):
    """Python value as R code: None, bool, numbers, strings and (nested) lists / tuples of them."""
    if value is None:
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}L"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NA_real_"
        if np.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return repr(float(value))
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are valid R escapes
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [r_literal(v) for v in value]
        scalars = all(not isinstance(v, (list, tuple, dict, np.ndarray)) for v in value)
        return ("c(" if scalars else "list(") + ", ".join(items) + ")"
    if isinstance(value, dict):
        return (
            "list("
            + ", ".join(
                f"{json.dumps(str(k))} = {r_literal(v)}" for k, v in value.items()
            )
            + ")"
        )
    raise TypeError(f"Cannot pass a value of type {type(value).__name__} to R.")


def run_r(script, workdir, rscript=None, timeout=None):
    """Run an R script (text) in ``workdir`` with Rscript; returns the stderr text (warnings and messages)."""
    exe = find_rscript(rscript)
    path = Path(workdir) / "script.R"
    path.write_text(script, encoding="utf-8")
    try:
        done = subprocess.run(
            [exe, str(path)],
            check=False,  # a failure is reported below, with the stderr of R
            cwd=str(workdir),
            env={
                **os.environ,
                "LANGUAGE": os.environ.get(ENV_LANGUAGE, "en"),
            },  # English, whatever the system
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"R did not finish within {timeout} seconds.") from e
    if done.returncode != 0:
        err = done.stderr or ""
        for line in err.splitlines():
            if _MISSING_PACKAGE in line:
                package = line.split(_MISSING_PACKAGE, 1)[1].strip().strip("'\"")
                raise ImportError(
                    f"The R package '{package}' is not installed: run install.packages('{package}') in R."
                )
        raise RuntimeError("R failed:\n" + "\n".join(err.strip().splitlines()[-15:]))
    return done.stderr or ""


_REQUIRE = (
    'if (!requireNamespace("%s", quietly = TRUE)) stop("'
    + _MISSING_PACKAGE
    + '%s", call. = FALSE)\n'
)


def r_require(*packages):
    """R code that stops with a recognizable message (turned into an ``ImportError``) if a package is missing."""
    return "".join(_REQUIRE % (package, package) for package in packages)
