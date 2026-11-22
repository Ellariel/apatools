"""Packaging smoke tests: the core needs only numpy/scipy, the other modules need their extra."""

import importlib
import importlib.util
import pkgutil
import re
import subprocess
import sys

import pytest

import apatools

# public module -> third-party packages it needs (installed by the extra of the same name)
REQUIRES = {
    "calc": [],
    "citation": [],
    "format": [],
    "lm": ["pandas", "statsmodels"],
    "lib": ["pandas", "statsmodels"],
    "slm": ["pandas", "statsmodels", "spreg", "libpysal"],
    "plm": ["pandas", "spreg", "libpysal", "patsy"],
    "sem": ["pandas", "semopy"],
    "psm": ["pandas", "psmpy"],
    "r": [],  # the package of the R bridge: its modules need pandas (and R) only when an R engine is used
}
CORE = [name for name, packages in REQUIRES.items() if not packages]

# every third-party package that is not part of the core
HEAVY = [
    "pandas", "statsmodels", "patsy", "spreg", "libpysal", "semopy", "psmpy",
    "sklearn", "linearmodels", "graphviz", "krippendorff", "matplotlib", "seaborn",
]  # fmt: skip


def installed(*packages):
    return all(importlib.util.find_spec(p) is not None for p in packages)


def run_python(code):
    """Run code in a fresh interpreter; returns the finished process."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def blocking(*packages):
    # a None entry in sys.modules makes `import <package>` raise ImportError
    return "import sys, warnings; warnings.simplefilter('ignore');" + "".join(
        f"sys.modules[{p!r}] = None;" for p in packages
    )


def test_version():
    assert isinstance(apatools.__version__, str) and apatools.__version__[0].isdigit()


def test_all_public_modules_are_listed():
    """Guards against a forgotten module (or a removed one that is still referenced)."""
    found = {
        m.name
        for m in pkgutil.iter_modules(apatools.__path__)
        if not m.name.startswith("_")
    }
    assert found == set(REQUIRES)


@pytest.mark.parametrize("name", REQUIRES)
def test_modules_import(name):
    if not installed(*REQUIRES[name]):
        pytest.skip(f"needs {REQUIRES[name]}")
    importlib.import_module(f"apatools.{name}")


def test_core_works_without_any_optional_package():
    """calc, format and citation must run with pandas, statsmodels, spreg, ... all missing."""
    code = (
        blocking(*HEAVY)
        + "from apatools.calc import r_from_z, p_from_r;"
        + "from apatools.format import format_p, format_r, get_stars;"
        + "from apatools.citation import cite;"
        + "import numpy as np;"
        + "assert abs(r_from_z(1.0) - np.tanh(1.0)) < 1e-12;"
        + "assert format_p(0.0004) == 'p < .001' and format_r(-0.5) == 'r = -.50';"
    )
    out = run_python(code)
    assert out.returncode == 0, out.stderr[-500:]


# module, a package it imports, the extra that provides it
EXTRAS = [
    ("lm", "statsmodels", "lm"),
    ("lib", "statsmodels", "lib"),
    ("slm", "spreg", "slm"),
    ("plm", "spreg", "plm"),
    ("sem", "semopy", "sem"),
    ("psm", "psmpy", "psm"),
]


@pytest.mark.parametrize("module, package, extra", EXTRAS)
def test_missing_dependencies_name_the_extra(module, package, extra):
    out = run_python(blocking(package) + f"import apatools.{module}")
    assert out.returncode != 0
    assert "ImportError" in out.stderr
    assert f'pip install "apatools[{extra}]"' in out.stderr, out.stderr[-500:]
    # the error names the first package that fails to import: the blocked one, or another required
    # package that happens to be missing as well (e.g. pandas in a core-only installation)
    named = re.search(rf"apatools\.{module} needs the package '(\w+)'", out.stderr)
    assert named, out.stderr[-500:]
    assert named.group(1) in REQUIRES[module]
    if all(installed(p) for p in REQUIRES[module] if p != package):
        assert named.group(1) == package


def test_common_names_pandas_when_it_is_missing():
    out = run_python(blocking("pandas") + "import apatools._common")
    assert out.returncode != 0 and "pip install pandas" in out.stderr


@pytest.mark.parametrize(
    "blocked, module",
    [
        ("linearmodels", "apatools.plm"),
        ("krippendorff", "apatools.lib"),
        ("graphviz", "apatools.sem"),
        ("networkx", "apatools.sem"),
        ("matplotlib", "apatools.sem"),
        ("sklearn", "apatools.lm"),
        ("sklearn", "apatools.lib"),  # mcdonald_omega fits its factor model itself
        ("factor_analyzer", "apatools.lib"),
    ],
)
def test_optional_packages_are_not_needed_at_import(blocked, module):
    """These packages are only needed when the feature that uses them is called."""
    needed = REQUIRES[module.split(".")[1]]
    if not installed(*needed):
        pytest.skip(f"needs {needed}")
    out = run_python(blocking(blocked) + f"import {module}")
    # semopy needs scikit-learn itself, so it cannot be tested through apatools.lm/sem
    assert out.returncode == 0, out.stderr[-500:]


def test_linearmodels_backend_raises_a_helpful_error_when_missing(monkeypatch):
    if not installed("pandas", "spreg", "libpysal", "patsy"):
        pytest.skip("needs the plm extra")
    import pandas as pd

    from apatools.plm import plm

    monkeypatch.setitem(sys.modules, "linearmodels", None)
    monkeypatch.setitem(sys.modules, "linearmodels.panel", None)
    df = pd.DataFrame(
        {
            "y": range(8),
            "x": [1.0, 3, 2, 5, 4, 6, 9, 7],
            "e": [0, 1] * 4,
            "t": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )
    with pytest.raises(ImportError, match=r"apatools\[linearmodels\]"):
        plm(
            df,
            "y",
            ["x"],
            entity="e",
            time="t",
            model="fe",
            backend="linearmodels",
            verbose=False,
        )
