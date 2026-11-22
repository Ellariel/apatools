"""Lets the test suite run with only the core dependencies installed (``pip install apatools``).

Test files whose third-party packages are missing are not collected, so that e.g.
``pip install apatools pytest && pytest`` tests ``calc`` and ``format`` and the packaging checks,
and ``pip install "apatools[lm]" pytest`` additionally tests ``lm``.
"""

import importlib.util

# test file -> third-party packages it needs (see the extras in pyproject.toml)
REQUIRES = {
    "test_common.py": ["pandas"],
    "test_lavaan.py": ["pandas"],
    "test_r_bridge.py": ["pandas"],
    "test_lib.py": ["pandas", "statsmodels"],
    "test_mcdonald.py": ["pandas", "statsmodels"],
    "test_lm.py": ["pandas", "statsmodels"],
    "test_r_lm.py": ["pandas", "statsmodels"],
    "test_slm.py": ["pandas", "statsmodels", "spreg", "libpysal"],
    "test_r_slm.py": ["pandas", "statsmodels", "spreg", "libpysal"],
    "test_plm.py": ["pandas", "spreg", "libpysal", "patsy"],
    "test_r_plm.py": ["pandas", "spreg", "libpysal", "patsy"],
    "test_splm.py": ["pandas", "spreg", "libpysal", "patsy"],
    "test_sem.py": ["pandas", "semopy"],
    "test_psm.py": [
        "pandas",
        "psmpy",
    ],  # the tests that download example data skip without statsmodels
}


def _installed(name):
    return importlib.util.find_spec(name) is not None


collect_ignore = [
    file
    for file, packages in REQUIRES.items()
    if not all(_installed(p) for p in packages)
]
