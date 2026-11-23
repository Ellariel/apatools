"""A toolbox for meta-analysis and APA-compliant reporting.

Modules: ``calc`` (conversions, confidence intervals), ``lib`` (bootstrap, reliability and
agreement coefficients), ``lm`` (OLS/RLM/GLM/QuantReg/MixedLM), ``slm`` (spatial
models), ``plm`` (panel models), ``sem`` (structural equation models), ``psm`` (propensity score
matching) and ``format`` (APA formatting). Import the modules you need, e.g.
``from apatools.lm import lm, lm_report``.

The package ``__init__`` imports none of them, so ``import apatools`` is cheap and works with
the core install (numpy and scipy only, enough for ``calc`` and ``format``). The
other modules need optional packages that are installed through extras named after the module,
for example ``pip install "apatools[lm]"``, ``"apatools[sem]"`` or ``"apatools[all]"``; if a
package is missing, importing the module raises an ``ImportError`` naming the extra.
SEM path diagrams additionally need the ``graphviz`` extra and the Graphviz executables.
Reports follow the conventions of the *Publication Manual of the American Psychological
Association* (7th ed., 2020): confidence levels are given as e.g. ``0.95`` (not as
significance levels), p-values and correlations without leading zero.

Attributes
----------
__version__ : str
    The installed package version, or ``"0+unknown"`` when run from a source tree that is not
    installed.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("apatools")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
