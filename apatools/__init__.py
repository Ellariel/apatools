"""A toolbox for meta-analysis and APA-compliant reporting.

Modules: ``calc`` (conversions, bootstrap), ``lm`` (OLS/RLM/GLM/QuantReg/MixedLM), ``slm`` (spatial
models), ``plm`` (panel models), ``sem`` (structural equation models), ``psm`` (propensity score
matching) and ``format`` (APA formatting). Import the modules you need, e.g.
``from apatools.lm import lm, lm_report``.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("apatools")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
