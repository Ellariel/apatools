"""Assorted helpers (extra ``lib``): everything is imported from ``apatools.lib``.

* ``misc``: ``bootstrap``, ``gini``, ``gini_from_percentiles``, ``log_transform``, ``z_transform``,
  ``htmt_ratio``
* ``fleiss``: ``fleiss_kappa``
* ``krippendorff``: ``krippendorffs_alpha``
* ``correlations``: ``average_correlations``, ``dependent_correlations``,
  ``independent_correlations``, ``rz_ci``, ``rho_rxy_rxz``
* ``cronbach``: ``cronbachs_alpha``
* ``mcdonald``: ``mcdonald_omega``, ``mcdonald_omega_bifactor``
"""

from importlib import import_module

from .._deps import missing_dependency

try:  # checked once for the whole package, so that the message names the extra to install
    import_module("pandas")
    import_module("statsmodels")
except ImportError as e:  # pragma: no cover
    missing_dependency("lib", "lib", e)

from .correlations import (
    average_correlations,
    dependent_correlations,
    independent_correlations,
    rho_rxy_rxz,
    rz_ci,
)
from .cronbach import cronbachs_alpha
from .fleiss import fleiss_kappa
from .krippendorff import krippendorffs_alpha
from .mcdonald import mcdonald_omega, mcdonald_omega_bifactor
from .misc import (
    bootstrap,
    gini,
    gini_from_percentiles,
    htmt_ratio,
    log_transform,
    z_transform,
)

__all__ = [
    "average_correlations",
    "bootstrap",
    "cronbachs_alpha",
    "dependent_correlations",
    "fleiss_kappa",
    "gini",
    "gini_from_percentiles",
    "htmt_ratio",
    "independent_correlations",
    "krippendorffs_alpha",
    "log_transform",
    "mcdonald_omega",
    "mcdonald_omega_bifactor",
    "rho_rxy_rxz",
    "rz_ci",
    "z_transform",
]
