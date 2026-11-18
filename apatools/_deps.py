"""Helper for the optional dependencies of the apatools modules (installed through extras)."""


def missing_dependency(module, extra, error, hint=None):
    """
    Raise an ``ImportError`` that tells which extra provides the missing package.

    Parameters
    ----------
    module : str
        Name of the apatools module that was imported, e.g. ``"lm"``.
    extra : str
        Name of the extra that installs its dependencies, e.g. ``"lm"``
        (``pip install "apatools[lm]"``).
    error : ImportError
        The original error; its ``name`` is the missing package.
    hint : str, optional
        Replaces the ``pip install "apatools[<extra>]"`` suggestion.
    """
    package = (getattr(error, "name", None) or str(error)).split(".")[0]  # "statsmodels.api" -> "statsmodels"
    how = hint or f'pip install "apatools[{extra}]"'
    raise ImportError(
        f"apatools.{module} needs the package '{package}', which is an optional dependency "
        f"and is not installed. Install it with: {how}"
    ) from error
