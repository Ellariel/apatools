"""Helper for the optional dependencies of the apatools modules (installed through extras).

The core of apatools (``calc``, ``format``, ``citation``) needs only numpy and scipy. Modules
such as ``lm``, ``slm``, ``plm``, ``sem``, ``psm`` and ``lib`` import their packages (pandas,
statsmodels, spreg, semopy, psmpy, ...) in a ``try`` block and call :func:`missing_dependency`
when an import fails, so that the user is told which extra to install instead of seeing a bare
``ModuleNotFoundError``. Private: not part of the documented API.
"""


def missing_dependency(module, extra, error, hint=None):
    """
    Raise an ``ImportError`` that tells which extra provides the missing package.

    The message reads ``apatools.<module> needs the package '<package>', which is an optional
    dependency and is not installed. Install it with: pip install "apatools[<extra>]"``. The
    function never returns normally; the original error is chained as ``__cause__``.

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
        Replaces the ``pip install "apatools[<extra>]"`` suggestion (then ``extra`` is not used
        and may be None).

    Raises
    ------
    ImportError
        Always.
    """
    package = (getattr(error, "name", None) or str(error)).split(".")[0]  # "statsmodels.api" -> "statsmodels"
    how = hint or f'pip install "apatools[{extra}]"'
    raise ImportError(
        f"apatools.{module} needs the package '{package}', which is an optional dependency "
        f"and is not installed. Install it with: {how}"
    ) from error
