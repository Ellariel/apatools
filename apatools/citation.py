"""Container for the literature behind a module."""

from typing import NamedTuple


class Citation(NamedTuple):
    """Reference for a method or software package used by an apatools module.

    Modules such as :mod:`apatools.sem` and :mod:`apatools.psm` expose a module-level
    ``CITATION`` constant, so users know what to cite when they report results obtained
    with the underlying package. A ``Citation`` is an immutable named tuple, so it can be
    unpacked, indexed or compared like a tuple.

    Attributes
    ----------
    APA : str
        The full reference in APA style (7th edition), including the DOI where one exists.

    Examples
    --------
    >>> from apatools.citation import Citation
    >>> Citation(APA="Author, A. (2020). Title. Journal, 1(1), 1-2.").APA
    'Author, A. (2020). Title. Journal, 1(1), 1-2.'
    """

    APA: str
