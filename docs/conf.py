"""Sphinx configuration of the apatools documentation."""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # import apatools from the source tree

project = "apatools"
author = "apatools contributors"
copyright = "apatools contributors, MIT license"
try:
    release = version("apatools")
except PackageNotFoundError:  # not installed: take the version from pyproject.toml
    import tomllib

    release = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "apatools_docs",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
master_doc = "index"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {"show-inheritance": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
}

html_theme = "furo"
html_title = f"apatools {release}"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
