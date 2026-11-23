"""Sphinx extension of the apatools documentation: pages generated from the code at build time.

* ``api/<module>.md``: the API reference of every public module (one ``autofunction`` / ``autoclass`` per public
  function, so that the module docstrings, which contain section headings, are not pulled into the page);
* ``_generated/references_<module>.md``: the references found in the ``References`` sections of the docstrings of a
  module (its functions and, for ``lib``, its submodules), de-duplicated; every guide page includes its own list under
  the heading *Literature*.

Both are written before the build (``builder-inited``), so they cannot drift from the code.
"""

import importlib
import inspect
import pkgutil
import re
import warnings
from pathlib import Path

MODULES = ["calc", "format", "lm", "slm", "plm", "sem", "psm", "lib"]
TITLES = {
    "calc": "calc: conversions and confidence intervals",
    "format": "format: APA formatting",
    "lm": "lm: linear models (statsmodels)",
    "slm": "slm: spatial regression (spreg)",
    "plm": "plm: panel regression (spreg, linearmodels, R)",
    "sem": "sem: structural equation models (semopy, lavaan)",
    "psm": "psm: propensity score matching",
    "lib": "lib: reliability, agreement, correlations, bootstrap",
}


def public_members(module):
    """Names of the public functions and classes of a module (its ``__all__``, or what it defines itself)."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [
            n
            for n, o in vars(module).items()
            if not n.startswith("_")
            and (inspect.isfunction(o) or inspect.isclass(o))
            and getattr(o, "__module__", "") == module.__name__
        ]
    return names


def write_api(srcdir):
    out = Path(srcdir) / "api"
    out.mkdir(exist_ok=True)
    for name in MODULES:
        module = importlib.import_module(f"apatools.{name}")
        lines = [
            f"# {TITLES[name]}",
            "",
            f"The user guide of this module: [{name}](../guide/{name}.md).",
            "",
        ]
        for member in public_members(module):
            obj = getattr(module, member)
            kind = "autoclass" if inspect.isclass(obj) else "autofunction"
            options = "   :members:\n" if kind == "autoclass" else ""
            lines += [
                "```{eval-rst}",
                f".. {kind}:: apatools.{name}.{member}",
                options.rstrip("\n"),
                "```",
                "",
            ]
        text = "\n".join(
            line for i, line in enumerate(lines) if line != "" or lines[i - 1] != ""
        )
        (out / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    index = [
        "# API reference",
        "",
        "Generated from the docstrings.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
    ]
    index += [f"{name}" for name in MODULES] + ["```", ""]
    (out / "index.md").write_text("\n".join(index), encoding="utf-8")


REFERENCE_SECTION = re.compile(
    r"\n\s*References\n\s*-{3,}\n(.*?)(?=\n\s*[A-Z][A-Za-z ]+\n\s*-{3,}\n|\Z)", re.S
)


def docstrings(module):
    yield module.__name__, inspect.getdoc(module) or ""
    for member in public_members(module):
        obj = getattr(module, member)
        yield f"{module.__name__}.{member}", inspect.getdoc(obj) or ""
    if hasattr(module, "__path__"):  # a package (lib): the docstrings of its submodules as well
        for info in pkgutil.iter_modules(module.__path__):
            if not info.name.startswith("_"):
                sub = importlib.import_module(f"{module.__name__}.{info.name}")
                yield sub.__name__, inspect.getdoc(sub) or ""


def references_of(name):
    """The de-duplicated references of the docstrings of ``apatools.<name>``, sorted: a list of strings."""
    found = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = importlib.import_module(f"apatools.{name}")
    for _, doc in docstrings(module):
        for section in REFERENCE_SECTION.findall("\n" + doc):
            for paragraph in re.split(r"\n\s*\n", section.strip()):
                text = " ".join(paragraph.split())
                if len(text) < 25 or not re.search(r"\((?:19|20)\d\d[a-z]?[,)]", text):
                    continue  # credits and bare web links in a References section are not literature
                key = re.sub(r"[^a-z0-9]", "", text.lower())[:45]  # authors, year, first words of the title
                if key not in found or len(text) > len(found[key]):  # keep the fullest version (with the DOI)
                    found[key] = text
    return sorted(found.values(), key=str.lower)


def write_references(srcdir):
    out = Path(srcdir) / "_generated"
    out.mkdir(exist_ok=True)
    for name in MODULES:
        lines = [f"- {text}" for text in references_of(name)]
        (out / f"references_{name}.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


# The references of the docstrings are written like APA text in Markdown style, *Journal, 45*(2): in reST the closing
# asterisk must not be followed directly by a letter, a digit or a bracket, so a backslash-space separates them.
EMPHASIS_BEFORE_TEXT = re.compile(
    r"(\*[^*\s](?:[^*\n]|\n(?!\s*\n))*[^*\s]\*)(?=[A-Za-z0-9(\[])"
)  # may wrap lines


def tidy_docstring(app, what, name, obj, options, lines):
    text = EMPHASIS_BEFORE_TEXT.sub(lambda m: m.group(1) + "\\ ", "\n".join(lines))
    lines[:] = text.split("\n")


def builder_inited(app):
    write_api(app.srcdir)
    write_references(app.srcdir)


def setup(app):
    app.connect("builder-inited", builder_inited)
    app.connect("autodoc-process-docstring", tidy_docstring)
    return {"parallel_read_safe": True}
