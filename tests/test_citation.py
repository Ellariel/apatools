"""apatools.citation: what to cite for a module and its engines (no optional dependency needed)."""

import doctest
import re

import pytest

import apatools.citation as citation_module
from apatools.citation import ENGINES, REFERENCES, cite


def test_the_table_is_consistent():
    used = {
        name
        for engines in ENGINES.values()
        for names in engines.values()
        for name in names
    }
    assert used == set(
        REFERENCES
    )  # nothing refers to a reference that does not exist, none is left over
    for name, text in REFERENCES.items():
        assert re.search(r"\((\d{4}|n\.d\.)\)\.", text), (
            name
        )  # APA: authors (year). title
        assert text.strip() == text and "  " not in text and "\\" not in text, name
        assert re.search(r"https?://\S+$", text), (
            name
        )  # every reference ends with a DOI or an URL
        assert " python " not in text and " python." not in text, (
            name
        )  # Python is a proper noun
    assert re.search(
        r"\(pp\. \d+–\d+\)", REFERENCES["statsmodels"]
    )  # page ranges with an en dash
    assert (
        "doi.org/10.25080/Majora-92bf1922-011" in REFERENCES["statsmodels"]
    )  # checked against Crossref


@pytest.mark.parametrize("module", sorted(ENGINES))
def test_cite_every_module_and_engine(module):
    refs = cite(module)
    assert refs and refs == sorted(set(refs)) and set(refs) <= set(REFERENCES.values())
    for engine in ENGINES[module]:
        assert (
            cite(module, engine=engine)
            == cite(module, engine=[engine])
            == cite(module, engine=(engine,))
        )
    assert cite(module) == cite(
        module, engine=next(iter(ENGINES[module]))
    )  # the default is the first engine


def test_references_depend_on_the_engines():
    assert cite("plm") == cite("plm", engine="spreg") == [REFERENCES["pysal"]]
    assert set(cite("plm", engine="linearmodels")) == {
        REFERENCES["pysal"],
        REFERENCES["linearmodels"],
    }
    assert set(cite("plm", engine="splm")) == {
        REFERENCES[k] for k in ("R", "plm", "splm", "spdep")
    }
    assert cite("sem") == cite("sem", engine="semopy") == [REFERENCES["semopy"]]
    assert set(cite("sem", engine="lavaan")) == {REFERENCES["R"], REFERENCES["lavaan"]}
    assert REFERENCES["semopy"] not in cite("sem", engine="lavaan")


def test_several_engines_are_listed_without_duplicates():
    both = cite("sem", engine=["lavaan", "semPlot"])
    assert (
        set(both) == {REFERENCES[k] for k in ("R", "lavaan", "semPlot")}
        and len(both) == 3
    )  # R only once
    assert (
        cite("sem", engine=["semPlot", "lavaan", "lavaan"]) == both
    )  # order and repetitions do not matter
    # fitted with semopy and drawn in Python, or drawn by Graphviz
    assert set(cite("sem", engine=["semopy", "networkx"])) == {
        REFERENCES[k] for k in ("semopy", "networkx", "matplotlib")
    }
    assert set(cite("sem", engine=("semopy", "graphviz"))) == {
        REFERENCES["semopy"],
        REFERENCES["graphviz"],
    }
    assert cite("sem", engine=[]) == []  # nothing asked, nothing cited


def test_cite_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="Unknown module 'calc', use one of"):
        cite("calc")
    with pytest.raises(
        ValueError,
        match=r"Unknown engine 'stata' for 'sem', use one of \['graphviz', 'lavaan'",
    ):
        cite("sem", engine="stata")
    with pytest.raises(ValueError, match="Unknown engine 'lavaan' for 'psm'"):
        cite("psm", engine=["psmpy", "lavaan"])
    with pytest.raises(TypeError):
        cite(
            "sem", plot_engine="networkx"
        )  # the engines of sem_plot are engines like the others now


def test_the_modules_no_longer_have_citation_constants():
    pytest.importorskip("pandas")
    import apatools.lm

    assert not hasattr(apatools.lm, "CITATION") and not hasattr(
        citation_module, "Citation"
    )


def test_docstring_examples():
    assert doctest.testmod(citation_module).failed == 0
