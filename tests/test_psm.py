import numpy as np
import pandas as pd
import pytest

from apatools.citation import cite
from apatools.psm import psm, psm_effect_size


def mtcars():
    """The example data of psmpy (downloaded by statsmodels; statsmodels is not a dependency of psm)."""
    sm = pytest.importorskip("statsmodels.api")
    return sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data.reset_index()


def test_psm_smoke():
    """Runs with the psm extra only (synthetic data, no download)."""
    rng = np.random.default_rng(0)
    n = 120
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d["treated"] = (
        rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * d.x1 - 0.4 * d.x2)))
    ).astype(int)
    d["y"] = d.x1 + 0.5 * d.treated + rng.normal(size=n)

    df, model = psm(
        d,
        treatment_variable="treated",
        include_vars=["x1", "x2"],
        return_model=True,
        seed=1,
    )
    assert set(df["treated"]) == {0, 1}
    assert {"x1", "x2", "y", "index"} <= set(df.columns)
    effects = psm_effect_size(model)
    assert list(effects.columns) == ["variable", "matching", "effect_size"]
    assert set(effects["matching"]) == {"before", "after"} and set(
        effects["variable"]
    ) == {"x1", "x2"}
    assert np.isfinite(effects["effect_size"]).all()
    # matching balances the covariates: smaller standardized differences afterwards
    by = effects.groupby("matching")["effect_size"].apply(lambda s: s.abs().sum())
    assert by["after"] < by["before"]
    # reproducible
    df2 = psm(d, treatment_variable="treated", include_vars=["x1", "x2"], seed=1)
    pd.testing.assert_frame_equal(df, df2)
    assert cite("psm")[0].startswith("Kline")


def test_psm():
    # https://pypi.org/project/psmpy/
    # https://github.com/adriennekline/psmpy
    test_data = mtcars()
    assert len(test_data) == 32

    print(cite("psm")[0])

    df, model = psm(
        test_data,
        treatment_variable="vs",
        include_vars=["mpg", "cyl"],
        exclude_vars=["am", "gear", "qsec", "hp"],
        index_variable="rownames",
        balance=False,
        replacement=True,
        return_model=True,
        seed=13,
    )

    effects = psm_effect_size(model)
    assert f"{effects.effect_size.sum():.3f}" == "4.499"

    df, model = psm(
        test_data,
        treatment_variable="vs",
        include_vars=[],
        exclude_vars=["am", "gear", "qsec", "hp"],
        index_variable="rownames",
        balance=False,
        replacement=True,
        return_model=True,
        seed=13,
    )

    effects = psm_effect_size(model)
    assert f"{effects.effect_size.sum():.3f}" == "12.349"


def test_psm_options():
    d = mtcars()
    kw = dict(
        treatment_variable="vs", index_variable="rownames", balance=False, seed=13
    )

    a = psm(d, include_vars=["mpg", "cyl"], **kw)
    # include_vars wins over exclude_vars
    b = psm(d, include_vars=["mpg", "cyl"], exclude_vars=["hp"], **kw)
    assert a.equals(b)
    # a single string is accepted
    assert len(psm(d, include_vars="mpg", **kw)) > 0
    # exclude_vars alone
    e = psm(
        d, exclude_vars=["am", "gear", "qsec", "hp", "carb", "drat", "disp", "wt"], **kw
    )
    assert len(e) > 0

    # stacked output holds treated and matched controls
    assert set(a["vs"]) == {0, 1}
    # coupled output has the matched columns side by side with a suffix
    coupled = psm(
        d, include_vars=["mpg", "cyl"], return_coupled=True, coupled_suffix="_m", **kw
    )
    assert "vs_m" in coupled.columns and "mpg_m" in coupled.columns
    assert (coupled["vs"] != coupled["vs_m"]).all()
    # no shared mutable default arguments
    assert psm.__defaults__[0] is None and psm.__defaults__[1] is None


if __name__ == "__main__":
    test_psm()
    print("Tests for apatools.psm are PASSED!")


def test_psm_seed_controls_psmpy_sampling():
    """With unequal groups and balance=True psmpy samples the larger group: the seed must reach it."""
    rng = np.random.default_rng(5)
    n = 300
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d["treated"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 * d.x1 - 2.5)))).astype(int)
    assert 10 < d["treated"].sum() < n / 4  # a clear minority
    kw = dict(treatment_variable="treated", include_vars=["x1", "x2"], balance=True)
    first = psm(d, seed=1, **kw)
    pd.testing.assert_frame_equal(first, psm(d, seed=1, **kw))
    others = [psm(d, seed=s, **kw) for s in (2, 3, 4)]
    assert any(not o.equals(first) for o in others)
