import statsmodels.api as sm

from apatools.psm import psm, psm_effect_size, CITATION


def test_psm():
    # https://pypi.org/project/psmpy/
    # https://github.com/adriennekline/psmpy
    test_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    test_data.reset_index(inplace=True)
    assert len(test_data) == 32

    print(CITATION.APA)

    df, model = psm(
        test_data,
        treatment_variable='vs',
        include_vars=['mpg', 'cyl'],
        exclude_vars=['am', 'gear', 
                      'qsec', 'hp'],
        index_variable='rownames',
        balance=False,
        replacement=True,
        return_model=True,
        seed=13,
    )

    effects = psm_effect_size(model)
    assert f"{effects.effect_size.sum():.3f}" == "4.499"

    df, model = psm(
        test_data,
        treatment_variable='vs',
        include_vars=[],
        exclude_vars=['am', 'gear', 
                      'qsec', 'hp'],
        index_variable='rownames',
        balance=False,
        replacement=True,
        return_model=True,
        seed=13,
    )

    effects = psm_effect_size(model)
    assert f"{effects.effect_size.sum():.3f}" == "12.349"

    

def test_psm_options():
    d = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data.reset_index()
    kw = dict(treatment_variable="vs", index_variable="rownames", balance=False, seed=13)

    a = psm(d, include_vars=["mpg", "cyl"], **kw)
    # include_vars wins over exclude_vars
    b = psm(d, include_vars=["mpg", "cyl"], exclude_vars=["hp"], **kw)
    assert a.equals(b)
    # a single string is accepted
    assert len(psm(d, include_vars="mpg", **kw)) > 0
    # exclude_vars alone
    e = psm(d, exclude_vars=["am", "gear", "qsec", "hp", "carb", "drat", "disp", "wt"], **kw)
    assert len(e) > 0

    # stacked output holds treated and matched controls
    assert set(a["vs"]) == {0, 1}
    # coupled output has the matched columns side by side with a suffix
    coupled = psm(d, include_vars=["mpg", "cyl"], return_coupled=True, coupled_suffix="_m", **kw)
    assert "vs_m" in coupled.columns and "mpg_m" in coupled.columns
    assert (coupled["vs"] != coupled["vs_m"]).all()
    # no shared mutable default arguments
    assert psm.__defaults__[0] is None and psm.__defaults__[1] is None


if __name__ == "__main__":
    test_psm()
    print("Tests for apatools.psm are PASSED!")
