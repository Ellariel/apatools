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
    assert f"{effects.effect_size.sum():.3f}" == "4.561"

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

    

if __name__ == "__main__":
    test_psm()
    print("Tests for metatools.psm are PASSED!")
