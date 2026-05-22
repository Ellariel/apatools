import numpy as np
from scipy.stats import spearmanr

from apatools.lib import *


def s(x):
    # print(x)
    return f"{x:.3f}"


def test_lib():



    # fleiss_kappa
    df = pd.DataFrame({
    "Expert1": ["for", "no decision", "for", "for", "no decision", np.nan],
    "Expert2": ["for", "no decision", "for", "no decision", "no decision", np.nan],
    "Expert3": ["for", "no decision", "for", "for", "for", np.nan],
    "Expert4": ["for", "no decision", "for", "for", "no decision", np.nan],
    })
    kappa, z_score, p_value, per_item = fleiss_kappa(df, return_per_item_agreement=True)
    assert s(kappa) == "0.727"
    assert s(z_score) == "10.808"
    assert s(p_value) == "0.000"
    assert len(per_item) == len(df)
   


if __name__ == "__main__":
    test_lib()
    print("Tests for apatools.lib are PASSED!")
