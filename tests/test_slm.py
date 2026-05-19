import numpy as np
import pandas as pd
import libpysal

from apatools.slm import slm#, slm_report


def s(x):
    return f"{x:.2f}"


def test_slm():
    db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    cols = ["HOVAL", "INC", "CRIME"]
    df = pd.DataFrame([db.by_col(c) for c in cols], index=cols).T
    assert len(df) == 49
    y, x = cols[0], cols[1:]

    #OLS
    results = slm(df, y, x, model='ols', 
                  standardize=False)
    assert s(results[0].betas[0][0]) == "46.43"
    
    
    

if __name__ == "__main__":
    test_slm()
    print("Tests for metatools.slm are PASSED!")
