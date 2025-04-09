import statsmodels.api as sm
import pandas as pd
from apatools.lm import lm, lm_report


def s(x):
    return f"{x:.2f}"


def test_lm():

    
    test_data = pd.read_csv('./g_data.csv')
    y = "ren_inst"
    x = ["cult_dist2"]
    
    #test_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    # assert len(test_data) == 32
    #y = "mpg"
    #x = ["wt", "vs"]
    
    results, metrics = lm(
        test_data,
        y,
        x,
        model=["rlm", "ols"],
        verbose=True,
        constant=False,
        standardized=True,
        vif=False,
        r_sq=True,
        pred_r_sq=True,
        ols_fit_cov_type="HC1",
        rlm_model_M=sm.robust.norms.HuberT(),#sm.robust.norms.RamsayE(),
        glm_fit_cov_type="HC1",
        glm_model_family=sm.families.Gaussian(),
    )

    results_rep = lm_report(results, metrics, format_pval=True, add_stars=True)
    print(results_rep, metrics)

    
if __name__ == "__main__":
    test_lm()
