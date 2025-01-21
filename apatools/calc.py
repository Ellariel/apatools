import numpy as np
import scipy


##############
# conversion #
##############


def d_from_r(r):
    # https://stats.stackexchange.com/questions/390549/converting-between-correlation-and-effect-size-cohens-d
    # assume equal groups
    # unequal here https://stats.stackexchange.com/questions/526789/convert-correlation-r-to-cohens-d-unequal-groups-of-known-size

    return 2 * r / np.sqrt(1 - r * r)


def r_from_d(d, n_groups=2):
    # Сorrelation r from Cohen's d
    # https://www.escal.site/
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    # https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/cohens-d/

    return d / np.sqrt(d * d + 2 * n_groups)  # assume equal groups


def t_from_r(r, n):
    # https://stats.stackexchange.com/questions/400146/how-to-derive-the-formula-of-the-t-test-of-the-correlation-coefficient
    # t-critical = stats.t.ppf(alpha/numOfTails, ddof)

    df = n - 2
    t = r / np.sqrt((1 - r * r) / df)
    return t


def r_from_t(t, n):
    # Сorrelation r from t-statistic
    # https://juls-dotcom.github.io/meta_analysis.html

    df = n - 2
    return t / np.sqrt(t * t + df)


def d_from_t(t, n):
    # Cohen’s d from t-statistic
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation

    df = n - 2
    return 2 * t / np.sqrt(df)


def t_from_d(d, n):
    # Cohen’s d from t-statistic
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation

    df = n - 2
    return d * np.sqrt(df) / 2


def z_from_r(r):
    # Fisher's z from correlation r
    # https://www.escal.site/
    # https://en.wikipedia.org/wiki/Fisher_transformation
    # np.log((1 + r)/(1 - r)) / 2

    return np.atanh(r)


def r_from_z(z):
    # Сorrelation r (Cohen’s r) from Fisher's z
    # https://www.escal.site/
    # https://www.statisticshowto.com/fisher-z/
    # https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions

    r = np.sqrt(1 - np.square(1 / np.cosh(z)))
    r = np.copysign(r, z)
    return r


def d_from_f(f, df_num, df_denom, n_groups=2):
    # Cohen’s d from F-statistic
    # Cohen’s d derived from F-statistic (Fritz et al., 2012)
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full

    eta_sq = (f * df_num) / (f * df_num + df_denom)
    f_sq = eta_sq / (1 - eta_sq)
    d = np.sqrt(f_sq * 2 * n_groups)
    return d


def r_from_f(f, df_num, df_denom):
    # Сorrelation r from F-statistic
    # f_stat = (r_sq / df_model) / ((1 - r_sq) / df_resid))
    # https://statproofbook.github.io/P/fstat-rsq.html

    r_sq = 1 - (1 / (1 + f * df_num / df_denom))
    return np.sqrt(r_sq)


###########
# p-value #
###########


def p_from_f(f, df_num, df_denom):
    # p value for F-statistic

    return scipy.stats.f.sf(f, df_num, df_denom)


def z_from_p(p, method="two-tailed"):
    # Fisher's z from p-value
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)
    # https://www.gigacalculator.com/calculators/p-value-to-z-score-calculator.php
    # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa

    p = p / 2 if method == "two-tailed" else p
    z = scipy.stats.norm.ppf(p)
    return np.abs(z)


def p_from_z(z, method="two-tailed"):
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)
    # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa

    p = scipy.stats.norm.sf(np.abs(z))
    p = p * 2 if method == "two-tailed" else p
    return p


def p_from_t(t, n, method="two-tailed"):
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)
    # t-critical = stats.t.ppf(alpha/numOfTails, ddof)
    # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa

    df = n - 2
    p = scipy.stats.t.sf(np.abs(t), df)
    p = p * 2 if method == "two-tailed" else p
    return p


def t_from_p(p, n, method="two-tailed"):
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)
    # t-critical = stats.t.ppf(alpha/numOfTails, ddof)
    # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa

    df = n - 2
    p = p / 2 if method == "two-tailed" else p
    t = scipy.stats.t.ppf(p, df)
    return np.abs(t)


def r_from_p(p, n, method="two-tailed"):
    # Cohen’s r from p-value
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation

    r = r_from_t(t_from_p(p, n, method=method), n)
    return r  # z_from_p(p, method=method) / math.sqrt(n)


def p_from_r(r, n, method="two-tailed"):
    p = p_from_t(t_from_r(r, n), n, method=method)
    return p  # z_from_p(p, method=method) / math.sqrt(n)


def d_from_p(p, n, method="two-tailed"):
    return d_from_r(r_from_p(p, n, method=method))


def p_from_d(d, n, method="two-tailed"):
    return p_from_r(r_from_d(d), n, method=method)


##############
# conflicted #
##############


def z_from_t(t, n, method=None):
    # t-statistic from Fisher's z
    # https://web.cortland.edu/andersmd/STATS/stdscore.html#:~:text=As%20evidenced%20above%2C%20zscores%20are,a%20T%20score%20of%2025.

    if method:
        p = p_from_t(t, n, method=method)
        z = z_from_p(p, method=method)
        return z
    return z_from_r(r_from_t(t, n))


def t_from_z(z, n, method=None):
    # t-statistic from Fisher's z
    # https://web.cortland.edu/andersmd/STATS/stdscore.html#:~:text=As%20evidenced%20above%2C%20zscores%20are,a%20T%20score%20of%2025.

    if method:
        p = p_from_z(z, method=method)
        t = t_from_p(p, n, method=method)
        return t
    return t_from_r(r_from_z(z), n)


def d_from_z(z):
    # https://easystats.github.io/effectsize/reference/t_to_r.html
    # assumed equal groups

    return d_from_r(r_from_z(z))


def z_from_d(d):
    # https://easystats.github.io/effectsize/reference/t_to_r.html
    # assumed equal groups

    return z_from_r(r_from_d(d))


########################
# Confidence intervals #
########################


def p_from_ci(estimate, cil, cir, n=None, alpha=0.95, method="two-tailed"):
    # https://genometoolbox.blogspot.com/2013/11/how-to-estimate-p-value-from-confidence.html
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)

    se = (cil - cir) / 2
    if n is None:
        se /= z_from_p(1 - alpha, method=method)  # z-critical is used
        z = estimate / se
        p = p_from_z(z, method=method)
    else:
        se /= t_from_p(1 - alpha, n, method=method)  # t-critical is used
        t = estimate / se
        p = p_from_t(t, n, method=method)
    return p


def ci_from_r(r, n, alpha=0.95, method="two-tailed"):
    """
    Bootstraping confidence intervals for the correlation coefficient
    using Fisher transformation
    https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    https://en.wikipedia.org/wiki/Fisher_transformation
    """

    se = z_from_p(1 - alpha, method=method) / np.sqrt(n - 3)  # z-critical is used
    return np.tanh(np.atanh(r) - se), np.tanh(np.atanh(r) + se)


def ci_from_p(p, n, alpha=0.95, method="two-tailed"):
    r = r_from_p(p, n, method=method)
    return ci_from_r(r, n, alpha=alpha, method=method)


def bootstrap(*args, func=np.mean, alpha=0.95, n_rep=1000, seed=13):
    """
    Bootstraping confidence intervals for the mean/median value
    https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    """

    np.random.seed(seed)
    data = np.asanyarray(args)
    idx = np.arange(0, data.shape[1], 1)
    sample = [
        func(*np.take(data, np.random.choice(idx, size=data.shape[1], replace=True), 1))
        for _ in range(n_rep)
    ]
    lp, m, rp = (1 - alpha) / 2, 0.5, 1 - (1 - alpha) / 2
    return np.percentile(sample, [lp * 100, m * 100, rp * 100])


##############
# Cohen r, d #
##############


def cohen_r_from_p(p, n, method="two-tailed"):
    return r_from_p(p, n, method=method)


def cohen_r_from_d(d, n_groups=2):
    return r_from_d(d, n_groups=n_groups)


def cohen_r_from_z(z):
    return r_from_z(z)


def cohen_r_from_t(t, n):
    return r_from_t(t, n)


def cohen_r_from_f(f, df_num, df_denom):
    # Сorrelation r from F-statistic
    return r_from_f(f, df_num, df_denom)


def cohen_d_from_p(p, n, method="two-tailed"):
    return d_from_p(p, n, method=method)


def cohen_d_from_r(r):
    return d_from_r(r)


def cohen_d_from_z(z):
    return d_from_z(z)


def cohen_d_from_t(t, n):
    return d_from_t(t, n)


def cohen_d_from_f(f, df_num, df_denom, n_groups=2):
    # Cohen's d from F-statistic
    return d_from_f(f, df_num, df_denom, n_groups=n_groups)


###########
#  other  #
###########


def t_from_b(b, se):
    # t-statistic from beta-coefficient
    return b / se


def unbiased_z(z):
    # unbiased Z
    # https://juls-dotcom.github.io/meta_analysis.html
    # sample sizes < 20 or 10 in each group, see Nakagawa & Cuthill, 2007
    # https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-185X.2007.00027.x
    # (zru) value using the equation proposed by Hedges & Olkin, 1985
    # https://www.sciencedirect.com/book/9780080570655/statistical-methods-for-meta-analysis
    return z


def logit_from_probit(x):
    # https://statmodeling.stat.columbia.edu/2006/06/06/take_logit_coef/
    # https://www.statalist.org/forums/forum/general-stata-discussion/general/1704819-calculating-odds-ration-in-probit-model
    return -1.7 * x


def OR_from_logit(x):
    # odds ratio from log odds ratio
    # odds = EXP(0.873) = 2.394;
    # https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
    return np.exp(x)


def logit_from_cohen_d(d):
    # log odds ratio from Cohen's d
    # https://www.escal.site/
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return np.pi * d / np.sqrt(3)


def cohen_d_from_logit(x):
    # Cohen's d from log odds ratio
    # https://stats.stackexchange.com/questions/68290/converting-odds-ratios-to-cohens-d-for-meta-analysis
    # https://www.um.es/metaanalysis/pdf/7078.pdf
    return x * np.sqrt(3) / np.pi


def logit_from_OR(x):
    return np.log(x)
