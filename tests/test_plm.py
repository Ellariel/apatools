import importlib.util
import warnings

import libpysal
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import spreg

import apatools.plm as plm_module
from apatools.plm import (
    SPREG_MODELS,
    PanelData,
    base_metrics,
    hausman,
    make_panel,
    mundlak,
    plm,
    plm_report,
    plm_tests,
    pred_metrics,
)

HAS_LM = importlib.util.find_spec("linearmodels") is not None
needs_lm = pytest.mark.skipif(not HAS_LM, reason="linearmodels is not installed")
BACKENDS = ["spreg", pytest.param("linearmodels", marks=needs_lm)]

SIDE, T = 6, 6
N = SIDE * SIDE


def simulate(rho=0.4, correlated=False, seed=7):
    """Long panel (shuffled rows) with entity effects and, optionally, a spatial lag."""
    rng = np.random.default_rng(seed)
    w = libpysal.weights.lat2W(SIDE, SIDE, rook=True)
    w.transform = "r"
    W = w.sparse.toarray()
    a = rng.normal(size=N)  # entity component of x1: gives x1 between-entity variation
    x1, x2 = a + rng.normal(size=(T, N)), rng.normal(size=(T, N))
    mu = rng.normal(size=N) + (2.0 * a if correlated else 0.0)  # effects correlated with x1 or not
    y = np.vstack(
        [np.linalg.solve(np.eye(N) - rho * W, x1[t] - 0.5 * x2[t] + mu + rng.normal(size=N)) for t in range(T)]
    )
    df = pd.DataFrame(
        {
            "y": y.ravel(),
            "x1": x1.ravel(),
            "x2": x2.ravel(),
            "county": np.tile(np.arange(N), T),
            "year": np.repeat(2000 + np.arange(T), N),
        }
    )
    return df.sample(frac=1, random_state=1).reset_index(drop=True), w


@pytest.fixture(scope="module")
def sim():
    return simulate()


@pytest.fixture(scope="module")
def panel(sim):
    df, w = sim
    return make_panel(df, "y", ["x1", "x2"], entity="county", time="year", w=w)


def fit(df, w=None, model="fe", quiet=True, **kwargs):
    """plm() on the simulated data; ``quiet=False`` lets warnings through (for pytest.warns)."""
    kwargs.setdefault("verbose", False)
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        return plm(df, "y", ["x1", "x2"], w=w, entity="county", time="year", model=model, **kwargs)


def coef(r):
    return plm_module._param_table(r)["coef"]


def within_ols(panel, idx=None):
    """Reference within estimator in plain numpy."""
    y, x = panel.y.to_numpy(), panel.x.to_numpy()
    ent = panel.ent
    cnt = np.bincount(ent)
    ym = np.bincount(ent, weights=y) / cnt
    xm = np.column_stack([np.bincount(ent, weights=x[:, j]) / cnt for j in range(x.shape[1])])
    beta = np.linalg.lstsq(x - xm[ent], y - ym[ent], rcond=None)[0]
    return beta, ym - xm @ beta


################
#  panel data  #
################


def test_make_panel_layout_is_independent_of_row_order(sim):
    df, w = sim
    shuffled = make_panel(df, "y", ["x1", "x2"], entity="county", time="year", w=w)
    ordered = make_panel(df.sort_values(["year", "county"]), "y", ["x1", "x2"], entity="county", time="year", w=w)
    pd.testing.assert_series_equal(shuffled.y, ordered.y)
    pd.testing.assert_frame_equal(shuffled.x, ordered.x)
    assert isinstance(shuffled, PanelData) and (shuffled.N, shuffled.T) == (N, T)
    # period blocks, entities in the order of w
    assert np.array_equal(shuffled.ent, np.tile(np.arange(N), T))
    assert np.array_equal(shuffled.tim, np.repeat(np.arange(T), N))
    assert shuffled.entities == list(w.id_order)
    # values end up at the right place
    row = df.iloc[0]
    k = int(np.flatnonzero((shuffled.y.index.get_level_values(0) == row.county)
                           & (shuffled.y.index.get_level_values(1) == row.year))[0])
    assert shuffled.y.iloc[k] == row.y and shuffled.x["x1"].iloc[k] == row.x1


def test_make_panel_accepts_multiindex_and_formula(sim):
    df, _ = sim
    a = make_panel(df, "y", ["x1", "x2"], entity="county", time="year")
    b = make_panel(df.set_index(["county", "year"]), "y", ["x1", "x2"])
    c = make_panel(df, formula="y ~ x1 + x2", entity="county", time="year")
    for other in (b, c):
        np.testing.assert_allclose(a.y.to_numpy(), other.y.to_numpy())
        np.testing.assert_allclose(a.x.to_numpy(), other.x.to_numpy())
    with pytest.raises(ValueError, match="both"):
        make_panel(df, "y", ["x1"], entity="county")
    with pytest.raises(ValueError, match="index"):
        make_panel(df, "y", ["x1"])
    with pytest.raises(ValueError, match="no predictors"):
        make_panel(df, formula="y ~ 1", entity="county", time="year")


def test_make_panel_weights_alignment(sim):
    df, w = sim
    # entity labels that are not the ids of w
    labelled = df.assign(county="c" + df["county"].astype(str))
    with pytest.raises(ValueError, match="w_ids"):
        make_panel(labelled, "y", ["x1"], entity="county", time="year", w=w)
    p = make_panel(labelled, "y", ["x1"], entity="county", time="year", w=w,
                   w_ids=[f"c{i}" for i in w.id_order])
    assert p.entities == [f"c{i}" for i in range(N)]
    with pytest.raises(ValueError, match="labels"):
        make_panel(labelled, "y", ["x1"], entity="county", time="year", w=w, w_ids=["a"])


def test_make_panel_balance_and_duplicates(sim):
    df, w = sim
    unbalanced = df.drop(index=df.index[(df.county == 5) & (df.year == 2003)])
    with pytest.raises(ValueError, match="not balanced"):
        make_panel(unbalanced, "y", ["x1"], entity="county", time="year", w=w)
    with pytest.warns(UserWarning) as rec:
        p = make_panel(unbalanced, "y", ["x1"], entity="county", time="year", w=w, balance="drop")
    assert any("Dropped 1 entities" in str(r.message) for r in rec)
    assert any("restricted" in str(r.message) for r in rec)
    assert p.N == N - 1 and p.w.n == N - 1 and 5 not in p.entities
    assert p.w.id_order == list(range(N - 1))
    # NA rows are removed like missing rows
    na = df.copy()
    na.loc[0, "x1"] = np.nan
    assert make_panel(na, "y", ["x1"], entity="county", time="year", balance="drop").N == N - 1
    with pytest.raises(ValueError, match="not balanced"):
        make_panel(na, "y", ["x1"], entity="county", time="year")
    with pytest.raises(ValueError, match="more than once"):
        make_panel(pd.concat([df, df.iloc[:1]]), "y", ["x1"], entity="county", time="year")
    with pytest.raises(ValueError, match="balance"):
        make_panel(df, "y", ["x1"], entity="county", time="year", balance="x")


def test_panel_subset_keeps_layout_and_weights(panel):
    keep = ~np.isin(panel.ent, [0, 7])
    sub = panel.subset(keep)
    assert (sub.N, sub.T, sub.w.n) == (N - 2, T, N - 2)
    assert np.array_equal(sub.ent, np.tile(np.arange(N - 2), T))
    sub_t = panel.subset(panel.tim != 2)
    assert (sub_t.N, sub_t.T) == (N, T - 1) and sub_t.w is panel.w
    with pytest.raises(ValueError, match="whole entities"):
        panel.subset(np.arange(len(panel.y)) != 3)


#################
#  estimation   #
#################


@needs_lm
def test_fe_and_pooled_match_manual_estimators(sim, panel):
    df, w = sim
    beta, _ = within_ols(panel)
    X1 = np.column_stack([np.ones(len(panel.y)), panel.x.to_numpy()])
    pooled = np.linalg.lstsq(X1, panel.y.to_numpy(), rcond=None)[0]
    for backend in ("spreg", "linearmodels"):
        (r_fe, r_p), (m_fe, m_p) = fit(df, w, ["fe", "pooled"], backend=backend)
        np.testing.assert_allclose(coef(r_fe)[["x1", "x2"]].to_numpy(), beta, rtol=1e-6)
        c = coef(r_p)
        np.testing.assert_allclose([c.iloc[0], c["x1"], c["x2"]], pooled, rtol=1e-6)
        assert m_fe["backend"] == backend and m_fe["model"] == "fe" and m_p["model"] == "pooled"


@needs_lm
def test_backends_report_the_same_metrics(sim):
    df, w = sim
    (_, _), (a_fe, a_p) = fit(df, w, ["fe", "pooled"], backend="spreg")
    (_, _), (b_fe, b_p) = fit(df, w, ["fe", "pooled"], backend="linearmodels")
    for a, b in ((a_fe, b_fe), (a_p, b_p)):
        for key in ("r_sq", "r_sq_adj", "f_stat", "f_pvalue", "mae", "mad"):
            assert a[key] == pytest.approx(b[key], rel=1e-6), key
        for key in ("df_model", "df_resid", "n_obs", "n_entities", "n_periods", "n_params", "r_sq_type"):
            assert a[key] == b[key], key
    assert (a_fe["n_obs"], a_fe["n_entities"], a_fe["n_periods"]) == (N * T, N, T)
    # FE: the N absorbed effects are subtracted from the residual degrees of freedom
    assert a_fe["df_resid"] == N * T - N - 2 and a_fe["df_model"] == 2
    assert a_fe["r_sq_type"] == "within" and a_p["r_sq_type"] == "ols"
    # reference within R2
    p = make_panel(df, "y", ["x1", "x2"], entity="county", time="year")
    beta, _ = within_ols(p)
    y_dm = p.y.to_numpy() - (np.bincount(p.ent, weights=p.y.to_numpy()) / T)[p.ent]
    x_dm = p.x.to_numpy() - np.column_stack(
        [(np.bincount(p.ent, weights=p.x.to_numpy()[:, j]) / T)[p.ent] for j in range(2)])
    resid = y_dm - x_dm @ beta
    assert a_fe["r_sq"] == pytest.approx(1 - resid @ resid / (y_dm @ y_dm))
    assert a_fe["mae"] == pytest.approx(np.mean(np.abs(resid)))


def test_nonspatial_models_work_without_weights(sim):
    df, w = sim
    with_w, _ = fit(df, w, ["pooled", "fe", "re"])
    without_w, m = fit(df, None, ["pooled", "fe", "re"])
    for a, b in zip(with_w, without_w):
        np.testing.assert_allclose(coef(a).to_numpy(), coef(b).to_numpy(), rtol=1e-6)
    assert len(m) == 3


def test_spatial_models(sim):
    df, w = sim
    results, metrics = fit(df, w, ["slm_fe", "slm_re", "sem_fe", "sem_pooled", "sem_pooled_gm", "sem_re_gm"])
    names = [m["model"] for m in metrics]
    assert names == ["slm_fe", "slm_re", "sem_fe", "sem_pooled", "sem_pooled_gm", "sem_re_gm"]
    # spatial lag: rho is recovered (true value 0.4) and is not counted as a slope
    for r, m in zip(results[:2], metrics[:2]):
        assert abs(coef(r)["rho"] - 0.4) < 0.15
        assert abs(coef(r)["x1"] - 1.0) < 0.15 and abs(coef(r)["x2"] + 0.5) < 0.15
        assert m["n_params"] == 2 and m["r_sq_type"] == "pseudo"
    for r, m in zip(results[2:], metrics[2:]):
        assert "lambda" in coef(r).index and m["n_params"] == 2
        assert m["n_obs"] == N * T
    # the full table is returned: constants, rho / lambda, variance components
    assert "phi" in coef(results[1]).index
    assert {"sigma2_v", "sigma2_1"} <= set(coef(results[5]).index)


def test_slx_lags_add_lagged_predictors(sim):
    df, w = sim
    (r,), (m,) = fit(df, w, "slm_fe", slx_lags=1)
    assert {"x1", "x2", "W_x1", "W_x2", "rho"} <= set(coef(r).index)
    assert m["n_params"] == 4
    with pytest.raises(NotImplementedError):
        fit(df, w, "fe", backend="linearmodels", slx_lags=1)


@pytest.mark.xfail(raises=TypeError, strict=False,
                   reason="spreg 1.9.0 ML_ErrorRE fails with complex numbers; use sem_re_gm")
def test_sem_re_ml_upstream_bug(sim):
    df, w = sim
    fit(df, w, "sem_re")


@needs_lm
def test_linearmodels_options(sim):
    df, w = sim
    (a,), _ = fit(df, w, "fe", backend="linearmodels")
    (b,), (mb,) = fit(df, w, "fe", backend="linearmodels",
                      fe_fit_cov_type="clustered", fe_fit_cluster_entity=True)
    (c,), _ = fit(df, w, "fe", backend="linearmodels", fe_model_time_effects=True)
    np.testing.assert_allclose(coef(a).to_numpy(), coef(b).to_numpy())
    assert not np.allclose(a.std_errors.to_numpy(), b.std_errors.to_numpy())
    assert b.std_errors.tolist() == plm_report([b], [mb], decimal=None, add_stars=False)[0]["se"].tolist()
    assert mb["f_stat"] == pytest.approx(float(b.f_statistic_robust.stat))  # robust F for robust SEs
    assert not np.allclose(coef(a).to_numpy(), coef(c).to_numpy())  # two-way effects change the estimates


def test_errors(sim):
    df, w = sim
    with pytest.raises(NotImplementedError, match="spatial models need backend='spreg'"):
        fit(df, w, "slm_fe", backend="linearmodels")
    with pytest.raises(NotImplementedError, match="not implemented"):
        fit(df, w, "xyz")
    with pytest.raises(ValueError, match="needs spatial weights"):
        fit(df, None, "slm_fe")
    with pytest.raises(ValueError, match="backend"):
        fit(df, w, "fe", backend="xyz")
    with pytest.raises(ValueError, match="explicitly"):
        plm(df, entity="county", time="year", verbose=False)


def test_verbose_output(sim, capsys):
    df, w = sim
    plm(df, "y", ["x1", "x2"], w=w, entity="county", time="year", model="fe", verbose=True)
    out = capsys.readouterr().out
    assert f"N={N} entities, T={T} periods, N*T={N * T}" in out
    assert "model: fe (spreg)" in out
    plm(df, "y", ["x1", "x2"], w=w, entity="county", time="year", model="fe", verbose=False)
    assert capsys.readouterr().out == ""


################
#   reporting  #
################


@needs_lm
def test_report(sim):
    df, w = sim
    models = ["pooled", "fe", "re", "slm_fe"]
    reports = {}
    for backend in ("spreg", "linearmodels"):
        names = models if backend == "spreg" else models[:3]
        r, m = fit(df, w, names, backend=backend)
        reports[backend] = (plm_report(r, m), m)
    for backend, (tables, metrics) in reports.items():
        for t, m in zip(tables, metrics):
            assert {"coef", "se", "p-value", "cil", "cir", "sig", "model"} <= set(t.columns)
            assert isinstance(t.iloc[0]["model"], str) and t["model"].iloc[1:].isna().all()
            assert f"N = {N * T}, n = {N}, T = {T}" in t.iloc[0]["model"]
            assert t.loc["x1", "p-value"] == "<.001" and t.loc["x1", "sig"] == "***"
            assert (t["cil"] <= t["coef"]).all() and (t["coef"] <= t["cir"]).all()
            if m["model"] == "fe":
                assert "Intercept" not in t.index  # no constant in fixed-effects models
                assert t.iloc[0]["model"].startswith("R²within = .")
            elif m["model"] == "pooled":
                assert "Intercept" in t.index and "const" not in t.index and "CONSTANT" not in t.index
                assert t.iloc[0]["model"].startswith("R² = .")
    assert reports["linearmodels"][0][2].iloc[0]["model"].startswith("R²overall")
    slm_table = reports["spreg"][0][3]
    assert "rho" in slm_table.index and slm_table.iloc[0]["model"].startswith("R²pseudo")
    assert "F(" not in slm_table.iloc[0]["model"]
    # the FE summary is identical for both backends up to AIC/BIC/LL, which linearmodels lacks
    a, b = (reports[k][0][1].iloc[0]["model"] for k in ("spreg", "linearmodels"))
    assert a.split(", AIC")[0] == b.split(", LL")[0]
    # options
    r, m = fit(df, w, "fe")
    t = plm_report(r, m, format_pval=False, add_stars=False, decimal=2, add_aic=False, add_bic=False,
                   add_llf=False, add_mae=False, add_mad=False, add_panel_dims=False)[0]
    assert "sig" not in t.columns and isinstance(t.loc["x1", "p-value"], float)
    summary = t.iloc[0]["model"]
    assert summary.startswith("R²within = .") and ", R²adj = ." in summary and "F(2, 178) = " in summary
    assert summary.endswith(f"N = {N * T}")
    assert not any(s in summary for s in ("AIC", "BIC", "LL", "MAE", "MAD", ", n = "))
    # results without metrics can still be reported
    assert len(plm_report(r)) == 1


################
#    tests     #
################


@pytest.mark.parametrize("backend", BACKENDS)
def test_correlated_effects_are_detected(backend):
    corr, w = simulate(correlated=True)
    uncorr, _ = simulate(correlated=False, seed=11)
    (fe, re), _ = fit(corr, w, ["fe", "re"], backend=backend)
    # Mundlak: effects are correlated with x1, so random effects are rejected
    stat, df, p = mundlak(re)
    assert df == 2 and stat > 0 and p < 0.001
    # classic Hausman: either rejects too, or is undefined (V_fe - V_re not positive definite),
    # which must be reported with a warning and not turned into a made-up number
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        h, k, hp = hausman(fe, re)
    assert k == 2
    assert (np.isnan(h) and np.isnan(hp) and any("positive definite" in str(x.message) for x in rec)) or hp < 0.05
    # uncorrelated effects: neither test rejects
    (fe, re), _ = fit(uncorr, w, ["fe", "re"], backend=backend)
    assert mundlak(re)[2] > 0.05
    assert hausman(fe, re)[2] > 0.05


@needs_lm
def test_mundlak_matches_manual_refit(sim, panel):
    df, w = sim
    (re,), _ = fit(df, w, "re", backend="linearmodels")
    from linearmodels.panel import RandomEffects

    means = panel.x.groupby(panel.ent).transform("mean").add_suffix("_mean")
    X = pd.concat([panel.x, means], axis=1).assign(const=1.0)
    res = RandomEffects(panel.y, X).fit()
    b = res.params[["x1_mean", "x2_mean"]].to_numpy()
    v = res.cov.loc[["x1_mean", "x2_mean"], ["x1_mean", "x2_mean"]].to_numpy()
    stat = float(b @ np.linalg.solve(v, b))
    assert mundlak(re) == pytest.approx((stat, 2, scipy.stats.chi2.sf(stat, 2)))


def test_mundlak_needs_plm_results_and_time_varying_x(sim):
    df, w = sim
    (re,), _ = fit(df, w, "re")
    del re.plm_spec
    with pytest.raises(ValueError, match="come from plm"):
        mundlak(re)
    # predictors that do not vary over time cannot be tested: x2 constant within entity
    d = df.assign(x2=df["county"] / N)
    (re2,), _ = fit(d, w, "re")
    stat, k, p = mundlak(re2)
    assert k == 1 and np.isfinite(stat)  # only x1 is tested
    d = d.assign(x1=(df["county"] % 5) / 5)  # now no predictor varies over time
    (re3,), _ = fit(d, w, "re")
    with pytest.raises(ValueError, match="varies over time"):
        mundlak(re3)


def test_hausman_and_mundlak_for_spatial_models(sim):
    df, w = sim
    results, _ = fit(df, w, ["slm_fe", "slm_re", "sem_re_gm"])
    h, k, p = hausman(results[0], results[1])
    assert k == 2 and (np.isnan(h) or (h >= 0 and 0 <= p <= 1))
    for r in results[1:]:
        stat, k, p = mundlak(r)
        assert k == 2 and stat >= 0 and 0 <= p <= 1


@needs_lm
def test_plm_tests_table(sim):
    df, w = sim
    r_s, _ = fit(df, w, ["pooled", "fe", "re"])
    r_l, _ = fit(df, w, ["pooled", "fe", "re"], backend="linearmodels")
    ts, tl = plm_tests(r_s), plm_tests(r_l)
    assert list(ts.columns) == ["test", "model", "statistic", "df", "df_denom", "p-value", "report"]
    # F test of entity effects: spreg (computed here) equals linearmodels (f_pooled)
    fs = ts[ts.test == "F test of entity effects"].iloc[0]
    fl = tl[tl.test == "F test of entity effects"].iloc[0]
    assert fs.statistic == pytest.approx(fl.statistic, rel=1e-6)
    assert (fs.df, fs.df_denom) == (N - 1, N * T - N - 2) == (fl.df, fl.df_denom)
    assert fs.report.startswith(f"F({N - 1}, {N * T - N - 2}) = ")
    assert (ts.test == "Hausman (FE vs RE)").sum() == 1 and (tl.test == "Hausman (FE vs RE)").sum() == 1
    for t_, r_ in ((ts, r_s), (tl, r_l)):
        row = t_[t_.test == "Mundlak (correlated effects)"]
        assert len(row) == 1 and row.iloc[0].model == "re" and row.iloc[0].df == 2
        assert row.iloc[0].statistic == pytest.approx(mundlak(r_[2])[0])
    # LM tests need spreg results and weights
    assert "LM lag" in set(ts.test) and "LM lag" not in set(tl.test)
    lm_rows = ts[ts.test.str.startswith(("LM", "Robust"))]
    assert len(lm_rows) == 7 and ((lm_rows["p-value"] >= 0) & (lm_rows["p-value"] <= 1)).all()
    # strong spatial dependence in the data (rho = .4) and entity effects are detected
    assert ts.set_index("test").loc["LM lag", "p-value"] < 0.05
    assert ts.set_index("test").loc["LM Marginal RE (LM1)", "p-value"] < 0.001
    # no weights, no LM tests; weights can also be passed explicitly
    r_nw, _ = fit(df, None, ["pooled", "fe"])
    assert not plm_tests(r_nw).test.str.startswith("LM").any()
    for r in r_nw:
        del r.plm_panel
    assert plm_tests(r_nw, w=w).test.str.startswith("LM").any()
    # nothing to compare: empty table with the same columns
    assert plm_tests(r_s[:1][:0]).empty


################
# pred metrics #
################


def manual_loo(panel, group, model, exclude_neighbors=False):
    """Plain numpy leave-one-group-out predictions for pooled and FE models."""
    y, x = panel.y.to_numpy(), panel.x.to_numpy()
    err = []
    groups = panel.N if group == "entity" else panel.T
    for g in range(groups):
        if group == "entity":
            drop = {g} | ({int(panel.w.id2i[j]) for j in panel.w.neighbors[g]} if exclude_neighbors else set())
            train, test = ~np.isin(panel.ent, list(drop)), panel.ent == g
        else:
            train, test = panel.tim != g, panel.tim == g
        if model == "pooled":
            X = np.column_stack([np.ones(train.sum()), x[train]])
            b = np.linalg.lstsq(X, y[train], rcond=None)[0]
            pred = np.column_stack([np.ones(test.sum()), x[test]]) @ b
        else:  # fe, leave one period out
            ent = panel.ent[train]
            cnt = np.bincount(ent)
            ym = np.bincount(ent, weights=y[train]) / cnt
            xm = np.column_stack([np.bincount(ent, weights=x[train][:, j]) / cnt for j in range(2)])
            b = np.linalg.lstsq(x[train] - xm[ent], y[train] - ym[ent], rcond=None)[0]
            pred = x[test] @ b + (ym - xm @ b)[panel.ent[test]]
        err.append((np.flatnonzero(test), y[test] - pred))
    rows = np.concatenate([e[0] for e in err])
    e = np.concatenate([e[1] for e in err])
    sst = np.sum((y[rows] - y.mean()) ** 2)
    return {"pred_loo_r_sq": 1 - np.sum(e**2) / sst, "pred_loo_mae": np.mean(np.abs(e)),
            "pred_loo_mad": np.median(np.abs(e - np.median(e)))}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("group,model", [("time", "pooled"), ("time", "fe"), ("entity", "pooled")])
def test_pred_metrics_match_manual(sim, panel, backend, group, model):
    df, w = sim
    _, (m,) = fit(df, w, model, backend=backend, pred_metrics=True, pred_group=group)
    for k, v in manual_loo(panel, group, model).items():
        assert m[k] == pytest.approx(v, rel=1e-6), k
    assert m["pred_loo_r_sq"] < m["r_sq"] or model == "fe"  # out of sample is worse than in sample


def test_pred_metrics_exclude_neighbors_creates_islands():
    # star graph: removing the hub (a neighbour of every leaf) isolates the other leaves
    rng = np.random.default_rng(0)
    n, t = 6, 5
    star = libpysal.weights.W({0: [1, 2, 3, 4, 5], **{i: [0] for i in range(1, n)}})
    star.transform = "r"
    d = pd.DataFrame({"county": np.tile(np.arange(n), t), "year": np.repeat(np.arange(t), n),
                      "x1": rng.normal(size=n * t), "x2": rng.normal(size=n * t)})
    d["y"] = d.x1 - d.x2 + rng.normal(size=n * t)
    with pytest.warns(UserWarning, match="islands"):
        plm(d, "y", ["x1", "x2"], w=star, entity="county", time="year", model="pooled", verbose=False,
            pred_metrics=True, pred_exclude_neighbors=True)


def test_pred_metrics_exclude_neighbors(sim, panel):
    df, w = sim
    _, (m,) = fit(df, w, "pooled", pred_metrics=True, pred_exclude_neighbors=True)
    for k, v in manual_loo(panel, "entity", "pooled", exclude_neighbors=True).items():
        assert m[k] == pytest.approx(v, rel=1e-6), k
    _, (plain,) = fit(df, w, "pooled", pred_metrics=True)
    assert plain["pred_loo_mae"] != m["pred_loo_mae"]
    with pytest.raises(ValueError, match="group='entity'"):
        fit(df, w, "pooled", pred_metrics=True, pred_group="time", pred_exclude_neighbors=True)
    with pytest.raises(ValueError, match="group='entity'"):
        fit(df, None, "pooled", pred_metrics=True, pred_exclude_neighbors=True)


def test_pred_metrics_fe_cannot_predict_new_entities(sim):
    df, w = sim
    with pytest.warns(UserWarning, match="cannot predict new entities"):
        _, ms = fit(df, w, ["fe", "pooled", "slm_fe"], quiet=False, pred_metrics=True, pred_group="entity")
    assert np.isnan(ms[0]["pred_loo_r_sq"]) and np.isnan(ms[2]["pred_loo_mae"])
    assert np.isfinite(ms[1]["pred_loo_r_sq"])


def test_pred_metrics_spatial_lag_matches_direct_spreg(sim, panel):
    """Leave one period out for slm_fe, with the refits done by calling spreg directly."""
    df, w = sim
    _, (m,) = fit(df, w, "slm_fe", pred_metrics=True, pred_group="time")
    y, x = panel.y.to_numpy(), panel.x.to_numpy()
    wy = np.asarray(w.sparse @ y.reshape(T, N).T).T.ravel()
    err, rows = [], []
    for g in range(T):
        train, test = panel.tim != g, panel.tim == g
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = spreg.ML_LagFE(y[train].reshape(-1, 1), x[train], w)
        b, rho = np.asarray(r.betas).ravel()[:2], float(np.asarray(r.betas).ravel()[2])
        resid = y - x @ b - rho * wy
        mu = np.bincount(panel.ent[train], weights=resid[train], minlength=N) / np.bincount(panel.ent[train], minlength=N)
        err.append(y[test] - (x[test] @ b + rho * wy[test] + mu[panel.ent[test]]))
        rows.append(np.flatnonzero(test))
    e, rows = np.concatenate(err), np.concatenate(rows)
    assert m["pred_loo_r_sq"] == pytest.approx(1 - np.sum(e**2) / np.sum((y[rows] - y.mean()) ** 2), rel=1e-6)
    assert m["pred_loo_mae"] == pytest.approx(np.mean(np.abs(e)), rel=1e-6)
    # the spatial lag helps: better than the non-spatial FE model
    _, (fe,) = fit(df, w, "fe", pred_metrics=True, pred_group="time")
    assert m["pred_loo_mae"] < fe["pred_loo_mae"]


def test_pred_metrics_report_and_failures(sim, monkeypatch, panel):
    df, w = sim
    r, m = fit(df, w, ["pooled", "fe"], pred_metrics=True, pred_group="time")
    for t in plm_report(r, m):
        assert all(s in t.iloc[0]["model"] for s in ("R²pred = ", "MAEpred = ", "MADpred = "))
    r, m = fit(df, w, ["pooled"])
    assert "R²pred" not in plm_report(r, m)[0].iloc[0]["model"]
    with pytest.raises(ValueError, match="group must be"):
        pred_metrics(r, panel, group="x")
    with pytest.raises(NotImplementedError, match="slx_lags"):
        pred_metrics(r, panel, slx_lags=1)

    def broken(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(plm_module, "_fit_one", broken)
    with pytest.warns(UserWarning, match="no convergence"):
        out = pred_metrics(r, panel, group="time")
    assert all(np.isnan(v) for v in out[0].values())


def test_model_registry_is_consistent():
    assert set(SPREG_MODELS) >= {"pooled", "fe", "re", "slm_fe", "slm_re", "sem_pooled", "sem_fe", "sem_re"}
    for cls_name, effects, spatial in SPREG_MODELS.values():
        assert hasattr(spreg, cls_name) and effects in ("pooled", "fe", "re") and spatial in (None, "lag", "error")
