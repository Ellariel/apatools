import io
import contextlib
import warnings
import scipy
import numpy as np
import spreg as spr
from libpysal.weights import WSP
from pandas import to_numeric
from itertools import zip_longest
from statsmodels.api import add_constant
# from sklearn.model_selection import LeaveOneOut
from statsmodels.formula.formulatools import handle_formula_data


from .citation import Citation
from .format import format_r, format_p, get_stars
from .utils import df_standardize, df_check_intercept


# https://pysal.org/spreg/index.html
CITATION = Citation(
    APA="Rey, S. J., & Anselin, L. (2007). \
PySAL: A python library of spatial analytical methods. \
The Review of Regional Studies, 37(1), 5-27."
)



def fit_model(model, Y, X, w, **kwargs):  # model.fit() generator function
    if isinstance(model, (list, tuple)):
        for m in model:
            yield next(fit_model(m, Y, X, w, **kwargs))
    elif isinstance(model, str):
        if model == "ols":
            yield next(fit_model(spr.OLS, Y, X, w, **kwargs))
        elif model == "slm":
            yield next(fit_model(spr.ML_Lag, Y, X, w, **kwargs))
        elif model == "sem":
            yield next(fit_model(spr.ML_Error, Y, X, w, **kwargs))
        else:
            raise NotImplementedError(
                f"'{model}' is not implemented, try 'ols', 'slm' or 'sem'."
            )
    else:
        verbose = kwargs.get("verbose", False)
        # `model` is a class here (not an instance), so compare with `is`
        if model is spr.OLS:
            if verbose:
                print("model: OLS")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("ols_")}
        elif model is spr.ML_Lag:
            if verbose:
                print("model: SLM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("slm_")}
        elif model is spr.ML_Error:
            if verbose:
                print("model: SEM")
            _kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("sem_")}
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:]: v for k, v in _kwargs.items() if k.startswith("model_")}

        if verbose:
            print("model_kwargs:", model_kwargs)

        yield model(Y, X, w, **model_kwargs)


def base_metrics(results):
    """
    Collect fit metrics (R²/pseudo R², adjusted R², F-test, AIC/BIC/LL, MAE, MAD) from a
    spreg results object.
    """

    outputs = {}
    resid = results.y.flatten() - results.predy.flatten()
    # spreg's `k` counts the constant and the predictors; it also counts rho for
    # ML_Lag but does NOT count lambda for ML_Error (added below).
    n_params = int(results.k)
    n_obs = int(results.n)
    
    if isinstance(results, spr.ML_Error):
        n_params += 1 # lambda counts
    df_model = n_params - 1  # predictors (+ rho / lambda), without the constant
    df_resid = n_obs - n_params  # n - k, as in spreg's OLS F-test
    
    if isinstance(results, (spr.ML_Lag, spr.ML_Error)):
        r_sq = results.pr2
        r_sq_adj = 1 - (1 - r_sq) * (n_obs - 1) / df_resid
        f_stat = (r_sq / df_model) / ((1 - r_sq) / df_resid)
        f_pvalue = scipy.stats.f.sf(f_stat, df_model, df_resid)
    else:
        r_sq = results.r2
        r_sq_adj = results.ar2
        f_stat, f_pvalue = results.f_stat
        
    outputs.update(
        {
            "r_sq": r_sq,
            "r_sq_adj": r_sq_adj,
            "df_model": df_model,
            "df_resid": df_resid,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "n_obs": n_obs,
            "n_params": n_params, 
            "mae": np.mean(np.abs(resid)),
            "mad": np.median(np.abs(resid - np.median(resid))),
            "aic": results.aic,
            "bic": results.schwarz,
            "llf": results.logll,
        }
    )
    return outputs


def w_subset_rows(w, positions):
    """
    Spatial weights restricted to the observations at ``positions``.

    Unlike ``libpysal.weights.w_subset`` the original weight values are kept (not
    replaced by binary ones), the transformation of ``w`` (e.g. row-standardization,
    ``w.transform``) is re-applied to the subset and the ids of the returned
    object are the new row positions ``0..len(positions)-1``.
    Observations that lose all their neighbours become islands (see ``.islands``).

    Parameters
    ----------
    w : libpysal.weights.W
        Weights whose ``id_order`` corresponds to the rows of the data.
    positions : array-like of int
        Row positions to keep (in the order of the data).
    """
    positions = np.asarray(positions, dtype=int)
    sub = WSP(w.sparse[positions][:, positions]).to_W(silence_warnings=True)
    sub.transform = w.transform
    return sub


def pred_metrics(results, model, Y, X, w, exclude_neighbors=False, **kwargs):
    """
    Leave-one-out prediction metrics (R²pred, MAEpred, MADpred) for spreg models.

    Every observation ``i`` is held out in turn, the model is refitted on the
    remaining observations (with the spatial weights restricted to them, see
    :func:`w_subset_rows`) and ``y_i`` is predicted. The prediction depends on the model:

    * ``ols`` and ``sem`` (spatial error): ``x_i·β`` (the trend; the error process
      is not used for prediction, which is also what spreg's ``predy`` is);
    * ``slm`` (spatial lag): ``ρ·(Wy)_i + x_i·β``, i.e. the observed ``y`` values
      of the neighbours of ``i`` are used through the *original* weights row of ``i``.
      This predictor uses more information than the one for ols/sem, so the
      R²pred values of the models are not strictly comparable.

    .. warning::
       Spatial data are autocorrelated. In plain leave-one-out the neighbours of the
       held-out observation stay in the training sample, and their (correlated)
       values leak into the fit, so R²pred is *optimistic* with respect to predicting
       a new, unseen region. Use ``exclude_neighbors=True`` for a more honest
       estimate: the neighbours of ``i`` (according to ``w``) are removed from
       the training sample as well (``i`` is still the only observation that is
       predicted). Because the refit has fewer rows this variant is slightly
       pessimistic for small samples and may create islands in the training weights.

    Cost: one model fit per observation and model (``n`` fits). For ML lag/error
    models each fit computes the log-determinant of ``W``; pass e.g.
    ``slm_model_method="LU"`` / ``sem_model_method="LU"`` or ``"ord"`` for large ``n``.

    Parameters
    ----------
    results : list
        Fitted spreg models returned by the full-sample fit, one per entry of ``model``
        (used for the design matrix, so that a constant added by spreg is handled).
    model : str or list of str
        The models that were fitted (``"ols"``, ``"slm"``, ``"sem"``).
    Y, X : array-like
        Data used for the full-sample fit; ``w`` must be aligned with the rows.
    w : libpysal.weights.W or None
        Spatial weights. ``None`` is only possible for plain ``ols``.
    exclude_neighbors : bool
        Also remove the neighbours of the held-out observation from the training sample.
    **kwargs
        The same model arguments as passed to :func:`slm`. Diagnostics of OLS are
        switched off inside the loop for speed.

    Returns
    -------
    list of dict
        One dict per model with ``pred_loo_r_sq`` (clipped to [-1, 1], as in
        :func:`apatools.lm.pred_metrics`), ``pred_loo_mae`` and ``pred_loo_mad``
        (median absolute deviation of the prediction errors). ``nan`` if every
        refit failed. Failed refits are skipped with a warning.
    """
    y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    x = np.asarray(X, dtype=float)
    n = y.shape[0]
    if w is None and exclude_neighbors:
        raise ValueError("exclude_neighbors=True requires spatial weights `w`.")
    if w is not None and w.n != n:
        raise ValueError(f"Weights have {w.n} observations but the data {n}.")

    _kwargs = kwargs.copy()
    _kwargs.update(verbose=False, ols_model_spat_diag=False, ols_model_nonspat_diag=False)
    designs = [r.x for r in results]  # n x k design (incl. constant added by spreg)
    wy = None if w is None else w.sparse @ y  # (Wy)_i from the original weights

    errors = {}
    fit_fails = []
    n_islands = 0
    all_pos = np.arange(n)
    for i in range(n):
        drop = {i}
        if exclude_neighbors:
            drop |= {int(w.id2i[j]) for j in w.neighbors[w.id_order[i]]}
        train = np.array([k for k in all_pos if k not in drop])
        w_train = None if w is None else w_subset_rows(w, train)
        if w_train is not None and len(w_train.islands):
            n_islands += 1
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")  # spreg is chatty on every refit
                fitted = list(fit_model(model, y[train], x[train], w_train, **_kwargs))
        except Exception as e:  # e.g. singular design, no convergence
            fit_fails.append(str(e))
            continue
        for idx, r in enumerate(fitted):
            d = designs[idx]
            beta = np.asarray(r.betas, dtype=float).ravel()[: d.shape[1]]
            pred = float(d[i] @ beta)
            if isinstance(r, spr.ML_Lag):
                pred += float(r.rho) * float(wy[i, 0])
            errors.setdefault(idx, []).append((i, y[i, 0] - pred))

    if fit_fails:
        warnings.warn(
            f"Some attempts to calculate LeaveOneOut metrics failed ({len(fit_fails)}): {set(fit_fails)}",
            UserWarning,
        )
    if n_islands:
        warnings.warn(
            f"In {n_islands} leave-one-out refits the training weights contained islands "
            "(observations without neighbours).",
            UserWarning,
        )

    ybar = y.mean()
    outputs = []
    for idx in range(len(results)):
        if idx not in errors:
            outputs.append({k: np.nan for k in ("pred_loo_r_sq", "pred_loo_mae", "pred_loo_mad")})
            continue
        pos, err = zip(*errors[idx])
        err = np.asarray(err)
        sst = np.sum((y[list(pos), 0] - ybar) ** 2)
        outputs.append(
            {
                "pred_loo_r_sq": float(np.clip(1 - np.sum(err**2) / sst, -1.0, 1.0)),
                "pred_loo_mae": float(np.mean(np.abs(err))),
                "pred_loo_mad": float(np.median(np.abs(err - np.median(err)))),
            }
        )
    return outputs


def slm(data, y=None, x=None, w=None, model="ols", formula=None, **kwargs):
    """
    Fit OLS, spatial lag (SLM, ``ML_Lag``) and spatial error (SEM, ``ML_Error``) models with spreg.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Dependent variable; ignored if ``formula`` is given.
    x : list of str
        Predictors; ignored if ``formula`` is given.
    w : libpysal.weights.W
        Spatial weights (required for SLM/SEM and for OLS spatial diagnostics).
    model : str or list of str
        ``"ols"``, ``"slm"``, ``"sem"`` or a list of them.
    formula : str, optional
        patsy formula; overrides ``y``/``x``.
    **kwargs
        Function options (below) and spreg arguments named ``<model>_model_<argument>``
        with ``<model>`` in ols/slm/sem, e.g. ``ols_model_spat_diag=True``.

    Returns
    -------
    results : list of spreg model objects
    metrics : list of dict
        r_sq (pseudo R² for SLM/SEM), r_sq_adj, f_stat, f_pvalue, df_model, df_resid
        (n - k), aic, bic, llf, mae, mad (median absolute deviation of residuals), n_obs.
        With ``pred_metrics=True`` also pred_loo_r_sq, pred_loo_mae, pred_loo_mad
        (see :func:`pred_metrics` for how the predictions are made and for the
        caveat on spatial autocorrelation).

    Function options (``**kwargs``)
    -------------------------------
    verbose : bool (default True)
        Print the sample size, the formula and the model summaries.
    dropna : bool (default True)
        Drop rows with NAs (not with ``formula``: patsy always drops them).
        ``w`` is restricted to the remaining rows automatically.
    intercept : bool (default False)
        Ignored by spreg, which always adds a constant.
    standardize : False, True/'z' or callable
        z-transform (or apply a callable to) the numeric columns.
    base_metrics : bool (default True)
        Compute :func:`base_metrics` for every model.
    pred_metrics : bool (default False)
        Add leave-one-out prediction metrics, which needs one refit per observation.
        Leave-one-out is optimistic for spatially autocorrelated data, see
        :func:`pred_metrics`.
    pred_exclude_neighbors : bool (default False)
        With ``pred_metrics``: also remove the neighbours of the held-out observation
        from the training sample (less optimistic; needs ``w``).
    There is no VIF for spreg models.

    Example
    -------
    slm(test_data, Y, X, w, model=['ols', 'slm'],
                    verbose=True,
                    dropna=True,
                    intercept=False, # ignored, spreg adds it by default
                    standardize='z' # keeps np.number or bool columns only
                    base_metrics=True,
                    pred_metrics=True, # leave-one-out R²pred, MAEpred, MADpred
                    pred_exclude_neighbors=False,

                    ols_model_spat_diag=True,
                    ols_model_nonspat_diag=True,
    """

    verbose = kwargs.get("verbose", True)
    dropna = kwargs.get("dropna", True)
    constant = kwargs.pop("intercept", False)
    standardize = kwargs.pop("standardize", False)
    add_base_metrics = kwargs.pop("base_metrics", True)
    add_pred_metrics = kwargs.pop("pred_metrics", False)
    pred_exclude_neighbors = kwargs.pop("pred_exclude_neighbors", False)

    if formula is None:
        if x is None or y is None or len(x) == 0:
            raise ValueError(
                "Either formula or x,y have to be explicitly defined."
            )

        df = data[[y] + x]
        if dropna:
            keep = df.notna().all(axis=1).to_numpy()
            df = df[keep]
            if w is not None and not keep.all():
                # keep the weights aligned with the rows that survived NA-dropping
                w = w_subset_rows(w, np.flatnonzero(keep))
        if len(df) != len(data):
            warnings.warn(
                f"Rows with NAs were dropped. Ntotal={len(data)}",
                UserWarning,
            )

        if standardize:
            df = df_standardize(df, func=standardize)

        X, Y = df[x], df[y]

        if verbose:
            print(f"N={len(Y)}")
            print(f"Formula: {y} ~ {'1 + ' if constant else ''}" + " + ".join(x))

        if constant:
            X = add_constant(X) # ignoredin spreg
    else:
        (Y, X), _, _ = handle_formula_data(data, X=None, formula=formula)
        if w is not None and len(Y) != len(data):
            # patsy dropped rows with NAs: restrict the weights to the remaining rows
            if not data.index.is_unique:
                raise ValueError("A unique data index is required to align the weights after NA-dropping.")
            w = w_subset_rows(w, data.index.get_indexer(Y.index))
            warnings.warn(
                f"Rows with NAs were dropped. Ntotal={len(data)}",
                UserWarning,
            )
        if verbose:
            print(f"N={len(Y)}")
            print(f"Specified formula: {formula}")
           
        if standardize: # need a test
            X = df_standardize(X, func=standardize)
            Y = df_standardize(Y, func=standardize)

    if standardize == 'z' or (isinstance(standardize, bool) and standardize):
        warnings.warn(
            "Using z-transformation sets the intercept estimate to zero. This may lead to wrong results.",
            UserWarning,
        )
    
    results = []
    for r in fit_model(model, Y, X, w, **kwargs):
        results.append(r)
        if verbose:
            print(r.summary)
    
    metrics = []
    if add_base_metrics or add_pred_metrics:
        metrics = [base_metrics(r) for r in results]
        if add_pred_metrics:
            metrics = [
                {**m, **p}
                for m, p in zip(
                    metrics,
                    pred_metrics(
                        results, model, Y, X, w,
                        exclude_neighbors=pred_exclude_neighbors, **kwargs
                    ),
                )
            ]

    return results, metrics


def slm_report(results, metrics={}, format_pval=True, add_stars=True, decimal=3, 
              add_ftest=True, add_aic=True, add_bic=True, add_llf=True, add_mae=True, add_mad=True,
              add_pred_loo_mae=True, add_pred_loo_mad=True, # shown only if pred_metrics=True was used; no vif for spreg
              add_n_obs=True, add_n_groups=False, add_vif=False,
              intercept_name='Intercept'):
    """
    Build one APA-style coefficient table per model from :func:`slm` output.

    Columns: coef, se, z/t, p-value, cil, cir (95% CI from the normal approximation),
    optional ``sig``. The first row's ``model`` column holds the fit summary, e.g.
    ``R² = .35, R²adj = .31, R²pred = .22, F(3, 45) = 8.06, p < .001, AIC = ..., N = 49``;
    R²pred, MAEpred and MADpred appear only if :func:`slm` was run with
    ``pred_metrics=True`` (leave-one-out, see :func:`pred_metrics`).
    """

    output = []
    for r, i in zip_longest(results, metrics, fillvalue={}):
        s = []
        if "r_sq" in i:
            s.append(f"R² {format_r(i['r_sq'], use_letter=False)}")
        if "r_sq_adj" in i:
            s.append(f"R²adj {format_r(i['r_sq_adj'], use_letter=False)}")
        if "pred_loo_r_sq" in i:
            s.append(f"R²pred {format_r(i['pred_loo_r_sq'], use_letter=False)}")
        if add_ftest and "f_stat" in i and "df_model" in i:
            s.append(
                f"F({i['df_model']}, {i['df_resid']}) = {i['f_stat']:.2f}, {format_p(i['f_pvalue'])}"
            )
        if add_aic and "aic" in i:
            s.append(f"AIC = {i['aic']:.1f}")
        if add_bic and "bic" in i:
            s.append(f"BIC = {i['bic']:.1f}")
        if add_llf and "llf" in i:
            s.append(f"LL = {i['llf']:.1f}")
        if add_mae and "mae" in i:
            s.append(f"MAE = {i['mae']:.3f}")
        if add_mad and "mad" in i:
            s.append(f"MAD = {i['mad']:.3f}")
        if add_pred_loo_mae and "pred_loo_mae" in i:
            s.append(f"MAEpred = {i['pred_loo_mae']:.3f}")
        if add_pred_loo_mad and "pred_loo_mad" in i:
            s.append(f"MADpred = {i['pred_loo_mad']:.3f}")
        if add_n_obs and "n_obs" in i:
            s.append(f"N = {i['n_obs']}")
        if add_n_groups and "n_groups" in i:
            s.append(f"G = {i['n_groups']}")

        s = ", ".join(s)
        params = r.output.set_index('var_names')
        params = params.apply(to_numeric, errors="coerce").rename(
            columns={
                "coefficients": "coef",
                "prob": "p-value",
                "std_err": "se",
                "zt_stat": "z",
            },
        )
        alpha_z = scipy.stats.norm.ppf(0.975) # 95% CI
        params["cil"] = params["coef"] - alpha_z * params["se"]
        params["cir"] = params["coef"] + alpha_z * params["se"]
        if decimal:
            for c in ["coef", "se", "cil", "cir"]:
                params[c] = params[c].round(decimal)
        if add_stars:
            add_stars = add_stars if callable(add_stars) else get_stars
            params["sig"] = [add_stars(c) for c in params["p-value"]]
        if format_pval:
            format_pval = (
                format_pval
                if callable(format_pval)
                else lambda x: format_p(
                    x, use_letter=False, keep_spaces=False, no_equals=True
                )
            )
            params["p-value"] = [format_pval(c) for c in params["p-value"]]
        if add_vif and "vif" in i and i["vif"] is not None:
            params = params.join(i["vif"])
        if len(i):
            params.loc[params.index[0], "model"] = s
        const_name = df_check_intercept(params, params.index)  
        if const_name != intercept_name:
            params.index = params.index.str.replace(const_name, intercept_name)
        output.append(params)

    return output