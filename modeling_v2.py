"""Modeling v2 (upgrade item 3): Optuna tuning logged to MLflow, probability calibration, grouped SHAP,
and the serving helpers shared by the notebook and the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from housing_utils import FORECAST_YEARS, NUM_FEATURES, CAT_FEATURES, engineer_features


# ------------------------------------------------------------------------------------------ pipeline
def make_pipe(model, num, cat):
    prep = ColumnTransformer([("num", StandardScaler(), list(num)),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), list(cat))])
    return Pipeline([("prep", prep), ("model", model)])


# ------------------------------------------------------------------------------------------ Optuna + MLflow
def _suggest(trial, task):
    """Deliberately regularised ranges: labels here are noisy and shift over time, so shallow trees, small learning
    rates and a large min_child_weight beat memorisation."""
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                max_depth=trial.suggest_int("max_depth", 2, 7),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 5, 200, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 3.0))


def _estimator(task, params, seed):
    base = dict(tree_method="hist", n_jobs=-1, random_state=seed)
    if task == "clf":
        return XGBClassifier(**base, eval_metric="logloss", **params)
    return XGBRegressor(**base, **params)


def tune_xgb(task, X, y, folds, num, cat, n_trials=25, seed=42, mlflow_module=None, run_name=None):
    """Bayesian search (TPE) over XGBoost hyper-parameters.
    Objective: mean validation log-loss (clf, calibration-friendly) or RMSE (reg) across the supplied folds.
    `folds` = list of (train_idx, val_idx) from purged_expanding_cv (real data) or StratifiedKFold (legacy mode).
    Every trial is a nested MLflow run; the study summary is logged on the parent run."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ml = mlflow_module

    def objective(trial):
        params = _suggest(trial, task)
        scores = []
        for tr, va in folds:
            pipe = make_pipe(_estimator(task, params, seed), num, cat).fit(X.loc[tr], y.loc[tr])
            if task == "clf":
                scores.append(log_loss(y.loc[va], pipe.predict_proba(X.loc[va])[:, 1]))
            else:
                scores.append(float(np.sqrt(mean_squared_error(y.loc[va], pipe.predict(X.loc[va])))))
        mean, sd = float(np.mean(scores)), float(np.std(scores))
        if ml is not None:
            with ml.start_run(run_name=f"{task}_trial_{trial.number:03d}", nested=True):
                ml.log_params(params)
                ml.log_metrics({"cv_mean": mean, "cv_std": sd, **{f"fold{i}": s for i, s in enumerate(scores)}})
                ml.set_tags({"task": task, "stage": "optuna_trial"})
        trial.set_user_attr("cv_std", sd)
        return mean

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name=f"{task}_xgb")
    if ml is not None:
        with ml.start_run(run_name=run_name or f"optuna_{task}") as parent:
            ml.set_tags({"task": task, "stage": "optuna_study"})
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            ml.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            ml.log_metrics({"best_cv_objective": study.best_value, "n_trials": n_trials})
            try:
                import optuna.visualization as ov
                ml.log_figure(ov.plot_optimization_history(study), f"{task}_optuna_history.html")
                if n_trials >= 10:
                    ml.log_figure(ov.plot_param_importances(study), f"{task}_optuna_param_importance.html")
            except Exception as e:                                     # plots are a bonus, never fatal
                print("optuna plots skipped:", e)
            parent_id = parent.info.run_id
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        parent_id = None
    return study, parent_id


def fit_best(task, study, X, y, num, cat, seed=42):
    return make_pipe(_estimator(task, study.best_params, seed), num, cat).fit(X, y)


# ------------------------------------------------------------------------------------------ calibration
def ece_score(y, p, bins=10):
    """Expected calibration error (equal-width bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CalibratedPipe:
    """Fitted pipeline + monotone probability map (Platt/sigmoid or isotonic). predict_proba mimics sklearn."""

    def __init__(self, pipe, calibrator, method):
        self.pipe, self.calibrator, self.method_ = pipe, calibrator, method

    def _map(self, p):
        if self.method_ == "isotonic":
            return self.calibrator.predict(p)
        return self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def predict_proba(self, X):
        p = self._map(self.pipe.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def calibrate_oof(make_model, X, y, folds, final_pipe, num, cat, method="auto"):
    """Calibrate WITHOUT touching the test set and without a separate calibration block.

    Out-of-fold probabilities are collected from the tuning folds (each fold model only saw the past when `folds`
    come from purged_expanding_cv); a monotone map  raw p -> observed rate  is then fitted on them and attached to
    the final model. Isotonic needs plenty of data (sklearn guidance: > ~1000 rows) -> 'auto' picks sigmoid below 5000.
    """
    oof_p, oof_y = [], []
    for tr, va in folds:
        pipe = make_pipe(make_model(), num, cat).fit(X.loc[tr], y.loc[tr])
        oof_p.append(pipe.predict_proba(X.loc[va])[:, 1])
        oof_y.append(y.loc[va].to_numpy())
    p, t = np.concatenate(oof_p), np.concatenate(oof_y)
    if method == "auto":
        method = "isotonic" if len(t) >= 5000 else "sigmoid"
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, t)
    else:
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), t)
    return CalibratedPipe(final_pipe, cal, method), len(t)


def calibration_table(y, probas: dict) -> pd.DataFrame:
    rows = {k: {"Log-loss": log_loss(y, p), "Brier": brier_score_loss(y, p), "ECE": ece_score(y, p),
                "ROC_AUC": roc_auc_score(y, p)} for k, p in probas.items()}
    return pd.DataFrame(rows).T


def reliability_curve(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = [(p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum())) for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(out, columns=["mean_predicted", "observed_rate", "n"])


# ------------------------------------------------------------------------------------------ SHAP (grouped back to raw features)
def _root_name(feature_out, cat):
    """'num__City_Tier' -> 'City_Tier'  but  'cat__City_Mumbai' -> 'City'.
    (Matching on the name alone would wrongly fold the numeric City_Tier into the categorical City.)"""
    kind, n = feature_out.split("__", 1)
    if kind == "num":
        return n
    return max((c for c in cat if n.startswith(c + "_")), key=len, default=n)


def shap_grouped(pipe, X, cat, sample=None, seed=42, explainer=None):
    """TreeSHAP for the XGBoost step of `pipe`. One-hot columns are summed back to the ORIGINAL feature, so
    'City' is one bar, not 42. Units: log-odds (classifier) or INR Lakh (regressor).
    Returns (values DataFrame [n x raw features], base_value, display DataFrame of raw feature values)."""
    import shap
    if sample is not None and len(X) > sample:
        X = X.sample(sample, random_state=seed)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = prep.transform(X)
    names = prep.get_feature_names_out()
    expl = explainer or shap.TreeExplainer(model)
    sv = expl.shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    roots = [_root_name(n, cat) for n in names]
    G = pd.DataFrame(sv, columns=names, index=X.index).T.groupby(roots).sum().T
    base = float(np.ravel(expl.expected_value)[0])
    return G, base, X[G.columns]


def shap_global_figure(G, top=15, title="Global SHAP importance (mean |SHAP|)"):
    import plotly.graph_objects as go
    s = G.abs().mean().sort_values().tail(top)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#4F46E5"))
    fig.update_layout(title=title, height=460, margin=dict(l=40, r=20, t=60, b=40), template="plotly_white")
    return fig


def shap_beeswarm_figure(G, Xdisp, num, top=12, max_points=1500, seed=42, title="SHAP beeswarm"):
    """Plotly beeswarm for numeric features (colour = feature value rank)."""
    import plotly.graph_objects as go
    order = G.abs().mean().sort_values().tail(top).index
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    sel = rng.choice(len(G), min(max_points, len(G)), replace=False)
    for i, f in enumerate(order):
        v = G[f].to_numpy()[sel]
        if f in num:
            col = pd.Series(Xdisp[f].to_numpy()[sel]).rank(pct=True).to_numpy()
            cs, hover = "RdBu_r", np.round(Xdisp[f].to_numpy()[sel], 2)
        else:
            col, cs, hover = np.full(len(v), 0.5), "Greys", Xdisp[f].astype(str).to_numpy()[sel]
        fig.add_trace(go.Scattergl(x=v, y=i + rng.uniform(-0.3, 0.3, len(v)), mode="markers", showlegend=False,
                                   marker=dict(size=4, opacity=.6, color=col, colorscale=cs, cmin=0, cmax=1),
                                   text=hover, hovertemplate=f"{f}=%{{text}}<br>SHAP %{{x:.3f}}<extra></extra>"))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=list(order))
    fig.update_layout(title=title, height=520, template="plotly_white", margin=dict(l=40, r=20, t=60, b=40),
                      xaxis_title="SHAP value (impact on model output)")
    return fig


def shap_waterfall_figure(g_row: pd.Series, base: float, disp_row: pd.Series, top=8, unit="log-odds", title=""):
    """Waterfall: model base value -> top contributions -> 'all other features' -> final model output."""
    import plotly.graph_objects as go
    order = g_row.abs().sort_values(ascending=False)
    head, rest = order.index[:top], order.index[top:]
    names = [f"{f} = {disp_row[f]:.4g}" if isinstance(disp_row[f], (int, float, np.floating, np.integer)) else f"{f} = {disp_row[f]}"
             for f in head]
    vals = [float(g_row[f]) for f in head]
    if len(rest):
        names.append(f"{len(rest)} other features"); vals.append(float(g_row[rest].sum()))
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        y=["average listing"] + names + ["this listing"], x=[base] + vals + [base + sum(vals)],
        connector={"line": {"color": "#94A3B8"}},
        increasing={"marker": {"color": "#10B981"}}, decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#4F46E5"}}))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, height=420, template="plotly_white", margin=dict(l=40, r=20, t=60, b=40),
                      xaxis_title=f"model output ({unit})")
    return fig


def local_explanation(g_row: pd.Series, disp_row: pd.Series, top=8) -> pd.DataFrame:
    d = pd.DataFrame({"feature": g_row.index, "value": [disp_row[f] for f in g_row.index], "shap": g_row.values})
    return d.reindex(d["shap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


# ------------------------------------------------------------------------------------------ serving
def serving_row(listing: dict, meta: dict) -> pd.DataFrame:
    """RAW listing dict -> model-ready 1-row frame, identical to training (incl. external features)."""
    row = pd.DataFrame([listing])
    if meta.get("age_base") == "listing_year":
        year = int(pd.Timestamp.today().year)
        row["Listing_Year"] = year
        row["Age_of_Property"] = max(year - int(listing["Year_Built"]), 0)
    f = engineer_features(row, meta["city_median_pps"])
    if "Crime_Rate_per_100k" in meta["num_features"]:
        f["Crime_Rate_per_100k"] = meta["city_crime"].get(listing["City"], meta["crime_fallback"])
    return f


def predict_v2(listing: dict, clf_cal, reg, meta: dict) -> dict:
    f = serving_row(listing, meta)
    X = f[meta["num_features"] + meta["cat_features"]]
    p = float(clf_cal.predict_proba(X)[0, 1])
    raw = float(reg.predict(X)[0])
    price = float(f["Price_in_Lakhs"].iloc[0])
    future = price * float(np.exp(raw)) if meta.get("reg_target") == "log_multiple" else raw   # real mode predicts log growth multiple
    return {"good_prob": p, "is_good": p >= meta.get("decision_threshold", 0.5), "future_price": future, "price": price,
            "implied_cagr": (future / price) ** (1 / meta.get("horizon_years", FORECAST_YEARS)) - 1, "features": f.iloc[0], "X": X}
