"""Train 6 classifiers + 6 regressors (+ ensembles) and export ONE serving bundle.
Usage:  python train_models.py --csv india_housing_prices.csv   [--fast for a 60k-row smoke run]"""
import argparse, os, time, warnings
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error)
from housing_utils import *

warnings.filterwarnings("ignore")
SEED = 42
ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="india_housing_prices.csv"); ap.add_argument("--fast", action="store_true")
a = ap.parse_args()

# ---------- data + targets (identical logic to the notebook / housing_utils) ----------
raw = pd.read_csv(a.csv)
df = raw.drop_duplicates(subset=[c for c in raw.columns if c != "ID"]).reset_index(drop=True)
df["PPS_INR"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
city_pps = df.groupby("City")["PPS_INR"].median().round(2).to_dict()
df = add_targets(engineer_features(df.drop(columns=["Price_per_SqFt"], errors="ignore"), city_pps), seed=SEED)
X, yc, yr = df[NUM_FEATURES + CAT_FEATURES], df["Good_Investment"], df["Future_Price_5Y"]
Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te = train_test_split(X, yc, yr, test_size=0.2, random_state=SEED, stratify=yc)
if a.fast: Xtr, yc_tr, yr_tr = Xtr.iloc[:60000], yc_tr.iloc[:60000], yr_tr.iloc[:60000]
print(f"train {len(Xtr):,} · test {len(Xte):,}")

def pipe(m):
    return Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), CAT_FEATURES)])), ("model", m)])
tt = lambda m: TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
CLF = {"Linear (Logistic/Ridge)": LogisticRegression(max_iter=500),
       "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20, max_samples=0.3, n_jobs=-1, random_state=SEED),
       "Extra Trees": ExtraTreesClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20, max_samples=0.3, bootstrap=True, n_jobs=-1, random_state=SEED),
       "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED),
       "XGBoost": XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
       "Neural Net (MLP)": MLPClassifier((64, 32), early_stopping=True, max_iter=60, random_state=SEED)}
REG = {"Linear (Logistic/Ridge)": Ridge(alpha=1.0),
       "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=10, max_samples=0.3, n_jobs=-1, random_state=SEED),
       "Extra Trees": ExtraTreesRegressor(n_estimators=60, max_depth=14, min_samples_leaf=10, max_samples=0.3, bootstrap=True, n_jobs=-1, random_state=SEED),
       "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=SEED),
       "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
       "Neural Net (MLP)": tt(MLPRegressor((64, 32), early_stopping=True, max_iter=60, random_state=SEED))}

def grouped_imp(p):
    m = p.named_steps["model"]
    if not hasattr(m, "feature_importances_"): return None
    s = pd.Series(m.feature_importances_, index=p.named_steps["prep"].get_feature_names_out())
    root = lambda n: next((c for c in CAT_FEATURES if n.split("__", 1)[1].startswith(c + "_")), n.split("__", 1)[1])
    s = s.groupby(root).sum(); return (s / s.sum()).sort_values().tail(12).round(4).to_dict()

# ---------- classifiers ----------
cp, cm, roc, cmat, fit_s, imp_c, probs = {}, {}, {}, {}, {}, {}, {}
for n, m in CLF.items():
    t = time.time(); cp[n] = pipe(m).fit(Xtr, yc_tr); fit_s[n] = time.time() - t; probs[n] = cp[n].predict_proba(Xte)[:, 1]
    imp_c[n] = grouped_imp(cp[n]); print(f"clf {n:<24} {fit_s[n]:5.1f}s")
probs["🏆 Ensemble (mean)"] = np.mean([probs[n] for n in CLF], axis=0)
for n, p in probs.items():
    pred = (p >= .5).astype(int)
    cm[n] = dict(Accuracy=accuracy_score(yc_te, pred), Precision=precision_score(yc_te, pred), Recall=recall_score(yc_te, pred),
                 F1=f1_score(yc_te, pred), ROC_AUC=roc_auc_score(yc_te, p))
    f, t_, _ = roc_curve(yc_te, p); i = np.linspace(0, len(f) - 1, min(120, len(f))).astype(int)
    roc[n] = dict(fpr=f[i].tolist(), tpr=t_[i].tolist()); cmat[n] = confusion_matrix(yc_te, pred).tolist()

# ---------- regressors ----------
rp, rm, imp_r, preds = {}, {}, {}, {}
for n, m in REG.items():
    t = time.time(); rp[n] = pipe(m).fit(Xtr, yr_tr); fit_s[n + " (reg)"] = time.time() - t; preds[n] = rp[n].predict(Xte)
    imp_r[n] = grouped_imp(rp[n]); print(f"reg {n:<24} {fit_s[n + ' (reg)']:5.1f}s")
preds["🏆 Ensemble (mean)"] = np.mean([preds[n] for n in REG], axis=0)
for n, p in preds.items():
    rm[n] = dict(RMSE=float(np.sqrt(mean_squared_error(yr_te, p))), MAE=mean_absolute_error(yr_te, p), R2=r2_score(yr_te, p),
                 MAPE=mean_absolute_percentage_error(yr_te, p))
idx = np.random.default_rng(SEED).choice(len(yr_te), min(1500, len(yr_te)), replace=False)
scatter = {"actual": yr_te.to_numpy()[idx].tolist(), **{n: p[idx].tolist() for n, p in preds.items()}}

# ---------- bundle ----------
meta = dict(city_median_pps=city_pps, state_cities=df.groupby("State")["City"].unique().apply(sorted).to_dict(),
            options={c: sorted(df[c].unique().tolist()) for c in ["Property_Type", "Furnished_Status", "Public_Transport_Accessibility", "Facing", "Owner_Type", "Availability_Status"]},
            amenities=["Gym", "Pool", "Clubhouse", "Garden", "Playground"],
            ranges=dict(size=[int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max())], price=[float(df.Price_in_Lakhs.min()), float(df.Price_in_Lakhs.max())]),
            clf_metrics=cm, reg_metrics=rm, roc=roc, cmat=cmat, scatter=scatter, imp_clf=imp_c, imp_reg=imp_r, fit_s=fit_s,
            n_rows=len(df), n_train=len(Xtr), n_test=len(Xte), threshold=GROWTH_THRESHOLD, years=FORECAST_YEARS,
            note="Targets are engineered from documented growth rules (the CSV has no price history), so scores measure rule recovery, not real-market skill.")
os.makedirs("models", exist_ok=True); os.makedirs("data", exist_ok=True)
joblib.dump(dict(clf=cp, reg=rp, meta=meta), "models/bundle.joblib", compress=3)
keep = [c for c in ["ID", "State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "PPS_INR", "Age_of_Property", "Furnished_Status",
        "Public_Transport_Accessibility", "Infrastructure_Score", "Availability_Status", "Owner_Type", "Simulated_Annual_Growth", "Future_Price_5Y", "Good_Investment"] if c in df]
df[keep].sample(min(30000, len(df)), random_state=SEED).to_csv("data/market_sample.csv", index=False)
print("✅ models/bundle.joblib · data/market_sample.csv  →  streamlit run app.py")
