"""Input-feature drift monitoring (upgrade item 4).

Compares live/recent model inputs with the TRAINING reference sample using
  * PSI  (Population Stability Index)  - the industry-standard bucketed shift score
  * KS test (numeric) / chi-square test (categorical) - significance of the shift
PSI rule of thumb:  < 0.10 stable   0.10-0.25 moderate shift (watch)   >= 0.25 major shift (investigate / retrain)

CLI:  python monitoring.py --reference models_v2/reference_sample.csv.gz --current logs/inference_log.csv
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
PSI_WARN, PSI_ALERT = 0.10, 0.25
MIN_CURRENT_ROWS = 200


def psi_numeric(ref, cur, bins=10):
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    if ref.nunique() <= bins:                            # flags / small integers (Metro_Within_1km, City_Tier, BHK):
        return psi_categorical(ref, cur)                 # quantile bins degenerate, so compare value frequencies
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur):
    ref, cur = pd.Series(ref).astype(str), pd.Series(cur).astype(str)
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=EPS)
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def _status(psi):
    return "DRIFT" if psi >= PSI_ALERT else "WARN" if psi >= PSI_WARN else "OK"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, num, cat, pred_ref=None, pred_cur=None) -> pd.DataFrame:
    rows = []
    for f in num:
        if f not in current or current[f].dropna().empty:
            continue
        p = psi_numeric(reference[f], current[f])
        ks = stats.ks_2samp(reference[f].dropna(), current[f].dropna())
        rows.append({"feature": f, "type": "numeric", "psi": p, "test": "KS", "statistic": ks.statistic,
                     "p_value": ks.pvalue, "ref_mean": reference[f].mean(), "cur_mean": current[f].mean(), "status": _status(p)})
    for f in cat:
        if f not in current or current[f].dropna().empty:
            continue
        p = psi_categorical(reference[f], current[f])
        cats = sorted(set(reference[f].astype(str)) | set(current[f].astype(str)))
        ct = np.vstack([reference[f].astype(str).value_counts().reindex(cats).fillna(0),
                        current[f].astype(str).value_counts().reindex(cats).fillna(0)])
        chi = stats.chi2_contingency(ct[:, ct.sum(0) > 0])
        rows.append({"feature": f, "type": "categorical", "psi": p, "test": "chi2", "statistic": chi[0],
                     "p_value": chi[1], "ref_mean": np.nan, "cur_mean": np.nan, "status": _status(p)})
    if pred_ref is not None and pred_cur is not None:
        p = psi_numeric(pred_ref, pred_cur)
        ks = stats.ks_2samp(pred_ref, pred_cur)
        rows.append({"feature": "» predicted probability", "type": "prediction", "psi": p, "test": "KS",
                     "statistic": ks.statistic, "p_value": ks.pvalue, "ref_mean": float(np.mean(pred_ref)),
                     "cur_mean": float(np.mean(pred_cur)), "status": _status(p)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def summarise(report: pd.DataFrame, n_current: int) -> dict:
    return {"n_current": int(n_current), "reliable": n_current >= MIN_CURRENT_ROWS,
            "drift": int((report.status == "DRIFT").sum()), "warn": int((report.status == "WARN").sum()),
            "ok": int((report.status == "OK").sum()),
            "overall": "DRIFT" if (report.status == "DRIFT").any() else "WARN" if (report.status == "WARN").any() else "OK"}


def plot_drift(report: pd.DataFrame):
    import plotly.graph_objects as go
    r = report.sort_values("psi")
    colors = r.status.map({"OK": "#10B981", "WARN": "#F59E0B", "DRIFT": "#EF4444"})
    fig = go.Figure(go.Bar(x=r.psi, y=r.feature, orientation="h", marker_color=colors))
    for x, c in [(PSI_WARN, "#F59E0B"), (PSI_ALERT, "#EF4444")]:
        fig.add_vline(x=x, line_dash="dash", line_color=c)
    fig.update_layout(title="Feature drift vs training reference (PSI)", height=max(380, 24 * len(r) + 120),
                      template="plotly_white", margin=dict(l=40, r=20, t=60, b=40), xaxis_title="PSI")
    return fig


# ------------------------------------------------------------------------------------------ inference log
def log_inference(X_row: pd.DataFrame, prob: float, path=None):
    """Append one served request (model inputs + predicted probability). On ephemeral hosts (Cloud Run / Community
    Cloud) swap this for Cloud Logging / BigQuery / a bucket - the file is lost on container restart."""
    path = path or os.environ.get("INFERENCE_LOG_PATH", "logs/inference_log.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = X_row.copy()
    r["pred_prob"] = prob
    r["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def read_inference_log(path=None) -> pd.DataFrame | None:
    path = path or os.environ.get("INFERENCE_LOG_PATH", "logs/inference_log.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description="Feature-drift report")
    ap.add_argument("--reference", default="models_v2/reference_sample.csv.gz")
    ap.add_argument("--meta", default="models_v2/meta_v2.json")
    ap.add_argument("--current", required=True)
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 when any feature is in DRIFT (for CI / cron)")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    ref, cur = pd.read_csv(a.reference), pd.read_csv(a.current)
    rep = drift_report(ref, cur, meta["num_features"], meta["cat_features"],
                       ref["pred_prob"] if "pred_prob" in ref else None, cur["pred_prob"] if "pred_prob" in cur else None)
    print(rep.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    s = summarise(rep, len(cur))
    print("\n", s)
    if not s["reliable"]:
        print(f"note: only {len(cur)} current rows (< {MIN_CURRENT_ROWS}); PSI is noisy at this size")
    raise SystemExit(1 if (a.fail_on_drift and s["overall"] == "DRIFT") else 0)


if __name__ == "__main__":
    main()
