import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import monitoring as mon
import modeling_v2 as m2
from housing_utils import BASE_YEAR

MODEL_DIR = os.environ.get("MODEL_DIR", "models_v2")

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

# ------------------------------------------------------------------ branding
COMPANY = "EstateIQ"            # <- change to your company name
TAGLINE = "Data-driven property investing"
LOGO_SVG = """<svg width="54" height="54" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E0E7FF"/></linearGradient></defs>
<rect x="2" y="2" width="60" height="60" rx="16" fill="rgba(255,255,255,.18)" stroke="white" stroke-opacity=".6"/>
<path d="M14 32 L32 15 L50 32 V50 H14 Z" fill="url(#g)"/>
<rect x="22" y="36" width="5" height="14" fill="#4F46E5"/><rect x="30" y="30" width="5" height="20" fill="#7C3AED"/>
<rect x="38" y="24" width="5" height="26" fill="#EC4899"/></svg>"""


def logo_html():
    """Custom logo: drop assets/logo.png next to this file (or upload in the sidebar) to replace the SVG."""
    import base64
    up = st.session_state.get("logo_upload")
    data = up.getvalue() if up is not None else None
    if data is None and os.path.exists("assets/logo.png"):
        data = open("assets/logo.png", "rb").read()
    if data:
        return f'<img src="data:image/png;base64,{base64.b64encode(data).decode()}" height="54" style="border-radius:12px"/>'
    return LOGO_SVG


with st.sidebar:
    st.markdown(f"### {COMPANY}")
    st.file_uploader("Upload company logo (PNG)", type=["png"], key="logo_upload")
    st.caption("Tip: or place `assets/logo.png` in the repo.")
    st.markdown("---")
    st.markdown("**Stack:** XGBoost · scikit-learn · SHAP · Plotly · Streamlit")
    st.markdown("**Models:** calibrated classifier, price regressor, 4-model arena, drift monitor")


# ------------------------------------------------------------------ assets
@st.cache_resource
def load_assets():
    import shap
    clf = joblib.load(f"{MODEL_DIR}/clf_calibrated.joblib")       # CalibratedPipe (pipeline + probability map)
    reg = joblib.load(f"{MODEL_DIR}/regressor.joblib")
    meta = json.load(open(f"{MODEL_DIR}/meta_v2.json"))
    ref = pd.read_csv(f"{MODEL_DIR}/reference_sample.csv.gz")
    expl = {"clf": shap.TreeExplainer(clf.pipe.named_steps["model"]),
            "reg": shap.TreeExplainer(reg.named_steps["model"])}
    return clf, reg, meta, ref, expl


MARKET_COLS = ["ID", "State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "PPS_INR",
               "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
               "Public_Transport_Accessibility", "Facing", "Good_Investment"]


@st.cache_data(show_spinner="Loading market data…")
def load_market(max_rows=60000):
    """Memory-safe loader: only needed columns, chunked, downcast, and tolerant of a malformed file."""
    import csv
    path = "data/market_sample.csv"
    if not os.path.exists(path):
        st.error(f"`{path}` not found. Run `python make_market_sample.py` first (see README).")
        st.stop()
    size_mb = os.path.getsize(path) / 1e6
    attempts = [dict(engine="c"),
                dict(engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip")]   # fallback: unbalanced quotes
    last_err = None
    for kw in attempts:
        try:
            parts = []
            for ch in pd.read_csv(path, usecols=lambda c: c in MARKET_COLS, chunksize=50_000, **kw):
                for c in ch.select_dtypes("float64"):
                    ch[c] = pd.to_numeric(ch[c], downcast="float")
                for c in ch.select_dtypes("object"):
                    ch[c] = ch[c].astype("category")
                parts.append(ch)
            df = pd.concat(parts, ignore_index=True)
            for c in df.select_dtypes("category"):
                df[c] = df[c].astype(str)
            if len(df) > max_rows:
                df = df.sample(max_rows, random_state=1).reset_index(drop=True)
            if "ID" not in df:
                df["ID"] = range(len(df))
            return df
        except (pd.errors.ParserError, MemoryError) as e:
            last_err = e
    st.error(f"Could not read `{path}` ({size_mb:,.0f} MB): {last_err}. The file is probably the full dataset, "
             "corrupted, or has an unbalanced quote. Regenerate it with `python make_market_sample.py`.")
    st.stop()


@st.cache_resource(show_spinner="Training model arena…")
def train_arena():
    df = market.copy()
    if "Good_Investment" not in df or df["Good_Investment"].nunique() < 2:
        return None
    num = [c for c in ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "PPS_INR", "Year_Built", "Floor_No", "Total_Floors",
                       "Nearby_Schools", "Nearby_Hospitals"] if c in df]
    cat = [c for c in ["State", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility", "Facing"] if c in df]
    cols = num + cat
    df = df.sample(min(25000, len(df)), random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(df[cols], df["Good_Investment"].astype(int), test_size=0.25,
                                          stratify=df["Good_Investment"], random_state=7)
    pre = lambda: ColumnTransformer([("n", StandardScaler(), num),
                                     ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    algos = {"Logistic Regression": LogisticRegression(max_iter=500),
             "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=7),
             "Hist. Gradient Boosting": HistGradientBoostingClassifier(random_state=7)}
    try:
        from xgboost import XGBClassifier
        algos["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9, n_jobs=-1,
                                         eval_metric="logloss")
    except Exception:
        pass
    models, rows, roc = {}, [], []
    for n, a in algos.items():
        p = Pipeline([("pre", pre()), ("model", a)]).fit(Xtr, ytr)
        pr = p.predict_proba(Xte)[:, 1]
        models[n] = p
        rows.append({"Model": n, "ROC-AUC": roc_auc_score(yte, pr), "F1": f1_score(yte, pr > .5),
                     "Accuracy": accuracy_score(yte, pr > .5)})
        fpr, tpr, _ = roc_curve(yte, pr)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))   # thin the curve: faster, lighter
        roc.append(pd.DataFrame({"fpr": fpr[keep], "tpr": tpr[keep], "Model": n}))
    return {"models": models, "cols": cols, "metrics": pd.DataFrame(rows).set_index("Model"),
            "roc": pd.concat(roc), "n_test": len(yte)}


clf, reg, meta, ref, expl = load_assets()
market = load_market()
REAL = meta["mode"] == "real"
HORIZON = meta["horizon_years"]
HAS_METRO = "Dist_to_Metro_km" in meta["num_features"]
HAS_RERA = "RERA_Status" in meta["cat_features"]

st.markdown(f"""
<style>
.hero{{background:linear-gradient(135deg,#4F46E5,#7C3AED 55%,#EC4899);padding:20px 28px;border-radius:14px;color:white;margin-bottom:14px;display:flex;align-items:center;gap:18px}}
.hero h1{{margin:0;color:white;font-size:30px}}.hero p{{margin:4px 0 0 0;color:#E0E7FF}}
.brand{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#C7D2FE;font-weight:600}}
.verdict{{padding:16px 20px;border-radius:12px;font-size:20px;font-weight:700}}
.good{{background:#DCFCE7;color:#14532D;border:2px solid #4ADE80}}.bad{{background:#FEE2E2;color:#7F1D1D;border:2px solid #F87171}}
</style>
<div class="hero">{logo_html()}<div><div class="brand">{COMPANY} · {TAGLINE}</div>
<h1>Real Estate Investment Advisor</h1>
<p>Calibrated probabilities · SHAP explanations · multi-model arena · live drift monitoring</p></div></div>
""", unsafe_allow_html=True)

if not REAL:
    st.warning("**Legacy mode:** this build was trained on the rule-based growth targets (no dated price history was "
               "available). Scores measure rule recovery, not real-market skill.")

tab_adv, tab_mkt, tab_heat, tab_arena, tab_model, tab_mon = st.tabs(
    ["🔮 Investment Advisor", "🗺️ Market Explorer", "🔥 Heatmaps & Trends", "🤖 Model Arena",
     "🧠 Model Insights", "📡 Monitoring"])

# ================================================================== TAB 1: advisor
with tab_adv:
    with st.form("listing"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("📍 Location")
            states = sorted(meta["state_cities"])
            state = st.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
            city = st.selectbox("City", meta["state_cities"][state])
            prop = st.selectbox("Property type", meta["options"]["Property_Type"])
            facing = st.selectbox("Facing", meta["options"]["Facing"])
            transport = st.select_slider("Public transport", ["Low", "Medium", "High"], value="Medium")
            if HAS_METRO:
                metro_km = st.slider("Distance to nearest metro (km)", 0.0, 25.0, 3.0, 0.1,
                                     help="25 km = no metro in the city")
        with c2:
            st.subheader("🏗️ Property")
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", *meta["ranges"]["size"], value=1800, step=50)
            price = st.number_input("Asking price (₹ Lakh)", 10.0, 500.0, 150.0, step=5.0)
            built = st.slider("Year built", 1990, BASE_YEAR - 2, 2015)
            status = st.selectbox("Availability", meta["options"]["Availability_Status"])
            owner = st.selectbox("Seller type", meta["options"]["Owner_Type"])
            if HAS_RERA:
                rera = st.selectbox("RERA status", meta["options"]["RERA_Status"])
        with c3:
            st.subheader("🏢 Building & neighbourhood")
            total_floors = st.slider("Total floors", 1, 30, 12)
            floor = st.slider("Floor number", 0, 30, 5)
            schools = st.slider("Nearby schools", 1, 10, 6)
            hospitals = st.slider("Nearby hospitals", 1, 10, 5)
            furnished = st.selectbox("Furnishing", meta["options"]["Furnished_Status"])
            parking = st.checkbox("Parking", True)
            security = st.checkbox("Security", True)
            amen = st.multiselect("Amenities", meta["amenities"], default=["Gym", "Pool"])
        go_btn = st.form_submit_button("Analyse this property", type="primary", width="stretch")

    if go_btn:
        listing = dict(State=state, City=city, Property_Type=prop, BHK=bhk, Size_in_SqFt=size, Price_in_Lakhs=price,
                       Year_Built=built, Furnished_Status=furnished, Floor_No=floor, Total_Floors=total_floors,
                       Nearby_Schools=schools, Nearby_Hospitals=hospitals, Public_Transport_Accessibility=transport,
                       Parking_Space="Yes" if parking else "No", Security="Yes" if security else "No",
                       Amenities=", ".join(amen), Facing=facing, Owner_Type=owner, Availability_Status=status)
        if HAS_METRO:
            listing["Dist_to_Metro_km"] = float(metro_km)
            listing["Metro_Within_1km"] = int(metro_km <= 1.0)
        if HAS_RERA:
            listing["RERA_Status"] = rera
        out = m2.predict_v2(listing, clf, reg, meta)
        f, X = out["features"], out["X"]
        mon.log_inference(X, out["good_prob"])                      # feeds the Monitoring tab

        cls = "good" if out["is_good"] else "bad"
        txt = "✅ Good investment" if out["is_good"] else "⚠️ Not a good investment"
        st.markdown(f'<div class="verdict {cls}">{txt} — calibrated probability {out["good_prob"]:.0%}</div>',
                    unsafe_allow_html=True)
        st.caption(f"Probability is calibrated ({clf.method_}): among listings scored ≈{out['good_prob']:.0%}, "
                   f"about that share turned out to be good investments in held-out data.")

        m1, m2_, m3, m4 = st.columns(4)
        m1.metric("Asking price", f"₹{out['price']:,.1f} L")
        m2_.metric(f"Estimated price in {HORIZON} yrs", f"₹{out['future_price']:,.1f} L",
                   f"{out['future_price']/out['price']-1:+.0%}")
        m3.metric("Implied annual growth", f"{out['implied_cagr']:.1%}")
        m4.metric("₹/sq ft vs city median", f"{f['PPS_vs_City']:.2f}×", "cheaper" if f["PPS_vs_City"] < 1 else "pricier",
                  delta_color="inverse")

        # ---- gauge + 5-year scenario + price sensitivity
        g1, g2 = st.columns([1, 2])
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=out["good_prob"] * 100,
                                       number={"suffix": "%"}, title={"text": "Investment confidence"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"},
                                              "steps": [{"range": [0, 40], "color": "#FEE2E2"},
                                                        {"range": [40, 60], "color": "#FEF9C3"},
                                                        {"range": [60, 100], "color": "#DCFCE7"}]}))
        g1.plotly_chart(gauge.update_layout(height=300, margin=dict(t=60, b=10)), width="stretch")
        yrs = np.arange(0, HORIZON + 1)
        base = out["implied_cagr"]
        sc = pd.DataFrame({"Year": yrs + BASE_YEAR})
        for name, d_ in [("Bear (−3 pp)", -0.03), ("Base (model)", 0.0), ("Bull (+3 pp)", 0.03)]:
            sc[name] = out["price"] * (1 + base + d_) ** yrs
        g2.plotly_chart(px.line(sc, x="Year", y=["Bear (−3 pp)", "Base (model)", "Bull (+3 pp)"], markers=True,
                                title=f"{HORIZON}-year value scenarios (₹ Lakh)", template="plotly_white",
                                color_discrete_sequence=["#EF4444", "#4F46E5", "#10B981"]).update_layout(height=300),
                        width="stretch")
        sens = []
        for mult in np.linspace(0.7, 1.3, 13):
            try:
                o2 = m2.predict_v2({**listing, "Price_in_Lakhs": price * mult}, clf, reg, meta)
                sens.append({"Asking price (₹ L)": price * mult, "P(good investment)": o2["good_prob"]})
            except Exception:
                break
        if sens:
            st.plotly_chart(px.line(pd.DataFrame(sens), x="Asking price (₹ L)", y="P(good investment)", markers=True,
                                    title="What-if: how negotiation changes the verdict", template="plotly_white")
                            .add_hline(y=0.5, line_dash="dash", line_color="grey").update_layout(height=320),
                            width="stretch")
        # ---- model consensus (arena models score the same listing)
        arena = train_arena()
        if arena:
            row = pd.DataFrame([{**listing, "PPS_INR": price * 1e5 / size}])
            cons = {n: float(p.predict_proba(row[arena["cols"]])[0, 1]) for n, p in arena["models"].items()}
            cons["Production (calibrated)"] = out["good_prob"]
            cdf = pd.DataFrame({"Model": list(cons), "P(good)": list(cons.values())})
            st.plotly_chart(px.bar(cdf, x="P(good)", y="Model", orientation="h", range_x=[0, 1], template="plotly_white",
                                   title=f"Model consensus — {sum(v >= .5 for v in cons.values())}/{len(cons)} models say 'good'",
                                   color="P(good)", color_continuous_scale="RdYlGn").update_layout(height=320),
                            width="stretch")

        st.markdown("### 🔍 Why this verdict? (SHAP)")
        Gc, bc, dc = m2.shap_grouped(clf.pipe, X, meta["cat_features"], explainer=expl["clf"])
        Gr, br, dr = m2.shap_grouped(reg, X, meta["cat_features"], explainer=expl["reg"])
        e1, e2 = st.columns(2)
        e1.plotly_chart(m2.shap_waterfall_figure(Gc.iloc[0], bc, dc.iloc[0], unit="log-odds",
                                                 title="Drivers of the investment verdict"), width="stretch")
        REG_UNIT = meta.get("reg_unit", "₹ Lakh")
        e2.plotly_chart(m2.shap_waterfall_figure(Gr.iloc[0], br, dr.iloc[0], unit=REG_UNIT,
                                                 title=f"Drivers of the {HORIZON}-year {'growth multiple' if REG_UNIT != '₹ Lakh' else 'price'}"),
                        width="stretch")
        st.caption("Green bars push the model towards 'good investment' / a higher price, red bars away from it. "
                   "SHAP values are on the raw model scale (log-odds); calibration is a monotone re-mapping, so the "
                   "direction of every driver is unchanged.")
        st.caption(f"Infrastructure score {f['Infrastructure_Score']:.0f}/100 · City tier {int(f['City_Tier'])} · "
                   f"City benchmark ₹{f['City_Median_PPS']:,.0f}/sq ft. Model output, not financial advice.")

# ================================================================== TAB 2: explorer
with tab_mkt:
    st.subheader("Filter the market")
    f1, f2, f3, f4 = st.columns(4)
    sel_states = f1.multiselect("State", sorted(market.State.unique()))
    sel_types = f2.multiselect("Property type", sorted(market.Property_Type.unique()))
    sel_bhk = f3.multiselect("BHK", sorted(market.BHK.unique()))
    lo, hi = f4.slider("Price (₹ Lakh)", 10, 500, (10, 500))
    d = market[market.Price_in_Lakhs.between(lo, hi)]
    if sel_states: d = d[d.State.isin(sel_states)]
    if sel_types: d = d[d.Property_Type.isin(sel_types)]
    if sel_bhk: d = d[d.BHK.isin(sel_bhk)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching listings", f"{len(d):,}")
    k2.metric("Avg price", f"₹{d.Price_in_Lakhs.mean():,.0f} L" if len(d) else "–")
    k3.metric("Median ₹/sq ft", f"₹{d.PPS_INR.median():,.0f}" if len(d) else "–")
    k4.metric("Good-investment share", f"{d.Good_Investment.mean():.0%}" if len(d) else "–")
    if len(d):
        a, b = st.columns(2)
        cs = d.groupby("City").agg(Avg_Price=("Price_in_Lakhs", "mean"), Good=("Good_Investment", "mean"), N=("ID", "size")).reset_index()
        a.plotly_chart(px.bar(cs.sort_values("Avg_Price", ascending=False).head(20), x="City", y="Avg_Price", color="Good",
                              color_continuous_scale="Viridis", title="Average price by city (colour = good-investment share)"),
                       width="stretch")
        b.plotly_chart(px.treemap(d.groupby(["State", "City"]).size().reset_index(name="Listings"),
                                  path=[px.Constant("India"), "State", "City"], values="Listings", title="Market map (listings)"),
                       width="stretch")
        st.dataframe(d.head(500), width="stretch", hide_index=True)
    else:
        st.info("No listings match these filters.")

# ================================================================== TAB 3: heatmaps & trends
with tab_heat:
    st.subheader("Where is the opportunity?")
    h = market.copy()
    if "Size_in_SqFt" in h:
        h["PPS"] = h["PPS_INR"] if "PPS_INR" in h else h["Price_in_Lakhs"] * 1e5 / h["Size_in_SqFt"]
    else:
        h["PPS"] = h.get("PPS_INR", h["Price_in_Lakhs"])
    top_c = h.City.value_counts().head(15).index
    a, b = st.columns(2)
    piv = h[h.City.isin(top_c)].pivot_table(index="City", columns="BHK", values="PPS", aggfunc="median")
    a.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="Turbo", text_auto=".0f",
                             title="Median ₹/sq ft — city × BHK").update_layout(height=480), width="stretch")
    piv2 = h.pivot_table(index="State", columns="Property_Type", values="Good_Investment", aggfunc="mean")
    b.plotly_chart(px.imshow(piv2, aspect="auto", color_continuous_scale="RdYlGn", text_auto=".0%",
                             title="Good-investment share — state × property type").update_layout(height=480),
                   width="stretch")
    c, d = st.columns(2)
    if "Year_Built" in h:
        tr = h.groupby("Year_Built").agg(PPS=("PPS", "median"), Good=("Good_Investment", "mean")).reset_index()
        c.plotly_chart(px.line(tr, x="Year_Built", y="PPS", markers=True, template="plotly_white",
                               title="Median ₹/sq ft by construction year").update_layout(height=380), width="stretch")
        d.plotly_chart(px.area(tr, x="Year_Built", y="Good", template="plotly_white",
                               title="Good-investment share by construction year").update_layout(height=380,
                               yaxis_tickformat=".0%"), width="stretch")
    st.plotly_chart(px.sunburst(h.groupby(["State", "Property_Type"]).agg(N=("Price_in_Lakhs", "size"),
                                Price=("Price_in_Lakhs", "mean")).reset_index(), path=["State", "Property_Type"],
                                values="N", color="Price", color_continuous_scale="Viridis",
                                title="Market composition (size = listings, colour = avg price ₹ L)")
                    .update_layout(height=520), width="stretch")
    st.markdown("**Price vs size vs ₹/sq ft** — bubble size = ₹/sq ft, colour = good investment")
    samp = h.dropna(subset=["PPS", "Price_in_Lakhs"]).pipe(lambda x: x.sample(min(2000, len(x)), random_state=3)).copy()
    samp["Good_Investment"] = samp["Good_Investment"].astype(int).map({1: "Good", 0: "Not good"})
    sz = "Size_in_SqFt" if "Size_in_SqFt" in samp else "BHK"
    st.plotly_chart(px.scatter(samp, x=sz, y="Price_in_Lakhs", size="PPS", color="Good_Investment",
                               hover_data=["City", "BHK"], opacity=.6, size_max=14, render_mode="svg",
                               template="plotly_white",
                               color_discrete_map={"Good": "#10B981", "Not good": "#EF4444"})
                    .update_layout(height=520), width="stretch")
    st.markdown("**Where the market concentrates** — price vs size density with distributions (no WebGL needed)")
    dens = samp[[sz, "Price_in_Lakhs"]].dropna()
    from plotly.subplots import make_subplots
    dfig = make_subplots(rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.18, 0.82],
                         shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.01, vertical_spacing=0.01)
    dfig.add_trace(go.Histogram(x=dens[sz], nbinsx=40, marker_color="#7C3AED", showlegend=False), row=1, col=1)
    dfig.add_trace(go.Histogram2d(x=dens[sz], y=dens["Price_in_Lakhs"], nbinsx=40, nbinsy=40, colorscale="Viridis",
                                  colorbar=dict(title="Listings")), row=2, col=1)
    dfig.add_trace(go.Histogram(y=dens["Price_in_Lakhs"], nbinsy=40, marker_color="#4F46E5", showlegend=False),
                   row=2, col=2)
    dfig.update_xaxes(title_text=sz, row=2, col=1)
    dfig.update_yaxes(title_text="Price (₹ Lakh)", row=2, col=1)
    st.plotly_chart(dfig.update_layout(height=560, template="plotly_white", bargap=0.02), width="stretch")

# ================================================================== TAB 4: model arena
with tab_arena:
    st.subheader("Model arena — four algorithms, one held-out test set")
    ar = train_arena()
    if not ar:
        st.info("Market sample has no usable Good_Investment label.")
    else:
        st.caption(f"Trained live on `data/market_sample.csv` (stratified 75/25 split, {ar['n_test']:,} test rows). "
                   "These are benchmark models; the calibrated, chronologically validated model in Model Insights "
                   "is the production one.")
        a, b = st.columns([1, 1])
        a.dataframe(ar["metrics"].style.format("{:.4f}").highlight_max(axis=0, color="#DCFCE7"), width="stretch")
        fig = px.line(ar["roc"], x="fpr", y="tpr", color="Model", template="plotly_white", title="ROC curves",
                      render_mode="svg")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        b.plotly_chart(fig.update_layout(height=380), width="stretch")
        best = ar["metrics"]["ROC-AUC"].idxmax()
        st.success(f"Best AUC: **{best}** ({ar['metrics'].loc[best, 'ROC-AUC']:.3f}). "
                   "If every model scores ~1.0, the label is likely a deterministic function of the inputs "
                   "(rule-based target) — flag that honestly rather than celebrate it.")
        pick = st.selectbox("Feature importance for", [m for m in ar["models"] if m != "Logistic Regression"])
        pp = ar["models"][pick]
        mdl, names = pp.named_steps["model"], pp.named_steps["pre"].get_feature_names_out()
        imp = getattr(mdl, "feature_importances_", None)
        if imp is not None:
            s_ = pd.Series(imp, index=[n.split("__")[-1] for n in names]).nlargest(15).sort_values()
            st.plotly_chart(px.bar(x=s_.values, y=s_.index, orientation="h", template="plotly_white",
                                   title=f"Top 15 features — {pick}").update_layout(height=450), width="stretch")
        else:
            st.info("This model does not expose built-in feature importances.")

# ================================================================== TAB 5: model insights
with tab_model:
    st.subheader("How good (and how honest) are the models?")
    st.write(f"**Training mode:** {'real price history, purged chronological split' if REAL else 'legacy rule-based targets, random split'}"
             f"  ·  **Trained:** {meta['trained_at']}")
    st.caption(meta["split_note"])
    a, b = st.columns(2)
    a.markdown("**Classifier on the held-out test block — raw vs calibrated**")
    a.dataframe(pd.DataFrame(meta["calibration"]).T.style.format("{:.4f}"), width="stretch")
    b.markdown("**Price regressor on the held-out test block (₹ Lakh)**")
    b.dataframe(pd.DataFrame([meta["reg_metrics"]], index=["XGBoost (tuned)"]).style.format("{:.3f}"), width="stretch")

    rel = pd.DataFrame(meta["reliability"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="perfect"))
    for name, g in rel.groupby("model"):
        fig.add_trace(go.Scatter(x=g.mean_predicted, y=g.observed_rate, mode="lines+markers", name=name))
    fig.update_layout(title="Reliability diagram (test block)", height=380, xaxis_title="predicted probability",
                      yaxis_title="observed rate", template="plotly_white")
    c, e = st.columns(2)
    c.plotly_chart(fig, width="stretch")
    th = pd.DataFrame({"trial": range(len(meta["tuning"]["clf_history"])), "log-loss": meta["tuning"]["clf_history"]})
    th["best so far"] = th["log-loss"].cummin()
    e.plotly_chart(px.line(th, x="trial", y=["log-loss", "best so far"], title="Optuna search (classifier, CV log-loss)",
                           template="plotly_white").update_layout(height=380), width="stretch")
    g1, g2 = st.columns(2)
    for col, key, title, color in [(g1, "shap_clf", "Global SHAP — investment verdict", "#4F46E5"),
                                   (g2, "shap_reg", "Global SHAP — future price / growth", "#10B981")]:
        s = pd.Series(meta[key]).sort_values()
        col.plotly_chart(px.bar(x=s.values, y=s.index, orientation="h", title=title, color_discrete_sequence=[color])
                         .update_layout(height=460, template="plotly_white"), width="stretch")
    st.info(meta["honesty_note"])

# ================================================================== TAB 6: monitoring
with tab_mon:
    st.subheader("Input-feature drift vs the training reference")
    st.caption("PSI < 0.10 stable · 0.10–0.25 watch · ≥ 0.25 investigate / retrain. KS (numeric) and χ² (categorical) "
               "give the significance; with large samples read PSI first.")
    src = st.radio("Compare against", ["Live requests served by this app", "Upload a CSV of recent listings",
                                       "Stress-test (simulated price premium)"], horizontal=True)
    num, cat = meta["num_features"], meta["cat_features"]
    cur = None
    if src.startswith("Live"):
        cur = mon.read_inference_log()
        if cur is None:
            st.info("No requests logged yet — analyse a few properties in the Advisor tab first.")
    elif src.startswith("Upload"):
        up = st.file_uploader("CSV with the model input columns (see below)", type="csv")
        st.code(", ".join(num + cat), language=None)
        if up is not None:
            cur = pd.read_csv(up)
            miss = [c for c in num + cat if c not in cur.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
                cur = None
    else:
        k = st.slider("Simulated price premium vs city benchmark", 1.0, 2.5, 1.4, 0.05)
        cur = ref.sample(min(1500, len(ref)), random_state=1).copy()
        for c in ("Price_in_Lakhs", "PPS_INR"):               # only present when they are model inputs (legacy mode)
            if c in cur:
                cur[c] = cur[c] * k
        cur["Log_PPS_vs_City"] = cur["Log_PPS_vs_City"] + np.log(k)
        cur["pred_prob"] = np.clip(cur["pred_prob"] - (k - 1) * 0.1, 0, 1)
        st.caption("Demonstration only: makes every listing in a reference resample dearer than its city benchmark by the chosen "
                   "factor, to show how the monitor reacts.")
    if cur is not None and len(cur):
        rep = mon.drift_report(ref, cur, num, cat, ref["pred_prob"], cur["pred_prob"] if "pred_prob" in cur else None)
        s = mon.summarise(rep, len(cur))
        m1, m2_, m3, m4 = st.columns(4)
        m1.metric("Rows compared", f"{s['n_current']:,}")
        m2_.metric("Features in DRIFT", s["drift"])
        m3.metric("Features on WATCH", s["warn"])
        m4.metric("Overall", s["overall"])
        if not s["reliable"]:
            st.warning(f"Only {s['n_current']} rows — below {mon.MIN_CURRENT_ROWS}, PSI is noisy at this size.")
        st.plotly_chart(mon.plot_drift(rep), width="stretch")
        st.dataframe(rep.style.format({"psi": "{:.3f}", "statistic": "{:.3f}", "p_value": "{:.3g}",
                                       "ref_mean": "{:,.2f}", "cur_mean": "{:,.2f}"}), width="stretch", hide_index=True)