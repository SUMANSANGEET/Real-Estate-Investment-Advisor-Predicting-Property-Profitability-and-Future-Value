import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from housing_utils import predict_listing, BASE_YEAR, FORECAST_YEARS

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

# ------------------------------------------------------------------ assets
@st.cache_resource
def load_assets():
    clf = joblib.load("models/classifier.joblib")
    reg = joblib.load("models/regressor.joblib")
    meta = json.load(open("models/meta.json"))
    return clf, reg, meta

@st.cache_data
def load_market():
    return pd.read_csv("data/market_sample.csv")

clf, reg, meta = load_assets()
market = load_market()

st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#4F46E5,#7C3AED 55%,#EC4899);padding:22px 28px;border-radius:14px;color:white;margin-bottom:14px}
.hero h1{margin:0;color:white}.hero p{margin:4px 0 0 0;color:#E0E7FF}
.verdict{padding:16px 20px;border-radius:12px;font-size:20px;font-weight:700}
.good{background:#DCFCE7;color:#14532D;border:2px solid #4ADE80}.bad{background:#FEE2E2;color:#7F1D1D;border:2px solid #F87171}
</style>
<div class="hero"><h1>🏠 Real Estate Investment Advisor</h1>
<p>Is it a good investment? What will it be worth in 5 years? ML-powered answers for Indian property listings.</p></div>
""", unsafe_allow_html=True)

tab_adv, tab_mkt, tab_model = st.tabs(["🔮 Investment Advisor", "🗺️ Market Explorer", "🧠 Model Insights"])

# ================================================================== TAB 1: advisor
with tab_adv:
    with st.form("listing"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("📍 Location")
            state = st.selectbox("State", sorted(meta["state_cities"]), index=sorted(meta["state_cities"]).index("Maharashtra"))
            city = st.selectbox("City", meta["state_cities"][state])
            prop = st.selectbox("Property type", meta["options"]["Property_Type"])
            facing = st.selectbox("Facing", meta["options"]["Facing"])
            transport = st.select_slider("Public transport", ["Low", "Medium", "High"], value="Medium")
        with c2:
            st.subheader("🏗️ Property")
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", *meta["ranges"]["size"], value=1800, step=50)
            price = st.number_input("Asking price (₹ Lakh)", 10.0, 500.0, 150.0, step=5.0)
            built = st.slider("Year built", 1990, BASE_YEAR - 2, 2015)
            status = st.selectbox("Availability", meta["options"]["Availability_Status"])
            owner = st.selectbox("Seller type", meta["options"]["Owner_Type"])
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
        out = predict_listing(listing, clf, reg, meta["city_median_pps"])
        f = out["features"]

        cls = "good" if out["is_good"] else "bad"
        txt = "✅ Good investment" if out["is_good"] else "⚠️ Not a good investment"
        st.markdown(f'<div class="verdict {cls}">{txt} — model confidence {max(out["good_prob"], 1-out["good_prob"]):.0%}</div>', unsafe_allow_html=True)
        st.write("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Asking price", f"₹{out['price']:,.1f} L")
        m2.metric(f"Estimated price in {FORECAST_YEARS} yrs", f"₹{out['future_price']:,.1f} L", f"{out['future_price']/out['price']-1:+.0%}")
        m3.metric("Implied annual growth", f"{out['implied_cagr']:.1%}")
        m4.metric("₹/sq ft vs city median", f"{f['PPS_vs_City']:.2f}×", "cheaper" if f["PPS_vs_City"] < 1 else "pricier", delta_color="inverse")

        g1, g2 = st.columns(2)
        with g1:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=out["good_prob"] * 100, number={"suffix": "%"},
                                           title={"text": "Probability of being a good investment"},
                                           gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"},
                                                  "steps": [{"range": [0, 50], "color": "#FEE2E2"}, {"range": [50, 100], "color": "#DCFCE7"}],
                                                  "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            gauge.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(gauge, width="stretch")
        with g2:
            yrs = np.arange(0, FORECAST_YEARS + 1)
            proj = go.Figure(go.Scatter(x=BASE_YEAR + yrs, y=out["price"] * (1 + out["implied_cagr"]) ** yrs, mode="lines+markers",
                                        line=dict(color="#4F46E5", width=4)))
            proj.update_layout(height=320, margin=dict(t=60, b=10), title=f"Projected value (₹ Lakh), {out['implied_cagr']:.1%} per year",
                               xaxis=dict(dtick=1), yaxis_title="₹ Lakh")
            st.plotly_chart(proj, width="stretch")

        st.caption(f"Infrastructure score: {f['Infrastructure_Score']:.0f}/100 · City tier: {int(f['City_Tier'])} · "
                   f"City median: ₹{f['City_Median_PPS']:,.0f}/sq ft. Forecasts rest on rule-based growth assumptions, not guaranteed returns — not financial advice.")

# ================================================================== TAB 2: explorer
with tab_mkt:
    st.subheader("Filter the market")
    f1, f2, f3, f4 = st.columns(4)
    sel_states = f1.multiselect("State", sorted(market.State.unique()))
    sel_types = f2.multiselect("Property type", sorted(market.Property_Type.unique()))
    sel_bhk = f3.multiselect("BHK", sorted(market.BHK.unique()))
    lo, hi = f4.slider("Price (₹ Lakh)", 10, 500, (10, 500))

    d = market[(market.Price_in_Lakhs.between(lo, hi))]
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
        city_stats = d.groupby("City").agg(Avg_Price=("Price_in_Lakhs", "mean"), Good=("Good_Investment", "mean"), N=("ID", "size")).reset_index()
        a.plotly_chart(px.bar(city_stats.sort_values("Avg_Price", ascending=False).head(20), x="City", y="Avg_Price", color="Good",
                              color_continuous_scale="Viridis", title="Average price by city (colour = good-investment share)"), width="stretch")
        b.plotly_chart(px.treemap(d.groupby(["State", "City"]).size().reset_index(name="Listings"), path=[px.Constant("India"), "State", "City"],
                                  values="Listings", title="Market map (listings)"), width="stretch")
        heat = d.pivot_table(index="City", columns="Property_Type", values="Good_Investment", aggfunc="mean")
        st.plotly_chart(px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="RdYlGn",
                                  title="Share of good investments — city × property type"), width="stretch")
        st.dataframe(d.head(500), width="stretch", hide_index=True)
    else:
        st.info("No listings match these filters.")

# ================================================================== TAB 3: model insights
with tab_model:
    st.subheader("How good are the models?")
    best = meta["best_models"]
    st.write(f"**Classifier:** {best['classifier']}  ·  **Regressor:** {best['regressor']}  (registered in MLflow as `champion`)")
    a, b = st.columns(2)
    a.markdown("**Classification (test set)**"); a.dataframe(pd.DataFrame(meta["clf_metrics"]).T.style.format("{:.3f}"), width="stretch")
    b.markdown("**Regression (test set, ₹ Lakh)**"); b.dataframe(pd.DataFrame(meta["reg_metrics"]).T.style.format("{:.3f}"), width="stretch")
    c, e = st.columns(2)
    for col, key, title, color in [(c, "importance_clf", "What drives 'good investment'", "#4F46E5"), (e, "importance_reg", "What drives future price", "#10B981")]:
        s = pd.Series(meta[key]).sort_values()
        col.plotly_chart(px.bar(x=s.values, y=s.index, orientation="h", title=title, color_discrete_sequence=[color]).update_layout(xaxis_tickformat=".0%"),
                         width="stretch")
    st.info("Targets are engineered from documented growth rules (the source data has no price history), so scores measure rule recovery, not real-market performance.")
