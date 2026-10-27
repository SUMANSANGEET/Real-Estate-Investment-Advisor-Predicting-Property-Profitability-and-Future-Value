import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from housing_utils import engineer_features, NUM_FEATURES, CAT_FEATURES, BASE_YEAR, FORECAST_YEARS

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

@st.cache_resource
def load():
    b = joblib.load("models/bundle.joblib"); return b["clf"], b["reg"], b["meta"]
@st.cache_data
def market(): return pd.read_csv("data/market_sample.csv")
CLF, REG, M = load(); MK = market(); ENS = "🏆 Ensemble (mean)"

st.markdown("""<style>.hero{background:linear-gradient(135deg,#4F46E5,#7C3AED 55%,#EC4899);padding:22px 28px;border-radius:14px;color:#fff;margin-bottom:12px}
.hero h1{margin:0;color:#fff}.hero p{margin:4px 0 0;color:#E0E7FF}.verdict{padding:16px 20px;border-radius:12px;font-size:20px;font-weight:700}
.good{background:#DCFCE7;color:#14532D;border:2px solid #4ADE80}.bad{background:#FEE2E2;color:#7F1D1D;border:2px solid #F87171}
.tip{border-left:5px solid #4F46E5;background:#EEF2FF;color:#1E1B4B;padding:10px 14px;border-radius:6px;margin:6px 0}</style>
<div class="hero"><h1>🏠 Real Estate Investment Advisor</h1><p>6 ML models vote live · every widget re-scores instantly · what-if, ROI and model arena</p></div>""", unsafe_allow_html=True)

# ------------------------------------------------ sidebar = live inputs (no submit button: Streamlit reruns on every change)
sb = st.sidebar; sb.header("🎛️ Property (live)")
states = sorted(M["state_cities"]); state = sb.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
city = sb.selectbox("City", M["state_cities"][state]); O = M["options"]
prop = sb.selectbox("Property type", O["Property_Type"]); bhk = sb.slider("BHK", 1, 5, 3)
size = sb.slider("Size (sq ft)", *M["ranges"]["size"], 1800, 50); price = sb.slider("Asking price (₹ Lakh)", 10.0, 500.0, 150.0, 5.0)
built = sb.slider("Year built", 1990, BASE_YEAR - 2, 2015); status = sb.selectbox("Availability", O["Availability_Status"])
transport = sb.select_slider("Public transport", ["Low", "Medium", "High"], "Medium")
schools = sb.slider("Nearby schools", 1, 10, 6); hospitals = sb.slider("Nearby hospitals", 1, 10, 5)
tf = sb.slider("Total floors", 1, 30, 12); fl = sb.slider("Floor number", 0, 30, 5)
with sb.expander("More details"):
    furn = st.selectbox("Furnishing", O["Furnished_Status"]); facing = st.selectbox("Facing", O["Facing"]); owner = st.selectbox("Seller", O["Owner_Type"])
    parking = st.checkbox("Parking", True); security = st.checkbox("Security", True); amen = st.multiselect("Amenities", M["amenities"], ["Gym", "Pool"])
sel = sb.multiselect("🤖 Model committee", list(CLF), list(CLF)) or list(CLF)

L = dict(State=state, City=city, Property_Type=prop, BHK=bhk, Size_in_SqFt=size, Price_in_Lakhs=price, Year_Built=built, Furnished_Status=furn,
         Floor_No=fl, Total_Floors=tf, Nearby_Schools=schools, Nearby_Hospitals=hospitals, Public_Transport_Accessibility=transport,
         Parking_Space="Yes" if parking else "No", Security="Yes" if security else "No", Amenities=", ".join(amen), Facing=facing,
         Owner_Type=owner, Availability_Status=status)
def feats(rows): return engineer_features(pd.DataFrame(rows), M["city_median_pps"])
def committee(rows):
    X = feats(rows)[NUM_FEATURES + CAT_FEATURES]
    return {n: CLF[n].predict_proba(X)[:, 1] for n in sel}, {n: REG[n].predict(X) for n in sel}

P, F = committee([L]); P = {k: float(v[0]) for k, v in P.items()}; F = {k: float(v[0]) for k, v in F.items()}
prob, fut = np.mean(list(P.values())), np.mean(list(F.values())); cagr = (fut / price) ** (1 / FORECAST_YEARS) - 1
f0 = feats([L]).iloc[0]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Live Advisor", "💰 ROI Simulator", "🗺️ Market Explorer", "🧠 Model Arena", "📘 About"])

with tab1:
    good = prob >= .5
    st.markdown(f'<div class="verdict {"good" if good else "bad"}">{"✅ Good investment" if good else "⚠️ Not a good investment"} — committee probability {prob:.0%} ({len(sel)} models)</div>', unsafe_allow_html=True)
    c = st.columns(4); c[0].metric("Asking price", f"₹{price:,.0f} L"); c[1].metric(f"Price in {FORECAST_YEARS} yrs", f"₹{fut:,.0f} L", f"{fut/price-1:+.0%}")
    c[2].metric("Implied annual growth", f"{cagr:.1%}"); c[3].metric("₹/sq ft vs city median", f"{f0['PPS_vs_City']:.2f}×", "cheaper" if f0["PPS_vs_City"] < 1 else "pricier", delta_color="inverse")
    a, b = st.columns(2)
    g = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number={"suffix": "%"}, title={"text": "P(good investment)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"}, "steps": [{"range": [0, 50], "color": "#FEE2E2"}, {"range": [50, 100], "color": "#DCFCE7"}], "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
    a.plotly_chart(g.update_layout(height=320, margin=dict(t=60, b=10)), width="stretch")
    yrs = np.arange(FORECAST_YEARS + 1); path = lambda v: price * (v / price) ** (yrs / FORECAST_YEARS)
    pj = go.Figure([go.Scatter(x=BASE_YEAR + yrs, y=path(max(F.values())), line=dict(width=0), showlegend=False),
                    go.Scatter(x=BASE_YEAR + yrs, y=path(min(F.values())), fill="tonexty", fillcolor="rgba(79,70,229,.15)", line=dict(width=0), name="model range"),
                    go.Scatter(x=BASE_YEAR + yrs, y=path(fut), mode="lines+markers", line=dict(color="#4F46E5", width=4), name="committee")])
    b.plotly_chart(pj.update_layout(height=320, margin=dict(t=60, b=10), title=f"Projected value (₹ Lakh) — band = model disagreement", xaxis=dict(dtick=1)), width="stretch")

    d1, d2 = st.columns(2)
    cons = pd.Series(P).sort_values(); cf = px.bar(x=cons.values, y=cons.index, orientation="h", title="Model consensus — each model's verdict", color=cons.values, color_continuous_scale="RdYlGn", range_color=[0, 1])
    d1.plotly_chart(cf.add_vline(x=.5, line_dash="dash").update_layout(height=340, coloraxis_showscale=False, xaxis_tickformat=".0%"), width="stretch")
    prices = np.linspace(max(10, price * .5), min(500, price * 1.5), 25); sp, _ = committee([{**L, "Price_in_Lakhs": float(x)} for x in prices]); curve = np.mean(list(sp.values()), axis=0)
    sf = go.Figure([go.Scatter(x=prices, y=curve, mode="lines", line=dict(color="#EC4899", width=4), name="P(good)"), go.Scatter(x=[price], y=[prob], mode="markers", marker=dict(size=14, color="#4F46E5"), name="you")])
    d2.plotly_chart(sf.add_hline(y=.5, line_dash="dash").update_layout(height=340, title="What-if: how the verdict moves with the asking price", xaxis_title="Asking price (₹ Lakh)", yaxis=dict(tickformat=".0%", range=[0, 1])), width="stretch")
    ok = [x for x, q in zip(prices, curve) if q >= .5]
    st.markdown(f'<div class="tip">💡 <b>Negotiation insight:</b> '
                + (f"the committee keeps this a <b>good investment up to ≈ ₹{max(ok):,.0f} L</b> ({max(ok)/price-1:+.0%} vs the ask)." if ok else "no price within ±50% of the ask turns this into a good investment — location/infrastructure drive the verdict.") + "</div>", unsafe_allow_html=True)

    cities = sorted(M["city_median_pps"]); cp_, cf_ = committee([{**L, "City": x} for x in cities])
    cdf = pd.DataFrame({"City": cities, "P(good)": np.mean(list(cp_.values()), axis=0), "Future ₹L": np.mean(list(cf_.values()), axis=0)}); cdf["CAGR"] = (cdf["Future ₹L"] / price) ** (1 / FORECAST_YEARS) - 1
    cdf["Selected"] = np.where(cdf.City == city, "your city", "other")
    st.plotly_chart(px.scatter(cdf, x="P(good)", y="CAGR", text="City", color="Selected", size=np.where(cdf.City == city, 16, 9), color_discrete_map={"your city": "#EC4899", "other": "#4F46E5"},
                    title="Same property, every city — where would it be the better bet?").update_traces(textposition="top center").update_layout(height=470, xaxis_tickformat=".0%", yaxis_tickformat=".0%"), width="stretch")
    st.caption(f"Infrastructure score {f0['Infrastructure_Score']:.0f}/100 · city tier {int(f0['City_Tier'])} · city median ₹{f0['City_Median_PPS']:,.0f}/sq ft. Rule-based scenarios, not financial advice.")

with tab2:
    st.subheader("💰 Buy-and-hold simulator (uses the committee's 5-year price)")
    r = st.columns(4); dp = r[0].slider("Down payment %", 10, 100, 20); rate = r[1].slider("Loan rate %", 6.0, 13.0, 8.75, .05); ten = r[2].slider("Tenure (yrs)", 5, 30, 20); yld = r[3].slider("Rental yield % / yr", 0.0, 6.0, 2.5, .1)
    loan = price * (100 - dp) / 100; i = rate / 1200; n = ten * 12; emi = loan * i * (1 + i) ** n / ((1 + i) ** n - 1) if loan and i else 0
    bal, rows = loan, []
    for y in range(1, FORECAST_YEARS + 1):
        paid_int = 0
        for _ in range(12): it = bal * i; bal -= (emi - it); paid_int += it
        val = price * (fut / price) ** (y / FORECAST_YEARS); rows.append(dict(Year=BASE_YEAR + y, Value=val, Loan=bal, Equity=val - bal, Interest=paid_int, Rent=price * yld / 100))
    R = pd.DataFrame(rows); inv = price * dp / 100; gain = R.Equity.iloc[-1] - inv + R.Rent.sum() - R.Interest.sum(); roi = gain / inv if inv else 0
    k = st.columns(4); k[0].metric("Monthly EMI", f"₹{emi*1e5:,.0f}"); k[1].metric("Cash invested", f"₹{inv:,.1f} L"); k[2].metric("Net gain (5y)", f"₹{gain:,.1f} L"); k[3].metric("ROI on equity", f"{roi:.0%}", f"{(1+roi)**(1/FORECAST_YEARS)-1:.1%} / yr")
    st.plotly_chart(go.Figure([go.Bar(x=R.Year, y=R.Equity, name="Equity", marker_color="#10B981"), go.Bar(x=R.Year, y=R.Loan, name="Loan outstanding", marker_color="#CBD5E1"),
                    go.Scatter(x=R.Year, y=R.Value, name="Property value", mode="lines+markers", line=dict(color="#4F46E5", width=4))]).update_layout(barmode="stack", height=400, title="Equity build-up vs loan (₹ Lakh)"), width="stretch")
    st.caption("Simplified: constant rent, no taxes/maintenance/stamp duty; sale value = committee forecast.")

with tab3:
    f = st.columns(4); ss = f[0].multiselect("State ", sorted(MK.State.unique())); tt_ = f[1].multiselect("Type", sorted(MK.Property_Type.unique())); bb = f[2].multiselect("BHK ", sorted(MK.BHK.unique())); lo, hi = f[3].slider("Price (₹ L)", 10, 500, (10, 500))
    d = MK[MK.Price_in_Lakhs.between(lo, hi)]
    for col, v in [("State", ss), ("Property_Type", tt_), ("BHK", bb)]: d = d[d[col].isin(v)] if v else d
    if len(d):
        k = st.columns(4); k[0].metric("Listings", f"{len(d):,}"); k[1].metric("Avg price", f"₹{d.Price_in_Lakhs.mean():,.0f} L"); k[2].metric("Median ₹/sq ft", f"₹{d.PPS_INR.median():,.0f}"); k[3].metric("Good-investment share", f"{d.Good_Investment.mean():.0%}")
        a, b = st.columns(2); cs = d.groupby("City").agg(Avg=("Price_in_Lakhs", "mean"), Good=("Good_Investment", "mean"), N=("ID", "size")).reset_index()
        a.plotly_chart(px.bar(cs.sort_values("Avg", ascending=False).head(20), x="City", y="Avg", color="Good", color_continuous_scale="Viridis", title="Avg price by city (colour = good share)"), width="stretch")
        b.plotly_chart(px.treemap(d.groupby(["State", "City"]).size().reset_index(name="n"), path=[px.Constant("India"), "State", "City"], values="n", title="Market map"), width="stretch")
        st.plotly_chart(px.imshow(d.pivot_table(index="City", columns="Property_Type", values="Good_Investment", aggfunc="mean"), text_auto=".0%", aspect="auto", color_continuous_scale="RdYlGn", title="Good-investment share: city × type").update_layout(height=720), width="stretch")
        st.plotly_chart(px.scatter(d.sample(min(3000, len(d)), random_state=1), x="Infrastructure_Score", y="Simulated_Annual_Growth", color="City_Tier" if "City_Tier" in d else "Public_Transport_Accessibility", opacity=.5, title="Infrastructure vs expected growth"), width="stretch")
    else: st.info("No listings match these filters.")

with tab4:
    st.subheader("🧠 Model Arena — every model on the same held-out test set")
    st.caption(f"{M['n_train']:,} train / {M['n_test']:,} test rows · training time per model is logged below")
    a, b = st.columns(2)
    a.markdown("**Classification**"); a.dataframe(pd.DataFrame(M["clf_metrics"]).T.style.format("{:.4f}").background_gradient(cmap="Greens"), width="stretch")
    b.markdown("**Regression (₹ Lakh)**"); b.dataframe(pd.DataFrame(M["reg_metrics"]).T.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.4f}", "MAPE": "{:.2%}"}).background_gradient(cmap="Greens"), width="stretch")
    pick = st.multiselect("ROC curves", list(M["roc"]), list(M["roc"])[-3:]); rf = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="chance"))
    for n in pick: rf.add_trace(go.Scatter(x=M["roc"][n]["fpr"], y=M["roc"][n]["tpr"], name=f"{n} (AUC {M['clf_metrics'][n]['ROC_AUC']:.3f})", mode="lines"))
    c, d = st.columns(2); c.plotly_chart(rf.update_layout(height=400, title="ROC", xaxis_title="False positive rate", yaxis_title="True positive rate"), width="stretch")
    mdl = d.selectbox("Confusion matrix / scatter model", list(M["roc"]), index=len(M["roc"]) - 1)
    d.plotly_chart(px.imshow(M["cmat"][mdl], text_auto=True, x=["Pred: no", "Pred: good"], y=["Actual: no", "Actual: good"], color_continuous_scale="Purples", title=f"Confusion matrix — {mdl}").update_layout(height=340), width="stretch")
    e, g_ = st.columns(2); sc = M["scatter"]; lim = [min(sc["actual"]), max(sc["actual"])]
    e.plotly_chart(go.Figure([go.Scatter(x=sc["actual"], y=sc[mdl], mode="markers", marker=dict(opacity=.4, color="#4F46E5"), name="test listings"), go.Scatter(x=lim, y=lim, line=dict(dash="dash", color="red"), name="perfect")]).update_layout(height=380, title=f"Actual vs predicted 5-yr price — {mdl}", xaxis_title="Actual ₹L", yaxis_title="Predicted ₹L"), width="stretch")
    ft = pd.Series(M["fit_s"]).sort_values(); g_.plotly_chart(px.bar(x=ft.values, y=ft.index, orientation="h", title="Training time (s)", color_discrete_sequence=["#EC4899"]).update_layout(height=380), width="stretch")
    m_ = st.selectbox("Feature importance — model", [n for n in M["imp_clf"] if M["imp_clf"][n]], index=0); h1, h2 = st.columns(2)
    for col, key, ttl in [(h1, "imp_clf", "drives 'good investment'"), (h2, "imp_reg", "drives future price")]:
        s = pd.Series(M[key][m_]).sort_values(); col.plotly_chart(px.bar(x=s.values, y=s.index, orientation="h", title=f"{m_}: what {ttl}").update_layout(xaxis_tickformat=".0%"), width="stretch")

with tab5:
    st.markdown(f"""### Pipeline
**Data** → cleaning & feature engineering (`housing_utils.py`, shared by training and serving) → rule-based targets → **6 classifiers + 6 regressors + ensemble** (`train_models.py`) → one joblib bundle → this Streamlit app.

**Models:** Logistic/Ridge · Random Forest · Extra Trees · HistGradientBoosting · XGBoost · Neural Net (MLP) · mean-ensemble.

> ⚠️ {M['note']} The source data also looks synthetic (near-independent columns). The notebook quantifies this instead of hiding it.""")
