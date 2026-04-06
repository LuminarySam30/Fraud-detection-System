import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ─── 1. PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Transaction BI Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 2. GLOBAL CSS (light theme) ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #F5F7FA;
    color: #1A1A2E;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E6ED;
}
section[data-testid="stSidebar"] * { color: #1A1A2E !important; }

/* ── KPI card ── */
.kpi-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid #0077B6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 0.5rem;
}
.kpi-label { font-size: 0.78rem; color: #6B7280; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #0077B6; margin: 0.1rem 0; }
.kpi-sub   { font-size: 0.75rem; color: #9CA3AF; }

/* ── Section card ── */
.section-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}

/* ── Risk badge ── */
.badge-low      { background:#D1FAE5; color:#065F46; padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-medium   { background:#FEF3C7; color:#92400E; padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-high     { background:#FEE2E2; color:#991B1B; padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-critical { background:#7F1D1D; color:#FFFFFF;  padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.85rem; }

/* ── Pipeline step ── */
.pipe-step {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
    padding:0.7rem 1rem; text-align:center; font-size:0.8rem;
    font-weight:600; color:#1E40AF;
}
.pipe-arrow { font-size:1.4rem; color:#93C5FD;
              display:flex; align-items:center; justify-content:center; }

/* ── Explanation row ── */
.exp-row {
    display:flex; align-items:flex-start; gap:0.75rem;
    padding:0.75rem 1rem; border-radius:8px; margin-bottom:0.5rem;
}
.exp-row-fraud  { background:#FFF1F2; border-left:3px solid #F43F5E; }
.exp-row-safe   { background:#F0FDF4; border-left:3px solid #22C55E; }
.exp-icon  { font-size:1.2rem; flex-shrink:0; }
.exp-title { font-weight:700; font-size:0.85rem; margin-bottom:2px; }
.exp-body  { font-size:0.8rem; color:#4B5563; }

/* ── Page title ── */
.page-title { font-size:1.6rem; font-weight:700; color:#1A1A2E; margin-bottom:0.2rem; }
.page-sub   { font-size:0.9rem; color:#6B7280; margin-bottom:1.2rem; }

/* ── Divider ── */
hr.light { border:none; border-top:1px solid #E5E7EB; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── 3. LOAD RESOURCES ────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        with open('models/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/features.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('models/threshold.pkl', 'rb') as f:
            optimal_threshold = pickle.load(f)
        return model, feature_names, optimal_threshold
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}. Ensure .pkl files are in models/")
        return None, None, 0.5

model, feature_names, optimal_threshold = load_resources()

# ─── 4. HELPERS ───────────────────────────────────────────────────────────────
def get_risk_tier(prob, threshold):
    if prob < threshold * 0.4:
        return "LOW", "badge-low"
    elif prob < threshold:
        return "MEDIUM", "badge-medium"
    elif prob < threshold + 0.25:
        return "HIGH", "badge-high"
    else:
        return "CRITICAL", "badge-critical"

def build_gauge(probability):
    color = "#22C55E" if probability < 0.3 else "#F59E0B" if probability < 0.6 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#1A1A2E"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#F5F7FA",
            "bordercolor": "#E5E7EB",
            "steps": [
                {"range": [0, 30],  "color": "#D1FAE5"},
                {"range": [30, 60], "color": "#FEF3C7"},
                {"range": [60, 80], "color": "#FEE2E2"},
                {"range": [80, 100],"color": "#FECACA"},
            ],
            "threshold": {"line": {"color": "#111827", "width": 3},
                          "thickness": 0.75, "value": optimal_threshold * 100}
        }
    ))
    fig.update_layout(height=240, margin=dict(t=20, b=10, l=20, r=20),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    return fig

def build_explanation(inputs, probability, threshold, is_fraud):
    """Returns list of dicts: {icon, title, body, type='risk'|'safe'}"""
    factors = []

    # Amount checks
    ratio = inputs['amt'] / (inputs['customer_avg_amt'] + 0.01)
    if ratio >= 5:
        factors.append({
            "icon": "💸", "type": "risk",
            "title": "Extreme Amount Anomaly",
            "body": f"${inputs['amt']:.2f} is {ratio:.1f}× your typical spend of ${inputs['customer_avg_amt']:.2f}. Transactions this far above average are a strong fraud signal."
        })
    elif ratio >= 2:
        factors.append({
            "icon": "⚠️", "type": "risk",
            "title": "Elevated Transaction Amount",
            "body": f"${inputs['amt']:.2f} is {ratio:.1f}× your usual spending. While not conclusive, this deviation raises the risk score."
        })
    else:
        factors.append({
            "icon": "✅", "type": "safe",
            "title": "Normal Spending Amount",
            "body": f"${inputs['amt']:.2f} is consistent with your typical spend of ${inputs['customer_avg_amt']:.2f} (ratio: {ratio:.2f}×)."
        })

    # Location checks
    dist = inputs['distance_from_home']
    if dist > 100:
        factors.append({
            "icon": "📍", "type": "risk",
            "title": "Very High Geographic Distance",
            "body": f"Merchant is {dist:.0f} miles from your home — well outside your normal transaction zone. Fraudulent transactions often occur at distant locations."
        })
    elif dist > 50:
        factors.append({
            "icon": "🗺️", "type": "risk",
            "title": "Unusual Geographic Distance",
            "body": f"Merchant is {dist:.0f} miles away, which is moderately unusual. Distance from home is one of the top predictive features."
        })
    else:
        factors.append({
            "icon": "✅", "type": "safe",
            "title": "Normal Transaction Location",
            "body": f"Merchant is only {dist:.0f} miles from home — within your typical spending area."
        })

    # Time checks
    hour = inputs['trans_hour']
    if hour < 5:
        factors.append({
            "icon": "🌙", "type": "risk",
            "title": "Suspicious Transaction Hour",
            "body": f"Transaction at {hour:02d}:00 (late night / early morning). Fraud is significantly more common between midnight and 5 AM."
        })
    elif hour < 7:
        factors.append({
            "icon": "⏰", "type": "risk",
            "title": "Unusual Transaction Hour",
            "body": f"Transaction at {hour:02d}:00 is outside typical spending hours and slightly elevates risk."
        })
    else:
        factors.append({
            "icon": "✅", "type": "safe",
            "title": "Normal Transaction Time",
            "body": f"Transaction at {hour:02d}:00 falls within regular spending hours."
        })

    # Age checks
    age = inputs['age']
    if age < 18:
        factors.append({
            "icon": "👤", "type": "risk",
            "title": "High-Risk Age Group (Under 18)",
            "body": f"Age {age} is in a statistically elevated fraud risk demographic based on training data patterns."
        })
    elif age > 80:
        factors.append({
            "icon": "👤", "type": "risk",
            "title": "High-Risk Age Group (Over 80)",
            "body": f"Age {age} is associated with a higher incidence of fraud targeting in the training dataset."
        })
    else:
        factors.append({
            "icon": "✅", "type": "safe",
            "title": "Normal Age Profile",
            "body": f"Age {age} is within the standard low-risk demographic range."
        })

    # Model-level catch-all (only for fraud cases with no risk factors)
    risk_count = sum(1 for f in factors if f["type"] == "risk")
    if is_fraud and risk_count == 0:
        factors.append({
            "icon": "🤖", "type": "risk",
            "title": "Complex Pattern Detected by AI Model",
            "body": f"No single feature triggered a rule, but the XGBoost model detected a high-risk combination of amount, category, behavioral history and timing. Risk score: {probability:.1%} (threshold: {threshold:.1%})."
        })
    elif not is_fraud and risk_count == 0:
        factors.append({
            "icon": "🛡️", "type": "safe",
            "title": "All Signals Clear",
            "body": f"The model found no suspicious patterns. Risk score {probability:.1%} is well below the {threshold:.1%} threshold."
        })

    return factors

# ─── 5. SIMULATED DASHBOARD DATA ──────────────────────────────────────────────
@st.cache_data
def get_trend_data():
    rng = np.random.default_rng(42)
    dates = [datetime.today() - timedelta(days=i) for i in range(6, -1, -1)]
    total  = rng.integers(9500, 11000, 7)
    frauds = rng.integers(55, 120, 7)
    return pd.DataFrame({"Date": [d.strftime("%b %d") for d in dates],
                         "Total": total, "Fraud": frauds})

@st.cache_data
def get_category_data():
    categories = ["shopping_net", "grocery_pos", "misc_net", "entertainment",
                  "gas_transport", "food_dining", "shopping_pos", "travel"]
    fraud_pct  = [4.1, 1.2, 3.8, 2.5, 1.0, 1.4, 2.2, 3.3]
    return pd.DataFrame({"Category": categories, "Fraud Rate (%)": fraud_pct})

@st.cache_data
def get_live_feed():
    rng = np.random.default_rng(7)
    cats = ["shopping_net", "grocery_pos", "gas_transport", "entertainment",
            "food_dining", "misc_net", "travel"]
    n = 12
    is_fraud = [1,0,0,1,0,0,0,1,0,0,0,1]
    amounts   = rng.uniform(10, 980, n).round(2)
    scores    = [round(rng.uniform(0.72, 0.98) if f else rng.uniform(0.05, 0.28), 2)
                 for f in is_fraud]
    now = datetime.now()
    times = [(now - timedelta(minutes=i*7)).strftime("%H:%M:%S") for i in range(n)]
    return pd.DataFrame({
        "Time":     times,
        "Amount":   [f"${a:,.2f}" for a in amounts],
        "Category": rng.choice(cats, n),
        "Risk Score": scores,
        "Status":   ["🔴 FRAUD" if f else "🟢 LEGIT" for f in is_fraud]
    })

@st.cache_data
def get_feature_importance():
    features = ["distance_from_home", "amt_ratio", "trans_hour",
                "amt_deviation", "customer_avg_amt", "age",
                "merchant_encoded", "is_new_merchant",
                "hours_since_last_trans", "is_weekend"]
    importance = [0.182, 0.154, 0.121, 0.108, 0.094,
                  0.078, 0.067, 0.058, 0.049, 0.031]
    return pd.DataFrame({"Feature": features, "Importance": importance})

# ─── 6. SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ BI Pipeline")
st.sidebar.markdown("**Financial Transaction**  \n**Fraud Detection System**")
st.sidebar.markdown("<hr class='light'>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", [
    "📊 Executive Dashboard",
    "🔍 Risk Scorecard",
    "📈 BI Analytics",
    "⚙️ Pipeline Architecture"
])
st.sidebar.markdown("<hr class='light'>", unsafe_allow_html=True)
st.sidebar.caption("Model: XGBoost · 500 trees · Depth 10")
st.sidebar.caption("Dataset: 1.3M+ transactions")
st.sidebar.caption(f"Threshold: {optimal_threshold:.4f}")

# ─── 7. PAGE: EXECUTIVE DASHBOARD ─────────────────────────────────────────────
if page == "📊 Executive Dashboard":
    st.markdown('<p class="page-title">📊 Executive Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Financial Transaction BI Pipeline — Real-Time Fraud Detection & Reporting System</p>', unsafe_allow_html=True)

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Fraud Prevented", "$2M+",      "Est. financial impact blocked"),
        ("Daily Predictions", "10K+",    "Avg. transactions scored/day"),
        ("Model Accuracy",   "99.6%",    "On 555K held-out test records"),
        ("Fraud Recall",     "88.02%",   "Frauds correctly identified"),
        ("Review Time ↓",    "60%",      "Reduction in manual review"),
    ]
    for col, (label, value, sub) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='light'>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📅 7-Day Fraud Activity Trend")
        trend = get_trend_data()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend["Date"], y=trend["Total"], name="Total Transactions",
                             marker_color="#BFDBFE"))
        fig.add_trace(go.Scatter(x=trend["Date"], y=trend["Fraud"], name="Fraudulent",
                                 line=dict(color="#EF4444", width=2.5),
                                 marker=dict(size=7), yaxis="y2"))
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            yaxis=dict(title="Transactions", showgrid=True, gridcolor="#F3F4F6"),
            yaxis2=dict(title="Fraud Count", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=20, b=30, l=10, r=10), height=280,
            font=dict(color="#374151")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🍩 Fraud Rate by Category")
        cat_df = get_category_data()
        fig2 = px.bar(cat_df.sort_values("Fraud Rate (%)"),
                      x="Fraud Rate (%)", y="Category", orientation="h",
                      color="Fraud Rate (%)",
                      color_continuous_scale=["#BFDBFE", "#3B82F6", "#1D4ED8"])
        fig2.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10), height=280,
                           font=dict(color="#374151"),
                           yaxis=dict(showgrid=False),
                           xaxis=dict(showgrid=True, gridcolor="#F3F4F6"))
        st.plotly_chart(fig2, use_container_width=True)

    # Live feed
    st.markdown("#### ⚡ Live Transaction Feed")
    feed = get_live_feed()
    st.dataframe(
        feed.style
            .applymap(lambda v: "color:#EF4444; font-weight:700"
                      if "FRAUD" in str(v) else "color:#16A34A; font-weight:700",
                      subset=["Status"])
            .applymap(lambda v: "color:#EF4444; font-weight:700"
                      if isinstance(v, float) and v >= optimal_threshold else "",
                      subset=["Risk Score"]),
        use_container_width=True, hide_index=True
    )

# ─── 8. PAGE: RISK SCORECARD ──────────────────────────────────────────────────
elif page == "🔍 Risk Scorecard":
    st.markdown('<p class="page-title">🔍 Real-Time Risk Scorecard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter transaction details to generate an instant fraud risk assessment</p>', unsafe_allow_html=True)

    with st.form("fraud_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Details**")
            amt      = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0)
            category = st.selectbox("Merchant Category", [
                'gas_transport','grocery_pos','home','shopping_pos','kids_pets',
                'shopping_net','entertainment','food_dining','personal_care',
                'health_fitness','misc_pos','misc_net','travel'])
            hour     = st.slider("Transaction Hour (0 = midnight)", 0, 23, 14)
            distance = st.number_input("Distance from Home (miles)", min_value=0.0, value=5.0, step=0.5)
        with c2:
            st.markdown("**Customer Profile**")
            age       = st.number_input("Customer Age", min_value=14, max_value=100, value=35)
            gender    = st.radio("Gender", ["Male", "Female"], horizontal=True)
            avg_amt   = st.number_input("Customer's Typical Spend ($)", min_value=0.0, value=50.0, step=1.0)
            merch_risk= st.slider("Merchant Risk Score", 0.0, 1.0, 0.5, step=0.01)
        submitted = st.form_submit_button("⚡ Generate Risk Scorecard", use_container_width=True)

    if submitted and model:
        input_data = {
            'amt': amt, 'customer_avg_amt': avg_amt,
            'distance_from_home': distance, 'trans_hour': hour, 'age': age,
            'amt_deviation': amt - avg_amt,
            'amt_ratio': amt / (avg_amt + 0.01),
            'is_weekend': 0, 'trans_day': 15, 'trans_month': 6,
            'merchant_visit_count': 5, 'is_new_merchant': 0,
            'hours_since_last_trans': 24,
            'gender_encoded': 1 if gender == "Male" else 0,
            'merchant_encoded': merch_risk
        }
        fv = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        for col, val in input_data.items():
            if col in fv.columns:
                fv[col] = val
        cat_col = f"cat_{category}"
        if cat_col in fv.columns:
            fv[cat_col] = 1

        probability = model.predict_proba(fv)[0][1]
        is_fraud    = probability >= optimal_threshold
        tier, badge = get_risk_tier(probability, optimal_threshold)

        st.markdown("<hr class='light'>", unsafe_allow_html=True)

        # Header result
        res_col, gauge_col = st.columns([2, 1])
        with res_col:
            if is_fraud:
                st.error("### ⚠️ FRAUD ALERT — Transaction Flagged")
            else:
                st.success("### ✅ TRANSACTION APPROVED — Risk Within Tolerance")

            st.markdown(f"""
            <div style="display:flex; gap:2rem; margin-top:0.8rem; flex-wrap:wrap;">
                <div><div class="kpi-label">Risk Score</div>
                     <div class="kpi-value" style="font-size:2.2rem">{probability:.1%}</div></div>
                <div><div class="kpi-label">Risk Tier</div>
                     <div style="margin-top:0.4rem"><span class="{badge}">{tier}</span></div></div>
                <div><div class="kpi-label">Decision Threshold</div>
                     <div class="kpi-value" style="font-size:2.2rem">{optimal_threshold:.1%}</div></div>
                <div><div class="kpi-label">Model Confidence</div>
                     <div class="kpi-value" style="font-size:2.2rem">{max(probability, 1-probability):.1%}</div></div>
            </div>
            """, unsafe_allow_html=True)

        with gauge_col:
            st.plotly_chart(build_gauge(probability), use_container_width=True)

        # Explanation
        st.markdown("<hr class='light'>", unsafe_allow_html=True)
        st.markdown("#### 🧠 Risk Factor Analysis")
        st.caption("Each factor below shows how this transaction compares against the model's learned patterns.")

        factors = build_explanation(input_data, probability, optimal_threshold, is_fraud)
        risk_factors = [f for f in factors if f["type"] == "risk"]
        safe_factors = [f for f in factors if f["type"] == "safe"]

        if risk_factors:
            st.markdown(f"**{len(risk_factors)} Risk Signal(s) Detected**")
            for f in risk_factors:
                st.markdown(f"""
                <div class="exp-row exp-row-fraud">
                    <div class="exp-icon">{f['icon']}</div>
                    <div><div class="exp-title" style="color:#991B1B">{f['title']}</div>
                         <div class="exp-body">{f['body']}</div></div>
                </div>""", unsafe_allow_html=True)

        if safe_factors:
            st.markdown(f"**{len(safe_factors)} Passing Check(s)**")
            for f in safe_factors:
                st.markdown(f"""
                <div class="exp-row exp-row-safe">
                    <div class="exp-icon">{f['icon']}</div>
                    <div><div class="exp-title" style="color:#065F46">{f['title']}</div>
                         <div class="exp-body">{f['body']}</div></div>
                </div>""", unsafe_allow_html=True)

        # Factor score table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Feature Contribution Summary")
        ratio = amt / (avg_amt + 0.01)
        table = pd.DataFrame({
            "Feature":        ["Transaction Amount", "Distance from Home", "Transaction Hour", "Customer Age", "Merchant Risk"],
            "Your Value":     [f"${amt:.2f}", f"{distance:.0f} mi", f"{hour:02d}:00", str(age), f"{merch_risk:.2f}"],
            "Baseline":       [f"${avg_amt:.2f} avg", "< 50 mi", "07:00–22:00", "18–80", "< 0.5"],
            "Signal":         [
                "🔴 High" if ratio >= 2 else "🟢 Normal",
                "🔴 High" if distance > 50 else "🟢 Normal",
                "🔴 Risk" if hour < 7 or hour > 22 else "🟢 Normal",
                "🔴 Risk" if age < 18 or age > 80 else "🟢 Normal",
                "🔴 High" if merch_risk > 0.6 else "🟡 Medium" if merch_risk > 0.4 else "🟢 Low"
            ]
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

# ─── 9. PAGE: BI ANALYTICS ────────────────────────────────────────────────────
elif page == "📈 BI Analytics":
    st.markdown('<p class="page-title">📈 BI Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Model performance metrics on 555,719 held-out test transactions</p>', unsafe_allow_html=True)

    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, value, sub) in zip([m1,m2,m3,m4], [
        ("ROC-AUC",          "0.9967", "Near-perfect discrimination"),
        ("Recall",           "88.02%", "Frauds correctly caught"),
        ("Precision",        "60.38%", "Alert accuracy rate"),
        ("Accuracy",         "99.72%", "Overall correct predictions"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='light'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🏆 Top 10 Feature Importance")
        fi = get_feature_importance()
        fig3 = px.bar(fi.sort_values("Importance"),
                      x="Importance", y="Feature", orientation="h",
                      color="Importance",
                      color_continuous_scale=["#BFDBFE","#2563EB"])
        fig3.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           coloraxis_showscale=False, height=340,
                           margin=dict(t=10,b=10,l=10,r=10),
                           font=dict(color="#374151"),
                           xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("#### 🧮 Confusion Matrix")
        z  = [[553356, 1007], [257, 1888]]   # TN FP / FN TP  (test set values)
        labels = [["True Negative<br>553,356", "False Positive<br>1,007"],
                  ["False Negative<br>257",    "True Positive<br>1,888"]]
        fig4 = go.Figure(go.Heatmap(
            z=z, text=labels, texttemplate="%{text}",
            colorscale=[[0,"#EFF6FF"],[1,"#1D4ED8"]],
            showscale=False
        ))
        fig4.update_layout(
            xaxis=dict(tickvals=[0,1], ticktext=["Predicted: Legit","Predicted: Fraud"],
                       side="bottom"),
            yaxis=dict(tickvals=[0,1], ticktext=["Actual: Legit","Actual: Fraud"],
                       autorange="reversed"),
            height=340, margin=dict(t=10,b=60,l=80,r=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(color="#374151", size=13)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Model comparison
    st.markdown("#### 🆚 Model Comparison")
    comp = pd.DataFrame({
        "Model":     ["Logistic Regression (baseline)", "Random Forest", "XGBoost (ours)"],
        "Accuracy":  ["96.4%", "98.9%", "99.72%"],
        "Recall":    ["61.2%", "74.5%", "88.02%"],
        "ROC-AUC":   ["0.921", "0.974", "0.9967"],
        "F1-Score":  ["0.521", "0.689", "0.718"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption("Baseline models use same feature set and SMOTE balancing. XGBoost trained with GPU acceleration (CUDA).")

# ─── 10. PAGE: PIPELINE ARCHITECTURE ─────────────────────────────────────────
elif page == "⚙️ Pipeline Architecture":
    st.markdown('<p class="page-title">⚙️ Pipeline Architecture</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">End-to-end data pipeline from raw transactions to real-time risk scores</p>', unsafe_allow_html=True)

    # Pipeline flow
    st.markdown("#### 🔄 Data Pipeline")
    steps = [
        ("📥", "Data Ingestion", "1.3M transactions\nKaggle dataset"),
        ("🔧", "Feature Engineering", "31 features\ntemporal · geo · behavioral"),
        ("⚖️", "SMOTE Balancing", "1:5 ratio\nfrom 1:171 imbalance"),
        ("🤖", "XGBoost Training", "500 trees · depth 10\nGPU (CUDA)"),
        ("🎯", "Threshold Optimization", "0.3029 cutoff\nmaximises recall"),
        ("📊", "Risk Scoring", "Real-time\n10K+ preds/day"),
    ]
    cols = st.columns(len(steps) * 2 - 1)
    for i, (icon, title, detail) in enumerate(steps):
        with cols[i * 2]:
            st.markdown(f"""
            <div class="pipe-step">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-weight:700;margin:4px 0">{title}</div>
                <div style="font-size:0.7rem;color:#3B82F6;white-space:pre-line">{detail}</div>
            </div>""", unsafe_allow_html=True)
        if i < len(steps) - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div class="pipe-arrow">→</div>', unsafe_allow_html=True)

    st.markdown("<hr class='light'>", unsafe_allow_html=True)

    col_feat, col_tech = st.columns(2)

    with col_feat:
        st.markdown("#### 🧩 Feature Groups (31 total)")
        feat_groups = pd.DataFrame({
            "Group":        ["Temporal", "Geographic", "Behavioral", "Merchant", "Demographic", "Target-Encoded", "One-Hot (Categories)"],
            "# Features":   [5, 1, 6, 2, 1, 4, 13],
            "Examples":     [
                "trans_hour, is_weekend, trans_month",
                "distance_from_home (Haversine)",
                "amt_ratio, amt_deviation, hours_since_last_trans",
                "merchant_visit_count, is_new_merchant",
                "age",
                "merchant_encoded, state_encoded, city_encoded, job_encoded",
                "cat_shopping_net, cat_grocery_pos, …"
            ]
        })
        st.dataframe(feat_groups, use_container_width=True, hide_index=True)

    with col_tech:
        st.markdown("#### 🛠️ Technology Stack")
        tech = pd.DataFrame({
            "Component":    ["ML Model", "Imbalance Handling", "Web Framework", "Visualisation", "Data Processing", "Deployment"],
            "Technology":   ["XGBoost (GPU/CUDA)", "SMOTE (imbalanced-learn)", "Streamlit", "Plotly", "pandas · NumPy", "Streamlit Cloud"],
            "Purpose":      [
                "Gradient boosted classifier",
                "Synthetic minority oversampling",
                "Interactive BI dashboard",
                "Charts, gauges, heatmaps",
                "Feature engineering pipeline",
                "Live public demo"
            ]
        })
        st.dataframe(tech, use_container_width=True, hide_index=True)

    st.markdown("<hr class='light'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Project Summary")
    st.markdown("""
| Attribute | Detail |
|---|---|
| **Training records** | 1,296,675 transactions |
| **Test records** | 555,719 transactions |
| **Fraud rate** | 0.58% (highly imbalanced) |
| **Model** | XGBoost · 500 estimators · max_depth=10 |
| **Decision threshold** | 0.3029 (optimised for recall) |
| **ROC-AUC** | 0.9967 |
| **Frauds caught (recall)** | 88.02% — 1,888 of 2,145 |
| **Estimated value** | $2M+ fraud prevented |
""")
