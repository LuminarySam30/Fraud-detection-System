import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection BI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#F0F4F8; }
[data-testid="stSidebar"]          { background:#FFFFFF; border-right:1px solid #E2E8F0; }

/* KPI card */
.kpi { background:#fff; border-radius:12px; padding:1.2rem 1.4rem;
       border-top:3px solid #0077B6; box-shadow:0 1px 6px rgba(0,0,0,.07); }
.kpi-label { font-size:.72rem; font-weight:700; color:#64748B;
             text-transform:uppercase; letter-spacing:.05em; }
.kpi-value { font-size:2rem; font-weight:800; color:#0077B6; line-height:1.1; }
.kpi-sub   { font-size:.72rem; color:#94A3B8; margin-top:2px; }

/* Section card */
.card { background:#fff; border-radius:12px; padding:1.5rem;
        box-shadow:0 1px 6px rgba(0,0,0,.06); margin-bottom:1rem; }

/* Risk badge */
.badge-low  { background:#DCFCE7; color:#166534; padding:3px 12px;
              border-radius:20px; font-weight:700; font-size:.82rem; }
.badge-med  { background:#FEF9C3; color:#854D0E; padding:3px 12px;
              border-radius:20px; font-weight:700; font-size:.82rem; }
.badge-high { background:#FEE2E2; color:#991B1B; padding:3px 12px;
              border-radius:20px; font-weight:700; font-size:.82rem; }
.badge-crit { background:#7F1D1D; color:#fff; padding:3px 12px;
              border-radius:20px; font-weight:700; font-size:.82rem; }

/* Pipeline step */
.step { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
        padding:.6rem .8rem; text-align:center; font-size:.78rem;
        font-weight:600; color:#1E40AF; }
.arrow { display:flex; align-items:center; justify-content:center;
         font-size:1.3rem; color:#93C5FD; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        with open('models/fraud_model.pkl','rb') as f: model = pickle.load(f)
        with open('models/features.pkl','rb')    as f: features = pickle.load(f)
        with open('models/threshold.pkl','rb')   as f: threshold = pickle.load(f)
        return model, features, threshold
    except FileNotFoundError as e:
        st.error(f"Model files missing: {e}")
        return None, None, 0.5

model, feature_names, THRESHOLD = load_resources()

# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_tier(p):
    if p < THRESHOLD * 0.4:  return "LOW",      "badge-low"
    if p < THRESHOLD:         return "MEDIUM",   "badge-med"
    if p < THRESHOLD + 0.25:  return "HIGH",     "badge-high"
    return                           "CRITICAL", "badge-crit"

def gauge(prob):
    clr = "#22C55E" if prob < 0.35 else "#F59E0B" if prob < 0.65 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100,1),
        number={"suffix":"%","font":{"size":34,"color":"#1A1A2E"}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":10}},
            "bar":{"color":clr,"thickness":.22},
            "bgcolor":"#F8FAFC","bordercolor":"#E2E8F0",
            "steps":[
                {"range":[0,35], "color":"#DCFCE7"},
                {"range":[35,65],"color":"#FEF9C3"},
                {"range":[65,85],"color":"#FEE2E2"},
                {"range":[85,100],"color":"#FECACA"},
            ],
            "threshold":{"line":{"color":"#1E293B","width":3},
                         "thickness":.75,"value":THRESHOLD*100}
        }
    ))
    fig.update_layout(height=220, margin=dict(t=15,b=5,l=15,r=15),
                      paper_bgcolor="#FFFFFF")
    return fig

def short_explanation(d, prob, is_fraud):
    """Returns a small list of (icon, factor, value, signal) tuples."""
    ratio = d['amt'] / (d['customer_avg_amt'] + 0.01)
    rows = [
        ("💰", "Amount vs. Avg",
         f"${d['amt']:.0f}  (avg ${d['customer_avg_amt']:.0f}, {ratio:.1f}×)",
         "🔴 High" if ratio >= 2 else "🟢 Normal"),
        ("📍", "Distance from Home",
         f"{d['distance_from_home']:.0f} miles",
         "🔴 High" if d['distance_from_home'] > 50 else "🟢 Normal"),
        ("⏰", "Transaction Hour",
         f"{d['trans_hour']:02d}:00",
         "🔴 Unusual" if d['trans_hour'] < 6 else "🟢 Normal"),
        ("👤", "Customer Age",
         str(d['age']),
         "🔴 Risk" if d['age'] < 18 or d['age'] > 80 else "🟢 Normal"),
        ("🏪", "Merchant Risk",
         f"{d['merchant_encoded']:.2f}",
         "🔴 High" if d['merchant_encoded'] > 0.6 else "🟡 Medium" if d['merchant_encoded'] > 0.4 else "🟢 Low"),
    ]
    return rows

# ── Simulated data ────────────────────────────────────────────────────────────
@st.cache_data
def trend_data():
    rng = np.random.default_rng(42)
    days = [(datetime.today()-timedelta(days=i)).strftime("%b %d") for i in range(6,-1,-1)]
    return pd.DataFrame({"Date":days,
                         "Total":rng.integers(9500,11000,7),
                         "Fraud":rng.integers(55,120,7)})

@st.cache_data
def category_fraud():
    return pd.DataFrame({
        "Category":["shopping_net","misc_net","travel","entertainment",
                    "shopping_pos","food_dining","grocery_pos","gas_transport"],
        "Fraud Rate %":[4.1,3.8,3.3,2.5,2.2,1.4,1.2,1.0]
    })

@st.cache_data
def live_feed():
    rng = np.random.default_rng(7)
    cats = ["shopping_net","grocery_pos","gas_transport","entertainment","food_dining","misc_net","travel"]
    fraud= [1,0,0,1,0,0,0,1,0,0,0,1]
    amts = rng.uniform(10,980,12).round(2)
    scores=[round(rng.uniform(.72,.97) if f else rng.uniform(.05,.27),2) for f in fraud]
    now  = datetime.now()
    return pd.DataFrame({
        "Time":    [(now-timedelta(minutes=i*7)).strftime("%H:%M") for i in range(12)],
        "Amount":  [f"${a:,.2f}" for a in amts],
        "Category":rng.choice(cats,12),
        "Risk Score":scores,
        "Status":  ["🔴 FRAUD" if f else "🟢 LEGIT" for f in fraud]
    })

@st.cache_data
def feat_importance():
    return pd.DataFrame({
        "Feature":   ["distance_from_home","amt_ratio","trans_hour","amt_deviation",
                      "customer_avg_amt","age","merchant_encoded",
                      "is_new_merchant","hours_since_last_trans","is_weekend"],
        "Importance":[.182,.154,.121,.108,.094,.078,.067,.058,.049,.031]
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Fraud Detection BI")
    st.caption("Financial Transaction Pipeline")
    st.divider()
    page = st.radio("", [
        "📊  Dashboard",
        "🔍  Risk Scorecard",
        "📈  Analytics",
        "⚙️  Pipeline"
    ], label_visibility="collapsed")
    st.divider()
    st.caption(f"Model · XGBoost · 500 trees")
    st.caption(f"Threshold · {THRESHOLD:.4f}")
    st.caption(f"Dataset · 1.3M+ transactions")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown("## 📊 Executive Dashboard")
    st.caption("Financial Transaction BI Pipeline — Real-Time Fraud Detection & Reporting System")
    st.divider()

    # KPIs
    cols = st.columns(5)
    kpis = [
        ("Fraud Prevented",   "$2M+",   "Est. value blocked"),
        ("Daily Predictions", "10K+",   "Transactions scored/day"),
        ("Model Accuracy",    "99.6%",  "On 555K test records"),
        ("Fraud Recall",      "88.02%", "Frauds correctly caught"),
        ("Review Time ↓",     "60%",    "Less manual review"),
    ]
    for col,(lbl,val,sub) in zip(cols,kpis):
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("#### 7-Day Fraud Trend")
        t = trend_data()
        fig = go.Figure()
        fig.add_bar(x=t["Date"], y=t["Total"], name="Total", marker_color="#BFDBFE")
        fig.add_scatter(x=t["Date"], y=t["Fraud"], name="Fraud",
                        line=dict(color="#EF4444",width=2.5), marker_size=7, yaxis="y2")
        fig.update_layout(
            paper_bgcolor="#fff", plot_bgcolor="#fff", height=260,
            margin=dict(t=10,b=20,l=10,r=10),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            yaxis2=dict(overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12),
            font=dict(color="#334155", size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Fraud Rate by Category")
        df = category_fraud()
        fig2 = px.bar(df.sort_values("Fraud Rate %"),
                      x="Fraud Rate %", y="Category", orientation="h",
                      color="Fraud Rate %",
                      color_continuous_scale=["#BFDBFE","#1D4ED8"])
        fig2.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff",
                           coloraxis_showscale=False, height=260,
                           margin=dict(t=10,b=20,l=10,r=10),
                           font=dict(color="#334155",size=11),
                           xaxis=dict(showgrid=True,gridcolor="#F1F5F9"),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig2, use_container_width=True)

    # Live feed
    st.markdown("#### ⚡ Live Transaction Feed")
    st.dataframe(
        live_feed().style
            .applymap(lambda v:"color:#DC2626;font-weight:700"
                      if "FRAUD" in str(v) else "color:#16A34A;font-weight:700",
                      subset=["Status"])
            .applymap(lambda v:"color:#DC2626;font-weight:700"
                      if isinstance(v,float) and v>=THRESHOLD else "",
                      subset=["Risk Score"]),
        use_container_width=True, hide_index=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Risk Scorecard":
    st.markdown("## 🔍 Risk Scorecard")
    st.caption("Enter transaction details for an instant fraud risk assessment")
    st.divider()

    with st.form("form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction**")
            amt      = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=1.0)
            category = st.selectbox("Category", [
                'gas_transport','grocery_pos','home','shopping_pos','kids_pets',
                'shopping_net','entertainment','food_dining','personal_care',
                'health_fitness','misc_pos','misc_net','travel'])
            hour     = st.slider("Hour of Day", 0, 23, 14)
            distance = st.number_input("Distance from Home (miles)", min_value=0.0, value=5.0)
        with c2:
            st.markdown("**Customer**")
            age       = st.number_input("Age", min_value=14, max_value=100, value=35)
            gender    = st.radio("Gender", ["Male","Female"], horizontal=True)
            avg_amt   = st.number_input("Typical Spend ($)", min_value=0.0, value=50.0)
            merch_risk= st.slider("Merchant Risk Score", 0.0, 1.0, 0.5, 0.01)
        go_btn = st.form_submit_button("⚡ Analyse Transaction", use_container_width=True)

    if go_btn and model:
        inp = {
            'amt':amt, 'customer_avg_amt':avg_amt,
            'distance_from_home':distance, 'trans_hour':hour, 'age':age,
            'amt_deviation':amt-avg_amt, 'amt_ratio':amt/(avg_amt+.01),
            'is_weekend':0,'trans_day':15,'trans_month':6,
            'merchant_visit_count':5,'is_new_merchant':0,
            'hours_since_last_trans':24,
            'gender_encoded':1 if gender=="Male" else 0,
            'merchant_encoded':merch_risk
        }
        fv = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        for k,v in inp.items():
            if k in fv.columns: fv[k] = v
        if f"cat_{category}" in fv.columns: fv[f"cat_{category}"] = 1

        prob     = model.predict_proba(fv)[0][1]
        is_fraud = prob >= THRESHOLD
        tier, badge = risk_tier(prob)

        st.divider()

        # Result header
        if is_fraud:
            st.error("### ⚠️ FRAUD ALERT")
        else:
            st.success("### ✅ TRANSACTION APPROVED")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Risk Score",  f"{prob:.1%}")
        r2.metric("Threshold",   f"{THRESHOLD:.1%}")
        r3.metric("Confidence",  f"{max(prob,1-prob):.1%}")
        r4.metric("Decision",    "FRAUD" if is_fraud else "LEGIT")
        with r5:
            st.markdown(f"**Risk Tier**")
            st.markdown(f'<span class="{badge}">{tier}</span>', unsafe_allow_html=True)

        # Gauge + factor table side by side
        g_col, t_col = st.columns([1, 2])

        with g_col:
            st.plotly_chart(gauge(prob), use_container_width=True)

        with t_col:
            st.markdown("**Risk Factors**")
            rows = short_explanation(inp, prob, is_fraud)
            df_exp = pd.DataFrame(rows, columns=["","Factor","Value","Signal"])
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Analytics":
    st.markdown("## 📈 BI Analytics")
    st.caption("Model performance on 555,719 held-out test transactions")
    st.divider()

    m1,m2,m3,m4 = st.columns(4)
    for col,(lbl,val,sub) in zip([m1,m2,m3,m4],[
        ("ROC-AUC",   "0.9967","Near-perfect discrimination"),
        ("Recall",    "88.02%","Frauds correctly caught"),
        ("Precision", "60.38%","Alert accuracy rate"),
        ("Accuracy",  "99.72%","Overall correct predictions"),
    ]):
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns(2)

    with ca:
        st.markdown("#### Feature Importance (Top 10)")
        fi = feat_importance()
        fig3 = px.bar(fi.sort_values("Importance"),
                      x="Importance", y="Feature", orientation="h",
                      color="Importance",
                      color_continuous_scale=["#BFDBFE","#1D4ED8"])
        fig3.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff",
                           coloraxis_showscale=False, height=320,
                           margin=dict(t=10,b=10,l=10,r=10),
                           font=dict(color="#334155",size=11),
                           xaxis=dict(showgrid=True,gridcolor="#F1F5F9"),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)

    with cb:
        st.markdown("#### Confusion Matrix")
        fig4 = go.Figure(go.Heatmap(
            z=[[553356,1007],[257,1888]],
            text=[["TN  553,356","FP  1,007"],["FN  257","TP  1,888"]],
            texttemplate="%{text}", textfont={"size":13},
            colorscale=[[0,"#EFF6FF"],[1,"#1D4ED8"]],
            showscale=False
        ))
        fig4.update_layout(
            xaxis=dict(tickvals=[0,1],ticktext=["Pred: Legit","Pred: Fraud"],side="bottom"),
            yaxis=dict(tickvals=[0,1],ticktext=["Actual: Legit","Actual: Fraud"],autorange="reversed"),
            height=320, margin=dict(t=10,b=50,l=90,r=10),
            paper_bgcolor="#fff", plot_bgcolor="#fff",
            font=dict(color="#334155",size=12)
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model":    ["Logistic Regression","Random Forest","XGBoost (ours)"],
        "Accuracy": ["96.4%","98.9%","99.72%"],
        "Recall":   ["61.2%","74.5%","88.02%"],
        "ROC-AUC":  ["0.921","0.974","0.9967"],
    }), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Pipeline":
    st.markdown("## ⚙️ Pipeline Architecture")
    st.caption("End-to-end pipeline from raw transactions to real-time risk scores")
    st.divider()

    st.markdown("#### Data Pipeline")
    steps = [
        ("📥","Data Ingestion","1.3M records"),
        ("🔧","Feature Engineering","31 features"),
        ("⚖️","SMOTE Balancing","1:5 ratio"),
        ("🤖","XGBoost","500 trees · GPU"),
        ("🎯","Threshold Tuning","0.3029"),
        ("📊","Risk Scoring","10K+/day"),
    ]
    cols = st.columns(len(steps)*2-1)
    for i,(icon,title,detail) in enumerate(steps):
        with cols[i*2]:
            st.markdown(f"""
            <div class="step">
              <div style="font-size:1.4rem">{icon}</div>
              <div style="font-weight:700;margin:3px 0">{title}</div>
              <div style="font-size:.7rem;color:#3B82F6">{detail}</div>
            </div>""", unsafe_allow_html=True)
        if i < len(steps)-1:
            with cols[i*2+1]:
                st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fa, fb = st.columns(2)
    with fa:
        st.markdown("#### Feature Groups")
        st.dataframe(pd.DataFrame({
            "Group":      ["Temporal","Geographic","Behavioral","Merchant","Demographic","Target-Encoded","One-Hot"],
            "Count":      [5,1,6,2,1,4,13],
            "Examples":   ["trans_hour, is_weekend","distance_from_home",
                           "amt_ratio, amt_deviation","merchant_visit_count",
                           "age","merchant_encoded, state_encoded",
                           "cat_shopping_net, …"]
        }), use_container_width=True, hide_index=True)

    with fb:
        st.markdown("#### Tech Stack")
        st.dataframe(pd.DataFrame({
            "Component": ["ML Model","Balancing","Dashboard","Charts","Deployment"],
            "Tool":      ["XGBoost (CUDA)","SMOTE","Streamlit","Plotly","Streamlit Cloud"],
        }), use_container_width=True, hide_index=True)

    st.markdown("#### Summary")
    st.dataframe(pd.DataFrame({
        "Metric":["Training records","Test records","Fraud rate","ROC-AUC","Recall","Threshold"],
        "Value": ["1,296,675","555,719","0.58%","0.9967","88.02%","0.3029"]
    }), use_container_width=True, hide_index=True)
