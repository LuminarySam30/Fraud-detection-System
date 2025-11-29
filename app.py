# INTELLIGENT CREDIT CARD FRAUD DETECTION SYSTEM
# Streamlit Web Application for CS5100 Final Project

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configure page
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏦",
    layout="wide"
)

# Load model resources
@st.cache_resource
def load_resources():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return model, features, threshold

model, feature_names, optimal_threshold = load_resources()

# Sidebar navigation
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🔍 Fraud Checker", "📊 Performance", "ℹ️ About"])

# HOME PAGE
if page == "🏠 Home":
    st.title("🏦 Intelligent Credit Card Fraud Detection System")
    st.markdown("---")
    
    st.markdown("### Welcome to the AI-Powered Fraud Detection System")
    st.write("Detect fraudulent credit card transactions in real-time with machine learning.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Recall", "88.4%")
    with col2:
        st.metric("Frauds Caught", "1,888 / 2,145")
    with col3:
        st.metric("Features Used", len(feature_names))
    with col4:
        st.metric("Threshold", f"{optimal_threshold:.4f}")
    
    st.markdown("---")
    st.subheader("📖 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**🔍 Fraud Checker**\n\nEnter transaction details and get instant fraud prediction with AI explanation")
    
    with col2:
        st.info("**📊 Performance**\n\nView model performance metrics and statistics")

# FRAUD CHECKER PAGE
elif page == "🔍 Fraud Checker":
    st.title("🔍 Credit Card Fraud Detection")
    st.markdown("Enter transaction details below to check for potential fraud")
    st.markdown("---")
    
    with st.form("fraud_check_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Info")
            amt = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
            
            category = st.selectbox("Category", [
                'gas_transport', 'grocery_pos', 'home', 'shopping_pos',
                'kids_pets', 'shopping_net', 'entertainment', 'food_dining',
                'personal_care', 'health_fitness', 'misc_pos', 'travel',
                'misc_net', 'grocery_net'
            ])
            
            trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
            distance = st.number_input("Distance from Home (miles)", min_value=0.0, value=10.0, step=5.0)
        
        with col2:
            st.subheader("Customer Info")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.radio("Gender", ["Male", "Female"])
            customer_avg = st.number_input("Typical Spending ($)", min_value=0.0, value=75.0, step=5.0)
            merchant_risk = st.slider("Merchant Risk Score (0-1)", 0.0, 0.5, 0.01, step=0.01)
        
        submitted = st.form_submit_button("🔍 Check Transaction", use_container_width=True)
    
    if submitted:
        st.markdown("---")
        
        # Calculate derived features
        gender_encoded = 1 if gender == "Male" else 0
        amt_deviation = amt - customer_avg
        amt_ratio = amt / (customer_avg + 0.01)
        category_avg = 70.0
        amt_dev_category = amt - category_avg
        
        # Create category one-hot encoding
        cat_features = {f'cat_{c}': 0 for c in [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net',
            'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net',
            'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel'
        ]}
        cat_features[f'cat_{category}'] = 1
        
        # Build feature vector matching training
        features_dict = {
            'amt': amt,
            'city_pop': 50000,
            'unix_time': 1577836800,
            'trans_hour': trans_hour,
            'trans_day_of_week': 1,
            'is_weekend': 0,
            'trans_day': 15,
            'trans_month': 6,
            'distance_from_home': distance,
            'customer_avg_amt': customer_avg,
            'amt_deviation': amt_deviation,
            'amt_ratio': amt_ratio,
            'category_avg_amt': category_avg,
            'amt_deviation_from_category': amt_dev_category,
            'merchant_visit_count': 1,
            'is_new_merchant': 0,
            'age': age,
            'gender_encoded': gender_encoded,
            **cat_features,
            'merchant_encoded': merchant_risk,
            'state_encoded': 0.01,
            'city_encoded': 0.01,
            'job_encoded': 0.01,
            'hours_since_last_trans': 6.8
        }
        
        # Create dataframe in correct feature order
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[feature_names]
        
        # Predict
        fraud_prob = model.predict_proba(input_df)[0][1]
        is_fraud = fraud_prob >= optimal_threshold
        
        # Display result
        if is_fraud:
            st.error("⚠️ FRAUD DETECTED!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
            with col2:
                risk = "HIGH" if fraud_prob > 0.7 else "MEDIUM"
                st.metric("Risk Level", risk)
        else:
            st.success("✅ TRANSACTION APPROVED")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
            with col2:
                st.metric("Risk Level", "LOW")
        
        # AI Explanation
        st.markdown("---")
        st.subheader("💬 Why was this flagged?")
        
        reasons = []
        
        if amt_ratio > 3:
            reasons.append(f"🚨 Amount ${amt:.2f} is {amt_ratio:.1f}x higher than typical spending of ${customer_avg:.2f}")
        
        if distance > 100:
            reasons.append(f"📍 Merchant located {distance:.0f} miles from home (unusually far)")
        
        if trans_hour < 6 or trans_hour > 22:
            reasons.append(f"⏰ Transaction at {trans_hour}:00 is outside normal hours (6 AM - 10 PM)")
        
        if merchant_risk > 0.05:
            reasons.append(f"⚠️ Merchant has {merchant_risk*100:.1f}% historical fraud rate (above average)")
        
        if amt > 500:
            reasons.append(f"💰 High-value transaction (${amt:.2f}) requires additional scrutiny")
        
        if len(reasons) > 0:
            for reason in reasons:
                st.markdown(f"- {reason}")
        else:
            st.success("✅ All indicators appear normal for this transaction")

# PERFORMANCE PAGE
elif page == "📊 Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recall", "88.02%", help="Frauds successfully detected")
    with col2:
        st.metric("Precision", "60.38%", help="Accuracy of fraud predictions")
    with col3:
        st.metric("Frauds Caught", "1,888 / 2,145")
    with col4:
        st.metric("ROC-AUC", "0.9966")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Configuration")
        st.markdown("""
        - **Algorithm:** XGBoost Classifier
        - **Trees:** 500
        - **Max Depth:** 10
        - **Learning Rate:** 0.05
        - **Training Samples:** 1,547,002 (with SMOTE)
        - **Testing Samples:** 555,719
        """)
    
    with col2:
        st.subheader("📈 Key Metrics")
        st.markdown(f"""
        - **Optimal Threshold:** {optimal_threshold:.4f}
        - **Features Engineered:** {len(feature_names)}
        - **Training Time:** ~2 minutes
        - **Fraud Detection Rate:** 88.02%
        - **False Positives:** 1,239 transactions
        """)
    
    st.markdown("---")
    st.subheader("🔑 Most Important Features")
    
    st.markdown("""
    Based on model analysis, these features have highest impact:
    
    1. **amt_ratio** - Transaction amount compared to customer average
    2. **distance_from_home** - Geographic distance from customer location
    3. **merchant_encoded** - Historical merchant fraud rate
    4. **trans_hour** - Time of day pattern
    5. **customer_avg_amt** - Customer spending baseline
    6. **age** - Customer demographic
    7. **category** - Transaction type
    """)

# ABOUT PAGE
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ### Intelligent Credit Card Fraud Detection System
    
    **Course:** CS5100 - Foundations of Artificial Intelligence  
    **Project Type:** Application Project - Machine Learning Pipeline
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Dataset")
    st.markdown("""
    - **Source:** Kaggle - Credit Card Transactions Fraud Detection Dataset
    - **Author:** Kartik Shenoy
    - **Training Samples:** 1,296,675 transactions
    - **Testing Samples:** 555,719 transactions
    - **Features:** 22 original → 31 engineered features
    - **Class Balance:** 0.58% fraud (highly imbalanced)
    """)
    
    st.markdown("---")
    
    st.subheader("🔬 Methodology")
    
    st.markdown("""
    **1. Feature Engineering:**
    - Temporal features (hour, day, month)
    - Geographic distance (Haversine formula)
    - Spending behavior (deviation, ratio)
    - Merchant familiarity
    - Demographic encoding
    
    **2. Class Balancing:**
    - Applied SMOTE (Synthetic Minority Over-sampling)
    - Balanced from 1:100 to 1:5 ratio
    
    **3. Model Training:**
    - Algorithm: XGBoost with 500 trees
    - Max depth: 10 levels
    - Threshold optimization for 88%+ recall
    
    **4. Performance:**
    - Recall: 88.02% (catches 1,888 of 2,145 frauds)
    - Precision: 60.38%
    - ROC-AUC: 0.9966 (near-perfect)
    """)
    
    st.markdown("---")
    
    st.subheader("🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning:**
        - Python 3.10
        - XGBoost
        - Scikit-learn
        - Imbalanced-learn
        - Pandas & NumPy
        """)
    
    with col2:
        st.markdown("""
        **Web Application:**
        - Streamlit 1.51.0
        - Plotly (visualizations)
        - Rule-based AI explanations
        """)
    
    st.markdown("---")
    st.success("✅ This system demonstrates production-ready fraud detection with 88% recall rate")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**CS5100 Final Project**\n\nFoundations of Artificial Intelligence")