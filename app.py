import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from math import radians, sin, cos, sqrt, atan2

# 1. SETUP & CONFIGURATION
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better 
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD MODELS & ASSETS
@st.cache_resource
def load_resources():
    try:
        # Load Model
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load Features List
        with open('features.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        # Load Threshold
        with open('threshold.pkl', 'rb') as f:
            optimal_threshold = pickle.load(f)
            
        return model, feature_names, optimal_threshold
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please make sure .pkl files are in the same folder.")
        return None, None, 0.5

model, feature_names, optimal_threshold = load_resources()

# 3. HELPER FUNCTIONS
def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula placeholder (since we take distance as input in UI)
    # This is just for internal consistency if needed
    return 0 

def generate_explanation(inputs, probability, threshold):
    """
    Generates a human-readable explanation for the prediction.
    """
    reasons = []
    
    # 1. Check Amount Anomalies
    # Added spaces around the variables and text
    if inputs['amt'] > inputs['customer_avg_amt'] * 5:
        reasons.append(f"🚨 Transaction amount (${inputs['amt']}) is significantly higher than average (${inputs['customer_avg_amt']}).")
    elif inputs['amt'] > inputs['customer_avg_amt'] * 2:
        reasons.append(f"⚠️ Transaction amount (${inputs['amt']}) is double the typical spending.")

    # 2. Check Location
    if inputs['distance_from_home'] > 100:
        reasons.append(f"📍 Location is very far ({inputs['distance_from_home']} miles) from home address.")
    elif inputs['distance_from_home'] > 50:
        reasons.append(f"📍 Location is {inputs['distance_from_home']} miles away, which is unusual.")

    # 3. Check Time
    if inputs['trans_hour'] < 6 or inputs['trans_hour'] > 23:
        reasons.append(f"⏰ Transaction occurred at unusual hour ({inputs['trans_hour']}:00).")

    # 4. Check Age Risk
    if inputs['age'] < 18 or inputs['age'] > 80:
        reasons.append(f"👤 Customer age ({inputs['age']}) falls into a high-risk statistical demographic.")

    # 5. Fallback for High Probability
    if len(reasons) == 0 and probability >= threshold:
        reasons.append("🤖 AI Model detected a complex high-risk pattern (combination of amount, category, and user history).")
        reasons.append(f"📊 Risk Score is {probability:.2%}, exceeding the safety limit of {threshold:.2%}.")

    return reasons
# 4. SIDEBAR NAVIGATION
st.sidebar.title("🏛️ Navigation")
page = st.sidebar.radio("Go to", ["Home", "Fraud Checker", "Performance", "About"])

# 5. PAGE: HOME
if page == "Home":
    st.title("🛡️ Fraud Detection System")
    st.subheader("Next-Generation Financial Fraud Detection System")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Accuracy", value="99.6%")
    with col2:
        st.metric(label="Fraud Recall", value="88.0%")
    with col3:
        st.metric(label="Transactions Processed", value="1.2M+")
        
    st.markdown("---")
    st.info("👈 Select **'Fraud Checker'** from the sidebar to test transactions.")

# 6. PAGE: FRAUD CHECKER
elif page == "Fraud Checker":
    st.title("🔍 Credit Card Fraud Detection")
    st.write("Enter transaction details below to check for potential fraud")
    
    # --- INPUT FORM ---
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Info")
            amt = st.number_input("Amount ($)", min_value=0.0, value=100.0)
            category = st.selectbox("Category", [
                'gas_transport', 'grocery_pos', 'home', 'shopping_pos', 'kids_pets',
                'shopping_net', 'entertainment', 'food_dining', 'personal_care',
                'health_fitness', 'misc_pos', 'misc_net', 'travel'
            ])
            hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
            distance = st.number_input("Distance from Home (miles)", min_value=0.0, value=5.0)
            
        with col2:
            st.subheader("Customer Info")
            age = st.number_input("Age", min_value=14, max_value=100, value=35)
            gender = st.radio("Gender", ["Male", "Female"])
            avg_amt = st.number_input("Typical Spending ($)", min_value=0.0, value=50.0)
            merch_risk = st.slider("Merchant Risk Score (0-1)", 0.0, 1.0, 0.5)
            
        submit = st.form_submit_button("🔍 Check Transaction")
    
    # --- PROCESSING ---
    if submit and model:
        # 1. Prepare Data
        # We need to map inputs to the 31 features expected by the model
        # Note: In a real app, we would calculate all 31. Here we approximate for demo.
        
        input_data = {
            'amt': amt,
            'customer_avg_amt': avg_amt,
            'distance_from_home': distance,
            'trans_hour': hour,
            'age': age,
            # Calculated features
            'amt_deviation': amt - avg_amt,
            'amt_ratio': amt / (avg_amt + 0.01),
            'is_weekend': 0, # Default
            'trans_day': 15, # Default
            'trans_month': 6, # Default
            'merchant_visit_count': 5, # Default
            'is_new_merchant': 0,
            'hours_since_last_trans': 24,
            # Placeholders for encoded features
            'gender_encoded': 1 if gender == "Male" else 0,
            'merchant_encoded': merch_risk
        }
        
        # Create a full feature vector with 0s for missing columns
        feature_vector = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        
        # Fill known values
        for col in input_data:
            if col in feature_vector.columns:
                feature_vector[col] = input_data[col]
                
        # Handle One-Hot Encoding for Category
        cat_col = f"cat_{category}"
        if cat_col in feature_vector.columns:
            feature_vector[cat_col] = 1
            
        # 2. Prediction
        probability = model.predict_proba(feature_vector)[0][1]
        is_fraud = probability >= optimal_threshold
        
        # 3. Display Results
        st.markdown("---")
        
        if is_fraud:
            st.error("⚠️ FRAUD DETECTED!")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fraud Probability", f"{probability:.2%}")
            with c2:
                st.metric("Risk Level", "HIGH", delta_color="inverse")
                
            st.subheader("💬 Why was this flagged?")
            
            # --- FIXED EXPLANATION LOGIC ---
            reasons = generate_explanation(input_data, probability, optimal_threshold)
            
            for r in reasons:
                st.warning(r)
                
        else:
            st.success("✅ TRANSACTION APPROVED")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fraud Probability", f"{probability:.2%}")
            with c2:
                st.metric("Risk Level", "LOW")
                
            st.info("All indicators appear normal for this transaction.")

# 7. PAGE: PERFORMANCE
elif page == "Performance":
    st.title("📊 Model Performance")
    st.write("Performance metrics on 555,719 test transactions.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (Frauds Caught)", "88.02%")
    c2.metric("Precision", "60.38%")
    c3.metric("ROC-AUC", "0.9966")
    c4.metric("False Alarm Rate", "Low")
    
    st.info("The model is optimized to prioritize Recall (catching as many frauds as possible).")

# 8. PAGE: ABOUT
elif page == "About":
    st.title("ℹ️ About")
    st.write("### Intelligent Credit Card Fraud Detection System")
    st.write("This project utilizes XGBoost with GPU acceleration to detect fraudulent credit card transactions in real-time.")
    st.write("**Dataset:** Kaggle Credit Card Fraud Dataset (1.3M+ records)")
    st.write("**Methodology:**")
    st.markdown("""
    - **Feature Engineering:** 31 custom features including velocity, distance, and behavioral deviations.
    - **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique).
    - **Model:** XGBoost Classifier (500 trees, Depth 10).
    - **Optimization:** Custom thresholding for maximum recall.
    """)

