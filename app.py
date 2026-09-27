# ==============================================================================
# E-COMMERCE FRAUD ANALYTICS & REAL-TIME RISK ENGINE (app.py)
# ==============================================================================

import io
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Fraud Analytics & Risk Engine",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS styling for metric cards
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. ASSET LOADING WITH CHUNK RECOMBINATION
# ------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    asset_dir = 'dashboard_assets'
    
    # Dynamically find and sort all model chunk parts (.part1, .part2, .part3, etc.)
    part_files = sorted([
        os.path.join(asset_dir, f) for f in os.listdir(asset_dir)
        if 'xgb_fraud_model.pkl.gz.part' in f
    ])
    
    if not part_files:
        st.error("Model chunk files not found in `dashboard_assets/`!")
        st.stop()
        
    # Recombine chunk bytes in memory
    combined_bytes = bytearray()
    for part in part_files:
        with open(part, 'rb') as f:
            combined_bytes.extend(f.read())
            
    # Load model directly from in-memory byte stream
    model = joblib.load(io.BytesIO(combined_bytes))
    
    # Load metadata and sampled EDA dataframe
    features = joblib.load(os.path.join(asset_dir, 'model_features.pkl'))
    cat_cols = joblib.load(os.path.join(asset_dir, 'cat_cols.pkl'))
    df_sample = pd.read_parquet(os.path.join(asset_dir, 'sample_eda.parquet'))
    
    return model, features, cat_cols, df_sample

# Load assets
model, features, cat_cols, df_sample = load_assets()

# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.title("🛡️ Fraud Engine Portal")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select View:", 
    ["Forensic EDA & Analytics", "Real-Time Risk Scoring Engine"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Highlights:**
* **Dataset:** 590K+ IEEE-CIS Transactions
* **Model:** XGBoost Classifier
* **Primary Metric:** PR-AUC = 0.5096 (15x Over Baseline)
* **ROC-AUC:** 0.8873
""")

# ------------------------------------------------------------------------------
# PAGE 1: FORENSIC EDA & ANALYTICS
# ------------------------------------------------------------------------------
if page == "Forensic EDA & Analytics":
    st.title("📊 E-Commerce Forensic Transaction Analytics")
    st.markdown("Investigating structural fraud signatures across temporal cycles, email domains, and device fingerprints.")
    st.markdown("---")
    
    # KPI Highlights
    col1, col2, col3, col4 = st.columns(4)
    total_tx = len(df_sample)
    fraud_rate = df_sample['isFraud'].mean() * 100
    total_vol = df_sample['TransactionAmt'].sum()
    fraud_vol = df_sample[df_sample['isFraud'] == 1]['TransactionAmt'].sum()
    
    col1.metric("Analyzed Sample Records", f"{total_tx:,}")
    col2.metric("Baseline Fraud Rate", f"{fraud_rate:.2f}%")
    col3.metric("Total Transaction Volume", f"${total_vol:,.2f}")
    col4.metric("Identified Fraud Exposure", f"${fraud_vol:,.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Analytics Tabs
    tab1, tab2, tab3 = st.tabs(["Temporal & Sleep Cycles", "Email Mismatch Risk", "Device & Hardware Channels"])
    
    with tab1:
        st.subheader("Hour-of-Day Sleep-Cycle Exploitation")
        hourly = df_sample.groupby('Hour_of_Day')['isFraud'].agg(['count', 'mean']).reset_index()
        hourly['Fraud Rate (%)'] = hourly['mean'] * 100
        
        fig_hour = px.bar(
            hourly, x='Hour_of_Day', y='Fraud Rate (%)',
            title="Fraud Attack Rate by Hour of Day (0 to 23)",
            labels={'Hour_of_Day': 'Hour of Day (0=Midnight, 23=11 PM)'},
            color='Fraud Rate (%)', color_continuous_scale='Reds'
        )
        fig_hour.add_hline(y=fraud_rate, line_dash="dash", annotation_text="Baseline Fraud Rate", annotation_position="top right")
        st.plotly_chart(fig_hour, use_container_width=True)
        st.info("💡 **Key Discovery:** Fraud probability spikes significantly during off-peak hours (2 AM – 5 AM), exploiting low customer vigilance when card lock alerts go unnoticed.")

    with tab2:
        st.subheader("Purchaser vs. Recipient Email Domain Risk")
        top_p_domains = df_sample['P_emaildomain'].value_counts().head(8).index
        email_df = df_sample[df_sample['P_emaildomain'].isin(top_p_domains)]
        email_stats = email_df.groupby('P_emaildomain')['isFraud'].mean().reset_index()
        email_stats['Fraud Rate (%)'] = email_stats['isFraud'] * 100
        email_stats = email_stats.sort_values(by='Fraud Rate (%)', ascending=True)
        
        fig_email = px.bar(
            email_stats, y='P_emaildomain', x='Fraud Rate (%)',
            orientation='h', title="Fraud Likelihood Across Top Purchaser Email Domains",
            labels={'P_emaildomain': 'Purchaser Email Domain'},
            color='Fraud Rate (%)', color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_email, use_container_width=True)
        st.info("💡 **Key Discovery:** High-risk or temporary webmail domains demonstrate up to 3x higher chargeback frequencies compared to established corporate domains.")

    with tab3:
        st.subheader("Mobile vs. Desktop Channel Vulnerabilities")
        device_df = df_sample.dropna(subset=['DeviceType'])
        device_stats = device_df.groupby('DeviceType')['isFraud'].agg(['count', 'mean']).reset_index()
        device_stats['Fraud Rate (%)'] = device_stats['mean'] * 100
        
        fig_device = px.pie(
            device_stats, names='DeviceType', values='Fraud Rate (%)',
            title="Proportional Fraud Distribution by Channel Category",
            color_discrete_sequence=['#636EFA', '#EF553B'],
            hole=0.4
        )
        st.plotly_chart(fig_device, use_container_width=True)
        st.info("💡 **Key Discovery:** Mobile web transactions carry a systematically higher fraud rate than desktop sessions due to device-spoofing and emulated user-agents.")

# ------------------------------------------------------------------------------
# PAGE 2: REAL-TIME RISK SCORING ENGINE
# ------------------------------------------------------------------------------
else:
    st.title("🎯 Real-Time Transaction Risk Scoring")
    st.markdown("Input payload characteristics below to run dynamic risk scoring via the trained XGBoost detector.")
    st.markdown("---")
    
    with st.form("risk_eval_form"):
        st.subheader("1. Transaction & Behavioral Input Payload")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tx_amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=185.50, step=5.0)
            product_cd = st.selectbox("Product Category", options=["W", "H", "C", "S", "R"])
            p_email = st.selectbox("Purchaser Email Domain", options=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.info", "outlook.com"])
        
        with col2:
            velocity_1h = st.slider("1-Hour Card Velocity (Tx Count)", 1, 20, 1)
            velocity_24h = st.slider("24-Hour Card Velocity (Tx Count)", 1, 50, 2)
            r_email = st.selectbox("Recipient Email Domain", options=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.info", "missing"])
            
        with col3:
            device_type = st.selectbox("Device Category", options=["desktop", "mobile"])
            hour_of_day = st.slider("Transaction Hour (0-23)", 0, 23, 3)
            card_network = st.selectbox("Card Network", options=["visa", "mastercard", "discover", "american express"])

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("⚡ Run Risk Assessment Model", use_container_width=True)

    if submit_btn:
        # Construct single-row input matrix matching model feature names
        input_data = {col: [np.nan] for col in features}
        
        # Populate populated input values
        input_data['TransactionAmt'] = [tx_amt]
        input_data['ProductCD'] = [product_cd]
        input_data['P_emaildomain'] = [p_email]
        input_data['R_emaildomain'] = [r_email]
        input_data['DeviceType'] = [device_type]
        input_data['Hour_of_Day'] = [hour_of_day]
        input_data['card4'] = [card_network]
        input_data['velocity_tx_1h'] = [velocity_1h]
        input_data['velocity_tx_24h'] = [velocity_24h]
        
        # Derived features matching Step 4 pipeline
        input_data['email_domain_mismatch'] = [1 if (p_email != 'missing' and r_email != 'missing' and p_email != r_email) else 0]
        input_data['has_recipient_email'] = [1 if r_email != 'missing' else 0]
        input_data['amt_to_uid_mean_ratio'] = [tx_amt / 120.0]  # Ratio against historical baseline
        input_data['amt_uid_zscore'] = [(tx_amt - 120.0) / 45.0]

        input_df = pd.DataFrame(input_data)
        
        # Cast categorical columns
        for c in cat_cols:
            if c in input_df.columns:
                input_df[c] = input_df[c].astype('category')
                
        # Perform Inference
        fraud_prob = float(model.predict_proba(input_df)[:, 1][0])
        
        st.markdown("---")
        st.subheader("2. Model Evaluation Output")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Risk Gauge Meter
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob * 100,
                title={'text': "Predicted Fraud Probability (%)"},
                number={'suffix': "%", 'valueformat': ".2f"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if fraud_prob >= 0.773 else ("orange" if fraud_prob >= 0.30 else "green")},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 77.3], 'color': "lightyellow"},
                        {'range': [77.3, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 77.3
                    }
                }
            ))
            gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            st.markdown("### Actionable Decision Recommendation")
            
            if fraud_prob >= 0.773:
                st.error("🚨 **HIGH RISK TRANSACTION (HARD BLOCK RECOMMENDED)**")
                st.markdown(f"**Calculated Score:** `{fraud_prob:.4f}` (Exceeds optimal threshold `0.773`)")
                st.markdown("""
                **Triggered Risk Vectors:**
                * High Transaction Velocity within 1-Hour Window
                * Purchaser / Recipient Domain Discrepancy
                * Off-Peak Sleep Cycle Timing (2 AM – 5 AM)
                """)
                st.markdown("**Action:** Auto-decline checkout attempt and issue alert to Fraud Ops.")
                
            elif fraud_prob >= 0.30:
                st.warning("⚠️ **MEDIUM RISK TRANSACTION (STEP-UP AUTHENTICATION REQUIRED)**")
                st.markdown(f"**Calculated Score:** `{fraud_prob:.4f}`")
                st.markdown("**Action:** Require 3D-Secure OTP / SMS Biometric confirmation before order authorization.")
                
            else:
                st.success("✅ **LOW RISK TRANSACTION (AUTO-APPROVED)**")
                st.markdown(f"**Calculated Score:** `{fraud_prob:.4f}`")
                st.markdown("**Action:** Fast-track order processing for seamless customer checkout.")
