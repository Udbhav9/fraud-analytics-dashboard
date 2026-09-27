import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import xgboost as xgb
import shap

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Fraud Analytics & Risk Engine",
    page_icon="🛡️",
    layout="wide"
)

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('dashboard_assets/xgb_fraud_model.pkl')
    features = joblib.load('dashboard_assets/model_features.pkl')
    cat_cols = joblib.load('dashboard_assets/cat_cols.pkl')
    df_sample = pd.read_parquet('dashboard_assets/sample_eda.parquet')
    return model, features, cat_cols, df_sample

model, features, cat_cols, df_sample = load_assets()

# Sidebar Navigation
st.sidebar.title("🛡️ Fraud Engine Portal")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate View:", ["Forensic EDA & Analytics", "Real-Time Risk Scoring Engine"])

# ------------------------------------------------------------------------------
# PAGE 1: FORENSIC EDA & ANALYTICS
# ------------------------------------------------------------------------------
if page == "Forensic EDA & Analytics":
    st.title("📊 E-Commerce Forensic Transaction Analytics")
    st.markdown("Investigating structural fraud signatures across temporal cycles, email domains, and device fingerprints.")
    
    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyzed Transactions", f"{len(df_sample):,}")
    col2.metric("Baseline Fraud Rate", f"{df_sample['isFraud'].mean()*100:.2f}%")
    col3.metric("Total Dollar Volume", f"${df_sample['TransactionAmt'].sum():,.2f}")
    col4.metric("Flagged Fraud Volume", f"${df_sample[df_sample['isFraud']==1]['TransactionAmt'].sum():,.2f}")
    st.markdown("---")
    
    # Tabbed Visualizations
    tab1, tab2, tab3 = st.tabs(["Temporal & Sleep Cycles", "Email Mismatch Matrix", "Device & Channel Tiers"])
    
    with tab1:
        st.subheader("Hour-of-Day Sleep-Cycle Exploitation")
        hourly = df_sample.groupby('Hour_of_Day')['isFraud'].agg(['count', 'mean']).reset_index()
        hourly['Fraud Rate (%)'] = hourly['mean'] * 100
        
        fig_hour = px.bar(
            hourly, x='Hour_of_Day', y='Fraud Rate (%)',
            title="Fraud Spike Probability by Hour (0 to 23)",
            color='Fraud Rate (%)', color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        st.info("💡 **Insight:** Fraud rates peak significantly between 2 AM and 5 AM, exploiting off-peak monitoring hours.")

    with tab2:
        st.subheader("Purchaser vs. Recipient Email Domain Risk")
        top_domains = df_sample['P_emaildomain'].value_counts().head(8).index
        email_df = df_sample[df_sample['P_emaildomain'].isin(top_domains)]
        email_stats = email_df.groupby('P_emaildomain')['isFraud'].mean().reset_index()
        email_stats['Fraud Rate (%)'] = email_stats['isFraud'] * 100
        
        fig_email = px.bar(
            email_stats, y='P_emaildomain', x='Fraud Rate (%)',
            orientation='h', title="Fraud Rates across Primary Email Domains",
            color='Fraud Rate (%)', color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_email, use_container_width=True)

    with tab3:
        st.subheader("Mobile vs. Desktop Channel Risk")
        device_df = df_sample.dropna(subset=['DeviceType'])
        device_stats = device_df.groupby('DeviceType')['isFraud'].mean().reset_index()
        device_stats['Fraud Rate (%)'] = device_stats['isFraud'] * 100
        
        fig_device = px.pie(
            device_stats, names='DeviceType', values='Fraud Rate (%)',
            title="Fraud Proportions by Hardware Category",
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        st.plotly_chart(fig_device, use_container_width=True)

# ------------------------------------------------------------------------------
# PAGE 2: REAL-TIME RISK SCORING ENGINE
# ------------------------------------------------------------------------------
else:
    st.title("🎯 Real-Time Transaction Risk Scoring")
    st.markdown("Input payload characteristics below to run dynamic risk scoring via XGBoost.")
    
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tx_amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=150.0)
            product_cd = st.selectbox("Product Code", options=["W", "H", "C", "S", "R"])
            p_email = st.selectbox("Purchaser Email Domain", options=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.info"])
        
        with col2:
            velocity_1h = st.slider("1-Hour Transaction Velocity (Card)", 1, 20, 1)
            velocity_24h = st.slider("24-Hour Transaction Velocity (Card)", 1, 50, 2)
            r_email = st.selectbox("Recipient Email Domain", options=["gmail.com", "yahoo.com", "hotmail.com", "anonymous.info"])
            
        with col3:
            device_type = st.selectbox("Device Category", options=["desktop", "mobile"])
            hour_of_day = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
            card_type = st.selectbox("Card Network", options=["visa", "mastercard", "discover", "american express"])

        submit_btn = st.form_submit_button("Evaluate Transaction Risk")

    if submit_btn:
        # Construct Single-Row Payload Matching Expected Features
        row = {f: [np.nan] for f in features}
        row['TransactionAmt'] = [tx_amt]
        row['ProductCD'] = [product_cd]
        row['P_emaildomain'] = [p_email]
        row['R_emaildomain'] = [r_email]
        row['DeviceType'] = [device_type]
        row['Hour_of_Day'] = [hour_of_day]
        row['card4'] = [card_type]
        row['velocity_tx_1h'] = [velocity_1h]
        row['velocity_tx_24h'] = [velocity_24h]
        row['email_domain_mismatch'] = [1 if p_email != r_email else 0]
        row['amt_to_uid_mean_ratio'] = [tx_amt / 135.0]
        
        input_df = pd.DataFrame(row)
        for c in cat_cols:
            if c in input_df.columns:
                input_df[c] = input_df[c].astype('category')
                
        # Inference
        fraud_prob = float(model.predict_proba(input_df)[:, 1][0])
        
        st.markdown("---")
        st.subheader("Assessment Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob * 100,
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if fraud_prob > 0.773 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 77.3], 'color': "orange"},
                        {'range': [77.3, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 77.3
                    }
                }
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            if fraud_prob >= 0.773:
                st.error("🚨 **HIGH RISK TRANSACTION DETECTED**")
                st.markdown(f"**Action:** Hard Block / Trigger Step-Up 3D-Secure OTP.")
                st.markdown(f"**Score:** `{fraud_prob:.4f}` (Exceeds optimal risk threshold of `0.773`)")
            elif fraud_prob >= 0.30:
                st.warning("⚠️ **MEDIUM RISK TRANSACTION**")
                st.markdown(f"**Action:** Flag for Manual Risk Operations Review.")
                st.markdown(f"**Score:** `{fraud_prob:.4f}`")
            else:
                st.success("✅ **LOW RISK TRANSACTION**")
                st.markdown(f"**Action:** Auto-Approve (Frictionless Checkout).")
                st.markdown(f"**Score:** `{fraud_prob:.4f}`")
