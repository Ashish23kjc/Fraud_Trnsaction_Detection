import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fraud Transaction Detection", layout="centered")

st.title("Fraud Transaction Detection")
st.write("Enter transaction details to predict whether it's fraudulent using the trained Random Forest model.")

# Load the trained Random Forest model with threshold tuning
model = joblib.load("C:/Users/Asus/OneDrive/Desktop/UM_Fraud_Transaction_Detection/fraud_detection/fraud_detection_model.pkl")

# Sidebar to select the threshold
st.sidebar.header("Threshold Tuning")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

# Input fields for transaction features
st.subheader("Transaction Inputs")

TX_AMOUNT = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
weekday = st.number_input("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=3)

FRAUD_TERMINAL_COUNT_28D = st.number_input("Terminal Fraud Count (Past 28 Days)", min_value=0, value=0)
CUSTOMER_AVG_AMOUNT_14D = st.number_input("Customer Avg Amount (Past 14 Days)", min_value=0.0, value=100.0)

# Derived / engineered features
IS_HIGH_AMOUNT = int(TX_AMOUNT > 220)
FRAUD_TERMINAL_FLAG = int(FRAUD_TERMINAL_COUNT_28D > 0)
TX_AMOUNT_OVER_AVG = TX_AMOUNT / (CUSTOMER_AVG_AMOUNT_14D + 1e-6)
IS_AMOUNT_ANOMALY = int(TX_AMOUNT_OVER_AVG > 5)

# Create input DataFrame
input_data = pd.DataFrame([{
    'TX_AMOUNT': TX_AMOUNT,
    'IS_HIGH_AMOUNT': IS_HIGH_AMOUNT,
    'FRAUD_TERMINAL_COUNT_28D': FRAUD_TERMINAL_COUNT_28D,
    'FRAUD_TERMINAL_FLAG': FRAUD_TERMINAL_FLAG,
    'CUSTOMER_AVG_AMOUNT_14D': CUSTOMER_AVG_AMOUNT_14D,
    'TX_AMOUNT_OVER_AVG': TX_AMOUNT_OVER_AVG,
    'IS_AMOUNT_ANOMALY': IS_AMOUNT_ANOMALY,
    'hour': hour,
    'weekday': weekday
}])

st.markdown("---")
st.subheader(" Model Prediction")

# Predict on button click
if st.button("Predict Fraud"):
    proba = model.predict_proba(input_data)[0][1]  # Probability of fraud
    prediction = int(proba > threshold)

    st.write(f"**Fraud Probability:** `{proba:.4f}`")
    st.write(f"**Applied Threshold:** `{threshold}`")

    if prediction == 1:
        st.error(" Prediction: Fraudulent Transaction")
    else:
        st.success("Prediction: Legitimate Transaction")

st.markdown("---")
st.caption("Note: Features like Terminal Fraud Count or Customer Avg Amount are typically computed from historical logs.")
