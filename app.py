# app.py

import streamlit as st
import pandas as pd
import joblib
from utils.predictor import smartcalc_predict

# Load models and scalers
model_bill = joblib.load('models/model_bill.pkl')
scaler_bill = joblib.load('models/scaler_bill.pkl')

model_units = joblib.load('models/model_units.pkl')
scaler_units = joblib.load('models/scaler_units.pkl')

model_load = joblib.load('models/model_load.pkl')
scaler_load = joblib.load('models/scaler_load.pkl')

# Load feature means
feature_means = pd.read_csv('data/feature_means.csv').iloc[0].to_dict()
full_features = list(feature_means.keys())

# UI
st.set_page_config(page_title="SmartCalc AI", page_icon="⚡")
st.title("🔮 SmartCalc AI — Electricity Bill Predictor")
st.markdown("Enter any **two values** below to predict the third:")

col1, col2, col3 = st.columns(3)
load_input = col1.number_input("🔌 Connected Load (kW)", min_value=0.0, step=0.1, format="%.2f")
units_input = col2.number_input("⚡ Units Consumed (kWh)", min_value=0.0, step=1.0, format="%.2f")
bill_input = col3.number_input("💸 Electricity Bill (₹)", min_value=0.0, step=10.0, format="%.2f")

if st.button("Predict"):
    inputs = {
        'load': load_input if load_input > 0 else None,
        'units': units_input if units_input > 0 else None,
        'bill': bill_input if bill_input > 0 else None
    }

    result = smartcalc_predict(
        **inputs,
        model_bill=model_bill, scaler_bill=scaler_bill,
        model_units=model_units, scaler_units=scaler_units,
        model_load=model_load, scaler_load=scaler_load,
        feature_means=feature_means, full_features=full_features
    )

    st.success("Prediction Result:")
    st.write(result)