# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and features
model = joblib.load('house_model.pkl')
features = joblib.load('features.pkl')

st.title("🏠 House Price Predictor")

# Create input fields dynamically
user_input = []
st.subheader("Enter House Details:")
for feat in features:
    value = st.number_input(f"{feat}", value=0)
    user_input.append(value)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=features)
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: **${prediction:,.2f}**")
