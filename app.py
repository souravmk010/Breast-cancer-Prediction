# app.py

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Prediction")
st.write("Enter tumor measurements below:")

# Input fields
mean_radius = st.number_input("Mean Radius", min_value=0.0)
mean_texture = st.number_input("Mean Texture", min_value=0.0)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
mean_area = st.number_input("Mean Area", min_value=0.0)
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[mean_radius, mean_texture,
                            mean_perimeter, mean_area,
                            mean_smoothness]])
    
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"Benign Tumor ✅ (Confidence: {probability[0][1]*100:.2f}%)")
    else:
        st.error(f"Malignant Tumor ⚠️ (Confidence: {probability[0][0]*100:.2f}%)")