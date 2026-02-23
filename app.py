import streamlit as st
import pickle
import numpy as np

# Load models
risk_model = pickle.load(open("risk_model.pkl", "rb"))
disease_model = pickle.load(open("disease_model.pkl", "rb"))
severity_model = pickle.load(open("severity_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Smart Health Risk Predictor")

st.write("Enter patient details:")

# Input fields
age = st.number_input("Age", 20, 100)
bmi = st.number_input("BMI", 10.0, 50.0)
blood_pressure = st.number_input("Blood Pressure", 80.0, 200.0)
glucose = st.number_input("Glucose Level", 50.0, 300.0)
cholesterol = st.number_input("Cholesterol", 100.0, 400.0)

if st.button("Predict"):

    features = np.array([[age, bmi, blood_pressure, glucose, cholesterol]])

    # Risk prediction (needs scaling)
    scaled_features = scaler.transform(features)
    risk_pred = risk_model.predict(scaled_features)[0]

    # Disease prediction
    disease_pred = disease_model.predict(features)[0]

    # Severity prediction
    severity_pred = severity_model.predict(features)[0]

    st.subheader("Prediction Result")

    st.write("Health Risk:", "High" if risk_pred == 1 else "Low")
    st.write("Predicted Disease:", disease_pred)
    st.write("Severity Level:", severity_pred)