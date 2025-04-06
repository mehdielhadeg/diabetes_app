import streamlit as st
import numpy as np
import joblib

model = joblib.load("xgb_diabetes_model.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Fill in the patient details to predict diabetes (0 = No, 1 = Yes)")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 0, 120, 33)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'🟥 Diabetic (1)' if prediction == 1 else '🟩 Not Diabetic (0)'}")