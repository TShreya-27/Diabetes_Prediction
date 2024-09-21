import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your model
model = joblib.load('D:\\Diabetes_Prediction\\Diabetes_Prediction.pkl')


st.title("Diabetes Prediction App")

# Input fields for user data
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("The model predicts you have diabetes.")
    else:
        st.success("The model predicts you do not have diabetes.")
