import sys
import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Set the page config
st.set_page_config(
    page_title="Diabetes Prediction",  # Change this to your desired title
    page_icon="🍩",  # Optional: You can add an emoji or a path to an icon
    layout="wide"  # Optional: You can set the layout to "centered" or "wide"
)

# Set the default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = load_model('Diabetes_Prediction.h5')

# Set up Streamlit input fields for the 8 features
st.title('Diabetes Prediction App')

gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 0, 100, 25)
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
smoking_history = st.selectbox('Smoking History', ['Never', 'Former', 'Current'])
bmi = st.slider('BMI', 10.0, 50.0, 22.5)
HbA1c_level = st.slider('HbA1c Level', 2.0, 15.0, 5.0)
blood_glucose_level = st.slider('Blood Glucose Level', 50, 300, 100)

# Convert categorical inputs to numerical values
gender = 1 if gender == 'Male' else 0
hypertension = 1 if hypertension == 'Yes' else 0
heart_disease = 1 if heart_disease == 'Yes' else 0
smoking_history = {'Never': 0, 'Former': 1, 'Current': 2}[smoking_history]

# Prepare the input data for prediction
input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])

# Standardize the input data (ensure to match the scaler used during training)
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Make the prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    result = 'Diabetic' if prediction[0] > 0.5 else 'Not Diabetic'
    st.write(f'Prediction: {result}')
