# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the model, scaler, and feature names
model = joblib.load('titanic_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Streamlit app
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival odds with XGBoost!")

# Input fields with flair
pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = Luxury, 3 = Economy")
sex = st.selectbox("Sex", ["Female", "Male"], help="Gender matters!")
age = st.slider("Age", 0, 100, 30, help="Slide to your age!")
fare = st.number_input("Fare", min_value=0.0, max_value=200.0, value=30.0, help="How much did you pay?")
family_size = st.slider("Family Size (including self)", 1, 11, 1, help="Solo or squad?")
has_cabin = st.checkbox("Has Cabin?", help="Private room or not?")
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"], 
                        help="Where did you board?")
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], 
                     help="Your social status!")

# Prepare input data
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'Fare': [fare],
    'FamilySize': [family_size],
    'HasCabin': [1 if has_cabin else 0],
    'Sex_male': [1 if sex == "Male" else 0],
    'Embarked_Q': [1 if embarked == "Queenstown" else 0],
    'Embarked_S': [1 if embarked == "Southampton" else 0],
    'Title_Mr': [1 if title == "Mr" else 0],
    'Title_Mrs': [1 if title == "Mrs" else 0],
    'Title_Miss': [1 if title == "Miss" else 0],
    'Title_Master': [1 if title == "Master" else 0],
    'Title_Rare': [1 if title == "Rare" else 0],
    'IsAlone': [1 if family_size == 1 else 0]
}, columns=feature_names)

# Scale numerical features
numerical_cols = ['Age', 'Fare', 'FamilySize']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.success(f"🎉 Prediction: **{'Survived' if prediction == 1 else 'Did Not Survive'}**")
    st.write(f"Survival Probability: **{probability:.2%}**")
    if prediction == 1:
        st.balloons()
    else:
        st.snow()