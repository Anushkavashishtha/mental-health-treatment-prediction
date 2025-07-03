# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tech Mental Health Treatment Predictor", layout="centered")
st.title("Mental Health Treatment Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model/mh_model.pkl")

model = load_model()

# User inputs
st.subheader("Worker Profile")
age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", ["Male","Female","Other"])
family = st.selectbox("Family History of Mental Illness?", ["Yes","No"])
work_int = st.selectbox("Work Interference by Mental Health?", ["Never","Rarely","Sometimes","Often"])
self_emp = st.selectbox("Self-employed?", ["Yes","No"])
benefits = st.selectbox("Employer Mental Health Benefits?", ["Yes","No"])
leave = st.selectbox("Easy to take mental health leave?", ["Yes","No"])
ph_conseq = st.selectbox("Physical health consequences if discussing mental health?", ["Yes","No"])

# Prepare input
layout = {"Yes":1,"No":0,"Often":3,"Sometimes":2,"Rarely":1,"Never":0}
input_df = pd.DataFrame([{
    "Gender": 0 if gender == "Male" else (1 if gender == "Female" else 2),
    "family_history": layout[family],
    "work_interfere": layout[work_int],
    "self_employed": layout[self_emp],
    "benefits": layout[benefits],
    "leave": layout[leave],
    "phys_health_consequence": layout[ph_conseq],
    "Age": age
}])


# Predict
if st.button("Predict Treatment"):
 
    pred = model.predict(input_df)[0]
    st.success("Yes" if pred else "No")
