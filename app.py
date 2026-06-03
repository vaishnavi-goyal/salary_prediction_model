import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Job Salary Prediction")

job_title = st.selectbox(
    "Job Title",
    encoders["job_title"].classes_
)

education = st.selectbox(
    "Education Level",
    encoders["education_level"].classes_
)

industry = st.selectbox(
    "Industry",
    encoders["industry"].classes_
)

company_size = st.selectbox(
    "Company Size",
    encoders["company_size"].classes_
)

location = st.selectbox(
    "Location",
    encoders["location"].classes_
)

remote_work = st.selectbox(
    "Remote Work",
    encoders["remote_work"].classes_
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=1
)

skills = st.number_input(
    "Skills Count",
    min_value=1,
    max_value=20,
    value=5
)

certifications = st.number_input(
    "Certifications",
    min_value=0,
    max_value=5,
    value=0
)

if st.button("Predict Salary"):

    data = pd.DataFrame([{
        "job_title": encoders["job_title"].transform([job_title])[0],
        "experience_years": experience,
        "education_level": encoders["education_level"].transform([education])[0],
        "skills_count": skills,
        "industry": encoders["industry"].transform([industry])[0],
        "company_size": encoders["company_size"].transform([company_size])[0],
        "location": encoders["location"].transform([location])[0],
        "remote_work": encoders["remote_work"].transform([remote_work])[0],
        "certifications": certifications
    }])

    salary = model.predict(data)[0]

    st.success(f"Predicted Salary: ${salary:,.0f}")