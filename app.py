import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Job Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# -----------------------------
# Load Model & Encoders
# -----------------------------
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#EAF4FF,#F8FBFF);
}

/* Main Container */
.block-container{
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
}

/* Title */
.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E3A8A;
}

.subtitle{
    text-align:center;
    color:#6B7280;
    font-size:18px;
    margin-bottom:25px;
}

/* Button */
.stButton > button{
    width:100%;
    background:#2563EB;
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#1D4ED8;
    transition:0.3s;
}

/* Inputs */
.stSelectbox,
.stNumberInput{
    margin-bottom:12px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="title">💼 Job Salary Prediction</div>
<div class="subtitle">
Predict employee salary using Machine Learning
</div>
""", unsafe_allow_html=True)

# -----------------------------
# User Inputs
# -----------------------------
job_title = st.selectbox(
    "💼 Job Title",
    encoders["job_title"].classes_
)

education = st.selectbox(
    "🎓 Education Level",
    encoders["education_level"].classes_
)

industry = st.selectbox(
    "🏢 Industry",
    encoders["industry"].classes_
)

company_size = st.selectbox(
    "🏬 Company Size",
    encoders["company_size"].classes_
)

location = st.selectbox(
    "📍 Location",
    encoders["location"].classes_
)

remote_work = st.selectbox(
    "🏠 Remote Work",
    encoders["remote_work"].classes_
)

experience = st.number_input(
    "📈 Years of Experience",
    min_value=0,
    max_value=40,
    value=1
)

skills = st.number_input(
    "🛠 Skills Count",
    min_value=1,
    max_value=20,
    value=5
)

certifications = st.number_input(
    "📜 Certifications",
    min_value=0,
    max_value=5,
    value=0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Salary"):

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

    st.markdown(f"""
    <div style="
        background:#2563EB;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        margin-top:25px;
        box-shadow:0 8px 20px rgba(0,0,0,0.2);
    ">
        <h2>💰 Predicted Salary</h2>
        <h1>${salary:,.0f}</h1>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr>
<div style="text-align:center;color:gray;font-size:15px;">
Made with ❤️ by <b>Vaishnavi Goyal</b><br>
Machine Learning Salary Prediction Model
</div>
""", unsafe_allow_html=True)
