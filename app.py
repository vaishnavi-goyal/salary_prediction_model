import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💰",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #7c3aed);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: #081028;
}

/* Header */
.title{
    text-align:center;
    color:white;
    font-size:55px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#e5e7eb;
    font-size:20px;
    margin-bottom:20px;
}

/* Metric Cards */
[data-testid="metric-container"]{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.2);
}

/* Select Boxes */
.stSelectbox > div > div{
    border-radius:10px;
}

/* Slider */
.stSlider{
    padding-top:10px;
}

/* Predict Button */
.stButton > button{
    background: linear-gradient(
        90deg,
        #4f46e5,
        #d946ef
    );
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
    width:100%;
}

/* Salary Result Card */
.salary-card{
    background: linear-gradient(
        90deg,
        #1e1b4b,
        #4338ca,
        #9333ea
    );
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 8px 25px rgba(0,0,0,0.3);
}

/* Footer */
.footer{
    text-align:center;
    color:white;
    padding-top:30px;
}

</style>
""", unsafe_allow_html=True)
# ---------------- HEADER ----------------

st.markdown("""
<div class="title">
💰 Job Salary Prediction System
</div>

<div class="subtitle">
Predict Employee Salary using Machine Learning
</div>
""", unsafe_allow_html=True)

st.divider()
# ---------------- SIDEBAR ----------------
st.sidebar.header("📋 Enter Details")

job_title = st.sidebar.selectbox(
    "💼 Job Title",
    encoders["job_title"].classes_
)

education = st.sidebar.selectbox(
    "🎓 Education Level",
    encoders["education_level"].classes_
)

industry = st.sidebar.selectbox(
    "🏭 Industry",
    encoders["industry"].classes_
)

company_size = st.sidebar.selectbox(
    "🏢 Company Size",
    encoders["company_size"].classes_
)

location = st.sidebar.selectbox(
    "📍 Location",
    encoders["location"].classes_
)

remote_work = st.sidebar.selectbox(
    "🏠 Remote Work",
    encoders["remote_work"].classes_
)

experience = st.sidebar.slider(
    "📈 Experience (Years)",
    0, 40, 1
)

skills = st.sidebar.slider(
    "🛠 Skills Count",
    1, 20, 5
)

certifications = st.sidebar.slider(
    "📜 Certifications",
    0, 5, 0
)

# ---------------- DASHBOARD ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Experience", f"{experience} Years")

with col2:
    st.metric("Skills", skills)

with col3:
    st.metric("Certificates", certifications)

st.divider()

# ---------------- PREDICT BUTTON ----------------
if st.button("🚀 Predict Salary", use_container_width=True):

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

    st.success("✅ Salary Prediction Completed")

    st.markdown(
        f"""
        <div class="salary-card">
        Predicted Salary <br><br>
        💵 ${salary:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Salary Category
    if salary < 50000:
        st.warning("📉 Salary Category: Low")
    elif salary < 100000:
        st.info("📊 Salary Category: Medium")
    else:
        st.success("📈 Salary Category: High")

    st.balloons()

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
    Developed using Streamlit & Machine Learning 🚀
    </div>
    """,
    unsafe_allow_html=True
)
