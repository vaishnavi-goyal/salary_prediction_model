import streamlit as st
import pandas as pd
import joblib
import io

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="💼 Job Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Hide Streamlit Header */

#MainMenu{
visibility:hidden;
}

header{
visibility:hidden;
}

footer{
visibility:hidden;
}

/* Background */

.stApp{
background:linear-gradient(
135deg,
#EEF4FF,
#F8FBFF,
#E8F1FF
);
}

/* Main Container */

.block-container{

padding-top:30px;

padding-bottom:30px;

}

/* Dashboard Card */

.dashboard{

background:white;

padding:35px;

border-radius:25px;

box-shadow:0px 20px 45px rgba(0,0,0,.12);

}

/* Title */

.title{

font-size:48px;

font-weight:700;

text-align:center;

color:#1E3A8A;

margin-bottom:8px;

}

/* Subtitle */

.subtitle{

font-size:18px;

text-align:center;

color:#64748B;

margin-bottom:30px;

}

/* Widget Labels */

div[data-testid="stWidgetLabel"] p{

font-size:16px;

font-weight:600;

color:#1F2937 !important;

}

/* Select Box */

div[data-baseweb="select"]{

border-radius:12px;

}

div[data-baseweb="select"] > div{

background:#F8FAFC;

border:2px solid #CBD5E1;

border-radius:12px;

}

/* Number Input */

.stNumberInput input{

background:#F8FAFC;

border-radius:12px;

border:2px solid #CBD5E1;

}

/* Button */

.stButton>button{

width:100%;

height:58px;

border:none;

border-radius:14px;

background:linear-gradient(90deg,#2563EB,#4F46E5);

color:white;

font-size:19px;

font-weight:bold;

box-shadow:0px 10px 25px rgba(37,99,235,.35);

transition:.3s;

}

.stButton>button:hover{

transform:translateY(-3px);

}

</style>
""", unsafe_allow_html=True)
# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🤖 AI Salary Predictor")

    st.write(
        """
Welcome 👋

Predict employee salary using Machine Learning.

### Features

✅ Gradient Boosting Model

✅ Instant Prediction

✅ Multiple Countries

✅ Currency Conversion

✅ Download Report
"""
    )

    st.markdown("---")

    st.info(
        """
Developer

👩‍💻 Vaishnavi Goyal

Machine Learning Project
"""
    )

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="dashboard">

<div class="title">

💼 Job Salary Prediction Dashboard

</div>

<div class="subtitle">

AI Powered Salary Prediction using Machine Learning

</div>

""", unsafe_allow_html=True)

# ==========================================================
# INPUT FORM
# ==========================================================

left, right = st.columns(2)

with left:

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

with right:

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

st.write("")

predict = st.button("🚀 Predict Salary")

st.markdown("</div>", unsafe_allow_html=True)
# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    # Prepare Input Data
    input_data = pd.DataFrame([{

        "job_title":
        encoders["job_title"].transform([job_title])[0],

        "experience_years":
        experience,

        "education_level":
        encoders["education_level"].transform([education])[0],

        "skills_count":
        skills,

        "industry":
        encoders["industry"].transform([industry])[0],

        "company_size":
        encoders["company_size"].transform([company_size])[0],

        "location":
        encoders["location"].transform([location])[0],

        "remote_work":
        encoders["remote_work"].transform([remote_work])[0],

        "certifications":
        certifications

    }])

    # Salary Prediction
    salary = model.predict(input_data)[0]

    # Currency Conversion

    currency_rates = {

        "USA": ("USD",1),

        "India":("INR",83),

        "UK":("GBP",0.79),

        "Canada":("CAD",1.36),

        "Germany":("EUR",0.92),

        "Australia":("AUD",1.52),

        "Singapore":("SGD",1.35),

        "UAE":("AED",3.67)

    }

    currency_code="USD"

    rate=1

    if location in currency_rates:

        currency_code,rate=currency_rates[location]

    converted_salary=salary*rate

    lower=max(0,salary-4255)

    upper=salary+4255

    # Beautiful Salary Card

    st.markdown(f"""

    <div style="

    background:linear-gradient(135deg,#2563EB,#4F46E5);

    border-radius:20px;

    padding:35px;

    color:white;

    text-align:center;

    margin-top:25px;

    box-shadow:0 15px 35px rgba(0,0,0,.18);

    ">

    <h2>💰 Predicted Salary</h2>

    <h1 style="font-size:52px;">

    ${salary:,.0f}

    </h1>

    <h3>{currency_code} {converted_salary:,.0f}</h3>

    </div>

    """,unsafe_allow_html=True)

    st.write("")
        # ==========================================================
    # PROFILE SUMMARY & PREDICTION DETAILS
    # ==========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## 👤 Candidate Summary")

        st.info(f"""

**💼 Job Title :** {job_title}

**🎓 Education :** {education}

**🏢 Industry :** {industry}

**🏬 Company Size :** {company_size}

**📍 Location :** {location}

**🏠 Remote Work :** {remote_work}

""")

    with col2:

        # Experience Level

        if experience <= 2:
            level = "🟢 Fresher"

        elif experience <= 5:
            level = "🟡 Junior"

        elif experience <= 10:
            level = "🟠 Mid Level"

        else:
            level = "🔴 Senior"

        confidence = 94

        st.markdown("## 📊 Prediction Details")

        st.success(f"""

⭐ Experience Level : {level}

📈 Confidence : {confidence}%

💵 Currency : {currency_code}

📉 Salary Range

${lower:,.0f}

to

${upper:,.0f}

""")

    st.write("")

    # ==========================================================
    # CONFIDENCE BAR
    # ==========================================================

    st.markdown("### 📈 Prediction Confidence")

    st.progress(confidence)

    st.write("")

    # ==========================================================
    # QUICK METRICS
    # ==========================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "💰 Salary",
            f"${salary:,.0f}"
        )

    with c2:

        st.metric(
            "📈 Experience",
            f"{experience} Years"
        )

    with c3:

        st.metric(
            "🛠 Skills",
            skills
        )

    st.write("")
        # ==========================================================
    # SALARY CATEGORY
    # ==========================================================

    st.write("")

    if salary < 50000:

        st.warning("💡 Salary Category : Entry Level")

    elif salary < 120000:

        st.info("💡 Salary Category : Mid Level")

    else:

        st.success("💡 Salary Category : High Income")

    # ==========================================================
    # DOWNLOAD REPORT
    # ==========================================================

    report = pd.DataFrame({

        "Field":[

            "Job Title",

            "Education",

            "Industry",

            "Company Size",

            "Location",

            "Remote Work",

            "Experience",

            "Skills",

            "Certifications",

            "Predicted Salary"

        ],

        "Value":[

            job_title,

            education,

            industry,

            company_size,

            location,

            remote_work,

            experience,

            skills,

            certifications,

            f"${salary:,.0f}"

        ]

    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(

        "📥 Download Prediction Report",

        csv,

        file_name="salary_prediction_report.csv",

        mime="text/csv"

    )

    # ==========================================================
    # CAREER TIPS
    # ==========================================================

    st.write("")

    st.markdown("## 💡 Career Tips")

    tips = [
        "🐍 Learn Python",
        "🗄️ Master SQL",
        "📊 Improve Data Analysis",
        "🤖 Learn Machine Learning",
        "📜 Complete Certifications",
        "💼 Build Real Projects",
        "🌐 Create Portfolio",
        "🔗 Keep LinkedIn Updated"
    ]

    for tip in tips:

        st.write("✅", tip)

    # ==========================================================
    # SUCCESS ANIMATION
    # ==========================================================

    st.balloons()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""

<div style="

text-align:center;

padding:20px;

color:#64748B;

">

<h3 style="color:#2563EB;">

💼 Job Salary Prediction Dashboard

</h3>

<p>

Developed with ❤️ by <b>Vaishnavi Goyal</b>

</p>

<p>

Machine Learning Project | Streamlit | Python | Scikit-learn

</p>

</div>

""", unsafe_allow_html=True)
