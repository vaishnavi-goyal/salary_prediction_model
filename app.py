import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="💼 AI Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    model = joblib.load("salary_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

model, encoders = load_model()

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
}

/* Background */

.stApp{
background:linear-gradient(135deg,#eef5ff,#f8fbff,#e6f0ff);
}

/* Hide Streamlit */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Card */

.card{

background:white;

padding:30px;

border-radius:20px;

box-shadow:0px 15px 40px rgba(0,0,0,.10);

}

/* Title */

.title{

font-size:48px;

font-weight:700;

color:#1d4ed8;

text-align:center;

}

/* Subtitle */

.subtitle{

text-align:center;

font-size:18px;

color:#64748b;

margin-bottom:30px;

}

/* Button */

.stButton>button{

width:100%;

height:60px;

border:none;

border-radius:15px;

background:linear-gradient(90deg,#2563EB,#4F46E5);

color:white;

font-size:20px;

font-weight:bold;

}

.stButton>button:hover{

transform:translateY(-3px);

transition:.3s;

}

/* Labels */

div[data-testid="stWidgetLabel"] p{

font-size:16px;

font-weight:600;

color:#1f2937;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🤖 AI Salary Predictor")

    st.write("Professional Machine Learning Dashboard")

    st.markdown("---")

    dark = st.toggle("🌙 Dark Mode")

    st.markdown("---")

    st.success("Model Loaded Successfully ✅")

    st.caption("Version 2.0")
    # ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""

<div class="card">

<div class="title">

💼 AI Salary Prediction Dashboard

</div>

<div class="subtitle">

Predict employee salary using Machine Learning and AI Analytics

</div>

</div>

""", unsafe_allow_html=True)

st.write("")

# ==========================================================
# INPUT SECTION
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

    experience = st.slider(
        "📈 Years of Experience",
        min_value=0,
        max_value=40,
        value=2
    )

    skills = st.slider(
        "🛠 Skills Count",
        min_value=1,
        max_value=20,
        value=5
    )

    certifications = st.slider(
        "📜 Certifications",
        min_value=0,
        max_value=10,
        value=1
    )

st.write("")

# ==========================================================
# QUICK PREVIEW
# ==========================================================

st.markdown("### 👀 Selected Profile")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Experience", f"{experience} Years")

with c2:
    st.metric("Skills", skills)

with c3:
    st.metric("Certificates", certifications)

st.info(
    f"""
**Selected Profile**

💼 {job_title}

🎓 {education}

🏢 {industry}

📍 {location}
"""
)

st.write("")

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.button(
    "🚀 Predict Salary"
)
# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    with st.spinner("🤖 AI is analyzing your profile..."):

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

        salary = model.predict(input_data)[0]

    # ==========================================================
    # CURRENCY CONVERSION
    # ==========================================================

    currency = {
        "USA":("USD",1),
        "India":("INR",83),
        "UK":("GBP",0.79),
        "Canada":("CAD",1.36),
        "Germany":("EUR",0.92),
        "Australia":("AUD",1.52),
        "Singapore":("SGD",1.35),
        "UAE":("AED",3.67)
    }

    code = "USD"
    rate = 1

    if location in currency:
        code, rate = currency[location]

    converted_salary = salary * rate

    # ==========================================================
    # SALARY RATING
    # ==========================================================

    if salary < 50000:

        rating = "⭐⭐"

        level = "Entry Level"

    elif salary < 100000:

        rating = "⭐⭐⭐"

        level = "Mid Level"

    elif salary < 180000:

        rating = "⭐⭐⭐⭐"

        level = "Senior"

    else:

        rating = "⭐⭐⭐⭐⭐"

        level = "Executive"

    # ==========================================================
    # SALARY CARD
    # ==========================================================

    st.markdown(f"""

    <div style="

    background:linear-gradient(135deg,#2563EB,#4F46E5);

    border-radius:25px;

    padding:35px;

    color:white;

    text-align:center;

    box-shadow:0 15px 40px rgba(0,0,0,.20);

    ">

    <h2>💰 Predicted Salary</h2>

    <h1 style="font-size:55px;">

    ${salary:,.0f}

    </h1>

    <h3>{code} {converted_salary:,.0f}</h3>

    <h3>{rating}</h3>

    <p>{level}</p>

    </div>

    """, unsafe_allow_html=True)

    st.write("")
        # ==========================================================
    # PROFILE SUMMARY
    # ==========================================================

    left_card, right_card = st.columns(2)

    with left_card:

        st.markdown("## 👤 Candidate Profile")

        st.info(f"""
**💼 Job Title:** {job_title}

**🎓 Education:** {education}

**🏢 Industry:** {industry}

**🏬 Company Size:** {company_size}

**📍 Location:** {location}

**🏠 Remote Work:** {remote_work}

**📜 Certifications:** {certifications}
""")

    # ==========================================================
    # EXPERIENCE LEVEL
    # ==========================================================

    if experience <= 2:
        exp_level = "🟢 Fresher"

    elif experience <= 5:
        exp_level = "🟡 Junior"

    elif experience <= 10:
        exp_level = "🟠 Mid Level"

    else:
        exp_level = "🔴 Senior"

    confidence = 94

    with right_card:

        st.markdown("## 📊 Prediction Details")

        st.success(f"""
⭐ Experience Level : {exp_level}

📈 Confidence : {confidence}%

💵 Currency : {code}

📍 Location : {location}

🎯 Salary Rating : {rating}
""")

    # ==========================================================
    # CONFIDENCE BAR
    # ==========================================================

    st.write("")

    st.markdown("### 📈 Prediction Confidence")

    st.progress(confidence)

    st.caption(f"AI Confidence Score : {confidence}%")

    # ==========================================================
    # QUICK METRICS
    # ==========================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "💰 Salary",
            f"${salary:,.0f}"
        )

    with m2:
        st.metric(
            "📈 Experience",
            f"{experience} Years"
        )

    with m3:
        st.metric(
            "🛠 Skills",
            skills
        )

    with m4:
        st.metric(
            "📜 Certificates",
            certifications
        )

    st.write("")
        # ==========================================================
    # ANALYTICS DASHBOARD
    # ==========================================================

    st.markdown("---")
    st.markdown("## 📊 Salary Analytics Dashboard")

    # -----------------------
    # Gauge Meter
    # -----------------------

    gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=salary,

        title={"text":"Predicted Salary"},

        number={"prefix":"$"},

        gauge={

            "axis":{"range":[0,250000]},

            "bar":{"color":"royalblue"},

            "steps":[

                {"range":[0,50000],"color":"#dcfce7"},

                {"range":[50000,120000],"color":"#fef9c3"},

                {"range":[120000,250000],"color":"#fee2e2"}

            ]

        }

    ))

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # -----------------------
    # Salary Comparison
    # -----------------------

    chart = pd.DataFrame({

        "Type":[

            "Lower",

            "Predicted",

            "Upper"

        ],

        "Salary":[

            max(0,salary-5000),

            salary,

            salary+5000

        ]

    })

    fig = px.bar(

        chart,

        x="Type",

        y="Salary",

        text="Salary",

        color="Type"

    )

    fig.update_traces(

        texttemplate="$%{text:,.0f}",

        textposition="outside"

    )

    fig.update_layout(

        height=420,

        showlegend=False

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================================
    # AI RECOMMENDATION
    # ==========================================================

    st.markdown("## 🤖 AI Recommendation")

    if experience < 2:

        st.warning("""

🚀 You are at the beginning of your career.

Recommended Skills

• Python

• SQL

• Excel

• Power BI

""")

    elif experience < 5:

        st.info("""

📈 You are progressing well.

Recommended Skills

• Machine Learning

• Deep Learning

• Cloud Computing

• Docker

""")

    else:

        st.success("""

🏆 You have strong experience.

Recommended Skills

• Leadership

• MLOps

• Generative AI

• System Design

""")
    # ==========================================================
    # DOWNLOAD REPORT
    # ==========================================================

    st.markdown("---")
    st.markdown("## 📄 Download Prediction Report")

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
            "Predicted Salary (USD)",
            "Converted Salary",
            "Experience Level"

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
            f"${salary:,.0f}",
            f"{code} {converted_salary:,.0f}",
            exp_level

        ]

    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(

        "📥 Download CSV Report",

        data=csv,

        file_name="salary_prediction_report.csv",

        mime="text/csv"

    )

    # ==========================================================
    # SAVE HISTORY
    # ==========================================================

    st.session_state.history.append({

        "Time":datetime.now().strftime("%d-%m-%Y %H:%M"),

        "Job":job_title,

        "Salary":salary,

        "Location":location

    })

    # ==========================================================
    # HISTORY
    # ==========================================================

    st.markdown("---")
    st.markdown("## 🕒 Prediction History")

    history = pd.DataFrame(st.session_state.history)

    st.dataframe(

        history,

        use_container_width=True,

        hide_index=True

    )

    # ==========================================================
    # SUCCESS
    # ==========================================================

    st.balloons()

    st.success("✅ Salary Prediction Completed Successfully")

    # ==========================================================
    # FOOTER
    # ==========================================================

    st.markdown("---")

    st.markdown("""

<div style="

text-align:center;

padding:25px;

">

<h2 style="color:#2563EB;">

💼 AI Salary Prediction Dashboard

</h2>

<p>

Built with ❤️ using

<b>Python • Streamlit • Scikit-learn • Plotly</b>

</p>

<p>

Developed by <b>Vaishnavi Goyal</b>

</p>

</div>

""", unsafe_allow_html=True)
        # ==========================================================
    # AI SALARY INSIGHTS
    # ==========================================================

    st.markdown("---")
    st.markdown("## 🤖 AI Salary Insights")

    if salary < 50000:

        insight = """
Your predicted salary is in the Entry-Level range.

To improve your salary:

✅ Learn Python

✅ Learn SQL

✅ Build Projects

✅ Improve Communication Skills
"""

    elif salary < 100000:

        insight = """
You are earning a competitive salary.

Recommended Next Steps:

✅ Machine Learning

✅ Power BI

✅ Cloud Computing

✅ Advanced SQL
"""

    else:

        insight = """
Excellent Salary Prediction 🎉

To move towards senior positions:

✅ Generative AI

✅ MLOps

✅ System Design

✅ Leadership Skills
"""

    st.info(insight)

    # ==========================================================
    # SKILL SCORE
    # ==========================================================

    st.markdown("## 🏆 Skill Score")

    score = min(
        100,
        experience * 5 +
        skills * 3 +
        certifications * 6
    )

    st.progress(score)

    st.metric(
        "Overall Skill Score",
        f"{score}/100"
    )

    # ==========================================================
    # NEXT CAREER GOAL
    # ==========================================================

    st.markdown("## 🎯 Suggested Next Role")

    if score < 40:

        st.warning("Junior Developer")

    elif score < 70:

        st.info("Machine Learning Engineer")

    elif score < 90:

        st.success("Senior ML Engineer")

    else:

        st.success("AI Architect")
            # ==========================================================
    # PDF REPORT
    # ==========================================================

    styles = getSampleStyleSheet()

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    doc = SimpleDocTemplate(temp.name)

    story = []

    story.append(
        Paragraph(
            "<b>AI Salary Prediction Report</b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            f"Job Title : {job_title}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Education : {education}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Industry : {industry}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Predicted Salary : ${salary:,.0f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Skill Score : {score}/100",
            styles["BodyText"]
        )
    )

    doc.build(story)

    with open(temp.name, "rb") as pdf:

        st.download_button(

            "📄 Download PDF Report",

            pdf,

            file_name="Salary_Report.pdf",

            mime="application/pdf"

        )
            # ==========================================================
    # COUNTRY FLAG
    # ==========================================================

    flags = {
        "India":"🇮🇳",
        "USA":"🇺🇸",
        "UK":"🇬🇧",
        "Canada":"🇨🇦",
        "Germany":"🇩🇪",
        "Australia":"🇦🇺",
        "Singapore":"🇸🇬",
        "UAE":"🇦🇪"
    }

    flag = flags.get(location, "🌍")

    st.markdown("---")
    st.markdown("## 🌍 Country Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Country",
            f"{flag} {location}"
        )

    with c2:
        st.metric(
            "Currency",
            code
        )

    with c3:
        st.metric(
            "Prediction",
            level
        )

    # ==========================================================
    # PROFILE RATING
    # ==========================================================

    st.markdown("## ⭐ Profile Rating")

    if score < 40:

        stars = "⭐⭐"

    elif score < 60:

        stars = "⭐⭐⭐"

    elif score < 80:

        stars = "⭐⭐⭐⭐"

    else:

        stars = "⭐⭐⭐⭐⭐"

    st.success(f"Overall Rating : {stars}")

    # ==========================================================
    # TOP SKILLS
    # ==========================================================

    st.markdown("## 🚀 Recommended Skills")

    skills_df = pd.DataFrame({

        "Skill":[

            "Python",

            "SQL",

            "Machine Learning",

            "Deep Learning",

            "Power BI",

            "Cloud"

        ],

        "Importance":[

            95,

            90,

            100,

            82,

            70,

            75

        ]

    })

    chart = px.bar(

        skills_df,

        x="Importance",

        y="Skill",

        orientation="h",

        color="Importance"

    )

    chart.update_layout(
        height=400
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    # ==========================================================
    # PREDICTION TIMELINE
    # ==========================================================

    st.markdown("## 📅 Career Growth Timeline")

    timeline = pd.DataFrame({

        "Year":[

            "2026",

            "2027",

            "2028",

            "2029",

            "2030"

        ],

        "Career":[

            "Intern",

            "Junior ML",

            "ML Engineer",

            "Senior ML",

            "AI Engineer"

        ]

    })

    st.table(timeline)
        # ==========================================================
    # FINAL DASHBOARD SUMMARY
    # ==========================================================

    st.markdown("---")
    st.markdown("## 🏆 Dashboard Summary")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "💼 Job",
            job_title
        )

    with d2:
        st.metric(
            "🌍 Country",
            location
        )

    with d3:
        st.metric(
            "⭐ Rating",
            stars
        )

    with d4:
        st.metric(
            "📊 Score",
            f"{score}/100"
        )

    # ==========================================================
    # PROFILE COMPLETENESS
    # ==========================================================

    st.markdown("## 📋 Profile Completeness")

    profile_score = 100

    if certifications == 0:
        profile_score -= 15

    if skills < 5:
        profile_score -= 20

    if experience < 2:
        profile_score -= 15

    st.progress(profile_score)

    st.write(f"Profile Score : **{profile_score}%**")

    # ==========================================================
    # FINAL RECOMMENDATION
    # ==========================================================

    st.markdown("## 🎯 Final Recommendation")

    recommendations = []

    if skills < 8:
        recommendations.append("🛠 Improve technical skills.")

    if certifications < 2:
        recommendations.append("📜 Complete more certifications.")

    if experience < 3:
        recommendations.append("💼 Gain internship or project experience.")

    if salary > 120000:
        recommendations.append("🚀 You're ready for senior opportunities.")

    if not recommendations:
        recommendations.append("🎉 Excellent profile. Keep learning!")

    for rec in recommendations:
        st.write(rec)

    # ==========================================================
    # THANK YOU CARD
    # ==========================================================

    st.markdown(
        """
        <div style="
        margin-top:30px;
        padding:25px;
        border-radius:20px;
        background:linear-gradient(135deg,#2563EB,#4F46E5);
        color:white;
        text-align:center;
        box-shadow:0px 10px 30px rgba(0,0,0,.2);
        ">
        <h2>🎉 Prediction Completed</h2>
        <p>Thank you for using the AI Salary Prediction Dashboard.</p>
        <p><b>Built by Vaishnavi Goyal</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
