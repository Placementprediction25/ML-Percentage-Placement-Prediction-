import streamlit as st
import pandas as pd
import joblib

# Load pipeline and model
pipeline = joblib.load("pipeline.joblib")   # preprocessing pipeline
model = joblib.load("Placement.joblib")         # trained ML model

st.set_page_config(page_title="Career Predictor 🎓💼", layout="wide")

# 🎨 Background styling
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1503264116251-35a269479413?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    position: relative;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.6); /* 🔥 dark overlay */
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1; /* ensures text stays above overlay */
    color: white; /* white text for visibility */
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🚀 Career Success Probability Predictor")
st.write("Fill in your academic & skill details, and get your predicted success probability 🎯")

# === User Inputs ===
col1, col2 = st.columns(2)

with col1:
    tenth = st.number_input("📘 10th Percentage", min_value=0.0, max_value=100.0, value=0.0)
    twelfth = st.number_input("📗 12th Percentage", min_value=0.0, max_value=100.0, value=0.0)
    cgpa = st.number_input("🎓 CGPA", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    internships = st.slider("💼 Internships", 0, 10, 1)
    projects = st.slider("📂 Projects", 0, 20, 2)
    aptitude = st.number_input("🧠 Aptitude Score", min_value=0, max_value=100, value=0)

with col2:
    soft_skills = st.number_input("💬 Soft Skills (0-10)", min_value=0, max_value=10, value=0)
    leadership = st.slider("👑 Leadership", 0, 10, 3)
    extracurricular = st.slider("⚽ Extracurricular Activities", 0, 10, 2)
    certifications = st.selectbox("📜 Certifications (0-10)", list(range(0, 11)), index=0)
    backlogs = st.selectbox("📉 Backlogs (0-10)", list(range(0, 11)), index=0)
    dsa = st.selectbox("💻 DSA Knowledge (0-10)", list(range(0, 11)), index=0)
    lang = st.selectbox("🌐 Programming Languages Known (0-10)", list(range(0, 11)), index=0)

# === Convert to DataFrame ===
user_data = {
    "Tenth": [tenth],
    "Twelfth": [twelfth],
    "CGPA": [cgpa],
    "Internships": [internships],
    "Projects": [projects],
    "Aptitude": [aptitude],
    "Soft skills": [soft_skills],
    "Leadership": [leadership],
    "Extracurricular": [extracurricular],
    "Certifications": [certifications],
    "Backlogs": [backlogs],
    "DSA": [dsa],
    "Lang": [lang]
}
user_df = pd.DataFrame(user_data)

# === Fix: Align with pipeline features ===
expected_features = pipeline.feature_names_in_
user_df = user_df.reindex(columns=expected_features, fill_value=0)

# === Prediction ===
if st.button("🔮 Predict Probability"):
    prepared_data = pipeline.transform(user_df)
    probability = model.predict_proba(prepared_data)[0][1] * 100  # Success probability %

    st.subheader(f"🎯 Predicted Success Probability: {probability:.2f}%")
    if probability > 70:
        st.success("🔥 Excellent chances of success! Keep it up!")
    elif probability > 50:
        st.info("👍 Good potential, keep improving skills!")
    else:
        st.warning("⚡ Work harder on skills & academics to increase your chances.")
