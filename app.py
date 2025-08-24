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

# Define feature names
feature_names = [
    "Tenth", "Twelfth", "CGPA", "Internships", "Projects", 
    "Aptitude", "Soft skills", "Leadership", "Extracurricular", 
    "Certifications", "Backlogs", "DSA", "Lang"
]

st.title("Placement Prediction Interface")

st.write("### Please enter your details:")

# # Step 1: User input
# user_input = {}
# for feature in feature_names:
#     user_input[feature] = st.number_input(f"{feature}:", value=0.0)

# # Step 2: Convert to DataFrame
# user_df = pd.DataFrame([user_input])
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
    dsa = st.number_input("💻 DSA Knowledge (0.0 - 100.0)", min_value=0.0, max_value=100.0, value=0.0, step=10.0)

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


# Step 3: Pass through pipeline
prepared_data = pipeline.transform(user_df)

# Step 4: Predict
if st.button("Predict"):
    prediction = model.predict(prepared_data)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(prepared_data)[0][1] * 100
        st.success(f"Prediction: {prediction[0]} (Probability: {prob:.2f}%)")
    else:
          st.success(f"Prediction: {prediction[0]}")


          # === Footer ===
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 16px; color: grey;">
        Managed by <b> Sparsh , Sumit , Sangam , Sunny , Aryan</b>
    </div>
    """,
    unsafe_allow_html=True
)
