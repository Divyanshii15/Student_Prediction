import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E3F2FD, #F3E5F5);
}

.main-title {
    text-align: center;
    color: #1E3A8A;
    font-size: 42px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #4B5563;
    font-size: 20px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: rgba(255,255,255,0.9);
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load Dataset
data = pd.read_csv("student-mat.csv")

# Create Pass/Fail Column
data['Result'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Features and Target
X = data[['studytime', 'absences', 'G1', 'G2']]
y = data['Result']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Header
st.markdown(
    '<p class="main-title">🎓 Student Performance Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Machine Learning Based Academic Success Predictor</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# Dashboard Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📚 Students Analyzed", len(data))

with col2:
    st.metric("🎓 Academic Insights", "Enabled")

with col3:
    st.metric("⚡ System Status", "Active")

st.info(
    "This intelligent prediction system analyzes study patterns, attendance "
    "behavior, and academic performance indicators to estimate whether a "
    "student is likely to pass or fail."
)

st.markdown("---")

# Sidebar
st.sidebar.header("📋 Student Information")

studytime = st.sidebar.slider(
    "Study Time",
    min_value=1,
    max_value=4,
    value=2
)

absences = st.sidebar.slider(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5
)

g1 = st.sidebar.slider(
    "G1 Score",
    min_value=0,
    max_value=20,
    value=10
)

g2 = st.sidebar.slider(
    "G2 Score",
    min_value=0,
    max_value=20,
    value=10
)

# Prediction Button
if st.button("🔍 Predict Student Result"):

    prediction = model.predict(
        [[studytime, absences, g1, g2]]
    )[0]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if prediction == 1:
        st.success("✅ Prediction Result: PASS")
        st.balloons()
        st.write(
            "Based on the provided academic indicators, "
            "the student is likely to successfully pass."
        )

    else:
        st.error("❌ Prediction Result: FAIL")
        st.write(
            "Based on the provided academic indicators, "
            "the student may require additional academic support."
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

st.success("✔ System Ready for Real-Time Student Performance Prediction")

st.caption("Developed by Divyanshi | Machine Learning Project")

