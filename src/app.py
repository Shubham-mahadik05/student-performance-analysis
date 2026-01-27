import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Intelligence System")
st.markdown("Interactive Data Science & Machine Learning Dashboard")

# ---------------- Load Data ---------------- #
df = pd.read_csv("outputs/processed_student_data.csv")
st.subheader("📂 Student Dataset Preview")
st.dataframe(df)

# ---------------- Risk Distribution ---------------- #
st.subheader("📊 Risk Level Distribution")
risk_count = df["RiskLevel"].value_counts()
st.bar_chart(risk_count)

# ---------------- Model Training ---------------- #
features = [
    "Maths",
    "Science",
    "English",
    "Attendance",
    "Percentage",
    "AttendanceImpact"
]

X = df[features]
y = df["RiskLevel"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y_encoded)

# ---------------- Prediction UI ---------------- #
st.subheader("🤖 Predict Student Risk Level")

col1, col2, col3 = st.columns(3)

with col1:
    maths = st.number_input("Maths Marks", 0, 100, 70)
    science = st.number_input("Science Marks", 0, 100, 70)

with col2:
    english = st.number_input("English Marks", 0, 100, 70)
    attendance = st.number_input("Attendance (%)", 0, 100, 80)

with col3:
    percentage = (maths + science + english) / 3
    attendance_impact = attendance * 0.3 + percentage * 0.7

    st.info(f"📌 Percentage: {percentage:.2f}")
    st.info(f"📌 Attendance Impact: {attendance_impact:.2f}")

if st.button("Predict Risk"):
    input_data = [[maths, science, english, attendance, percentage, attendance_impact]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    risk_label = encoder.inverse_transform(prediction)

    st.success(f"🎯 Predicted Risk Level: {risk_label[0]}")

