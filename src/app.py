import streamlit as st
import pandas as pd
from PIL import Image
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Intelligence System")
st.markdown("Interactive Data Science & Machine Learning Dashboard")

# Add custom CSS for better styling
st.markdown("""
<style>
    .chart-container {
        margin: 20px 0;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Create tabs for different sections
tabs = st.tabs(["📊 Overview", "📈 Visualizations", "🤖 Predictions", "📋 Data"])

# ================== TAB 1: Overview ================== #
with tabs[0]:
    st.subheader("Welcome to Student Performance Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis and visualization of student performance data.
    
    **Features:**
    - 📊 10 detailed visualizations of student performance
    - 🎯 Real-time risk level predictions
    - 📈 Statistical analysis and correlations
    - 👥 Individual student performance tracking
    """)
    
    # Load and display dataset summary
    df = pd.read_csv("outputs/processed_student_data.csv")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Pass Rate", f"{(df['Result'] == 'Pass').sum() / len(df) * 100:.1f}%")
    with col3:
        st.metric("Avg Performance", f"{df['Percentage'].mean():.1f}%")
    with col4:
        st.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")

# ================== TAB 2: Visualizations ================== #
with tabs[1]:
    st.subheader("📊 Performance Visualizations")
    
    # Define chart directory
    chart_dir = "outputs/charts"
    
    # Check if charts exist
    if os.path.exists(chart_dir):
        charts = sorted([f for f in os.listdir(chart_dir) if f.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_')) and f.endswith('.png')])
        
        if len(charts) > 0:
            st.success(f"✅ Found {len(charts)} visualizations")
            
            # Create 2-column layout for charts
            col1, col2 = st.columns(2)
            
            for idx, chart_file in enumerate(charts):
                chart_path = os.path.join(chart_dir, chart_file)
                
                # Clean up the filename for display
                display_name = chart_file.replace('_', ' ').replace('.png', '').replace('01 ', '').replace('02 ', '').replace('03 ', '').replace('04 ', '').replace('05 ', '').replace('06 ', '').replace('07 ', '').replace('08 ', '').replace('09 ', '').replace('10 ', '')
                
                try:
                    img = Image.open(chart_path)
                    
                    # Alternate between columns
                    if idx % 2 == 0:
                        with col1:
                            st.markdown(f"**Chart {idx + 1}: {display_name}**")
                            st.image(img, use_container_width=True)
                    else:
                        with col2:
                            st.markdown(f"**Chart {idx + 1}: {display_name}**")
                            st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {chart_file}: {e}")
        else:
            st.warning("⚠️ No charts found. Please run analysis.py first.")
    else:
        st.error("❌ Charts directory not found. Please generate charts first.")

# ================== TAB 3: Predictions ================== #
with tabs[2]:
    st.subheader("🤖 Predict Student Risk Level")
    
    # Load data for training
    df = pd.read_csv("outputs/processed_student_data.csv")
    
    # Model Training
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
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_encoded)
    
    st.markdown("### Enter Student Marks:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        maths = st.slider("Maths Marks", 0, 100, 70)
    
    with col2:
        science = st.slider("Science Marks", 0, 100, 70)
    
    with col3:
        english = st.slider("English Marks", 0, 100, 70)
    
    col4, col5 = st.columns(2)
    
    with col4:
        attendance = st.slider("Attendance (%)", 0, 100, 80)
    
    # Calculate derived metrics
    percentage = (maths + science + english) / 300 * 100
    attendance_impact = attendance * 0.3 + percentage * 0.7
    
    # Display calculated metrics
    st.markdown("### Calculated Metrics:")
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.info(f"📌 **Percentage**: {percentage:.2f}%")
    
    with col_metrics2:
        st.info(f"📌 **Attendance Impact**: {attendance_impact:.2f}")
    
    # Prediction button
    if st.button("🔮 Predict Risk Level", key="predict_btn"):
        try:
            input_data = [[maths, science, english, attendance, percentage, attendance_impact]]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            risk_label = encoder.inverse_transform(prediction)[0]
            
            # Display prediction with color coding
            if risk_label == "Low Risk":
                st.success(f"✅ **Predicted Risk Level: {risk_label}**", icon="✓")
            elif risk_label == "Medium Risk":
                st.warning(f"⚠️ **Predicted Risk Level: {risk_label}**", icon="⚠")
            else:
                st.error(f"❌ **Predicted Risk Level: {risk_label}**", icon="✕")
            
            # Show confidence scores
            st.markdown("### Confidence Scores:")
            risk_classes = encoder.classes_
            for class_name, prob in zip(risk_classes, probability[0]):
                st.progress(prob, text=f"{class_name}: {prob*100:.2f}%")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ================== TAB 4: Data ================== #
with tabs[3]:
    st.subheader("📋 Student Dataset")
    
    # Load data
    df = pd.read_csv("outputs/processed_student_data.csv")
    
    # Display full dataset
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download button for data
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name="student_performance_data.csv",
        mime="text/csv"
    )
    
    # Data Statistics
    st.subheader("📊 Data Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Summary:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Risk Level Distribution:**")
        risk_dist = df["RiskLevel"].value_counts()
        st.bar_chart(risk_dist)
    
    st.write("**Result Distribution:**")
    result_dist = df["Result"].value_counts()
    st.bar_chart(result_dist)

# ================== FOOTER ================== #
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666; font-size: 12px;'>
    <p>🎓 <b>Student Performance Intelligence System</b></p>
    <p>Developed by: <b>Shubham Mahadik</b></p>
    <p style='margin-top: 10px; font-size: 11px; color: #999;'>
        © 2026 | Data Science & Machine Learning Dashboard | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)

