import pandas as pd

# Load dataset
df = pd.read_csv("data/student_performance.csv")

print("\n✅ Dataset Loaded Successfully")

# ---------------- Data Cleaning ---------------- #

# Check missing values
if df.isnull().sum().sum() > 0:
    print("⚠ Missing values detected. Filling with column mean.")
    df.fillna(df.mean(numeric_only=True), inplace=True)

# Validate marks range (0–100)
subjects = ["Maths", "Science", "English"]
for col in subjects:
    df[col] = df[col].clip(0, 100)

# Attendance validation
df["Attendance"] = df["Attendance"].clip(0, 100)

print("✅ Data Validation Completed")

# ---------------- Feature Engineering ---------------- #

# Total and Percentage
df["Total"] = df[subjects].sum(axis=1)
df["Percentage"] = (df["Total"] / 300) * 100

# Grade Logic
def assign_grade(p):
    if p >= 85:
        return "A"
    elif p >= 70:
        return "B"
    elif p >= 55:
        return "C"
    elif p >= 40:
        return "D"
    else:
        return "F"

df["Grade"] = df["Percentage"].apply(assign_grade)

# Risk Level Detection
def risk_level(row):
    if row["Percentage"] < 40 or row["Attendance"] < 60:
        return "High Risk"
    elif row["Percentage"] < 60:
        return "Medium Risk"
    else:
        return "Low Risk"

df["RiskLevel"] = df.apply(risk_level, axis=1)

# Attendance Impact Score
df["AttendanceImpact"] = df["Attendance"] * 0.3 + df["Percentage"] * 0.7

# Save processed dataset
output_path = "outputs/processed_student_data.csv"
df.to_csv(output_path, index=False)

print(f"\n🚀 Preprocessing Completed!")
print(f"📂 Processed file saved at: {output_path}")

print("\n📊 Sample Processed Data:\n")
print(df.head())
