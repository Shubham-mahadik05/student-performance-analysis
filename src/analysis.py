import pandas as pd
import matplotlib.pyplot as plt

# Load dataset

df = pd.read_csv("data/student_performance.csv")

# Calculate total and average
df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
df["Average"] = df["Total"] / 3

# Sort by average
top_students = df.sort_values(by="Average", ascending=False)

print("\n📊 Student Performance Summary:\n")
print(top_students[["RollNo", "Name", "Average"]])

# Save result
top_students.to_csv("outputs/performance_report.csv", index=False)

# Visualization
plt.bar(df["Name"], df["Average"])
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Student Average Performance")
plt.show()