import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Features and target
X = df[["Maths", "Science", "English", "Attendance"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("✅ Model Accuracy:", accuracy)

# Predict new student
new_student = [[60, 55, 58, 70]]  # example data
result = model.predict(new_student)

print("🎯 Prediction for new student:", result[0])