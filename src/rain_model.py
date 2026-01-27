import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load processed data
df = pd.read_csv("outputs/processed_student_data.csv")

print("\n✅ Processed Dataset Loaded")

# Select features and target
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

# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Models ---------------- #

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

print("\n🚀 Training Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"📌 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

# Best Model Selection
best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model_name}")

