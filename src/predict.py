import joblib
import pandas as pd
import os
from preprocess import prepare_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "rf_with_grades.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading feature layout...")
feature_columns = joblib.load(FEATURES_PATH)

print("\n=== Student Performance Predictor ===\n")

age = int(input("Age: "))
sex = input("Gender (M/F): ").lower()
studytime = int(input("Study time (1-4): "))
failures = int(input("Past failures: "))
absences = int(input("Absences: "))
health = int(input("Health (1-5): "))
internet = input("Internet (yes/no): ").lower()
g1 = int(input("First exam (G1): "))
g2 = int(input("Second exam (G2): "))

data = {
    "age": age,
    "sex": sex,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "health": health,
    "internet": internet,
    "G1": g1,
    "G2": g2
}

df = pd.DataFrame([data])

X, _ = prepare_features(df, for_training=False)

# align to training columns
X = X.reindex(columns=feature_columns, fill_value=0)

prediction = model.predict(X)[0]

print("\nPredicted Final Grade (G3):", round(prediction, 2))
