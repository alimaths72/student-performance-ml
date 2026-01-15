import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from load_data import load_and_merge_data
from preprocess import prepare_features

# ===============================
# Project paths
# ===============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ===============================
# Load and preprocess
# ===============================
print("Loading data...")
data = load_and_merge_data()

print("Preparing features...")
X, y = prepare_features(data)

# Save feature layout for prediction
feature_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
joblib.dump(X.columns.tolist(), feature_path)

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train model
# ===============================
print("Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("R2:", r2)
print("RMSE:", rmse)

# ===============================
# Save model
# ===============================
model_path = os.path.join(MODELS_DIR, "rf_with_grades.pkl")
joblib.dump(model, model_path)

print("Model saved to:", model_path)
print("Feature columns saved to:", feature_path)
