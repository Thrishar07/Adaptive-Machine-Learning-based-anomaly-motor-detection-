
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# Locate project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "motor_sensor_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the simulated dataset
data = pd.read_csv(DATA_FILE)

features = ["temperature", "vibration", "current"]
X = data[features]
y = data["label"]

print("Dataset loaded successfully.")
print(f"Total readings: {len(data)}")

# Split NORMAL readings into training and testing sets.
# The model learns only from normal motor behavior.
normal_data = data[data["label"] == 0]
abnormal_data = data[data["label"] == 1]

X_normal_train, X_normal_test = train_test_split(
    normal_data[features],
    test_size=0.20,
    random_state=42
)

# Test on unseen normal readings and all simulated abnormal readings
X_test = pd.concat(
    [X_normal_test, abnormal_data[features]],
    ignore_index=True
)

y_test = np.concatenate([
    np.zeros(len(X_normal_test), dtype=int),
    np.ones(len(abnormal_data), dtype=int)
])

print(f"Training readings: {len(X_normal_train)}")
print(f"Testing readings: {len(X_test)}")

# Create and train the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

model.fit(X_normal_train)

# Isolation Forest returns 1 for normal and -1 for anomaly
raw_predictions = model.predict(X_test)

# Convert predictions to our label convention:
# 0 = normal, 1 = anomaly
predictions = np.where(raw_predictions == -1, 1, 0)

# Evaluate the model
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

print("\n--- MODEL EVALUATION ---")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

print("\nDetailed classification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

# Save the trained model
model_file = MODEL_DIR / "motor_anomaly_model.joblib"
joblib.dump(model, model_file)
print(f"Model saved to: {model_file}")

# Create a confusion matrix plot
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

display.plot(values_format="d")
plt.title("Motor Anomaly Detection - Confusion Matrix")
plt.tight_layout()

plot_file = REPORT_DIR / "anomaly_results.png"
plt.savefig(plot_file, dpi=150)
plt.close()

print(f"Confusion matrix saved to: {plot_file}")
print("\nTraining and evaluation completed.")
