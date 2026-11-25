import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Number of simulated readings
normal_count = 1800
abnormal_count = 200

# Normal readings: match the Wokwi ESP32 simulation
normal_data = pd.DataFrame({
    "temperature": np.random.uniform(37.0, 46.0, normal_count),
    "vibration": np.random.uniform(2.0, 3.5, normal_count),
    "current": np.random.uniform(1.8, 2.3, normal_count),
    "label": 0
})

# Abnormal readings: simulate increased sensor values
abnormal_data = pd.DataFrame({
    "temperature": np.random.uniform(57.0, 66.0, abnormal_count),
    "vibration": np.random.uniform(5.0, 6.5, abnormal_count),
    "current": np.random.uniform(3.3, 3.8, abnormal_count),
    "label": 1
})

# Combine and shuffle the dataset
data = pd.concat(
    [normal_data, abnormal_data],
    ignore_index=True
)

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save the dataset
project_root = Path(__file__).resolve().parent.parent
output_folder = project_root / "data"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "motor_sensor_data.csv"
data.to_csv(output_file, index=False)

print("Motor sensor dataset generated successfully!")
print(f"Saved to: {output_file}")
print(f"Total readings: {len(data)}")
print(f"Normal readings: {(data['label'] == 0).sum()}")
print(f"Abnormal readings: {(data['label'] == 1).sum()}")
print("\nFirst five rows:")
print(data.head())