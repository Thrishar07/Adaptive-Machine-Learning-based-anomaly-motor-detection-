
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Find project files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "motor_sensor_data.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "motor_anomaly_model.joblib"

FEATURES = ["temperature", "vibration", "current"]

st.set_page_config(
    page_title="Motor Anomaly Monitor",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Motor Anomaly Monitoring System")
st.caption(
    "Simulation-based motor condition monitoring "
    "using Isolation Forest"
)

# Check required files
if not DATA_FILE.exists():
    st.error("Dataset not found. Run src/generate_data.py first.")
    st.stop()

if not MODEL_FILE.exists():
    st.error("Trained model not found. Run src/train_model.py first.")
    st.stop()

# Load data and trained model
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

data = load_data(str(DATA_FILE))
model = load_model(str(MODEL_FILE))

# Keep replay state between Streamlit reruns
if "reading_index" not in st.session_state:
    st.session_state.reading_index = 0

if "monitor_running" not in st.session_state:
    st.session_state.monitor_running = False

# Sidebar controls
st.sidebar.header("Monitoring Controls")
st.sidebar.write("Replay the simulated sensor readings.")

if st.sidebar.button("▶ Start / Resume"):
    st.session_state.monitor_running = True

if st.sidebar.button("⏸ Pause"):
    st.session_state.monitor_running = False

if st.sidebar.button("↻ Reset"):
    st.session_state.reading_index = 0
    st.session_state.monitor_running = False

st.sidebar.write(
    f"Reading: {st.session_state.reading_index + 1} "
    f"of {len(data)}"
)

# Advance one reading when requested
if st.session_state.monitor_running:
    if st.sidebar.button("Next Reading"):
        if st.session_state.reading_index < len(data) - 1:
            st.session_state.reading_index += 1
        else:
            st.session_state.monitor_running = False
            st.sidebar.info("End of dataset reached.")

# Choose the current reading
index = st.session_state.reading_index
reading = data.iloc[index]
sensor_values = reading[FEATURES].to_frame().T

# Predict: Isolation Forest returns 1 for normal, -1 for anomaly
prediction = int(model.predict(sensor_values)[0])
is_anomaly = prediction == -1

# Top status
if is_anomaly:
    st.error("⚠️ ANOMALY DETECTED — Unusual sensor pattern")
else:
    st.success("✅ NORMAL — No anomaly detected by the model")

st.caption(
    f"Simulated reading {index + 1} of {len(data)}"
)

# Sensor measurements
col1, col2, col3 = st.columns(3)

col1.metric(
    "Temperature",
    f"{reading['temperature']:.1f} °C"
)

col2.metric(
    "Vibration",
    f"{reading['vibration']:.2f}"
)

col3.metric(
    "Motor Current",
    f"{reading['current']:.2f} A"
)

st.divider()

# Prepare predictions for the readings already viewed
viewed_data = data.iloc[:index + 1].copy()
viewed_predictions = model.predict(viewed_data[FEATURES])

viewed_data["predicted_status"] = [
    "Anomaly" if p == -1 else "Normal"
    for p in viewed_predictions
]

viewed_data["actual_status"] = viewed_data["label"].map({
    0: "Normal",
    1: "Abnormal"
})

# Sensor trends
st.subheader("Sensor Trends")

chart_data = viewed_data[
    ["temperature", "vibration", "current"]
].reset_index(drop=True)

st.line_chart(chart_data)

# History
st.subheader("Reading History")

st.dataframe(
    viewed_data[
        FEATURES + ["actual_status", "predicted_status"]
    ].tail(20),
    width="stretch",
    hide_index=True
)

# Summary for readings viewed so far
st.subheader("Monitoring Summary")

total_viewed = len(viewed_data)
detected = int((viewed_predictions == -1).sum())

col4, col5 = st.columns(2)
col4.metric("Readings Viewed", total_viewed)
col5.metric("Readings Flagged", detected)

st.info(
    "This dashboard replays prerecorded synthetic data. "
    "It does not yet receive live ESP32 sensor readings. "
    "An anomaly is a model prediction, not definitive proof "
    "of motor failure."
)
