
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "live_readings.csv"

st.set_page_config(
    page_title="Motor Anomaly Monitor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Live Motor Anomaly Detection")
st.caption("ESP32 simulation → MQTT → Machine-learning model → Dashboard")

# Refresh the dashboard every 2 seconds
st_autorefresh(interval=2000, key="live_refresh")

if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
    st.info("Waiting for sensor data. Keep Wokwi and mqtt_bridge.py running.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except (pd.errors.EmptyDataError, pd.errors.ParserError):
    st.warning("Sensor data is being written. Please wait for the next refresh.")
    st.stop()

if df.empty:
    st.info("Waiting for the first sensor reading...")
    st.stop()

latest = df.iloc[-1]
status = str(latest["predicted_status"])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Temperature", f"{latest['temperature']:.1f} °C")
c2.metric("Vibration", f"{latest['vibration']:.1f}")
c3.metric("Current", f"{latest['current']:.1f} A")

if status == "ANOMALY":
    c4.error("⚠️ ANOMALY DETECTED")
else:
    c4.success("✅ NORMAL")

st.subheader("Latest Prediction")
st.write(f"**Status:** {status}")
st.write(f"**Timestamp:** {latest['timestamp']}")

st.subheader("Live Sensor Trends")
chart_data = df.tail(60).copy()
chart_data["Reading"] = range(1, len(chart_data) + 1)

st.line_chart(
    chart_data.set_index("Reading")[["temperature", "vibration", "current"]]
)

st.subheader("Recent Sensor Readings")
st.dataframe(df.tail(20).iloc[::-1], use_container_width=True)

st.caption(
    f"Total readings received: {len(df)}. "
    "Predictions are based on a model trained on synthetic data "
    "and are for demonstration only."
)