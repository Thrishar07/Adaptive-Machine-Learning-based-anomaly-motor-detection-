
import json
import csv
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "motor_anomaly_model.joblib"
OUTPUT_FILE = ROOT / "data" / "live_readings.csv"

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "motor-rbp-demo/srush-8264/sensors"

FEATURES = ["temperature", "vibration", "current"]
COLUMNS = [
    "timestamp",
    "temperature",
    "vibration",
    "current",
    "predicted_status",
]

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_FILE}. Run src/train_model.py first."
    )

model = joblib.load(MODEL_FILE)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT connection result:", reason_code)
    if reason_code == 0:
        client.subscribe(TOPIC)
        print("Subscribed to:", TOPIC)
        print("Waiting for Wokwi sensor readings...")


def on_message(client, userdata, message):
    try:
        reading = json.loads(message.payload.decode("utf-8"))
        values = {name: float(reading[name]) for name in FEATURES}

        sample = pd.DataFrame([values], columns=FEATURES)
        prediction = int(model.predict(sample)[0])
        status = "ANOMALY" if prediction == -1 else "NORMAL"

        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            **values,
            "predicted_status": status,
        }

        exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0
        with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"{status} | Temperature: {values['temperature']:.1f} °C | "
            f"Vibration: {values['vibration']:.1f} | "
            f"Current: {values['current']:.1f} A"
        )

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print("Invalid sensor message:", error)


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="motor-python-subscriber-srush",
)
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to {BROKER}:{PORT} ...")

try:
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping MQTT bridge.")
except Exception as error:
    print("MQTT connection error:", error)
finally:
    client.disconnect()
