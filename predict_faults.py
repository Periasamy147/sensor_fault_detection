import boto3
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ✅ Force TensorFlow to Run on CPU Only
tf.config.set_visible_devices([], "GPU")

# ✅ AWS Configuration
AWS_REGION = "ap-south-1"
DYNAMODB_TABLE = "TemperatureReadings"

# ✅ Load BiLSTM Model
MODEL_PATH = "bilstm_fault_detection.keras"
model = load_model(MODEL_PATH)

# ✅ Connect to DynamoDB
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

# ✅ Feature Scaler
scaler = StandardScaler()

# ✅ Prediction Threshold
FAULT_THRESHOLD = 0.5  

# ✅ Fetch Real-Time Data
def fetch_latest_data():
    """
    Fetch latest temperature readings from DynamoDB (last 60 seconds).
    Uses alias `#ts` for `timestamp` to avoid reserved keyword error.
    """
    time_threshold = datetime.now(timezone.utc) - timedelta(seconds=10)

    response = table.scan(
        FilterExpression="#ts >= :time",
        ExpressionAttributeNames={"#ts": "timestamp"},  
        ExpressionAttributeValues={":time": time_threshold.isoformat()},
    )

    items = response.get("Items", [])
    return items

# ✅ Fix: Ensure Correct Input Shape
def preprocess_data(data):
    """
    Converts raw DynamoDB data into model-ready input.
    """
    if not data or len(data) < 10:  
        return None  # ✅ Not enough data

    df = pd.DataFrame(data)
    
    # ✅ Ensure correct data types
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna().sort_values(by="timestamp")  # ✅ Remove NaNs and sort

    if len(df) < 10:
        return None  # ✅ Still not enough data after cleaning

    # ✅ Fix: Ensure exactly 10 readings (for time step consistency)
    temp_values = df["temperature"].values[-10:].reshape(-1, 1)

    # ✅ Fix: Standardize and reshape correctly
    temp_values = scaler.fit_transform(temp_values)
    temp_values = np.array([temp_values])  # ✅ Shape must be (1, 10, 1)

    return temp_values

# ✅ Predict Fault Function
def predict_fault(data):
    """
    Runs the BiLSTM model on processed data and prints fault status.
    """
    prediction = model.predict(data)
    probability = prediction[0][0]

    if probability >= FAULT_THRESHOLD:
        print(f"⚠️ FAULT DETECTED! Probability: {probability:.4f}")
    else:
        print(f"✅ NORMAL. Probability: {probability:.4f}")

# ✅ Real-Time Monitoring Loop
print("\n🚀 **Real-time Fault Detection Started...**")
while True:
    raw_data = fetch_latest_data()
    processed_data = preprocess_data(raw_data)

    if processed_data is None:
        print("⚠️ Not enough data yet... Waiting for more readings.")
    else:
        predict_fault(processed_data)

    time.sleep(2)  
