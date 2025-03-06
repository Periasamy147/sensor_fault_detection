import boto3
import joblib
import numpy as np
import tensorflow as tf
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import time
import json
from datetime import datetime, timezone

# AWS DynamoDB Configuration
dynamodb = boto3.client('dynamodb', region_name='ap-south-1')  # Change region as needed
table_name = "TemperatureReadings"

# Load trained BiLSTM model
print("✅ Loading model...")
model = tf.keras.models.load_model("sensor_fault_model.keras")  # Use .keras format
print("✅ Model loaded successfully!")

# Store last N readings for feature computation
window_size = 5  # Adjust based on training data
temperature_history = deque(maxlen=window_size)

# Load the trained MinMaxScaler
try:
    scaler = joblib.load("scaler.pkl")  # Ensure scaler is properly trained
    print("✅ Scaler loaded successfully!")
except FileNotFoundError:
    print("❌ Error: scaler.pkl not found. Ensure it's created in preprocess_data.py.")
    exit()

# Function to compute features
def compute_features(temp):
    """
    Computes meaningful features from temperature readings.
    """
    temperature_history.append(temp)
    
    if len(temperature_history) < window_size:
        return None  # Not enough data yet

    temp_array = np.array(temperature_history)

    # Feature extraction
    moving_avg = np.mean(temp_array)
    rate_of_change = temp_array[-1] - temp_array[-2] if len(temp_array) > 1 else 0
    std_dev = np.std(temp_array)
    min_temp = np.min(temp_array)
    max_temp = np.max(temp_array)
    rolling_diff = np.diff(temp_array).mean() if len(temp_array) > 1 else 0

    return np.array([temp, moving_avg, rate_of_change, std_dev, min_temp, max_temp])

# Function to preprocess real-time data
def preprocess_data(temp):
    """
    Transforms raw temperature into a feature vector suitable for LSTM input.
    """
    features = compute_features(temp)

    if features is None:
        return None  # Skip prediction if not enough history

    # Normalize using the trained scaler
    try:
        features = scaler.transform(features.reshape(1, -1))  
    except ValueError as e:
        print(f"❌ Error in scaler.transform: {e}")
        return None

    return features.reshape(1, 1, 6)  # Reshape for LSTM input

# Function to fetch latest temperature data from DynamoDB
def fetch_latest_temperature():
    try:
        response = dynamodb.scan(
            TableName=table_name,
            Limit=1,  
            ScanFilter={
                "timestamp": {
                    "ComparisonOperator": "GT",
                    "AttributeValueList": [{"S": "0"}]  # Fetch latest record
                }
            }
        )
        
        items = response.get("Items", [])
        if items:
            return float(items[0]["temperature"]["N"])  # Convert to float
    except Exception as e:
        print(f"❌ Error fetching data from DynamoDB: {e}")

    return None

# Real-time fault detection loop
print("🔍 Starting real-time fault detection...")

while True:
    temp = fetch_latest_temperature()
    
    if temp is not None:
        input_data = preprocess_data(temp)
        
        if input_data is not None:
            prediction = model.predict(input_data)
            fault_label = np.argmax(prediction)  # Get predicted fault class
            
            # Display results
            timestamp = datetime.now(timezone.utc).isoformat()  # ✅ Fixed timestamp
            if fault_label != 0:  # Assuming 0 is "No Fault"
                print(f"⏱️ {timestamp} | 🚨 SENSOR FAULT DETECTED (Class {fault_label})")
            else:
                print(f"⏱️ {timestamp} | ✅ No fault detected.")
        else:
            print("⚠️ Not enough data yet for prediction.")
    else:
        print("⚠️ No new temperature data received.")

    time.sleep(2)  # Fetch data every 2 seconds