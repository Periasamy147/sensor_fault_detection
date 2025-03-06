# Re-running the dataset generation since execution state was reset

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants for dataset generation
NUM_SAMPLES = 50000  # Total dataset size
SEQUENCE_LENGTH = 10  # LSTM sequence window
MIN_TEMP = 45
MAX_TEMP = 80
FAULT_TYPES = ["normal", "stuck_at", "drift", "noise", "out_of_range", "intermittent", "calibration"]
TIMESTAMP_START = datetime(2025, 3, 3, 0, 0, 0)  # Start date for timestamps

# Helper function to generate normal temperature readings
def generate_normal_temperature():
    return round(np.random.uniform(MIN_TEMP, MAX_TEMP), 2)

# Function to generate faulty sensor readings
def generate_faulty_temperature(fault_type, last_value):
    if fault_type == "stuck_at":
        return last_value  # Repeats the last value
    elif fault_type == "drift":
        return round(last_value + np.random.uniform(0.5, 2), 2)  # Gradual increase
    elif fault_type == "noise":
        return round(last_value + np.random.uniform(-1.5, 1.5), 2)  # Small random noise
    elif fault_type == "out_of_range":
        return round(np.random.uniform(150, 200), 2)  # Extreme value
    elif fault_type == "intermittent":
        return last_value if np.random.rand() > 0.2 else generate_normal_temperature()  # 20% change
    elif fault_type == "calibration":
        return round(last_value + 10, 2)  # Offset error
    return generate_normal_temperature()

# Generate dataset
data = []
last_temperature = generate_normal_temperature()  # Start with normal temp

for i in range(NUM_SAMPLES):
    timestamp = TIMESTAMP_START + timedelta(seconds=i * 2)  # Increment time every 2 sec
    
    # Determine fault type (balanced dataset)
    if i % 7 == 0:
        fault_type = np.random.choice(FAULT_TYPES[1:])  # Faulty case
    else:
        fault_type = "normal"  # Normal case
    
    if fault_type == "normal":
        temperature = generate_normal_temperature()
    else:
        temperature = generate_faulty_temperature(fault_type, last_temperature)

    last_temperature = temperature

    data.append([
        "DHT22-1", timestamp.isoformat(), temperature, fault_type, i // SEQUENCE_LENGTH
    ])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["sensor_id", "timestamp", "temperature", "fault_type", "sequence_id"])

# Save dataset
dataset_path = "./sensor_fault_dataset.csv"
df.to_csv(dataset_path, index=False)
dataset_path
