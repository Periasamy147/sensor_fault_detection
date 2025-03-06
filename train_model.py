import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
file_path = "./sensor_fault_dataset.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Convert timestamps to datetime and sort by time
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by="timestamp")

# Compute natural Z-score BEFORE scaling
mean_temp = df["temperature"].mean()
std_temp = df["temperature"].std()
df["z_score"] = (df["temperature"] - mean_temp) / std_temp

# Normalize temperature separately
scaler = StandardScaler()
df["temperature"] = scaler.fit_transform(df[["temperature"]])
joblib.dump(scaler, "scaler.pkl")

# Keep Z-score unchanged
df["z_score"] = np.clip(df["z_score"], 0.85, 0.95)

# Encode fault labels
label_encoder = LabelEncoder()
df["fault_label"] = label_encoder.fit_transform(df["fault_type"])  
joblib.dump(label_encoder, "label_encoder.pkl")

# Select relevant features
df = df[["temperature", "z_score", "fault_label"]]

# Convert data into sequences for LSTM
SEQUENCE_LENGTH = 20  # Increased for better learning

def create_sequences(data, labels, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y)

# Convert data to sequences
X, y = create_sequences(df[["temperature", "z_score"]].values, df["fault_label"].values, SEQUENCE_LENGTH)
X = X.reshape((X.shape[0], X.shape[1], 2))  # (Samples, Timesteps, Features)
y = to_categorical(y)  # Convert to multi-class format

# Correct Multi-class Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        pt = tf.exp(-ce)
        return K.mean(alpha * (1 - pt) ** gamma * ce)
    return loss

# Build optimized BiLSTM Model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, activation="tanh"), input_shape=(SEQUENCE_LENGTH, 2)),
    BatchNormalization(),
    Dropout(0.1),

    Bidirectional(LSTM(64, return_sequences=False, activation="tanh")),
    BatchNormalization(),
    Dropout(0.1),

    Dense(64, activation="relu"),
    Dropout(0.1),

    Dense(y.shape[1], activation="softmax")  
])

# Compile with correct loss function
model.compile(
    optimizer=AdamW(learning_rate=0.0003, weight_decay=0.01),  # AdamW for stability
    loss=focal_loss(),
    metrics=["accuracy", tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
)

# Train model
EPOCHS = 40  # Increased for better convergence
BATCH_SIZE = 64

history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)

# Save trained model
model.save("bilstm_fault_detection.keras")

print("\n✅ Model training complete! Saved as bilstm_fault_detection.keras")

# Log Final Metrics
final_mae = history.history['val_mean_absolute_error'][-1]
final_mse = history.history['val_mean_squared_error'][-1]

print(f"📉 Final Validation Mean Absolute Error (MAE): {final_mae:.5f}")
print(f"📉 Final Validation Mean Squared Error (MSE): {final_mse:.5f}")

# Verify Z-score range after scaling
z_score_mean = df["z_score"].mean()
z_score_std = df["z_score"].std()
print(f"✅ Z-score After Scaling → Mean: {z_score_mean:.5f}, Std Dev: {z_score_std:.5f}")
