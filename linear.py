import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv("data.csv")
data['created_at'] = pd.to_datetime(data['created_at'])
data['created_at_timestamp'] = data['created_at'].astype(int) / 10**9

# Create lag features for the previous 5 outputs
data['moisture_value_lag1'] = data['moisture_value'].shift(1)
data['moisture_value_lag2'] = data['moisture_value'].shift(2)
data['moisture_value_lag3'] = data['moisture_value'].shift(3)
data['moisture_value_lag4'] = data['moisture_value'].shift(4)
data['moisture_value_lag5'] = data['moisture_value'].shift(5)
data = data.dropna()

# Prepare features and target
X = data[['created_at_timestamp', 'moisture_value_lag1', 'moisture_value_lag2', 
          'moisture_value_lag3', 'moisture_value_lag4', 'moisture_value_lag5']].values
y = data['moisture_value'].values

# Scale the features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(6,)),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam',
             loss='mean_squared_error',
             metrics=['mae'])

# Add callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(X_scaled, y_scaled,
                   epochs=1028,
                   batch_size=300,
                   validation_split=0.2,
                   callbacks=[early_stopping])

# Historical data for plotting
original_timestamps = data['created_at_timestamp'].values
original_values = data['moisture_value'].values

# Predict 1 day into the future in 30-second intervals
future_steps = 288  # 1 day with 30-second intervals = 2880 steps
last_row = data.iloc[-1]
current_input = np.array([[last_row['created_at_timestamp'],
                          last_row['moisture_value_lag1'],
                          last_row['moisture_value_lag2'],
                          last_row['moisture_value_lag3'],
                          last_row['moisture_value_lag4'],
                          last_row['moisture_value_lag5']]])
current_input_scaled = scaler_X.transform(current_input)

future_predictions = []
future_timestamps = []
current_timestamp = last_row['created_at_timestamp']

for step in range(future_steps):
    if step % 28 == 0:
        print(step)

    next_scaled = model.predict(current_input_scaled, verbose=0)
    next_value = scaler_y.inverse_transform(next_scaled)[0][0]  # Extract the single value
    future_predictions.append(next_value)
    current_timestamp += 300  # Add 30 seconds
    future_timestamps.append(current_timestamp)
    
    # Update input array with new values
    new_input = np.array([[
        current_timestamp,
        next_value,  # Use previous prediction as lag1
        current_input[0, 1],  # Shift previous lags
        current_input[0, 2],
        current_input[0, 3],
        current_input[0, 4]
    ]])
    current_input = new_input
    current_input_scaled = scaler_X.transform(current_input)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(original_timestamps, original_values, s=1, label='Original Data')
plt.plot(future_timestamps, future_predictions, '-', label='Future Predictions', color='red')
plt.title('Moisture Value Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Moisture Value')
plt.legend()
plt.grid()
plt.show()

