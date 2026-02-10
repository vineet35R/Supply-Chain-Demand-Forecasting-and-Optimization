import numpy as np
import pandas as pd
import tensorflow as tf
import os
from preprocessing import load_data, preprocess_data
from model import create_lstm_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Constants
# Use the file provided by the user
DATA_PATH = 'demand_forecasting_dataset.csv' 
MODEL_PATH = 'models/lstm_model.h5'
SEQUENCE_LENGTH = 30
EPOCHS = 20 # Sufficient for clean data
BATCH_SIZE = 32

def train():
    print("Loading data...", flush=True)
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    # Preprocess returns features list as 4th element
    X, y, scaler, features = preprocess_data(df, SEQUENCE_LENGTH)
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Input features: {features}")
    
    print("Creating model...")
    # Input shape is (Sequence Length, Number of Features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("Evaluating model...")
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Inverse transform to get actual values for metrics
    # We need to create dummy arrays because scaler expects (n_samples, n_features)
    # The scaler was fitted on: ['sales_units', 'price', ...] (7 features)
    # We only have the target (scaled_data[:, target_idx]) in y_test/y_pred.
    
    # 1. Reconstruct dummy variables with 0s
    n_features = len(features) 
    target_idx = features.index('sales_units')
    
    # helper to inverse transform
    def inverse_transform_target(y_scaled, scaler, n_features, target_idx):
        dummy = np.zeros((len(y_scaled), n_features))
        dummy[:, target_idx] = y_scaled.flatten()
        return scaler.inverse_transform(dummy)[:, target_idx]

    y_test_orig = inverse_transform_target(y_test, scaler, n_features, target_idx)
    y_pred_orig = inverse_transform_target(y_pred, scaler, n_features, target_idx)
    
    # Calculate metrics on ORIGINAL scale
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    # Calculate Accuracy (100 - MAPE)
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + epsilon))) * 100
    accuracy = 100 - mape
    
    print("\n" + "="*30)
    print("Model Performance Metrics (Original Scale)")
    print("="*30)
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2:       {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*30 + "\n")
    
    # Save metrics to a text file
    with open('model_metrics.txt', 'w') as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\nAccuracy: {accuracy:.2f}%\n")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig[:100], label='Actual Sales', color='blue') 
    plt.plot(y_pred_orig[:100], label='Predicted Sales', color='red', linestyle='--')
    plt.title('Demand Forecasting: Actual vs Predicted (First 100 Test Samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Sales Units')
    plt.legend()
    plt.savefig('model_performance.png')
    print("Performance plot saved to model_performance.png")

    print("Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
