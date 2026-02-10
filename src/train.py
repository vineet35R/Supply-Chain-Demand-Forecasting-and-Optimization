import numpy as np
import pandas as pd
import tensorflow as tf
import os
from preprocessing import load_data, preprocess_data
from model import create_lstm_model

# Constants
# Use the file provided by the user
DATA_PATH = 'demand_forecasting_dataset.csv' 
MODEL_PATH = 'models/lstm_model.h5'
SEQUENCE_LENGTH = 30
EPOCHS = 20
BATCH_SIZE = 32

def train():
    print("Loading data...")
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
    
    print("Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
