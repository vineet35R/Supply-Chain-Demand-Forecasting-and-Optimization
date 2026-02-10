import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from preprocessing import load_data, preprocess_data
import os

# Page config
st.set_page_config(page_title="Demand Forecasting System", layout="wide")

# Constants
DATA_PATH = 'demand_forecasting_dataset.csv'
MODEL_PATH = 'models/lstm_model.h5'
SEQUENCE_LENGTH = 30

@st.cache_data
def get_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file {DATA_PATH} not found.")
        return None, None, None, None
    
    df = load_data(DATA_PATH)
    X, y, scaler, features = preprocess_data(df, SEQUENCE_LENGTH)
    return df, X, scaler, features

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Please train the model first.")
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def make_multivariate_prediction(model, scaler, last_sequence_data, future_days, n_features, target_idx):
    """
    Generates future predictions for multivariate data.
    
    Strategy: 
    For future features (like price, weather), we simply repeat the last known value 
    or use a naive heuristic since we don't have a separate model for them. 
    In a real production system, you'd have separate forecasts for these drivers.
    """
    predictions = []
    
    # Start with the last known sequence
    current_sequence = last_sequence_data.copy() # Shape: (SEQUENCE_LENGTH, n_features)
    
    for _ in range(future_days):
        # Reshape for model input (1, SEQUENCE_LENGTH, n_features)
        input_seq = current_sequence.reshape(1, SEQUENCE_LENGTH, n_features)
        
        # Predict next target value (scaled)
        pred_scaled_value = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred_scaled_value)
        
        # Create next time step's feature vector
        # We need to construct a new row of features. 
        # The target (sales_units) is updated with our prediction.
        # Other features (price, etc.) are carried forward (naive approach)
        next_step_features = current_sequence[-1].copy()
        next_step_features[target_idx] = pred_scaled_value
        
        # Append new step and remove oldest
        current_sequence = np.vstack([current_sequence[1:], next_step_features])
        
    return predictions

def main():
    st.title("Supply Chain Demand Forecasting & Optimization")
    st.markdown("""
    **Multivariate LSTM System**
    Predicting future demand based on Sales, Price, Economic Index, and other factors.
    """)

    # Sidebar
    st.sidebar.header("Configuration")
    future_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
    
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Training model (~1 minute)..."):
            # Run training in a subprocess or import logic. Importing logic is safer in streamlit.
            from train import train
            train()
            st.success("Model trained!")
            st.cache_resource.clear() # Clear cache to reload model

    # Load data
    df, X_all, scaler, features = get_data()
    
    if df is not None:
        # Show raw data metrics
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Date Range", f"{df['date'].dt.date.min()} to {df['date'].dt.date.max()}")
        col3.metric("Features Used", len(features))
        col4.metric("Target Variable", "sales_units")

        with st.expander("View Raw Data"):
            st.dataframe(df.tail(10))
            
        # Plot History
        st.subheader("Historical Trends")
        feature_to_plot = st.selectbox("Select Feature to Visualize", features)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['date'], df[feature_to_plot], label=feature_to_plot)
        ax.set_title(f"Historical {feature_to_plot}")
        st.pyplot(fig)

        # Forecast
        model = get_model()
        
        if model is not None:
            if st.button("Generate Demand Forecast"):
                with st.spinner(f"Forecasting next {future_days} days..."):
                    
                    # Prepare input for prediction (last sequence from data)
                    last_sequence = X_all[-1] # Shape: (SEQUENCE_LENGTH, n_features)
                    n_features = len(features)
                    target_column = 'sales_units'
                    target_idx = features.index(target_column)
                    
                    # Get scaled predictions
                    scaled_predictions = make_multivariate_prediction(
                        model, scaler, last_sequence, future_days, n_features, target_idx
                    )
                    
                    # Inverse transform
                    # We need to construct a dummy matrix to inverse transform because scaler expects all features
                    # We will fill other features with 0s or last known values, but importantly we extract the target column
                    
                    # Create a matrix of shape (future_days, n_features)
                    dummy_matrix = np.zeros((len(scaled_predictions), n_features))
                    # Fill the target column with our predictions
                    dummy_matrix[:, target_idx] = scaled_predictions
                    
                    # Inverse transform
                    original_scale_matrix = scaler.inverse_transform(dummy_matrix)
                    
                    # Extract the target column predictions (sales_units)
                    final_predictions = original_scale_matrix[:, target_idx]
                    
                    # Ensure no negative demand
                    final_predictions = np.maximum(final_predictions, 0)
                    
                    # Dates
                    last_date = df['date'].iloc[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
                    
                    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Demand': final_predictions})
                    
                    # Visualize Forecast
                    st.subheader("Demand Forecast Analysis")
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    # Show last 90 days of history
                    history_view = df.iloc[-90:]
                    ax2.plot(history_view['date'], history_view['sales_units'], label='Historical Sales (Last 3 Months)')
                    ax2.plot(forecast_df['Date'], forecast_df['Predicted Demand'], label='Forecast', color='red', linestyle='--', marker='.')
                    ax2.set_title(f"Sales Forecast: Next {future_days} Days")
                    ax2.set_ylabel("Sales Units")
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    col_a, col_b = st.columns(2)
                    col_a.dataframe(forecast_df)
                    
                    with col_b:
                        st.subheader("Forecast Insights")
                        st.metric("Expected Total Demand", f"{final_predictions.sum():,.0f}")
                        st.metric("Peak Demand Day", f"{final_predictions.max():,.0f}")
                        st.metric("Lowest Demand Day", f"{final_predictions.min():,.0f}")

if __name__ == "__main__":
    main()
