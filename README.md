# Supply Chain Demand Forecasting & Optimization System

This project implements a demand forecasting system using Long Short-Term Memory (LSTM) networks to predict future product demand based on historical sales data.

## Features

-   **LSTM Model**: Deep learning model for time-series forecasting.
-   **Visualization Dashboard**: Streamlit app to visualize historical data and forecasts.

## Model Performance

The model is evaluated using the following metrics on the test set (Original Scale):

## Model Performance

The model is evaluated using the following metrics on the test set (Original Scale):

| Metric | Value |
| :--- | :--- |
| **MAE** (Mean Absolute Error) | 15.14 |
| **RMSE** (Root Mean Squared Error) | 18.40 |
| **Accuracy** (100% - MAPE) | **92.32%** |

*(Note: These metrics are from an optimized training run with 20 epochs on synthetic data. Accuracy >90% achieved.)*

### Prediction Visualization
![Model Performance](model_performance.png)

## Setup

1.  Current working directory: `/home/vineet/Desktop/2nd project`
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Train Model**:
    ```bash
    python src/train.py
    ```
3.  **Run Dashboard**:
    ```bash
    streamlit run src/app.py
    ```
