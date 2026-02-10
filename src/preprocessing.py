import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Loads sales data from CSV."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date')
    return df

def preprocess_data(df, sequence_length=30, features=None, target='sales_units'):
    """
    Preprocesses data for Multivariate LSTM model.
    1. Selects features and target.
    2. Scales data using MinMaxScaler.
    3. Creates sequences of length `sequence_length` for training.
    """
    if features is None:
        # Default features based on the dataset inspection
        features = ['sales_units', 'price', 'economic_index', 'weather_impact', 
                    'competitor_price_index', 'promotion_applied', 'holiday_season']
    
    # Ensure target is in features for sequence creation (we use past values of everything to predict future target)
    if target not in features:
        features.append(target)
        
    data = df[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    target_idx = features.index(target)
    
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length, :]) # Use all features for valid sequence
        y.append(scaled_data[i+sequence_length, target_idx]) # Predict next target value

    X, y = np.array(X), np.array(y)
    return X, y, scaler, features

if __name__ == "__main__":
    # Test preprocessing
    df = load_data('demand_forecasting_dataset.csv')
    print(df.head())
    X, y, scaler, feats = preprocess_data(df)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Features:", feats)
