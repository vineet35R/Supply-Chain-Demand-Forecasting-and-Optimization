import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    """Creates a Multivariate LSTM model."""
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1)) 
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Example input shape: (30 time steps, 7 features)
    model = create_lstm_model((30, 7))
    model.summary()
