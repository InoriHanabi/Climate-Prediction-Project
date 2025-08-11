# tf_keras_modules.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_lstm_forecaster_tf(input_shape, hidden_units, num_layers, output_units, 
                              dropout_rate=0.2, output_activation=None): # << ADDED output_activation
    """
    Creates a Keras LSTM forecasting model.
    """
    model = keras.Sequential(name="lstm_forecaster")
    
    for i in range(num_layers):
        return_sequences = True if i < num_layers - 1 else False 
        if i == 0:
            model.add(keras.layers.LSTM(hidden_units, 
                                        input_shape=input_shape, 
                                        return_sequences=return_sequences,
                                        dropout=dropout_rate if num_layers > 1 and dropout_rate > 0 else 0,
                                        name=f"lstm_layer_{i+1}"))
        else:
            model.add(keras.layers.LSTM(hidden_units, 
                                        return_sequences=return_sequences,
                                        dropout=dropout_rate if num_layers > 1 and dropout_rate > 0 else 0,
                                        name=f"lstm_layer_{i+1}"))

    model.add(keras.layers.Dense(output_units, activation=output_activation, name="output_dense")) # << USED output_activation
    
    return model

def create_tf_sequences(data, window_size, offset, target_col_idx=0): # << ADDED target_col_idx
    """
    Prepares sequences for training with Keras.
    Args:
        data (np.array): Scaled time series data (num_samples, num_features).
        window_size (int): Size of the input window.
        offset (int): Number of steps to predict ahead.
        target_col_idx (int): Index of the column in 'data' to be used as the target y.
    Returns:
        (np.array, np.array): X (sequences), y (targets)
    """
    X, y = [], []
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    for i in range(len(data) - window_size - offset + 1):
        X.append(data[i:(i + window_size), :]) 
        y.append(data[i + window_size : i + window_size + offset, target_col_idx]) # Use target_col_idx
                                                                     
    return np.array(X), np.array(y)