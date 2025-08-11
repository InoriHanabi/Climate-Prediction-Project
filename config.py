# config.py
import tensorflow as tf

# --- General Settings ---
SELECTED_STATION_IDX = 220
FIGURE_SAVE_DIR = "./plots/"
LOG_FILE_PATH = "output_log.txt" 

# --- Data File Paths ---
COUNTRIES_FILE_PATH = None
STATIONS_FILE_PATH = None

# --- Task 1: Climate Prediction (Monthly TMAX) ---
# NOTE: The current model_training.py doesn't use this, but it's good practice to keep it complete.
TMAX_CLIMATE_TF = {
    "WINDOW_SIZE": 12, "OFFSET": 12, "HIDDEN_UNITS": 64, 
    "NUM_LAYERS": 2, "DROPOUT_RATE": 0.2, "EPOCHS": 100,      
    "BATCH_SIZE": 16, "TRAIN_SPLIT_RATIO": 0.85,
    "LEARNING_RATE": 0.0005, 
    "EARLY_STOPPING_PATIENCE": 15,
    "LOSS_FUNCTION": "mse",
    "METRICS": ["mae"]
}

# --- Task 2: Weather Prediction (Daily TMAX) ---
TMAX_WEATHER_TF = {
    "WINDOW_SIZE": 30, "OFFSET": 1, "HIDDEN_UNITS": 100, 
    "NUM_LAYERS": 2, "DROPOUT_RATE": 0.2, "EPOCHS": 100,      
    "BATCH_SIZE": 64, "TRAIN_SPLIT_RATIO": 0.85,
    "EARLY_STOPPING_PATIENCE": 15,
    "LEARNING_RATE": 0.0005,
    # --- ADDED THESE LINES ---
    "LOSS_FUNCTION": "mse",
    "METRICS": ["mae"]
    # --- END ADDED LINES ---
}

# --- Task 3: Weather Prediction (Daily PRCP) ---
PRCP_CLASSIFICATION_TF = {
    "WINDOW_SIZE": 30, "OFFSET": 1, "HIDDEN_UNITS": 64, 
    "NUM_LAYERS": 2, "DROPOUT_RATE": 0.2, "EPOCHS": 50,       
    "BATCH_SIZE": 64, "TRAIN_SPLIT_RATIO": 0.85,
    "EARLY_STOPPING_PATIENCE": 10,
    "LEARNING_RATE": 0.001, 
    "LOSS_FUNCTION": "binary_crossentropy", 
    "METRICS": ["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    "OUTPUT_ACTIVATION": "sigmoid" 
}
PRCP_REGRESSION_TF = {
    "WINDOW_SIZE": 30, "OFFSET": 1, "HIDDEN_UNITS": 64, 
    "NUM_LAYERS": 2, "DROPOUT_RATE": 0.2, "EPOCHS": 100,      
    "BATCH_SIZE": 64, "TRAIN_SPLIT_RATIO": 0.85, 
    "EARLY_STOPPING_PATIENCE": 15,
    "LEARNING_RATE": 0.001, 
    "LOSS_FUNCTION": "mae", 
    "METRICS": ["mae"], 
    "OUTPUT_ACTIVATION": "relu" 
}