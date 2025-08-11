# model_training.py

# --- Suppress TensorFlow INFO and WARNING logs ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# --- End Suppression ---

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             mean_absolute_error, mean_squared_error)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

# --- Import Project Modules ---
try:
    import config
    from ghcn_helpers import GHNCD, getdata, fillholesT, fillPRCP, fillSN
    from tf_keras_modules import create_tf_sequences
except ImportError as e:
    print(f"FATAL ERROR: Could not import project modules. Error: {e}")
    sys.exit(1)


# --- Custom TQDM Progress Bar Callback for Keras ---
class TqdmProgressCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.tqdm_bar = tqdm(total=self.epochs, desc="Training", unit="epoch", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.tqdm_bar.set_postfix_str(metrics_str)
        self.tqdm_bar.update(1)
    def on_train_end(self, logs=None):
        self.tqdm_bar.close()


# --- Model Creation and Training Utilities ---
def create_recurrent_model(model_type, input_shape, hidden_units, num_layers, output_units, dropout_rate=0.2, output_activation=None):
    model = keras.Sequential(name=f"{model_type.upper()}_Forecaster")
    RecurrentLayer = keras.layers.LSTM if model_type.lower() == 'lstm' else keras.layers.GRU
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        if i == 0:
            model.add(RecurrentLayer(hidden_units, input_shape=input_shape, return_sequences=return_sequences,
                                     dropout=dropout_rate if num_layers > 1 else 0, name=f"{model_type}_layer_{i+1}"))
        else:
            model.add(RecurrentLayer(hidden_units, return_sequences=return_sequences,
                                     dropout=dropout_rate if num_layers > 1 else 0, name=f"{model_type}_layer_{i+1}"))
    model.add(keras.layers.Dense(output_units, activation=output_activation, name="output_dense"))
    return model

def train_and_evaluate_model(model, train_X, train_y, test_X, test_y, cfg, logger, class_weight_dict=None):
    logger.log_important(f"--- Training Model: {model.name} ---", to_console=False)
    optimizer = keras.optimizers.legacy.Adam(learning_rate=cfg.get('LEARNING_RATE', 0.001))
    model.compile(optimizer=optimizer, loss=cfg['LOSS_FUNCTION'], metrics=cfg.get('METRICS', []))
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['EARLY_STOPPING_PATIENCE'], restore_best_weights=True, verbose=0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)
    tqdm_callback = TqdmProgressCallback()
    
    print(f"Training {model.name} for up to {cfg.get('EPOCHS', 100)} epochs...")
    history = model.fit(train_X, train_y, validation_data=(test_X, test_y),
                        epochs=cfg.get('EPOCHS', 100), batch_size=cfg.get('BATCH_SIZE', 64),
                        callbacks=[early_stopping, reduce_lr, tqdm_callback],
                        class_weight=class_weight_dict, verbose=0)
    print("Training complete.")
    eval_results = model.evaluate(test_X, test_y, verbose=0)
    predictions = model.predict(test_X, verbose=0)
    return history, eval_results, predictions, model


# --- Selective Logger Class ---
class SelectiveLogger:
    def __init__(self, filepath):
        self.filepath = filepath; self.terminal = sys.stdout; self.log_file = None
    def __enter__(self): self.log_file = open(self.filepath, 'a'); return self
    def __exit__(self, type, value, traceback):
        if self.log_file: self.log_file.close()
    def log_important(self, message, to_console=True):
        full_message = message + "\n"
        if self.log_file and not self.log_file.closed: self.log_file.write(full_message)
        if to_console: self.terminal.write(full_message)


# --- Data Loading and Preparation (Master Function) ---
def load_and_prepare_full_dataset(ghn, data_fetcher, station_idx):
    print("Loading and preparing master dataset...")
    dates_max, tmax, _, tmin, _, _ = fillholesT(data_fetcher.TmaxTmin(station_idx))
    dates_prcp, prcp, _, _ = data_fetcher.PRCP(station_idx)
    dates_prcp, prcp, _ = fillPRCP((dates_prcp, prcp, None, None))
    dates_snow, snow, _, _ = data_fetcher.SNOW(station_idx)
    dates_snow, snow, _ = fillSN((dates_snow, snow, None, None))
    dates_snwd, snwd, _, _ = data_fetcher.SNWD(station_idx)
    dates_snwd, snwd, _ = fillSN((dates_snwd, snwd, None, None))
    
    df_temp = pd.DataFrame({'TMAX': tmax, 'TMIN': tmin}, index=pd.to_datetime(dates_max))
    df_prcp = pd.DataFrame({'PRCP': prcp}, index=pd.to_datetime(dates_prcp))
    df_snow = pd.DataFrame({'SNOW': snow}, index=pd.to_datetime(dates_snow))
    df_snwd = pd.DataFrame({'SNWD': snwd}, index=pd.to_datetime(dates_snwd))
    
    df_master = pd.concat([df_temp, df_prcp, df_snow, df_snwd], axis=1)
    df_master = df_master.ffill().bfill().fillna(0)
    print("Master dataset prepared.")
    return df_master


# --- Main Modeling Functions ---

def run_daily_tmax_modeling(df_master, station_info, station_idx, logger):
    task_name = "Daily TMAX Prediction (Multivariate)"
    logger.log_important(f"\n{'='*20}\n--- Starting Task: {task_name} ---\n{'='*20}", to_console=True)
    cfg = config.TMAX_WEATHER_TF

    # --- FIX: DataFrame was not created here ---
    df = df_master[['TMAX', 'TMIN']].copy() 
    # --- END FIX ---
    
    for w in [3, 7, 14]:
        df[f'TMAX_roll_mean_{w}'] = df['TMAX'].rolling(w, min_periods=1).mean()
        df[f'TMAX_roll_std_{w}'] = df['TMAX'].rolling(w, min_periods=1).std()
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df.bfill(inplace=True) # FIX: Use bfill() instead of fillna(method='bfill')
    
    tmax_scaler = MinMaxScaler()
    other_features_scaler = MinMaxScaler()
    df['TMAX_scaled'] = tmax_scaler.fit_transform(df[['TMAX']])
    other_cols = [c for c in df.columns if c not in ['TMAX', 'TMAX_scaled']]
    df[other_cols] = other_features_scaler.fit_transform(df[other_cols])
    
    feature_cols = ['TMAX_scaled'] + other_cols
    X, y = create_tf_sequences(df[feature_cols].values, cfg['WINDOW_SIZE'], cfg['OFFSET'], target_col_idx=0)
    split = int(X.shape[0] * cfg['TRAIN_SPLIT_RATIO'])
    train_X, test_X, train_y, test_y = X[:split], X[split:], y[:split], y[split:]
    logger.log_important(f"[{task_name}] Data prepared. Input shape: {train_X.shape}, Target shape: {train_y.shape}", to_console=False)
    
    results = {}
    for model_type in ['lstm', 'gru']:
        input_shape = (train_X.shape[1], train_X.shape[2])
        model = create_recurrent_model(model_type, input_shape, cfg['HIDDEN_UNITS'], cfg['NUM_LAYERS'], cfg['OFFSET'], cfg['DROPOUT_RATE'])
        _, _, preds_scaled, _ = train_and_evaluate_model(model, train_X, train_y, test_X, test_y, cfg, logger)
        preds_unscaled = tmax_scaler.inverse_transform(preds_scaled).flatten()
        actuals_unscaled = tmax_scaler.inverse_transform(test_y).flatten()
        mae = mean_absolute_error(actuals_unscaled, preds_unscaled)
        results[model_type] = {'mae': mae, 'preds': preds_unscaled, 'actuals': actuals_unscaled}
        logger.log_important(f"[{task_name} - {model_type.upper()}] Final Test MAE: {mae:.2f} C", to_console=True)
    
    persistence_mae = mean_absolute_error(results['lstm']['actuals'][1:], results['lstm']['actuals'][:-1])
    logger.log_important(f"[{task_name} - Persistence] Final Test MAE: {persistence_mae:.2f} C", to_console=True)
    best_model = min(results, key=lambda k: results[k]['mae'])
    logger.log_important(f"[{task_name}] Best model: {best_model.upper()}", to_console=True)

    plt.figure(figsize=(15, 7))
    plot_len = min(200, len(results[best_model]['actuals']))
    plt.plot(results[best_model]['actuals'][:plot_len], label='Actual TMAX', color='black')
    plt.plot(results['lstm']['preds'][:plot_len], label=f"LSTM (MAE: {results['lstm']['mae']:.2f})", linestyle='--')
    plt.plot(results['gru']['preds'][:plot_len], label=f"GRU (MAE: {results['gru']['mae']:.2f})", linestyle=':')
    plt.plot(results['lstm']['actuals'][:-1][:plot_len-1], label=f"Persistence (MAE: {persistence_mae:.2f})", color='gray', alpha=0.7)
    plt.title(f'Daily TMAX Prediction for {station_info.name.strip()}'); plt.ylabel('Temperature (°C)'); plt.xlabel('Test Set Day Index')
    plt.legend(); plt.grid(True)
    plot_path = os.path.join(config.FIGURE_SAVE_DIR, f"C_TMAX_s{station_idx}.png"); plt.savefig(plot_path); plt.close()
    logger.log_important(f"[{task_name}] Saved plot to {plot_path}", to_console=False)

def run_two_stage_modeling(df_master, var_name, station_info, station_idx, logger):
    task_name = f"Two-Stage Daily {var_name} Prediction"
    logger.log_important(f"\n{'='*20}\n--- Starting Task: {task_name} ---\n{'='*20}", to_console=True)
    cfg_clf = config.PRCP_CLASSIFICATION_TF
    cfg_reg = config.PRCP_REGRESSION_TF
    
    df = df_master.copy()
    df[f'{var_name}_Occurrence'] = (df[var_name] > 0.1).astype(int)
    df[f'{var_name}_log1p'] = np.log1p(df[var_name])
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    feature_cols_base = ['TMAX', 'TMIN', 'day_sin', 'day_cos']
    for lag in [1, 3, 7]: df[f'{var_name}_log_lag_{lag}'] = df[f'{var_name}_log1p'].shift(lag)
    feature_cols = feature_cols_base + [f'{var_name}_log_lag_{lag}' for lag in [1, 3, 7]]
    df.dropna(inplace=True)
    
    target_clf = f'{var_name}_Occurrence'
    scaler_clf = MinMaxScaler(); df[feature_cols] = scaler_clf.fit_transform(df[feature_cols])
    X_clf_all, y_clf_all = create_tf_sequences(df[[target_clf] + feature_cols].values, cfg_clf['WINDOW_SIZE'], 1, 0)
    split_clf = int(X_clf_all.shape[0] * cfg_clf['TRAIN_SPLIT_RATIO'])
    train_X_clf, test_X_clf, train_y_clf, test_y_clf = X_clf_all[:split_clf], X_clf_all[split_clf:], y_clf_all[:split_clf], y_clf_all[split_clf:]

    weights = compute_class_weight('balanced', classes=np.unique(train_y_clf.flatten()), y=train_y_clf.flatten())
    class_weight_dict = dict(enumerate(weights))
    
    clf_input_shape = (train_X_clf.shape[1], train_X_clf.shape[2])
    model_clf = create_recurrent_model('gru', clf_input_shape, cfg_clf['HIDDEN_UNITS'], cfg_clf['NUM_LAYERS'], 1, cfg_clf['DROPOUT_RATE'], cfg_clf['OUTPUT_ACTIVATION'])
    _, _, preds_proba_clf, _ = train_and_evaluate_model(model_clf, train_X_clf, train_y_clf, test_X_clf, test_y_clf, cfg_clf, logger, class_weight_dict)
    preds_binary_clf = (preds_proba_clf > 0.5).astype(int).flatten()
    f1 = f1_score(test_y_clf.flatten(), preds_binary_clf, zero_division=0)
    logger.log_important(f"[{task_name} - Classifier] Final Test F1-Score: {f1:.4f}", to_console=True)
    
    target_reg = f'{var_name}_log1p'
    scaler_reg = MinMaxScaler(); df[target_reg] = scaler_reg.fit_transform(df[[target_reg]])
    X_reg_all, y_reg_all = create_tf_sequences(df[[target_reg] + feature_cols].values, cfg_reg['WINDOW_SIZE'], 1, 0)
    
    y_orig_for_filtering = df[var_name].values[cfg_reg['WINDOW_SIZE']:]
    event_indices = np.where(y_orig_for_filtering > 0.1)[0]
    X_reg_event, y_reg_event = X_reg_all[event_indices], y_reg_all[event_indices]
    split_reg = int(X_reg_event.shape[0] * cfg_reg['TRAIN_SPLIT_RATIO'])
    train_X_reg, test_X_reg_event, train_y_reg, test_y_reg_event = X_reg_event[:split_reg], X_reg_event[split_reg:], y_reg_event[:split_reg], y_reg_event[split_reg:]

    model_reg = None
    if train_X_reg.shape[0] > 20:
        reg_input_shape = (train_X_reg.shape[1], train_X_reg.shape[2])
        model_reg = create_recurrent_model('gru', reg_input_shape, cfg_reg['HIDDEN_UNITS'], cfg_reg['NUM_LAYERS'], 1, cfg_reg['DROPOUT_RATE'], cfg_reg['OUTPUT_ACTIVATION'])
        _, _, _, model_reg = train_and_evaluate_model(model_reg, train_X_reg, train_y_reg, test_X_reg_event, test_y_reg_event, cfg_reg, logger)
    
    test_X_reg_all = X_reg_all[split_clf:]
    preds_amount_orig = np.zeros_like(preds_binary_clf, dtype=float)
    if model_reg:
        preds_amount_scaled_log = model_reg.predict(test_X_reg_all, verbose=0)
        preds_amount_log = scaler_reg.inverse_transform(preds_amount_scaled_log)
        preds_amount_orig = np.expm1(preds_amount_log).flatten()
        preds_amount_orig = np.maximum(0, preds_amount_orig)
    
    final_preds = preds_amount_orig * preds_binary_clf[:len(preds_amount_orig)]
    actuals_orig = df[var_name].values[cfg_clf['WINDOW_SIZE']:][split_clf:]
    min_len = min(len(final_preds), len(actuals_orig))
    
    final_mae = mean_absolute_error(actuals_orig[:min_len], final_preds[:min_len])
    final_rmse = np.sqrt(mean_squared_error(actuals_orig[:min_len], final_preds[:min_len]))
    logger.log_important(f"[{task_name} - Combined] Final Test MAE: {final_mae:.2f} mm, RMSE: {final_rmse:.2f} mm", to_console=True)
    
    plot_len = min(200, len(final_preds))
    plt.figure(figsize=(15, 7))
    plt.plot(actuals_orig[:plot_len], label=f'Actual {var_name}', color='black', alpha=0.8)
    plt.plot(final_preds[:plot_len], label=f"Two-Stage GRU (MAE: {final_mae:.2f})", linestyle='--')
    plt.title(f'Daily {var_name} Two-Stage Prediction for {station_info.name.strip()}'); plt.ylabel(f'{var_name} (mm)'); plt.legend(); plt.grid(True)
    plot_path = os.path.join(config.FIGURE_SAVE_DIR, f"C_{var_name}_s{station_idx}.png"); plt.savefig(plot_path); plt.close()
    logger.log_important(f"[{task_name}] Saved plot to {plot_path}", to_console=False)

def run_daily_snwd_modeling(df_master, station_info, station_idx, logger):
    task_name = "Daily SNWD Prediction (Multivariate)"
    logger.log_important(f"\n{'='*20}\n--- Starting Task: {task_name} ---\n{'='*20}", to_console=True)
    cfg = config.TMAX_WEATHER_TF
    
    df = df_master[['SNWD', 'TMAX', 'TMIN', 'SNOW']].copy()
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    snwd_scaler = MinMaxScaler()
    other_features_scaler = MinMaxScaler()
    df['SNWD_scaled'] = snwd_scaler.fit_transform(df[['SNWD']])
    other_cols = [c for c in df.columns if c not in ['SNWD', 'SNWD_scaled']]
    df[other_cols] = other_features_scaler.fit_transform(df[other_cols])
    
    feature_cols = ['SNWD_scaled'] + other_cols
    X, y = create_tf_sequences(df[feature_cols].values, cfg['WINDOW_SIZE'], 1, 0)
    split = int(X.shape[0] * cfg['TRAIN_SPLIT_RATIO'])
    train_X, test_X, train_y, test_y = X[:split], X[split:], y[:split], y[split:]
    
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = create_recurrent_model('gru', input_shape, cfg['HIDDEN_UNITS'], cfg['NUM_LAYERS'], 1, cfg['DROPOUT_RATE'], output_activation='relu')
    _, _, preds_scaled, _ = train_and_evaluate_model(model, train_X, train_y, test_X, test_y, cfg, logger)
    
    preds_unscaled = snwd_scaler.inverse_transform(preds_scaled).flatten()
    actuals_unscaled = snwd_scaler.inverse_transform(test_y).flatten()
    mae = mean_absolute_error(actuals_unscaled, preds_unscaled)
    logger.log_important(f"[{task_name} - GRU] Final Test MAE: {mae:.2f} mm", to_console=True)
    
    plot_len = min(200, len(preds_unscaled))
    plt.figure(figsize=(15, 7))
    plt.plot(actuals_unscaled[:plot_len], label='Actual SNWD', color='black')
    plt.plot(preds_unscaled[:plot_len], label=f"GRU (MAE: {mae:.2f})", linestyle='--')
    plt.title(f'Daily Snow Depth (SNWD) Prediction for {station_info.name.strip()}'); plt.ylabel('SNWD (mm)'); plt.legend(); plt.grid(True)
    plot_path = os.path.join(config.FIGURE_SAVE_DIR, f"C_SNWD_s{station_idx}.png"); plt.savefig(plot_path); plt.close()
    logger.log_important(f"[{task_name}] Saved plot to {plot_path}", to_console=False)


if __name__ == '__main__':
    # You need to have tqdm installed: pip install tqdm
    if not os.path.exists(config.FIGURE_SAVE_DIR): os.makedirs(config.FIGURE_SAVE_DIR)
    with open(config.LOG_FILE_PATH, 'w') as f:
        f.write(f"MODELING RESULTS LOG - {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    
    ghn = GHNCD(); ghn.readCountriesFile(); ghn.readStationsFile(justGSN=True)
    data_fetcher = getdata(ghn_instance=ghn)
    station_idx = config.SELECTED_STATION_IDX
    station_info = ghn.getStation(ghn.getStatKeyNames()[station_idx])
    
    df_master = load_and_prepare_full_dataset(ghn, data_fetcher, station_idx)
    
    if df_master is not None and not df_master.empty:
        with SelectiveLogger(config.LOG_FILE_PATH) as logger:
            logger.log_important(f"Starting modeling tasks for station: {station_info}", to_console=True)
            run_daily_tmax_modeling(df_master, station_info, station_idx, logger)
            run_two_stage_modeling(df_master, 'PRCP', station_info, station_idx, logger)
            run_two_stage_modeling(df_master, 'SNOW', station_info, station_idx, logger)
            run_daily_snwd_modeling(df_master, station_info, station_idx, logger)
            logger.log_important(f"\n{'='*20}\nAll Modeling Tasks Complete\n{'='*20}", to_console=True)
    else:
        print("\nAborting modeling: Master dataset could not be created.")

    print(f"\nModeling complete. See detailed results in '{config.LOG_FILE_PATH}' and plots in 'plots/'.")