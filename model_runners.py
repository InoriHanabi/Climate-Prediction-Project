# model_runners.py
import os
import sys # Needed for SelectiveLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import config 
from ghcn_helpers import fillPRCP, fillholesT 
from tf_keras_modules import create_lstm_forecaster_tf, create_tf_sequences
from training_utils_tf import train_tf_model, evaluate_tf_model

# --- SelectiveLogger Class (MOVED HERE) ---
class SelectiveLogger:
    def __init__(self, filepath, console_too=False):
        self.filepath = filepath
        self.console_too = console_too
        self.terminal = sys.stdout
        self.log_file = None

    def __enter__(self):
        self.log_file = open(self.filepath, 'a')
        # We are NOT redirecting sys.stdout globally here to allow Keras to print to console
        return self

    def __exit__(self, type, value, traceback):
        if self.log_file:
            self.log_file.close()
        # sys.stdout = self.terminal # No need to restore if not changed

    # write and flush are not strictly needed if sys.stdout isn't redirected to self
    # def write(self, message):
    #     self.terminal.write(message)
    # def flush(self):
    #     self.terminal.flush()

    def log_important(self, message, to_console=True):
        full_message = message + "\n"
        if self.log_file and not self.log_file.closed:
            self.log_file.write(full_message)
        if to_console:
            self.terminal.write(full_message)

def run_climate_prediction_tmax_tf(ghn_instance, data_fetcher_instance, station_idx, logger):
    task_name = "Task 1: Climate Prediction (Monthly TMAX)"
    logger.log_important(f"\n--- {task_name} ---", to_console=False)
    cfg = config.TMAX_CLIMATE_TF
    
    print(f"Starting {task_name} for station {ghn_instance.getStatKeyNames()[station_idx]}...") # Console feedback
    raw_tmax_data = data_fetcher_instance.TmaxTmin(station_idx)
    if not raw_tmax_data[0] and not raw_tmax_data[1]:
        logger.log_important(f"[{task_name}] No TMAX/TMIN data. Skipping.", to_console=True)
        return
    filled_dates_max, filled_tmax, _, _, _, _ = fillholesT(raw_tmax_data)
    if filled_tmax.size == 0:
        logger.log_important(f"[{task_name}] TMAX data empty after filling. Skipping.", to_console=True)
        return

    tmax_series = pd.Series(filled_tmax, index=pd.to_datetime(filled_dates_max))
    monthly_tmax = tmax_series.resample('ME').mean().dropna()
    if monthly_tmax.empty:
        logger.log_important(f"[{task_name}] Monthly TMAX empty after resampling. Skipping.", to_console=True)
        return
        
    monthly_tmax_values = monthly_tmax.values.astype(np.float32).reshape(-1, 1)
    scaler = MinMaxScaler()
    monthly_tmax_scaled = scaler.fit_transform(monthly_tmax_values)
    X, y = create_tf_sequences(monthly_tmax_scaled, cfg["WINDOW_SIZE"], cfg["OFFSET"], target_col_idx=0)

    if X.shape[0] == 0: logger.log_important(f"[{task_name}] Not enough data for sequences. Skipping.", to_console=True); return
    split_point = int(X.shape[0] * cfg["TRAIN_SPLIT_RATIO"])
    train_X, test_X = X[:split_point], X[split_point:]
    train_y, test_y = y[:split_point], y[split_point:]
    if train_X.shape[0] == 0 : logger.log_important(f"[{task_name}] Not enough train data. Skipping.", to_console=True); return
        
    model_input_shape = (cfg["WINDOW_SIZE"], X.shape[2])
    model = create_lstm_forecaster_tf(
        input_shape=model_input_shape, hidden_units=cfg["HIDDEN_UNITS"],
        num_layers=cfg["NUM_LAYERS"], output_units=cfg["OFFSET"],
        dropout_rate=cfg["DROPOUT_RATE"]
    )
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=cfg["LEARNING_RATE"]),
                  loss=config.TF_COMMON["LOSS_FUNCTION"], metrics=config.TF_COMMON["METRICS"])
    
    stringlist = []; model.summary(print_fn=lambda x: stringlist.append(x))
    logger.log_important(f"Model Summary ({task_name}):\n" + "\n".join(stringlist), to_console=False)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg["EARLY_STOPPING_PATIENCE"], restore_best_weights=True)
    
    print(f"Training {task_name} model (Keras verbose output to console)...") 
    history, model = train_tf_model(model, train_X, train_y, 
                                     validation_data=(test_X, test_y),
                                     epochs=cfg["EPOCHS"], batch_size=cfg["BATCH_SIZE"],
                                     callbacks=[early_stopping], verbose=1) 
    
    best_val_loss = min(history.history.get('val_loss', [np.nan]))
    best_val_mae = min(history.history.get('val_mae', [np.nan])) # Use 'val_mae' if 'mae' is in metrics
    best_epoch = np.argmin(history.history.get('val_loss', [0])) + 1
    logger.log_important(f"[{task_name}] Training finished. Best val_loss: {best_val_loss:.4f} (MAE: {best_val_mae:.4f}) at epoch {best_epoch}", to_console=True)

    eval_results, preds_scaled = evaluate_tf_model(model, test_X, test_y, batch_size=cfg["BATCH_SIZE"], verbose=0)
    logger.log_important(f"[{task_name}] Evaluation results (loss, MAE): {eval_results[0]:.4f}, {eval_results[1]:.4f}", to_console=True)
    
    preds_monthly = scaler.inverse_transform(preds_scaled)
    actuals_monthly = scaler.inverse_transform(test_y)
    plt.figure(figsize=(12, 6))
    if preds_monthly.shape[0] > 0 and actuals_monthly.shape[0] > 0:
        plt.plot(actuals_monthly[0, :], label="Actual Monthly TMAX (First Test Sequence)")
        plt.plot(preds_monthly[0, :], label="Predicted Monthly TMAX (First Test Sequence)")
    else:
        plt.text(0.5, 0.5, "Not enough data to plot predictions for the first test sequence.", 
                 horizontalalignment='center', verticalalignment='center')
    station_info = ghn_instance.getStation(ghn_instance.getStatKeyNames()[station_idx])
    plt.title(f"TF Monthly TMAX: {station_info.name if station_info else 'Unknown'}\n(First {cfg['OFFSET']} months of test)")
    plt.xlabel("Month into Future"); plt.ylabel("Avg TMAX (C)")
    plt.legend(); plt.grid(True)
    plot_filename = os.path.join(config.FIGURE_SAVE_DIR, f"tf_climate_monthly_tmax_s{station_idx}.png")
    plt.savefig(plot_filename); plt.close()
    logger.log_important(f"[{task_name}] Saved plot to {plot_filename}", to_console=False)


def run_weather_prediction_tmax_tf(ghn_instance, data_fetcher_instance, station_idx, logger):
    task_name = "Task 2: Weather Prediction (Daily TMAX)"
    logger.log_important(f"\n--- {task_name} ---", to_console=False)
    cfg = config.TMAX_WEATHER_TF

    print(f"Starting {task_name} for station {ghn_instance.getStatKeyNames()[station_idx]}...")
    raw_tmax_data = data_fetcher_instance.TmaxTmin(station_idx)
    if not raw_tmax_data[0] and not raw_tmax_data[1]: logger.log_important(f"[{task_name}] No TMAX/TMIN data. Skipping.", to_console=True); return
    filled_dates_max, filled_tmax, _, _, _, _ = fillholesT(raw_tmax_data)
    if filled_tmax.size == 0: logger.log_important(f"[{task_name}] TMAX data empty after filling. Skipping.", to_console=True); return
        
    tmax_series_pd = pd.Series(filled_tmax, index=pd.to_datetime(filled_dates_max))
    daily_tmax_values_unscaled = tmax_series_pd.dropna().values.astype(np.float32).reshape(-1, 1)
    dates_for_features = tmax_series_pd.dropna().index
    if daily_tmax_values_unscaled.shape[0] < cfg["WINDOW_SIZE"] + cfg["OFFSET"] + 20: logger.log_important(f"[{task_name}] Not enough daily TMAX data. Skipping.", to_console=True); return

    df_features = pd.DataFrame(daily_tmax_values_unscaled, columns=['TMAX'], index=dates_for_features)
    for window in [3, 7, 14]:
        df_features[f'TMAX_roll_mean_{window}'] = df_features['TMAX'].rolling(window=window, min_periods=1).mean()
        df_features[f'TMAX_roll_std_{window}'] = df_features['TMAX'].rolling(window=window, min_periods=1).std()
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    day_of_year = df_features.index.dayofyear.values
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32).reshape(-1, 1)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32).reshape(-1, 1)
    tmax_for_scaling = df_features['TMAX'].values.reshape(-1, 1)
    other_features_unscaled = df_features.drop(columns=['TMAX']).values
    tmax_scaler = MinMaxScaler()
    daily_tmax_scaled = tmax_scaler.fit_transform(tmax_for_scaling)
    other_features_scaler = MinMaxScaler()
    other_features_scaled = other_features_scaler.fit_transform(other_features_unscaled)
    all_features_for_model = np.concatenate([daily_tmax_scaled, other_features_scaled, day_of_year_sin, day_of_year_cos], axis=1)
    logger.log_important(f"[{task_name}] Shape of features for daily model: {all_features_for_model.shape}", to_console=False)

    X, y = create_tf_sequences(all_features_for_model, cfg["WINDOW_SIZE"], cfg["OFFSET"], target_col_idx=0)
    if X.shape[0] == 0: logger.log_important(f"[{task_name}] Not enough data for sequences. Skipping.", to_console=True); return
    split_point = int(X.shape[0] * cfg["TRAIN_SPLIT_RATIO"])
    train_X, test_X = X[:split_point], X[split_point:]
    train_y, test_y = y[:split_point], y[split_point:] 
    if train_X.shape[0] == 0 : logger.log_important(f"[{task_name}] Not enough train data. Skipping.", to_console=True); return

    model_input_shape = (cfg["WINDOW_SIZE"], X.shape[2])
    model = create_lstm_forecaster_tf(
        input_shape=model_input_shape, hidden_units=cfg["HIDDEN_UNITS"],
        num_layers=cfg["NUM_LAYERS"], output_units=cfg["OFFSET"], 
        dropout_rate=cfg["DROPOUT_RATE"]
    )
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=cfg["LEARNING_RATE"]),
                  loss=config.TF_COMMON["LOSS_FUNCTION"], metrics=config.TF_COMMON["METRICS"])
    stringlist = []; model.summary(print_fn=lambda x: stringlist.append(x))
    logger.log_important(f"Model Summary ({task_name}):\n" + "\n".join(stringlist), to_console=False)
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg["EARLY_STOPPING_PATIENCE"], restore_best_weights=True)
    print(f"Training {task_name} model (Keras verbose output to console)...")
    history, model = train_tf_model(model, train_X, train_y, validation_data=(test_X, test_y),
                                     epochs=cfg["EPOCHS"], batch_size=cfg["BATCH_SIZE"],
                                     callbacks=[early_stopping], verbose=1)
    best_val_loss = min(history.history.get('val_loss', [np.nan]))
    best_val_mae = min(history.history.get('val_mae', [np.nan]))
    best_epoch = np.argmin(history.history.get('val_loss', [0])) + 1
    logger.log_important(f"[{task_name}] Training finished. Best val_loss: {best_val_loss:.4f} (MAE: {best_val_mae:.4f}) at epoch {best_epoch}", to_console=True)

    eval_results, preds_scaled_lstm = evaluate_tf_model(model, test_X, test_y, batch_size=cfg["BATCH_SIZE"], verbose=0)
    logger.log_important(f"[{task_name}] Evaluation results (loss, MAE): {eval_results[0]:.4f}, {eval_results[1]:.4f}", to_console=True)
    
    preds_daily_lstm = tmax_scaler.inverse_transform(preds_scaled_lstm).flatten()
    actuals_for_lstm = tmax_scaler.inverse_transform(test_y).flatten()
    lstm_mae = np.mean(np.abs(preds_daily_lstm - actuals_for_lstm))
    lstm_rmse = np.sqrt(np.mean((preds_daily_lstm - actuals_for_lstm)**2))
    logger.log_important(f"[{task_name}] TF LSTM Daily TMAX MAE: {lstm_mae:.2f} C, RMSE: {lstm_rmse:.2f} C", to_console=True)
    
    persistence_mae, persistence_rmse = -1, -1 
    if len(actuals_for_lstm) >= 2:
        persistence_preds = actuals_for_lstm[:-1] 
        persistence_actuals = actuals_for_lstm[1:]  
        common_len = min(len(preds_daily_lstm), len(persistence_preds)) 
        if common_len > 0 :
            persistence_mae = np.mean(np.abs(persistence_preds[:common_len] - persistence_actuals[:common_len]))
            logger.log_important(f"[{task_name}] TF Persistence Daily TMAX MAE: {persistence_mae:.2f} C", to_console=True)
            if lstm_mae < persistence_mae: logger.log_important(f"[{task_name}] LSTM performs better than persistence.", to_console=True)
            else: logger.log_important(f"[{task_name}] Persistence performs better/equal to LSTM.", to_console=True)
        else: logger.log_important(f"[{task_name}] Not enough data for persistence MAE.", to_console=True)
    else: logger.log_important(f"[{task_name}] Not enough data for persistence baseline.", to_console=True)

    plt.figure(figsize=(15, 7))
    plot_len = min(200, len(preds_daily_lstm)) 
    if plot_len > 0:
        plt.plot(actuals_for_lstm[:plot_len], label="Actual Daily TMAX", alpha=0.9)
        plt.plot(preds_daily_lstm[:plot_len], label="TF LSTM Predicted TMAX (Roll+Time Feats)", linestyle='--')
        if persistence_mae != -1 and common_len > 0 and common_len >= plot_len -1 :
             plt.plot(persistence_preds[:plot_len-1], label="Persistence Baseline", linestyle=':', alpha=0.7)
    else:
        plt.text(0.5, 0.5, "Not enough data to plot daily TMAX predictions.", 
                 horizontalalignment='center', verticalalignment='center')
    station_info = ghn_instance.getStation(ghn_instance.getStatKeyNames()[station_idx])
    plt.title(f"TF Daily TMAX: {station_info.name if station_info else 'Unknown'} (Test Set w/ Roll+Time Feats)")
    plt.xlabel("Day"); plt.ylabel("TMAX (C)")
    plt.legend(); plt.grid(True)
    plot_filename = os.path.join(config.FIGURE_SAVE_DIR, f"tf_weather_daily_tmax_rolltimefeats_s{station_idx}.png")
    plt.savefig(plot_filename); plt.close()
    logger.log_important(f"[{task_name}] Saved plot to {plot_filename}", to_console=False)


def prepare_prcp_two_stage_data(daily_prcp_values_unscaled, dates_for_features, cfg_dict, logger, task_prefix):
    # This function prepares data for both stages of PRCP prediction
    # daily_prcp_values_unscaled should be 1D numpy array
    logger.log_important(f"[{task_prefix}] Preparing PRCP two-stage data...", to_console=False)

    df_features = pd.DataFrame(index=dates_for_features)
    df_features['PRCP_orig'] = daily_prcp_values_unscaled

    df_features['Rain_Occurrence'] = (df_features['PRCP_orig'] > 0.1).astype(int)
    df_features['PRCP_log1p'] = np.log1p(df_features['PRCP_orig'])

    for lag in [1, 2, 3, 7]:
         df_features[f'PRCP_orig_lag_{lag}'] = df_features['PRCP_orig'].shift(lag)
    
    for window in [3, 7, 14]:
        df_features[f'PRCP_roll_sum_{window}'] = df_features['PRCP_orig'].rolling(window=window, min_periods=1).sum()
    
    df_features = df_features.fillna(method='bfill').fillna(method='ffill') 

    day_of_year = df_features.index.dayofyear.values
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)

    # Classifier data
    clf_input_feature_cols = [col for col in df_features.columns if col not in ['PRCP_orig', 'Rain_Occurrence', 'PRCP_log1p']]
    clf_input_features_unscaled = df_features[clf_input_feature_cols].values
    clf_target = df_features['Rain_Occurrence'].values.reshape(-1,1)
    clf_input_scaler = MinMaxScaler()
    clf_input_features_scaled = clf_input_scaler.fit_transform(clf_input_features_unscaled)
    clf_all_data_for_seq = np.concatenate([clf_target, clf_input_features_scaled], axis=1)
    X_clf, y_clf = create_tf_sequences(clf_all_data_for_seq, cfg_dict["WINDOW_SIZE"], cfg_dict["OFFSET"], target_col_idx=0)

    # Regressor data
    reg_target_unscaled = df_features['PRCP_log1p'].values.reshape(-1,1)
    # Regressor uses the same input features as classifier
    reg_all_data_for_seq = np.concatenate([reg_target_unscaled, clf_input_features_scaled], axis=1) 
    reg_target_scaler = MinMaxScaler()
    reg_all_data_for_seq[:, 0] = reg_target_scaler.fit_transform(reg_all_data_for_seq[:, 0].reshape(-1,1)).flatten()
    X_reg_all, y_reg_all_scaled_log_amount = create_tf_sequences(reg_all_data_for_seq, cfg_dict["WINDOW_SIZE"], cfg_dict["OFFSET"], target_col_idx=0)
    
    # Filter for rainy days for regressor training/testing
    # original_prcp_for_y maps to the time steps of y_reg_all_scaled_log_amount
    start_idx_for_y = cfg_dict["WINDOW_SIZE"] # y values start after the first window
    original_prcp_for_y = daily_prcp_values_unscaled[start_idx_for_y : start_idx_for_y + len(y_reg_all_scaled_log_amount)]
    
    rainy_day_indices_for_reg_training = np.where(original_prcp_for_y.flatten() > 0.1)[0]

    X_reg_rainy = X_reg_all[rainy_day_indices_for_reg_training]
    y_reg_rainy_scaled_log_amount = y_reg_all_scaled_log_amount[rainy_day_indices_for_reg_training]

    logger.log_important(f"[{task_prefix}] Total sequences for regression: {len(y_reg_all_scaled_log_amount)}, Rainy day sequences for training regression: {len(y_reg_rainy_scaled_log_amount)}", to_console=False)

    data_pack = {
        "X_clf_all": X_clf, "y_clf_all": y_clf, "clf_input_scaler": clf_input_scaler,
        "X_reg_all": X_reg_all, 
        "X_reg_rainy": X_reg_rainy, "y_reg_rainy_scaled": y_reg_rainy_scaled_log_amount, 
        "reg_target_scaler": reg_target_scaler,
        "original_prcp_for_y_reg_all": original_prcp_for_y # For final eval on original scale
    }
    return data_pack


def run_prcp_classification_stage_tf(data_pack, cfg_clf, logger, task_prefix="PRCP Stage 1 (Classifier)"):
    logger.log_important(f"\n--- {task_prefix} ---", to_console=False)
    X_clf, y_clf = data_pack["X_clf_all"], data_pack["y_clf_all"]

    if X_clf.shape[0] == 0: logger.log_important(f"[{task_prefix}] No data. Skipping.", to_console=True); return None, None

    split_point = int(X_clf.shape[0] * cfg_clf["TRAIN_SPLIT_RATIO"])
    train_X_clf, test_X_clf = X_clf[:split_point], X_clf[split_point:]
    train_y_clf, test_y_clf = y_clf[:split_point], y_clf[split_point:]
    
    if train_X_clf.shape[0] == 0: 
        logger.log_important(f"[{task_prefix}] No train data after split. Skipping.", to_console=True)
        return None, test_y_clf # Return test_y_clf for potential length matching later

    logger.log_important(f"[{task_prefix}] train_X_clf shape: {train_X_clf.shape}, train_y_clf shape: {train_y_clf.shape}", to_console=False)
    logger.log_important(f"[{task_prefix}] test_X_clf shape: {test_X_clf.shape}, test_y_clf shape: {test_y_clf.shape}", to_console=False)
    
    # --- Calculate and Debug Class Weights ---
    class_weight_dict = None
    unique_train_labels, counts_train_labels = np.unique(train_y_clf.flatten(), return_counts=True)
    logger.log_important(f"[{task_prefix}] Training labels for classifier - Unique: {unique_train_labels}, Counts: {counts_train_labels}", to_console=True)

    if len(unique_train_labels) == 2: 
        # Ensure classes are [0, 1] for compute_class_weight if not already
        if not np.array_equal(unique_train_labels, np.array([0, 1])) and len(unique_train_labels) == 2:
             logger.log_important(f"[{task_prefix}] Warning: Unique classes are {unique_train_labels}, expected [0,1] for binary. Check data.", to_console=True)
             # Proceeding, but this might indicate an issue if only one class is present in training.
        
        class_weights_computed = compute_class_weight(
            class_weight='balanced', # Correct spelling
            classes=unique_train_labels, 
            y=train_y_clf.flatten()
        )
        class_weight_dict = dict(zip(unique_train_labels, class_weights_computed))
        logger.log_important(f"[{task_prefix}] Using class weights: {class_weight_dict}", to_console=True)
    elif len(unique_train_labels) == 1:
        logger.log_important(f"[{task_prefix}] Only one class ({unique_train_labels[0]}) present in training data for classifier. Cannot compute balanced class weights. Proceeding without.", to_console=True)
    else: # Should not happen for binary target
        logger.log_important(f"[{task_prefix}] Unexpected number of classes ({len(unique_train_labels)}) in training data for classifier. Proceeding without class weights.", to_console=True)
    # --- End Calculate Class Weights ---

    model_input_shape = (cfg_clf["WINDOW_SIZE"], X_clf.shape[2]) # X_clf.shape[2] is num_features
    model_clf = create_lstm_forecaster_tf(
        input_shape=model_input_shape, hidden_units=cfg_clf["HIDDEN_UNITS"],
        num_layers=cfg_clf["NUM_LAYERS"], output_units=1, 
        dropout_rate=cfg_clf["DROPOUT_RATE"], output_activation=cfg_clf["OUTPUT_ACTIVATION"]
    )
    model_clf.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=cfg_clf["LEARNING_RATE"]),
                      loss=cfg_clf["LOSS_FUNCTION"], metrics=cfg_clf.get("METRICS", ["accuracy"]))
    
    stringlist = []; model_clf.summary(print_fn=lambda x: stringlist.append(x))
    logger.log_important(f"Model Summary ({task_prefix}):\n" + "\n".join(stringlist), to_console=False)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg_clf["EARLY_STOPPING_PATIENCE"], restore_best_weights=True)
    print(f"Training {task_prefix} model (Keras verbose output to console)...")
    
    history, model_clf = train_tf_model(model_clf, train_X_clf, train_y_clf, 
                                         validation_data=(test_X_clf, test_y_clf),
                                         epochs=cfg_clf["EPOCHS"], batch_size=cfg_clf["BATCH_SIZE"],
                                         callbacks=[early_stopping], 
                                         class_weight=class_weight_dict, 
                                         verbose=1) 
    
    # Safely access history and metrics
    val_loss_history = history.history.get('val_loss', [np.nan])
    best_val_loss = min(val_loss_history)
    best_epoch = np.argmin(val_loss_history) + 1
    
    val_metrics_log_str = []
    for metric_obj in cfg_clf.get("METRICS", []):
        metric_name = metric_obj if isinstance(metric_obj, str) else metric_obj.name
        val_metric_history = history.history.get(f'val_{metric_name}', [np.nan])
        # For accuracy/precision/recall, we usually want the value at the best val_loss epoch, or max if it's an accuracy type metric
        metric_at_best_val_loss = val_metric_history[best_epoch-1] if len(val_metric_history) >= best_epoch else np.nan
        val_metrics_log_str.append(f"val_{metric_name}: {metric_at_best_val_loss:.4f}")
    
    logger.log_important(f"[{task_prefix}] Training finished. Best val_loss: {best_val_loss:.4f} ({', '.join(val_metrics_log_str)}) at epoch {best_epoch}", to_console=True)

    eval_results, preds_proba_clf = evaluate_tf_model(model_clf, test_X_clf, test_y_clf, batch_size=cfg_clf["BATCH_SIZE"], verbose=0)
    log_eval_metrics = [f"{model_clf.metrics_names[i]}: {eval_results[i]:.4f}" for i in range(len(eval_results))]
    logger.log_important(f"[{task_prefix}] Evaluation results on test set ({', '.join(log_eval_metrics)})", to_console=True)

    preds_binary_clf = (preds_proba_clf > 0.5).astype(int)
    if test_y_clf.size > 0: # Ensure test_y_clf is not empty
        acc = accuracy_score(test_y_clf, preds_binary_clf)
        f1 = f1_score(test_y_clf, preds_binary_clf, zero_division=0)
        precision_val = precision_score(test_y_clf, preds_binary_clf, zero_division=0)
        recall_val = recall_score(test_y_clf, preds_binary_clf, zero_division=0)
        logger.log_important(f"[{task_prefix}] Test Metrics (threshold 0.5): Acc: {acc:.4f}, F1: {f1:.4f}, P: {precision_val:.4f}, R: {recall_val:.4f}", to_console=True)
    
    return model_clf, test_X_clf, test_y_clf # Return test_X for consistent prediction length

def run_prcp_regression_stage_tf(data_pack, cfg_reg, logger, task_prefix="PRCP Stage 2 (Regressor)"):
    logger.log_important(f"\n--- {task_prefix} ---", to_console=False)
    X_reg_rainy, y_reg_rainy_scaled = data_pack["X_reg_rainy"], data_pack["y_reg_rainy_scaled"]

    if X_reg_rainy.size == 0 or y_reg_rainy_scaled.size == 0:
        logger.log_important(f"[{task_prefix}] Not enough rainy day data for regression. Skipping.", to_console=True)
        return None

    split_point = int(X_reg_rainy.shape[0] * cfg_reg["TRAIN_SPLIT_RATIO"])
    train_X_reg, test_X_reg_rainy_subset = X_reg_rainy[:split_point], X_reg_rainy[split_point:] # test_X for this subset
    train_y_reg, test_y_reg_rainy_subset_scaled = y_reg_rainy_scaled[:split_point], y_reg_rainy_scaled[split_point:]
    
    if train_X_reg.shape[0] == 0: logger.log_important(f"[{task_prefix}] No train data for regressor. Skipping.", to_console=True); return None

    logger.log_important(f"[{task_prefix}] Train X shape (rainy days): {train_X_reg.shape}, Train y shape: {train_y_reg.shape}", to_console=False)
    
    model_input_shape = (cfg_reg["WINDOW_SIZE"], X_reg_rainy.shape[2])
    model_reg = create_lstm_forecaster_tf(
        input_shape=model_input_shape, hidden_units=cfg_reg["HIDDEN_UNITS"],
        num_layers=cfg_reg["NUM_LAYERS"], output_units=1, 
        dropout_rate=cfg_reg["DROPOUT_RATE"], output_activation=cfg_reg.get("OUTPUT_ACTIVATION", None)
    )
    model_reg.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=cfg_reg["LEARNING_RATE"]),
                      loss=cfg_reg["LOSS_FUNCTION"], metrics=cfg_reg.get("METRICS", ["mae"]))
    stringlist = []; model_reg.summary(print_fn=lambda x: stringlist.append(x))
    logger.log_important(f"Model Summary ({task_prefix}):\n" + "\n".join(stringlist), to_console=False)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg_reg["EARLY_STOPPING_PATIENCE"], restore_best_weights=True)
    print(f"Training {task_prefix} model (Keras verbose output to console)...")
    history, model_reg = train_tf_model(model_reg, train_X_reg, train_y_reg, 
                                         validation_data=(test_X_reg_rainy_subset, test_y_reg_rainy_subset_scaled),
                                         epochs=cfg_reg["EPOCHS"], batch_size=cfg_reg["BATCH_SIZE"],
                                         callbacks=[early_stopping], verbose=1)
    
    best_val_loss = min(history.history.get('val_loss', [np.nan]))
    best_val_mae = min(history.history.get('val_mae', [np.nan]))
    logger.log_important(f"[{task_prefix}] Training finished. Best val_loss ({cfg_reg['LOSS_FUNCTION']}): {best_val_loss:.4f} (val_mae: {best_val_mae:.4f}) at epoch {np.argmin(history.history.get('val_loss',[0]))+1}", to_console=True)
    
    if test_X_reg_rainy_subset.shape[0] > 0: # Only evaluate if there's test data for rainy subset
        eval_results, _ = evaluate_tf_model(model_reg, test_X_reg_rainy_subset, test_y_reg_rainy_subset_scaled, batch_size=cfg_reg["BATCH_SIZE"], verbose=0)
        logger.log_important(f"[{task_prefix}] Evaluation on rainy test days (Loss '{cfg_reg['LOSS_FUNCTION']}': {eval_results[0]:.4f}, MAE_on_scaled_log: {eval_results[1]:.4f})", to_console=True)

    return model_reg


def run_weather_prediction_prcp_two_stage_tf(ghn_instance, data_fetcher_instance, station_idx, logger):
    task_name = "Task 3: Two-Stage PRCP Prediction"
    logger.log_important(f"\n--- {task_name} ---", to_console=False)
    cfg_clf = config.PRCP_CLASSIFICATION_TF
    cfg_reg = config.PRCP_REGRESSION_TF

    print(f"Starting {task_name} for station {ghn_instance.getStatKeyNames()[station_idx]}...")
    raw_prcp_data_tuple = data_fetcher_instance.PRCP(station_idx)
    if not raw_prcp_data_tuple[0] or not raw_prcp_data_tuple[1]:
        logger.log_important(f"[{task_name}] No PRCP data. Skipping.", to_console=True); return

    # Prepare data
    filled_dates, filled_prcp_orig, _ = fillPRCP(raw_prcp_data_tuple)
    if filled_prcp_orig.size == 0: logger.log_important(f"[{task_name}] PRCP data empty. Skipping.", to_console=True); return
    prcp_series_pd = pd.Series(filled_prcp_orig, index=pd.to_datetime(filled_dates))
    daily_prcp_values_unscaled = prcp_series_pd.dropna().values.astype(np.float32)
    dates_for_features = prcp_series_pd.dropna().index
    if daily_prcp_values_unscaled.shape[0] < cfg_clf["WINDOW_SIZE"] + cfg_clf["OFFSET"] + 20: # Use one of the cfgs
        logger.log_important(f"[{task_name}] Not enough PRCP data. Skipping.", to_console=True); return
    
    data_pack = prepare_prcp_two_stage_data(daily_prcp_values_unscaled, dates_for_features, cfg_clf, logger, task_name) # Use cfg_clf for window/offset
    if data_pack is None: return

    model_clf, test_X_clf_for_final_preds, test_y_occurrence_actual = run_prcp_classification_stage_tf(data_pack, cfg_clf, logger)
    if model_clf is None: logger.log_important(f"[{task_name}] Classifier training failed. Skipping.", to_console=True); return

    model_reg = run_prcp_regression_stage_tf(data_pack, cfg_reg, logger)

    # --- Combine Predictions ---
    # Use X_reg_all's test split for amount prediction inputs
    split_idx_all = int(data_pack["X_reg_all"].shape[0] * cfg_reg["TRAIN_SPLIT_RATIO"]) # Assuming TRAIN_SPLIT_RATIO is consistent
    test_X_reg_for_final_preds = data_pack["X_reg_all"][split_idx_all:]
    
    # Ensure consistent length for test sets for final evaluation
    min_len = min(len(test_X_clf_for_final_preds), len(test_X_reg_for_final_preds), len(test_y_occurrence_actual))
    test_X_clf_for_final_preds = test_X_clf_for_final_preds[:min_len]
    test_X_reg_for_final_preds = test_X_reg_for_final_preds[:min_len]
    test_y_occurrence_actual = test_y_occurrence_actual[:min_len] # True 0/1 rain for these test samples

    # Original PRCP values for the final test set
    test_y_prcp_original = data_pack["original_prcp_for_y_reg_all"][split_idx_all : split_idx_all + min_len]


    preds_occurrence_proba = model_clf.predict(test_X_clf_for_final_preds, verbose=0)
    preds_occurrence_binary = (preds_occurrence_proba > 0.5).astype(int).flatten()

    final_prcp_predictions = np.zeros_like(preds_occurrence_binary, dtype=float)
    if model_reg is not None and test_X_reg_for_final_preds.shape[0] > 0 :
        preds_amount_scaled_log = model_reg.predict(test_X_reg_for_final_preds, verbose=0)
        preds_amount_log = data_pack["reg_target_scaler"].inverse_transform(preds_amount_scaled_log)
        preds_amount_orig_scale = np.expm1(preds_amount_log).flatten()
        preds_amount_orig_scale = np.maximum(0, preds_amount_orig_scale)
        final_prcp_predictions = preds_amount_orig_scale * preds_occurrence_binary
    else:
        logger.log_important(f"[{task_name}] Regressor not available or no test data for it; amounts for predicted rain events will be 0.", to_console=True)

    if len(test_y_prcp_original) == len(final_prcp_predictions) and len(test_y_prcp_original) > 0 :
        combined_mae = np.mean(np.abs(final_prcp_predictions - test_y_prcp_original.flatten()))
        combined_rmse = np.sqrt(np.mean((final_prcp_predictions - test_y_prcp_original.flatten())**2))
        logger.log_important(f"[{task_name}] Combined Two-Stage PRCP MAE: {combined_mae:.2f} mm, RMSE: {combined_rmse:.2f} mm", to_console=True)
        
        # Re-evaluate occurrence on the aligned final test set
        acc = accuracy_score(test_y_occurrence_actual.flatten(), preds_occurrence_binary)
        f1 = f1_score(test_y_occurrence_actual.flatten(), preds_occurrence_binary, zero_division=0)
        precision_val = precision_score(test_y_occurrence_actual.flatten(), preds_occurrence_binary, zero_division=0)
        recall_val = recall_score(test_y_occurrence_actual.flatten(), preds_occurrence_binary, zero_division=0)
        logger.log_important(f"[{task_name}] Final Test Occurrence Metrics (on {min_len} samples): Acc: {acc:.4f}, F1: {f1:.4f}, P: {precision_val:.4f}, R: {recall_val:.4f}", to_console=True)

        # Plotting
        plt.figure(figsize=(15, 7))
        plot_len = min(200, len(final_prcp_predictions))
        plt.plot(test_y_prcp_original[:plot_len], label="Actual Daily PRCP", alpha=0.9, linewidth=1.5)
        plt.plot(final_prcp_predictions[:plot_len], label="Two-Stage LSTM Predicted PRCP", linestyle='--', linewidth=1)
        station_info = ghn_instance.getStation(ghn_instance.getStatKeyNames()[station_idx])
        plt.title(f"TF Two-Stage Daily PRCP: {station_info.name if station_info else 'Unknown'} (Test Set)")
        plt.xlabel("Day"); plt.ylabel("PRCP (mm)")
        plt.legend(); plt.grid(True)
        plot_filename = os.path.join(config.FIGURE_SAVE_DIR, f"tf_weather_daily_prcp_twostage_s{station_idx}.png")
        plt.savefig(plot_filename); plt.close()
        logger.log_important(f"[{task_name}] Saved plot to {plot_filename}", to_console=False)
    else:
        logger.log_important(f"[{task_name}] Not enough aligned data for final PRCP evaluation or plotting.", to_console=True)