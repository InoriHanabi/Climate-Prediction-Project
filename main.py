# main.py (slim version)
import os
import sys # Keep sys if SelectiveLogger was here, but not strictly needed in main if logger is passed
import pandas as pd
import tensorflow as tf 

import config
from ghcn_helpers import GHNCD, getdata 
from model_runners import (
    run_climate_prediction_tmax_tf,
    run_weather_prediction_tmax_tf,
    run_weather_prediction_prcp_two_stage_tf,
    SelectiveLogger # <<<<<<<<<<<<<<<<<<<<<<< ENSURE THIS IS CORRECTLY IMPORTED
)

if __name__ == '__main__':
    with open(config.LOG_FILE_PATH, 'w') as f:
        f.write(f"Log started at {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU with memory growth set.")
        except RuntimeError as e: print(f"GPU setup error: {e}")
    else:
        print("Running on CPU.")

    if not os.path.exists(config.FIGURE_SAVE_DIR):
        os.makedirs(config.FIGURE_SAVE_DIR)

    ghn = GHNCD()
    print("Reading countries file...")
    ghn.readCountriesFile(fileName=config.COUNTRIES_FILE_PATH)
    print("Reading stations file...")
    ghn.readStationsFile(fileName=config.STATIONS_FILE_PATH, justGSN=True)
    if not ghn.getStatKeyNames(): 
        print("No stations loaded. Exiting.")
        exit()
        
    data_fetcher = getdata(ghn_instance=ghn)
    station_idx_to_run = config.SELECTED_STATION_IDX
    station_id_to_run = ghn.getStatKeyNames()[station_idx_to_run]
    station_object = ghn.getStation(station_id_to_run)
    
    with SelectiveLogger(config.LOG_FILE_PATH, console_too=False) as logger: 
        if station_object:
            logger.log_important(f"\nProcessing for station: {station_object}", to_console=False)
        else:
            logger.log_important(f"\nStation with ID {station_id_to_run} not found. Exiting.", to_console=True)
            exit()

        run_climate_prediction_tmax_tf(ghn, data_fetcher, station_idx_to_run, logger)
        run_weather_prediction_tmax_tf(ghn, data_fetcher, station_idx_to_run, logger)
        run_weather_prediction_prcp_two_stage_tf(ghn, data_fetcher, station_idx_to_run, logger)
        
        logger.log_important("\nAll TensorFlow tasks complete.", to_console=True)

    print(f"\nSummary log written to {config.LOG_FILE_PATH}")
    print("Verbose Keras training output and TF debug messages were shown in the console only.")