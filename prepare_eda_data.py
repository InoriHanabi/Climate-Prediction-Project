# prepare_eda_data.py
import os
import pandas as pd
import numpy as np

# Assuming ghcn_helpers and config are in the same directory
import config
from ghcn_helpers import GHNCD, getdata, fillholesT, fillPRCP, fillSN

def create_and_save_eda_dataframe(ghn_instance, data_fetcher_instance, station_idx, output_filepath):
    """
    Loads, cleans, fills, and consolidates all key weather variables into a single
    DataFrame and saves it to a CSV file.
    """
    print(f"--- Starting data preparation for station index {station_idx} ---")
    
    # --- Load and fill TMAX and TMIN ---
    print("Loading and filling TMAX/TMIN...")
    tmax_tmin_tuple = data_fetcher_instance.TmaxTmin(station_idx)
    if not tmax_tmin_tuple[1] and not tmax_tmin_tuple[3]: # Check if both tmax and tmin value lists are empty
        print("Error: Initial TMAX and TMIN data is empty. Aborting.")
        return None
    dates_max, tmax, _, tmin, _, _ = fillholesT(tmax_tmin_tuple)
    if tmax.size == 0 or tmin.size == 0:
        print("Error: Temperature data is empty after filling. Aborting.")
        return None
    df_temp = pd.DataFrame({'TMAX': tmax, 'TMIN': tmin}, index=pd.to_datetime(dates_max))
    
    # --- Load and fill PRCP ---
    print("Loading and filling PRCP...")
    prcp_tuple = data_fetcher_instance.PRCP(station_idx)
    if not prcp_tuple[1]: # Check if value list is empty
        print("Warning: No initial PRCP data. Will create an empty column.")
        df_prcp = pd.DataFrame()
    else:
        dates_prcp, prcp, _ = fillPRCP(prcp_tuple)
        if prcp.size > 0:
            df_prcp = pd.DataFrame({'PRCP': prcp}, index=pd.to_datetime(dates_prcp))
        else:
            print("Warning: No PRCP data after filling.")
            df_prcp = pd.DataFrame()

    # --- Load and fill SNOW ---
    print("Loading and filling SNOW...")
    snow_tuple = data_fetcher_instance.SNOW(station_idx)
    if not snow_tuple[1]:
        print("Warning: No initial SNOW data. Will create an empty column.")
        df_snow = pd.DataFrame()
    else:
        dates_snow, snow, _ = fillSN(snow_tuple)
        if snow.size > 0:
            df_snow = pd.DataFrame({'SNOW': snow}, index=pd.to_datetime(dates_snow))
        else:
            print("Warning: No SNOW data after filling.")
            df_snow = pd.DataFrame()

    # --- Load and fill SNWD (Snow Depth) ---
    print("Loading and filling SNWD...")
    snwd_tuple = data_fetcher_instance.SNWD(station_idx)
    if not snwd_tuple[1]:
        print("Warning: No initial SNWD data. Will create an empty column.")
        df_snwd = pd.DataFrame()
    else:
        dates_snwd, snwd, _ = fillSN(snwd_tuple) # Reuse fillSN for SNWD
        if snwd.size > 0:
            df_snwd = pd.DataFrame({'SNWD': snwd}, index=pd.to_datetime(dates_snwd))
        else:
            print("Warning: No SNWD data after filling.")
            df_snwd = pd.DataFrame()

    # --- Combine all dataframes ---
    print("Consolidating all variables into a single DataFrame...")
    df_all = pd.concat([df_temp, df_prcp, df_snow, df_snwd], axis=1)
    
    # Fill any remaining NaNs from alignment mismatches and fill empty columns with 0
    df_all = df_all.ffill().bfill()
    df_all = df_all.fillna(0) # Fill any columns that were completely empty with 0

    print(f"Final DataFrame shape before saving: {df_all.shape}")
    print("DataFrame Head:")
    print(df_all.head())

    # --- Save to CSV ---
    try:
        df_all.to_csv(output_filepath)
        print(f"\nSuccessfully saved consolidated EDA data to '{output_filepath}'")
    except Exception as e:
        print(f"\nError saving data to CSV: {e}")

    return df_all


if __name__ == '__main__':
    # Initialize data loaders
    ghn = GHNCD()
    print("Reading countries file...")
    ghn.readCountriesFile(fileName=config.COUNTRIES_FILE_PATH)
    print("\nReading stations file...")
    ghn.readStationsFile(fileName=config.STATIONS_FILE_PATH, justGSN=True)
    if not ghn.getStatKeyNames(): 
        print("No stations loaded. Exiting.")
        exit()
        
    data_fetcher = getdata(ghn_instance=ghn)
    
    # Define station and output file
    station_idx_to_run = config.SELECTED_STATION_IDX
    station_id = ghn.getStatKeyNames()[station_idx_to_run]
    station_info = ghn.getStation(station_id)
    output_csv_path = "eda_station_data.csv"
    
    print(f"\nPreparing data for EDA. Station: {station_info}")
    
    # Run the preparation and save
    create_and_save_eda_dataframe(ghn, data_fetcher, station_idx_to_run, output_csv_path)