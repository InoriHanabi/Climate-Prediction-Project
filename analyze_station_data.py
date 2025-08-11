# analyze_MULTI_station_data.py

import numpy as np
import pandas as pd
from datetime import datetime, date
import os
import random # For picking a random station if a country has multiple GSN stations

from ghcn_helpers import GHNCD, getdata 

def generate_multi_station_data_report(ghn, data_fetcher, 
                                       station_ids_to_process, # Now this will be the curated list
                                       output_filename="country_representative_station_report.txt",
                                       csv_output_filename="country_representative_station_summary.csv"):
    """
    Generates a summary report and a CSV for a curated list of stations.
    """
    report_lines = []
    csv_data = [] # List of dictionaries for CSV

    report_lines.append(f"Representative Station Data Summary Report (One GSN station per country)")
    report_lines.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"Attempting to process {len(station_ids_to_process)} stations (one per country with GSN stations).\n")
    _write_report(report_lines, output_filename, mode='w')
    report_lines = [] 

    processed_station_count = 0
    failed_station_count = 0

    elements_to_analyze = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD'] 

    for idx, station_id in enumerate(station_ids_to_process):
        # Give console feedback for long runs
        # The print inside the loop is removed to make this console output the main progress indicator
        # The detailed station processing will go to the file.
        # sys.stdout.write(f"\rProcessing station {idx+1}/{len(station_ids_to_process)}: {station_id} ...")
        # sys.stdout.flush()
        # For simpler, less frequent console updates:
        if (idx + 1) % 5 == 0 or idx == 0 or (idx + 1) == len(station_ids_to_process):
             print(f"Processing station {idx+1}/{len(station_ids_to_process)}: {station_id} ...")


        station_object = ghn.getStation(station_id)
        current_station_report = []

        if not station_object:
            current_station_report.append(f"--- Station ID: {station_id} (Metadata not found) ---")
            current_station_report.append("  Skipping this station.\n")
            failed_station_count +=1
            _append_to_report(current_station_report, output_filename)
            continue
        
        current_station_report.append(f"--- Station ID: {station_object.sid} ---")
        current_station_report.append(f"Name: {station_object.name}")
        current_station_report.append(f"Country: {station_object.country}") # Explicitly add country
        current_station_report.append(f"Coords: Lat={station_object.lat}, Lon={station_object.lon}, Elev={station_object.el}m")

        try:
            station_idx_for_fetcher = data_fetcher.statNames.index(station_id)
        except ValueError:
            current_station_report.append("  Error: Station ID not found in data_fetcher's list. Skipping data fetch.\n")
            failed_station_count +=1
            _append_to_report(current_station_report, output_filename)
            continue
            
        # This print is useful for console during long runs to see which file is being fetched
        print(f"  Fetching .dly for {station_id}...")
        statDict, _ = data_fetcher._fetch_and_process_station(station_idx_for_fetcher, f"data for {station_id}")

        if not statDict:
            current_station_report.append("  Could not retrieve or process the .dly file for this station.\n")
            failed_station_count +=1
            _append_to_report(current_station_report, output_filename)
            continue
        
        current_station_report.append(f"  Parsed elements from .dly: {list(statDict.keys())}")
        station_has_any_key_data = False

        for element in elements_to_analyze:
            current_station_report.append(f"  -- Element: {element} --")
            raw_element_data = ghn.getVar(statDict, element)
            
            station_data_for_csv = { # Initialize CSV row data
                'Station_ID': station_object.sid,
                'Station_Name': station_object.name,
                'Country': station_object.country,
                'Element': element,
                'Start_Date': 'N/A', 'End_Date': 'N/A',
                'Days_in_Span': 0, 'Valid_Points': 0, 'Availability_Pct': 0.0,
                'count': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan,
                '25%': np.nan, '50%': np.nan, '75%': np.nan, 'max': np.nan,
                'zero_value_days': np.nan, 'zero_value_pct': np.nan
            }

            if not raw_element_data:
                current_station_report.append("    No valid data points found (all missing or filtered by QFLAG).")
                csv_data.append(station_data_for_csv) # Add empty stats for this element
                continue 

            station_has_any_key_data = True
            dates, values = zip(*raw_element_data)
            
            if not values: 
                current_station_report.append("    Data list for values is empty after unzip.")
                csv_data.append(station_data_for_csv)
                continue

            values_series = pd.Series(values) 
            dates_pd = pd.to_datetime(list(dates))

            station_data_for_csv['Start_Date'] = dates_pd.min().strftime('%Y-%m-%d')
            station_data_for_csv['End_Date'] = dates_pd.max().strftime('%Y-%m-%d')
            station_data_for_csv['Valid_Points'] = len(values)
            
            if station_data_for_csv['Valid_Points'] > 0:
                total_days_in_span = (dates_pd.max() - dates_pd.min()).days + 1
                station_data_for_csv['Days_in_Span'] = total_days_in_span
                percentage_present = (station_data_for_csv['Valid_Points'] / total_days_in_span) * 100 if total_days_in_span > 0 else 0
                station_data_for_csv['Availability_Pct'] = round(percentage_present, 1)
            
            current_station_report.append(f"    Observed Period (valid data): {station_data_for_csv['Start_Date']} to {station_data_for_csv['End_Date']}")
            current_station_report.append(f"    Days in span: {station_data_for_csv['Days_in_Span']}, Valid data points: {station_data_for_csv['Valid_Points']} ({station_data_for_csv['Availability_Pct']:.1f}% available)")
            
            stats = values_series.describe().to_dict()
            for stat_name, stat_val in stats.items():
                current_station_report.append(f"      {stat_name}: {stat_val:.2f}")
                if stat_name in station_data_for_csv: # Update CSV data
                    station_data_for_csv[stat_name] = round(stat_val, 2)

            if element in ['PRCP', 'SNOW', 'SNWD']:
                zero_days_count = np.sum(values_series == 0)
                zero_pct = (zero_days_count/station_data_for_csv['Valid_Points'])*100 if station_data_for_csv['Valid_Points'] > 0 else 0
                current_station_report.append(f"      zero_value_days: {zero_days_count} ({zero_pct:.1f}%)")
                station_data_for_csv['zero_value_days'] = zero_days_count
                station_data_for_csv['zero_value_pct'] = round(zero_pct, 1)
            
            csv_data.append(station_data_for_csv) # Add populated stats for this element
        
        if station_has_any_key_data:
            processed_station_count += 1
        else:
            current_station_report.append("  No processable data found for key elements TMAX, TMIN, PRCP, SNOW, SNWD in this station.")
            failed_station_count += 1
            
        current_station_report.append("\n") 
        _append_to_report(current_station_report, output_filename)

    # Write CSV file
    if csv_data:
        df_summary = pd.DataFrame(csv_data)
        df_summary.to_csv(csv_output_filename, index=False)
        print(f"Summary statistics saved to {csv_output_filename}")
    else:
        print("No data collected for CSV summary.")

    final_summary_lines = [
        "\n--- Overall Summary ---",
        f"Total unique countries with GSN stations targeted: {len(station_ids_to_process)}",
        f"Stations successfully processed (with some data): {processed_station_count}",
        f"Stations failed or with no key element data: {failed_station_count}"
    ]
    _append_to_report(final_summary_lines, output_filename)
    print("Multi-station data report generation complete.")

def _write_report(lines, filename, mode='w'):
    try:
        with open(filename, mode, encoding='utf-8') as f: # Added encoding
            for line in lines:
                f.write(line + "\n")
    except Exception as e:
        print(f"Error writing to report file {filename}: {e}")


def _append_to_report(lines, filename):
    _write_report(lines, filename, mode='a')


if __name__ == '__main__':
    ghn = GHNCD()
    print("Initializing GHCN data handlers...")
    print("Reading countries file...")
    ghn.readCountriesFile() 
    print("Reading stations file...")
    # Ensure justGSN=True to only load GSN stations, as per your earlier successful runs
    ghn.readStationsFile(justGSN=True) 

    if not ghn.getStatKeyNames():
        print("No stations loaded from GHNCD. Cannot generate report.")
    else:
        data_fetcher = getdata(ghn_instance=ghn)
        
        # --- Select one GSN station per country ---
        stations_by_country = {}
        for station_id in ghn.getStatKeyNames(): # Iterate over loaded GSN stations
            station_obj = ghn.getStation(station_id)
            if station_obj:
                country_name = station_obj.country
                if country_name not in stations_by_country:
                    stations_by_country[country_name] = []
                stations_by_country[country_name].append(station_id)
        
        representative_stations = []
        for country, station_list in stations_by_country.items():
            if station_list: # If there are stations for this country
                representative_stations.append(random.choice(station_list)) # Pick one randomly

        print(f"Selected {len(representative_stations)} representative GSN stations (one per country with GSN stations).")
        
        if representative_stations:
            generate_multi_station_data_report(ghn, data_fetcher, 
                                               station_ids_to_process=representative_stations,
                                               output_filename="country_representative_GSN_station_report.txt",
                                               csv_output_filename="country_representative_GSN_station_summary.csv")
        else:
            print("No representative stations selected. Check GSN station loading and country data.")