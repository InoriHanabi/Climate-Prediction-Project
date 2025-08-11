# generate_latex_table.py
import pandas as pd
import numpy as np
import os
import re # For more robust sanitization

def sanitize_for_latex(text_string):
    """Sanitizes a string for use in LaTeX by escaping special characters."""
    if not isinstance(text_string, str):
        text_string = str(text_string)
    
    # Order of replacements can matter
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_', # Underscore is very common
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}', # Backslash itself
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    # Create a regex that matches any of the special characters
    regex = re.compile('|'.join(re.escape(key) for key in conv.keys()))
    # For each match, look up the corresponding replacement
    return regex.sub(lambda match: conv[match.group(0)], text_string)


def create_latex_table_from_csv(csv_filepath, output_latex_filename="station_summary_table.tex",
                                num_stations_to_show=10, elements_to_show=['TMAX', 'PRCP']):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return

    unique_stations = df['Station_ID'].unique()
    if len(unique_stations) == 0:
        print("No unique stations found in the CSV.")
        return

    if num_stations_to_show >= len(unique_stations):
        selected_station_ids = unique_stations
        print(f"Selected all {len(unique_stations)} stations for the LaTeX table.")
    else:
        selected_station_ids = np.random.choice(unique_stations, num_stations_to_show, replace=False)
        print(f"Randomly selected {num_stations_to_show} stations for the LaTeX table: {list(selected_station_ids)}")
        
    df_subset = df[df['Station_ID'].isin(selected_station_ids) & df['Element'].isin(elements_to_show)].copy()

    if df_subset.empty:
        print("No data to display for the selected stations and elements.")
        return

    # Apply sanitization BEFORE formatting percentages, as % is a special char
    for col in ['Station_ID', 'Country', 'Element', 'Start_Date', 'End_Date']: # Columns likely to have text
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].apply(lambda x: sanitize_for_latex(x) if pd.notna(x) else 'N/A')
            
    # Format numeric columns AFTER potential text sanitization if they were objects
    numeric_cols_for_rounding = ['mean', 'std', 'min', 'max', '25%', '50%', '75%'] # Common numeric columns
    for col in numeric_cols_for_rounding:
        if col in df_subset.columns:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce').round(2)


    df_subset['Availability_Pct'] = pd.to_numeric(df_subset['Availability_Pct'], errors='coerce').round(1).astype(str) + '\\%'
    df_subset['zero_value_pct'] = pd.to_numeric(df_subset['zero_value_pct'], errors='coerce').apply(lambda x: f"{x:.1f}\\%" if pd.notna(x) else 'N/A')
    
    columns_to_display_map = { # Renamed to map for clarity
        'Station_ID': 'Station ID',
        'Country': 'Country',
        'Element': 'Element',
        'Start_Date': 'Start',
        'End_Date': 'End',
        'Valid_Points': 'Valid Pts',
        'Availability_Pct': 'Avail. (\\%)',
        'mean': 'Mean',
        'std': 'Std Dev',
        'min': 'Min',
        'max': 'Max',
        # Only include quantile columns if they exist in the CSV from describe()
        # Pandas describe() output for '25%', '50%', '75%' might vary based on pandas version
        # Let's assume they are '25%', '50%', '75%' in the CSV.
        # If they are 0.25, 0.50, 0.75, the describe in analyze_MULTI_station_data.py handles it.
        # This script assumes they are already strings '25%', etc. as keys.
        '25%': '25\\%', # Quantiles
        '50%': '50\\% (Median)',
        '75%': '75\\%',
        'zero_value_pct': 'Zero Val. (\\%)'
    }
    
    # Ensure only existing columns are selected and in the desired order
    final_columns_ordered = []
    for csv_col, display_col in columns_to_display_map.items():
        if csv_col in df_subset.columns:
            final_columns_ordered.append(csv_col)
            
    df_display = df_subset[final_columns_ordered].rename(columns=columns_to_display_map)

    latex_output_lines = []
    latex_output_lines.append("\\begin{table}[htbp]")
    latex_output_lines.append("    \\centering")
    latex_output_lines.append(f"    \\caption{{Summary Statistics for Selected GSN Stations and Elements (Sample of {len(selected_station_ids)} Stations).}}")
    latex_output_lines.append("    \\label{tab:station_summary_sample}")
    latex_output_lines.append("    \\resizebox{\\textwidth}{!}{%") 
    
    num_cols = len(df_display.columns)
    # Define column alignment: e.g., left for text, right for numbers
    # Example: 'lllr' + 'r'*(num_cols - 4) for 3 text cols, 1 left (%), rest right
    # Let's try a robust default: left for first few descriptive columns, then right for numbers.
    if num_cols >= 3: # Station ID, Country, Element
        col_format = 'l' * 3 
        if num_cols > 3 : # Start, End, Valid Pts
             col_format += 'l' * min(3, num_cols - 3) 
        if num_cols > 6: # Numerical stats (mean to max)
            col_format += 'r' * min(4, num_cols - 6) # Assuming up to 4 main stats
        if num_cols > 10: # Quantiles and Zero Val Pct
            col_format += 'r' * (num_cols - 10 -1) # Quantiles right
            col_format += 'l' # Zero Val Pct left
        if len(col_format) < num_cols: # Fallback for any remaining
            col_format += 'l' * (num_cols - len(col_format))

    else: # If very few columns
        col_format = 'l' * num_cols
    
    # Ensure col_format string length matches num_cols exactly
    if len(col_format) != num_cols:
        print(f"Warning: Column format length ({len(col_format)}) doesn't match number of columns ({num_cols}). Defaulting to all 'l'.")
        col_format = 'l' * num_cols

    latex_output_lines.append(f"    \\begin{{tabular}}{{{col_format}}}")
    latex_output_lines.append("        \\toprule")
    # Sanitize column headers for LaTeX
    sanitized_headers = [sanitize_for_latex(col) for col in df_display.columns]
    latex_output_lines.append(f"        {' & '.join(sanitized_headers)} \\\\")
    latex_output_lines.append("        \\midrule")
    
    for _, row in df_display.iterrows():
        # Values were already sanitized if they were text, or are numbers/formatted percentages
        # Convert all to string for joining, ensure NaNs are 'N/A'
        row_values = []
        for val in row.values:
            if pd.isna(val):
                row_values.append('N/A')
            else:
                # For numerical values that were not pre-formatted as strings (like mean, std, min, max from describe)
                # they might need specific formatting here if not done earlier.
                # However, df_subset already rounded them.
                # The percentage columns are already strings with '\%'.
                # Other text columns were sanitized.
                row_values.append(str(val)) 
        latex_output_lines.append(f"        {' & '.join(row_values)} \\\\")
        
    latex_output_lines.append("        \\bottomrule")
    latex_output_lines.append("    \\end{tabular}%")
    latex_output_lines.append("    }") 
    latex_output_lines.append("\\end{table}")
    
    final_latex_code = "\n".join(latex_output_lines)

    with open(output_latex_filename, 'w', encoding='utf-8') as f:
        f.write(final_latex_code)
    print(f"LaTeX table code saved to {output_latex_filename}")

if __name__ == '__main__':
    csv_file = "country_representative_GSN_station_summary.csv" 
    if os.path.exists(csv_file):
        try:
            df_temp = pd.read_csv(csv_file)
            num_unique_stations_in_csv = df_temp['Station_ID'].nunique()
            print(f"Found {num_unique_stations_in_csv} unique stations in {csv_file}.")
        except Exception as e:
            print(f"Could not read {csv_file} to determine station count: {e}")
            num_unique_stations_in_csv = 200 

        create_latex_table_from_csv(
            csv_filepath=csv_file,
            output_latex_filename="ALL_country_representative_stations_table.tex", 
            num_stations_to_show=10, 
            elements_to_show=['TMAX', 'PRCP', 'SNOW'] # Example with SNOW
        )
    else:
        print(f"Error: {csv_file} not found. Please run analyze_MULTI_station_data.py first.")