import os
import pandas as pd
import glob

os.makedirs("main_data", exist_ok=True)  # ADD THIS

# Your station coordinates
STATION_LAT = -24.1296  
STATION_LON = 15.8947
STATION_ID = "31201"

# Combine all monthly CSVs
path = "weather_station_data_csv"
all_files = glob.glob(os.path.join(path, "*.csv"))

# Read CSV files with semicolon delimiter
combined_df = pd.concat([pd.read_csv(f, delimiter=';') for f in all_files], ignore_index=True)

# The date column is named 'Date' (with capital D)
date_column = 'Date'

# Convert date format (handles "01 May 2017")
combined_df[date_column] = pd.to_datetime(combined_df[date_column], format='%d %b %Y')

# Create complete date range (Jan 1, 2017 to Jun 30, 2026)
all_dates = pd.date_range(start='2017-01-01', end='2026-06-30', freq='D')
all_dates = all_dates[all_dates.month.isin([1,2,3,4,5,6,7,8,9,10,11,12])]  # Keep only Jan-Dec

# Create complete dataframe with all dates
complete_df = pd.DataFrame({date_column: all_dates})

# Merge with your existing data
complete_df = complete_df.merge(combined_df, on=date_column, how='left')

# Add location columns
complete_df['latitude'] = STATION_LAT
complete_df['longitude'] = STATION_LON
complete_df['station_id'] = STATION_ID

# Keep only wind-related columns (ADD THIS)
wind_cols = ['Date', 'Wind speed  (vc avg)', 'Wind  direction  (vc avg)', 
             'Wind  speed (max)', 'Wind Dir.(Max wind speed)', 
             'latitude', 'longitude', 'station_id']
complete_df = complete_df[[col for col in wind_cols if col in complete_df.columns]]

# Convert dates back to original "01 May 2017" format
complete_df[date_column] = complete_df[date_column].dt.strftime('%d %b %Y')

# Save using semicolon delimiter to match original format
complete_df.to_csv("main_data/combined_weather_with_location.csv", sep=';', index=False)  # CHANGED PATH

print("Done! File saved as 'main_data/combined_weather_with_location.csv'")
print(f"Total dates in file: {len(complete_df)}")
print(f"Wind data present for: {complete_df['Wind speed  (vc avg)'].notna().sum()} dates")  # MODIFIED