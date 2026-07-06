import os
import glob
import pandas as pd
import geopandas as gpd

# Create main_data folder
output_folder = 'main_data'
os.makedirs(output_folder, exist_ok=True)

# Folder containing your playa files
playa_folder = r"playa_donuts"

# Load all playa files
filepaths = glob.glob(os.path.join(playa_folder, "*.geojson"))
all_gdfs = []

for filepath in sorted(filepaths):
    filename = os.path.basename(filepath)
    date_str = filename.replace(".geojson", "").split('_')[-1]  # adjust this if needed
    
    gdf = gpd.read_file(filepath)
    gdf['acquisition_date'] = date_str
    all_gdfs.append(gdf)

# Merge all files together
merged = pd.concat(all_gdfs, ignore_index=True)
merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=merged.crs)

# Export merged playa geojson
merged.to_file(os.path.join(output_folder, "merged_playa.geojson"), driver='GeoJSON')

# Dissolve by acquisition date
dissolved = merged.dissolve(by='acquisition_date', as_index=False)

# Export dissolved by date
dissolved.to_file(os.path.join(output_folder, "playa_by_date.geojson"), driver='GeoJSON')

print("Done! Exported: merged_playa.geojson and playa_by_date.geojson")