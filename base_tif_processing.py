"""
PRE-PROCESSING SCRIPT: Extract single band from GeoTIFFs for Folium overlay
Input: tif\sossusvlei_YYYY_MM_YYYY-MM-DD.tif
Output: Base_tif\sossusvlei_YYYY_MM.png + Base_tif\metadata.json
"""

import os
import json
import numpy as np
from PIL import Image
import rasterio
from pyproj import Transformer
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FOLDER = "tif"          # Folder containing original GeoTIFFs
OUTPUT_FOLDER = "Base_tif"    # Folder for processed PNGs and metadata
BAND_INDEX = 4                # Which band to extract (1-indexed)

# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_band_and_bounds(tif_path, band_idx=1):
    """
    Extract a single band from a GeoTIFF and get its geographic bounds.
    
    Args:
        tif_path: Path to the GeoTIFF file
        band_idx: Which band to extract (1-indexed)
    
    Returns:
        tuple: (normalized_8bit_array, bounds_dict, date_info)
    """
    with rasterio.open(tif_path) as src:
        # Read the specified band
        band_data = src.read(band_idx).astype(np.float32)
        
        # Get NoData value (if it exists)
        nodata = src.nodata
        if nodata is not None:
            # Mask out NoData values
            band_data = np.where(band_data == nodata, np.nan, band_data)
        
        # Get geographic bounds (left, bottom, right, top)
        # Transform from CRS to WGS84 (lat/lon) if needed
        transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
        left, bottom = transformer.transform(src.bounds.left, src.bounds.bottom)
        right, top = transformer.transform(src.bounds.right, src.bounds.top)
        bounds = {
                "left": left,
                "right": right,
                "bottom": bottom,
                "top": top
        }
        
        # Extract date from filename or from the file's metadata
        # Format: sossusvlei_2017_05_2017-05-20.tif
        filename = os.path.basename(tif_path)
        parts = filename.replace('.tif', '').split('_')
        
        # Expecting: sossusvlei_YYYY_MM_YYYY-MM-DD
        if len(parts) >= 4:
            year = int(parts[1])
            month = int(parts[2])
            # The full date is parts[3] (YYYY-MM-DD)
            date_full = parts[3]
        else:
            # Fallback: try to extract from metadata
            year = src.tags().get('year', 2020)
            month = src.tags().get('month', 1)
            date_full = f"{year}-{month:02d}-01"
        
        # Normalize band to 0-255 (for PNG)
        # Handle NaN values
        valid_data = band_data[~np.isnan(band_data)]
        if len(valid_data) == 0:
            # All values are NoData - create a blank image
            normalized = np.zeros(band_data.shape, dtype=np.uint8)
        else:
            # Min-max stretch (ignore NoData/NaN)
            data_min = np.nanmin(band_data)
            data_max = np.nanmax(band_data)
            
            if data_max - data_min > 0:
                normalized = (band_data - data_min) / (data_max - data_min) * 255
            else:
                normalized = np.zeros(band_data.shape)
            
            # Replace NaN with 0 (transparent in PNG)
            normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)
        
        return normalized, bounds, {
            "year": year,
            "month": month,
            "date_full": date_full,
            "filename": filename
        }

def save_as_png(array, output_path):
    """
    Save a 2D array as a grayscale PNG.
    
    Args:
        array: 2D numpy array (values 0-255)
        output_path: Path to save the PNG
    """
    # Create PIL image (mode='L' for grayscale)
    img = Image.fromarray(array, mode='L')
    img.save(output_path, format='PNG', compress_level=6)

def main():
    """
    Main processing function.
    """
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all GeoTIFF files
    tif_files = sorted(Path(INPUT_FOLDER).glob("*.tif"))
    
    if not tif_files:
        print(f"❌ No .tif files found in '{INPUT_FOLDER}'")
        return
    
    print(f"Found {len(tif_files)} GeoTIFF files to process")
    
    # Store metadata for all files
    all_metadata = {}
    
    # Process each file
    for tif_path in tif_files:
        try:
            print(f"Processing: {tif_path.name}")
            
            # Extract band and metadata
            band_array, bounds, date_info = extract_band_and_bounds(tif_path, BAND_INDEX)
            
            # Generate output filename: sossusvlei_YYYY_MM.png
            output_name = f"sossusvlei_{date_info['year']}_{date_info['month']:02d}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            
            # Save as PNG
            save_as_png(band_array, output_path)
            
            # Store metadata
            metadata_key = f"{date_info['year']}_{date_info['month']:02d}"
            all_metadata[metadata_key] = {
                "year": date_info["year"],
                "month": date_info["month"],
                "date_full": date_info["date_full"],
                "png_path": output_name,
                "bounds": bounds,
                "width": band_array.shape[1],
                "height": band_array.shape[0],
                "source_file": date_info["filename"]
            }
            
            print(f" Saved: {output_name} ({band_array.shape[1]}x{band_array.shape[0]} pixels)")
            
        except Exception as e:
            print(f" Error processing {tif_path.name}: {e}")
    
    # Save metadata as JSON
    metadata_path = os.path.join(OUTPUT_FOLDER, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n Processing complete!")
    print(f" PNGs saved in: '{OUTPUT_FOLDER}'")
    print(f" Metadata saved: '{metadata_path}'")
    print(f" Total processed: {len(all_metadata)} files")

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()