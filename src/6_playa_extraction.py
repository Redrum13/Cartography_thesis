"""
BATCH PLAYA EXTRACTION FOR ALL SENTINEL-2 IMAGES
Extracts pure salt flats and dissolves by date into single GeoJSON
"""

import glob
import os
import geopandas as gpd
import numpy as np
import rasterio
import pandas as pd
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage import measure, morphology
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_TIF_FOLDER = "data/raw/tif"
INPUT_TIF_GLOB = os.path.join(INPUT_TIF_FOLDER, "sossusvlei_*.tif")

SI1_PERCENTILE = 97
MIN_SIZE_PIXELS = 100
MIN_AREA_M2 = 1000
CHAIKIN_ITERATIONS = 1
SIMPLIFY_TOLERANCE_M = 1

OUTPUT_FOLDER = "data/processed"
OUTPUT_DISSOLVED_PLAYA = os.path.join(OUTPUT_FOLDER, "merged_playa.geojson")


# ---------------------------------------------------------------------------
def extract_playa_mask(image_path: str):
    with rasterio.open(image_path) as src:
        blue = src.read(1).astype(np.float32)
        red = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs

    si1 = np.sqrt(blue * red)
    mask = (si1 > np.percentile(si1, SI1_PERCENTILE))
    mask = morphology.remove_small_objects(mask, max_size=MIN_SIZE_PIXELS)
    mask = morphology.closing(mask, morphology.disk(3))
    mask = morphology.opening(mask, morphology.disk(2))
    return mask, transform, crs


# ---------------------------------------------------------------------------
def chaikin_smooth(coords, iterations=CHAIKIN_ITERATIONS):
    if len(coords) < 4 or iterations == 0:
        return coords
    pts = np.asarray(coords, dtype=np.float64)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
        reclose = True
    else:
        reclose = False
    for _ in range(iterations):
        n = len(pts)
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, axis=0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, axis=0)
        pts = np.empty((2 * n, 2), dtype=np.float64)
        pts[0::2] = q
        pts[1::2] = r
    if reclose:
        pts = np.vstack([pts, pts[0]])
    return pts


# ---------------------------------------------------------------------------
def vectorize_playa(mask, transform, crs):
    labeled, num_features = measure.label(mask, connectivity=2, return_num=True)
    polygons = []

    for i in range(1, num_features + 1):
        single = (labeled == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(single, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            continue

        hier = hierarchy[0]
        holes_by_parent = {}
        exterior_list = []

        for idx, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            parent_idx = hier[idx][3]
            if parent_idx == -1:
                exterior_list.append((idx, contour))
            else:
                holes_by_parent.setdefault(parent_idx, []).append(contour)

        for ext_idx, ext_contour in exterior_list:
            ext_world = []
            for pt in ext_contour:
                col, row = pt[0]
                x = transform[2] + col * transform[0]
                y = transform[5] + row * transform[4]
                ext_world.append((x, y))
            
            if len(ext_world) < 3:
                continue
                
            ext_smooth = chaikin_smooth(np.array(ext_world))
            if not np.allclose(ext_smooth[0], ext_smooth[-1]):
                ext_smooth = np.vstack([ext_smooth, ext_smooth[0]])
            ext_coords = ext_smooth.tolist()

            hole_coords_list = []
            for hole_c in holes_by_parent.get(ext_idx, []):
                hole_world = []
                for pt in hole_c:
                    col, row = pt[0]
                    x = transform[2] + col * transform[0]
                    y = transform[5] + row * transform[4]
                    hole_world.append((x, y))
                if len(hole_world) < 3:
                    continue
                hole_smooth = chaikin_smooth(np.array(hole_world))
                if not np.allclose(hole_smooth[0], hole_smooth[-1]):
                    hole_smooth = np.vstack([hole_smooth, hole_smooth[0]])
                hole_coords_list.append(hole_smooth.tolist())

            try:
                poly = Polygon(ext_coords, hole_coords_list)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue
                poly = poly.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
                if poly.geom_type == "Polygon" and poly.area >= MIN_AREA_M2:
                    polygons.append(poly)
                elif poly.geom_type == "MultiPolygon":
                    for part in poly.geoms:
                        if part.area >= MIN_AREA_M2:
                            polygons.append(part)
            except:
                continue

    return polygons


# ---------------------------------------------------------------------------
def extract_playa_gdf(image_path):
    try:
        mask, transform, crs = extract_playa_mask(image_path)
        if mask.sum() == 0:
            return None

        polygons = vectorize_playa(mask, transform, crs)
        if not polygons:
            return None

        basename = os.path.basename(image_path)
        acquisition_date = basename.replace(".tif", "").split('_')[-1]
        
        rows = []
        for idx, poly in enumerate(polygons):
            rows.append({
                "acquisition_date": acquisition_date,
                "geometry": poly,
            })

        return gpd.GeoDataFrame(rows, crs=crs)
    except:
        return None


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    image_files = sorted(glob.glob(INPUT_TIF_GLOB))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 70)
    print("PLAYA EXTRACTION - DISSOLVED BY DATE")
    print("=" * 70)
    print(f"  Images found: {len(image_files)}")
    print("=" * 70)

    all_gdfs = []

    for idx, path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {os.path.basename(path)}")
        gdf = extract_playa_gdf(path)
        if gdf is not None and not gdf.empty:
            all_gdfs.append(gdf)
            print(f"  Found {len(gdf)} polygons")

    if all_gdfs:
        print("\nMerging and dissolving by date...")
        merged = pd.concat(all_gdfs, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=merged.crs)
        
        # Dissolve by date
        dissolved = merged.dissolve(by='acquisition_date', aggfunc='first')
        dissolved = dissolved.reset_index()
        
        # Save
        dissolved.to_file(OUTPUT_DISSOLVED_PLAYA, driver='GeoJSON')
        
        print(f"\nSaved: {OUTPUT_DISSOLVED_PLAYA}")
        print(f"  Dates: {len(dissolved)}")
        print(f"  Total area: {dissolved.geometry.area.sum() / 10_000:.2f} ha")
        
        # Summary
        print("\nPer-Date Summary:")
        for _, row in dissolved.iterrows():
            area_ha = row.geometry.area / 10_000
            print(f"  {row['acquisition_date']}: {area_ha:.2f} ha")
    else:
        print("\nNo playa polygons extracted!")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)