"""
BATCH PLAYA EXTRACTION FOR ALL SENTINEL-2 IMAGES
Extracts pure salt flats with holes preserved as interior rings (donut shapes)
"""

import glob
import os

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
from skimage import measure, morphology

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CONFIGS = {
    "high_purity": {
        "si1_percentile": 97,
        "min_size_pixels": 100,
        "min_area_m2": 1000,
        "chaikin_iterations": 3,
    },
    "balanced": {
        "si1_percentile": 95,
        "min_size_pixels": 100,
        "min_area_m2": 1000,
        "chaikin_iterations": 2,
    },
    "inclusive": {
        "si1_percentile": 90,
        "min_size_pixels": 50,
        "min_area_m2": 200,
        "chaikin_iterations": 1,
    },
}

ACTIVE_CONFIG = "high_purity"  # Change to "balanced" or "inclusive" as needed


# ---------------------------------------------------------------------------
# 1. Binary mask
# ---------------------------------------------------------------------------
def extract_playa_mask(
    image_path: str,
    si1_percentile: float = 95,
    min_size_pixels: int = 100,
    closing_radius: int = 3,
    opening_radius: int = 2,
):
    """Return (binary_mask, transform, crs) for pure salt-flat pixels."""
    with rasterio.open(image_path) as src:
        blue  = src.read(1).astype(np.float32)
        red   = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs

    si1 = np.sqrt(blue * red)

    mask = (si1 > np.percentile(si1, si1_percentile))

    mask = morphology.remove_small_objects(mask, max_size=min_size_pixels)
    if closing_radius > 0:
        mask = morphology.closing(mask, morphology.disk(closing_radius))
    if opening_radius > 0:
        mask = morphology.opening(mask, morphology.disk(opening_radius))

    return mask, transform, crs


# ---------------------------------------------------------------------------
# 2. Smoothing — Chaikin corner-cutting
# ---------------------------------------------------------------------------
def chaikin_smooth(coords: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Chaikin corner-cutting: each iteration replaces every edge AB with
    two new points at 1/4 and 3/4 along the edge. Ring-aware (wraps).
    """
    if len(coords) < 4 or iterations == 0:
        return coords
    pts = np.asarray(coords, dtype=np.float64)
    # Drop duplicate closing vertex if present
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


def _anti_alias_contour(contour: np.ndarray, approx_tolerance: float = 0.5) -> np.ndarray:
    """Remove staircase artefacts from cv2 contours before smoothing."""
    epsilon = approx_tolerance
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approx


# ---------------------------------------------------------------------------
# 3. Contour → world-space coords
# ---------------------------------------------------------------------------
def _contour_to_world(contour: np.ndarray, transform) -> list:
    coords = []
    for pt in contour:
        col, row = pt[0]
        x = transform[2] + col * transform[0]
        y = transform[5] + row * transform[4]
        coords.append((x, y))
    return coords


# ---------------------------------------------------------------------------
# 4. Vectorize polygons with holes (donut shapes)
# ---------------------------------------------------------------------------
def vectorize_playa_donuts(
    mask: np.ndarray,
    transform,
    crs,
    min_area_m2: float = 500,
    simplify_tolerance_m: float = 5,
    chaikin_iterations: int = 2,
) -> list:
    """Vectorize playa polygons with holes subtracted (donut shapes)."""
    
    labeled, num_features = measure.label(mask, connectivity=2, return_num=True)
    polygons = []

    for i in range(1, num_features + 1):
        single = (labeled == i).astype(np.uint8)

        contours, hierarchy = cv2.findContours(
            single, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or hierarchy is None:
            continue

        hier = hierarchy[0]

        # Partition into exteriors and their children (holes)
        holes_by_parent: dict[int, list] = {}
        exterior_list: list[tuple[int, np.ndarray]] = []

        for idx, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            parent_idx = hier[idx][3]
            if parent_idx == -1:
                exterior_list.append((idx, contour))
            else:
                holes_by_parent.setdefault(parent_idx, []).append(contour)

        for ext_idx, ext_contour in exterior_list:
            # Process exterior
            ext_contour = _anti_alias_contour(ext_contour)
            ext_world   = _contour_to_world(ext_contour, transform)
            if len(ext_world) < 3:
                continue
            ext_smooth = chaikin_smooth(np.array(ext_world), chaikin_iterations)
            if not np.allclose(ext_smooth[0], ext_smooth[-1]):
                ext_smooth = np.vstack([ext_smooth, ext_smooth[0]])
            ext_coords = ext_smooth.tolist()

            # Collect holes as interior rings
            hole_coords_list = []
            for hole_c in holes_by_parent.get(ext_idx, []):
                hole_c      = _anti_alias_contour(hole_c)
                hole_world  = _contour_to_world(hole_c, transform)
                if len(hole_world) < 3:
                    continue

                hole_smooth = chaikin_smooth(np.array(hole_world), chaikin_iterations)
                if not np.allclose(hole_smooth[0], hole_smooth[-1]):
                    hole_smooth = np.vstack([hole_smooth, hole_smooth[0]])
                hole_coords_list.append(hole_smooth.tolist())

            try:
                # Create donut polygon (exterior with holes subtracted)
                poly = Polygon(ext_coords, hole_coords_list)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue

                poly = poly.simplify(simplify_tolerance_m, preserve_topology=True)

                # Handle MultiPolygon results
                if poly.geom_type == "MultiPolygon":
                    for part in poly.geoms:
                        if part.area >= min_area_m2:
                            polygons.append(part)
                elif poly.area >= min_area_m2:
                    polygons.append(poly)

            except Exception as exc:
                print(f"    Warning: skipping polygon — {exc}")

    return polygons


# ---------------------------------------------------------------------------
# 5. Export donut polygons (holes subtracted, only net area)
# ---------------------------------------------------------------------------
def export_playa_donuts(polygons: list, output_path: str, crs, acquisition_date: str) -> bool:
    """Export playa donut polygons (holes subtracted)."""
    if not polygons:
        print("  ✗ No polygons to save")
        return False

    rows = []
    for idx, poly in enumerate(polygons):
        # poly.area is already the net area (donut area with holes subtracted)
        rows.append({
            "id":          idx,
            "area_m2":     poly.area,
            "area_ha":     poly.area / 10_000,
            "perimeter_m": poly.length,
            "compactness": (4 * np.pi * poly.area / poly.length ** 2)
                           if poly.length > 0 else 1.0,
            "acquisition_date": acquisition_date,
            "geometry":    poly,
        })

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.to_file(output_path, driver="GeoJSON")

    total_area = gdf["area_ha"].sum()
    print(f"  ✓ {len(polygons)} playa donuts (holes subtracted) → {output_path}")
    print(f"    Total net area: {total_area:.2f} ha")
    return True


# ---------------------------------------------------------------------------
# 6. Main extraction function (donut shapes with holes subtracted)
# ---------------------------------------------------------------------------
def extract_playa_donuts(
    image_path: str,
    output_path: str,
    si1_percentile: float = 95,
    min_size_pixels: int = 100,
    min_area_m2: float = 500,
    simplify_tolerance_m: float = 2,
    chaikin_iterations: int = 2,
) -> bool:
    """Extract playa donut polygons with holes subtracted."""
    try:
        print("  Step 1: binary mask…")
        mask, transform, crs = extract_playa_mask(
            image_path,
            si1_percentile=si1_percentile,
            min_size_pixels=min_size_pixels,
        )
        n_pixels = mask.sum()
        if n_pixels == 0:
            print("  ✗ No playa pixels detected")
            return False
        print(f"  ✓ {n_pixels} playa pixels")

        print("  Step 2: creating donut polygons (holes subtracted)…")
        polygons = vectorize_playa_donuts(
            mask, transform, crs,
            min_area_m2=min_area_m2,
            simplify_tolerance_m=simplify_tolerance_m,
            chaikin_iterations=chaikin_iterations,
        )
        
        if not polygons:
            print("  ✗ No polygons after filtering")
            return False

        print("  Step 3: exporting donut polygons…")
        # Extract acquisition date from filename
        import os
        basename = os.path.basename(image_path)
        acquisition_date = basename.replace(".tif", "").split('_')[-1]
        return export_playa_donuts(polygons, output_path, crs, acquisition_date)

    except Exception as exc:
        import traceback
        print(f"  ✗ Error: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = CONFIGS[ACTIVE_CONFIG]
    image_files = sorted(glob.glob("tif/sossusvlei_*.tif"))

    os.makedirs("playa_donuts", exist_ok=True)

    print("=" * 70)
    print("PLAYA DONUT EXTRACTION (HOLES SUBTRACTED)")
    print("=" * 70)
    print(f"  Config            : {ACTIVE_CONFIG}")
    print(f"  SI-1 percentile   : {config['si1_percentile']}")
    print(f"  Min polygon area  : {config['min_area_m2']} m²")
    print(f"  Chaikin iterations: {config['chaikin_iterations']}")
    print(f"  Output            : DONUT shapes (holes subtracted)")
    print(f"  Images found      : {len(image_files)}")
    print("=" * 70)

    ok_count = fail_count = 0

    for idx, path in enumerate(image_files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        tag  = "_".join(stem.split("_")[1:])
        out  = f"playa_donuts/playa_donut_{ACTIVE_CONFIG}_{tag}.geojson"

        print(f"\n[{idx}/{len(image_files)}] {os.path.basename(path)}")
        if extract_playa_donuts(
            path, out,
            si1_percentile=config["si1_percentile"],
            min_size_pixels=config["min_size_pixels"],
            min_area_m2=config["min_area_m2"],
            simplify_tolerance_m=2,
            chaikin_iterations=config["chaikin_iterations"],
        ):
            ok_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print(f"Done — {ok_count} succeeded, {fail_count} failed")
    print(f"Output  : playa_donuts/playa_donut_{ACTIVE_CONFIG}_*.geojson")
    print("=" * 70)