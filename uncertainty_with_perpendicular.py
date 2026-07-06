import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import nearest_points
import pandas as pd

# ---------------------------------------------------------------------------
# Calculate uncertainty between multiple GNSS reference lines and detected lines
# ---------------------------------------------------------------------------
def calculate_multiline_uncertainty(gnss_lines_gdf, detected_lines_gdf, 
                                   sampling_interval=5.0, perp_length=50.0, 
                                   line_id_field='line_id'):
    """
    Calculate uncertainty between multiple GNSS reference lines and detected edge lines.
    Returns perpendicular lines at sample locations with uncertainty values.
    """
    
    # Ensure CRS consistency
    if gnss_lines_gdf.crs != detected_lines_gdf.crs:
        print(f"  CRS mismatch: {gnss_lines_gdf.crs} vs {detected_lines_gdf.crs}")
        print("  Reprojecting detected lines to match GNSS reference...")
        detected_lines_gdf = detected_lines_gdf.to_crs(gnss_lines_gdf.crs)
    
    # Get CRS from GNSS data
    crs = gnss_lines_gdf.crs
    
    # Store results for each GNSS line
    results_by_line = {}
    all_lines_list = []
    
    # Process each GNSS line
    for idx, gnss_row in gnss_lines_gdf.iterrows():
        # Get line identifier
        if line_id_field in gnss_row:
            line_id = gnss_row[line_id_field]
        else:
            line_id = f"line_{idx}"
        
        print(f"\n{'='*60}")
        print(f"Processing GNSS line: {line_id}")
        print(f"{'='*60}")
        
        # Create GeoDataFrame for this single GNSS line
        gnss_single_gdf = gpd.GeoDataFrame(
            [gnss_row], 
            geometry=[gnss_row.geometry], 
            crs=crs
        )
        
        # Calculate uncertainty for this line
        line_results = calculate_line_uncertainty(
            gnss_line_gdf=gnss_single_gdf,
            detected_lines_gdf=detected_lines_gdf,
            sampling_interval=sampling_interval,
            perp_length=perp_length,
            crs=crs
        )
        
        # Add line identifier to lines
        if line_results['perpendicular_lines'] is not None and len(line_results['perpendicular_lines']) > 0:
            line_results['perpendicular_lines']['gnss_line_id'] = line_id
            all_lines_list.append(line_results['perpendicular_lines'])
        
        # Store results
        results_by_line[line_id] = line_results
    
    # Combine all perpendicular lines
    combined_lines = pd.concat(all_lines_list, ignore_index=True) if all_lines_list else gpd.GeoDataFrame()
    
    # Calculate and print overall summary statistics
    if len(combined_lines) > 0:
        valid_uncertainties = combined_lines['uncertainty_m'].dropna()
        
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total GNSS lines processed: {len(results_by_line)}")
        print(f"Total sample lines: {len(combined_lines)}")
        print(f"Valid measurements: {len(valid_uncertainties)}")
        print(f"Measurement rate: {len(valid_uncertainties)/len(combined_lines)*100:.1f}%")
        print(f"\nUncertainty Statistics:")
        print(f"  Mean: {valid_uncertainties.mean():.2f}m")
        print(f"  Median: {valid_uncertainties.median():.2f}m")
        print(f"  Std Dev: {valid_uncertainties.std():.2f}m")
        print(f"  Min: {valid_uncertainties.min():.2f}m")
        print(f"  Max: {valid_uncertainties.max():.2f}m")
        print(f"{'='*60}")
        
        # Per-line summary
        print(f"\nPER-LINE SUMMARY")
        print(f"{'='*60}")
        for line_id, line_res in results_by_line.items():
            stats = line_res['summary_stats']
            print(f"\n  {line_id}:")
            print(f"    Mean uncertainty: {stats['mean_uncertainty_m']:.2f}m")
            print(f"    Valid samples: {stats['valid_samples']}/{stats['total_samples']}")
            print(f"    Left/Right: {stats['left_measurements']}/{stats['right_measurements']}")
        print(f"{'='*60}")
    else:
        print("No valid measurements found!")
    
    return {
        'all_perpendicular_lines': combined_lines,
        'by_line': results_by_line
    }


# ---------------------------------------------------------------------------
# Calculate uncertainty for a single GNSS line
# ---------------------------------------------------------------------------
def calculate_line_uncertainty(gnss_line_gdf, detected_lines_gdf, sampling_interval=5.0, 
                               perp_length=50.0, crs=None):
    """
    Calculate uncertainty between a single GNSS reference line and detected edge lines.
    Returns perpendicular lines at sample locations with uncertainty values.
    """
    
    # Extract GNSS reference line
    gnss_geom = gnss_line_gdf.geometry.iloc[0]
    if gnss_geom.geom_type == 'MultiLineString':
        gnss_line = max(gnss_geom.geoms, key=lambda x: x.length)
    else:
        gnss_line = gnss_geom
    
    print(f"  GNSS reference line length: {gnss_line.length:.1f}m")
    print(f"  Detected lines count: {len(detected_lines_gdf)}")
    
    # Extract all detected lines (flatten MultiLineStrings)
    detected_segments = []
    for idx, row in detected_lines_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                detected_segments.append(line)
        elif geom.geom_type == 'LineString':
            detected_segments.append(geom)
    
    # Sample points along GNSS reference line
    sample_points = []
    distances_along = []
    total_length = gnss_line.length
    current_dist = 0
    
    while current_dist <= total_length:
        point = gnss_line.interpolate(current_dist)
        sample_points.append(point)
        distances_along.append(current_dist)
        current_dist += sampling_interval
    
    # Also add the last point if not already included
    if distances_along[-1] < total_length:
        point = gnss_line.interpolate(total_length)
        sample_points.append(point)
        distances_along.append(total_length)
    
    print(f"  Sample points along GNSS line: {len(sample_points)}")
    
    # Calculate uncertainty at each sample point
    results = []
    perpendicular_lines = []
    
    for i, (dist_along, ref_point) in enumerate(zip(distances_along, sample_points)):
        # Get direction at this point (tangent)
        offset = min(5.0, total_length * 0.02)
        dist_before = max(0, dist_along - offset)
        dist_after = min(total_length, dist_along + offset)
        
        point_before = gnss_line.interpolate(dist_before)
        point_after = gnss_line.interpolate(dist_after)
        
        # Calculate tangent direction
        dx = point_after.x - point_before.x
        dy = point_after.y - point_before.y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            if i > 0:
                # Use previous result's direction
                prev_perp_dx = results[-1].get('perp_dx', 1.0)
                prev_perp_dy = results[-1].get('perp_dy', 0.0)
                perp_dx, perp_dy = prev_perp_dx, prev_perp_dy
            else:
                perp_dx, perp_dy = 1.0, 0.0
        else:
            # Normalize tangent
            dx /= length
            dy /= length
            # Perpendicular direction (rotate 90 degrees)
            perp_dx = -dy
            perp_dy = dx
        
        # Create perpendicular line
        perp_start = Point(ref_point.x - perp_dx * perp_length, 
                          ref_point.y - perp_dy * perp_length)
        perp_end = Point(ref_point.x + perp_dx * perp_length, 
                        ref_point.y + perp_dy * perp_length)
        perpendicular = LineString([perp_start, perp_end])
        
        # Find closest detected line point
        closest_distance = None
        signed_dist = None
        closest_point = None
        
        for segment in detected_segments:
            # Check intersection with perpendicular
            intersection = perpendicular.intersection(segment)
            
            if not intersection.is_empty:
                if intersection.geom_type == 'MultiPoint':
                    points_to_check = intersection.geoms
                elif intersection.geom_type == 'Point':
                    points_to_check = [intersection]
                elif intersection.geom_type == 'LineString':
                    points_to_check = [intersection.interpolate(0.5, normalized=True)]
                else:
                    continue
                
                for intersect_point in points_to_check:
                    vec_x = intersect_point.x - ref_point.x
                    vec_y = intersect_point.y - ref_point.y
                    s_dist = vec_x * perp_dx + vec_y * perp_dy
                    abs_dist = abs(s_dist)
                    
                    if closest_distance is None or abs_dist < closest_distance:
                        closest_distance = abs_dist
                        signed_dist = s_dist
                        closest_point = intersect_point
            
            # Also check nearest point on segment (for gaps)
            nearest = nearest_points(ref_point, segment)[1]
            proj_dist = perpendicular.distance(nearest)
            
            if proj_dist < 1.0:  # Within 1 meter of perpendicular
                vec_x = nearest.x - ref_point.x
                vec_y = nearest.y - ref_point.y
                s_dist = vec_x * perp_dx + vec_y * perp_dy
                abs_dist = abs(s_dist)
                
                if closest_distance is None or abs_dist < closest_distance:
                    closest_distance = abs_dist
                    signed_dist = s_dist
                    closest_point = nearest
        
        # Determine left or right for closest point
        if signed_dist is not None:
            left_or_right = 'right' if signed_dist > 0 else 'left'
        else:
            left_or_right = None
        
        # Store result
        result = {
            'distance_along': dist_along,
            'sample_point': ref_point,
            'uncertainty_m': closest_distance if closest_distance is not None else np.nan,
            'left_or_right': left_or_right,
            'signed_distance': signed_dist if closest_distance is not None else np.nan,
            'is_measured': closest_distance is not None,
            'perp_dx': perp_dx,
            'perp_dy': perp_dy
        }
        results.append(result)
        
        # Create perpendicular line data for GeoDataFrame
        line_data = {
            'distance_along_m': dist_along,
            'uncertainty_m': closest_distance if closest_distance is not None else np.nan,
            'signed_distance_m': signed_dist if closest_distance is not None else np.nan,
            'left_or_right': left_or_right,
            'is_measured': closest_distance is not None,
            'gnss_point_x': ref_point.x,
            'gnss_point_y': ref_point.y,
            'intersection_x': closest_point.x if closest_point is not None else np.nan,
            'intersection_y': closest_point.y if closest_point is not None else np.nan
        }
        perpendicular_lines.append({'geometry': perpendicular, 'data': line_data})
    
    # Create GeoDataFrame of perpendicular lines
    lines_geometry = []
    lines_data = []
    
    for pl in perpendicular_lines:
        lines_geometry.append(pl['geometry'])
        lines_data.append(pl['data'])
    
    lines_gdf = gpd.GeoDataFrame(lines_data, geometry=lines_geometry, crs=crs)
    
    # Summary statistics (only for measured points)
    valid_uncertainties = lines_gdf[lines_gdf['is_measured']]['uncertainty_m'].dropna()
    summary_stats = {
        'mean_uncertainty_m': valid_uncertainties.mean() if len(valid_uncertainties) > 0 else np.nan,
        'median_uncertainty_m': valid_uncertainties.median() if len(valid_uncertainties) > 0 else np.nan,
        'std_uncertainty_m': valid_uncertainties.std() if len(valid_uncertainties) > 0 else np.nan,
        'max_uncertainty_m': valid_uncertainties.max() if len(valid_uncertainties) > 0 else np.nan,
        'min_uncertainty_m': valid_uncertainties.min() if len(valid_uncertainties) > 0 else np.nan,
        'percent_measured': (len(valid_uncertainties) / len(results)) * 100 if len(results) > 0 else 0,
        'total_samples': len(results),
        'valid_samples': len(valid_uncertainties),
        'left_measurements': len(lines_gdf[lines_gdf['left_or_right'] == 'left']) if len(lines_gdf) > 0 else 0,
        'right_measurements': len(lines_gdf[lines_gdf['left_or_right'] == 'right']) if len(lines_gdf) > 0 else 0
    }
    
    print(f"\n  Uncertainty Summary for this line:")
    print(f"    Mean: {summary_stats['mean_uncertainty_m']:.2f}m")
    print(f"    Valid measurements: {summary_stats['valid_samples']}/{summary_stats['total_samples']}")
    
    return {
        'perpendicular_lines': lines_gdf,
        'summary_stats': summary_stats,
        'raw_results': results
    }


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    
    # Load your data
    gnss_file = "star_dune_crsts.geojson" 
    detected_file = "crests/crests_2026_03_2026-03-21.geojson"
    
    # Read GeoJSON files
    gnss_gdf = gpd.read_file(gnss_file)
    detected_gdf = gpd.read_file(detected_file)
    
    print(f"Loaded {len(gnss_gdf)} GNSS reference lines")
    print(f"Loaded {len(detected_gdf)} detected line features")
    print(f"GNSS CRS: {gnss_gdf.crs}")
    
    # Calculate uncertainty
    uncertainty_results = calculate_multiline_uncertainty(
        gnss_lines_gdf=gnss_gdf,
        detected_lines_gdf=detected_gdf,
        sampling_interval=10.0,
        perp_length=50.0,
        line_id_field='name'
    )
    
    # Create data folder if it doesn't exist
    os.makedirs("main_data", exist_ok=True)
    
    # Save perpendicular lines to GeoJSON
    output_file = "main_data/multiline_uncertainty_perpendicular_lines.geojson"
    uncertainty_results['all_perpendicular_lines'].to_file(output_file, driver='GeoJSON')
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")