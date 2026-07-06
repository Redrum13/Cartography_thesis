================================================================================
STAR DUNE DYNAMICS DASHBOARD - README
================================================================================

PROJECT OVERVIEW
--------------------------------------------------------------------------------
This dashboard supports an MSc Cartography thesis on aeolian star dune dynamics
in the Namib Desert (Sossusvlei area, ~2017-2026). It is an interactive
Streamlit application that visualizes, on a single monthly-navigable map:

  - Dune crest lines (extracted from Sentinel-2 imagery)
  - Crest movement / centerline points across a monthly time series
  - Playa (salt flat) polygons
  - Wind rose and wind data completeness (Gantt chart)
  - GNSS-based positional error assessment of the crest line extraction method

Three dunes are covered: the Star Dune (which has GNSS ground-truth validation
of extraction error), Big Mommy Dune, and Inverted Y Dune (manually
digitized in QGIS; error assessment visualization was dropped for these two due to
scope).

The error assessment framework itself is built from a single-epoch GNSS validation
(March 2026) that is extended as a representative assumption across the full
monthly time series, under the stated assumption of consistent sensor,
preprocessing, and extraction methodology. Error distributions were found to
be right-skewed (mean > median, with outliers), so a normal-distribution
assumption should be treated with caution in any written methodology.

PIPELINE AT A GLANCE
--------------------------------------------------------------------------------
  1. python.py                 -> downloads monthly Sentinel-2 imagery (GEE)
  2. base_tif_processing.py    -> converts GeoTIFFs to PNG overlays + metadata
  3. crest_extraction.py       -> extracts raw crest lines per image (Canny +
                                   skeleton/graph centerline vectorization)
  4. crest_post_process.py     -> dissolves, merges dates, gap-fills, and
                                   builds centerline/movement points
  5. playa_extraction.py       -> extracts playa polygons per image
  6. playa_post_process.py     -> merges + dissolves playa polygons by date
  7. wind_data_pre.py           -> cleans/combines station wind CSVs
  8. error assessment_with_gap_csv.py -> compares GNSS reference lines to detected
                                   crest lines, outputs error assessment vectors
                                   and summary statistics
  9. dune_dash.py              -> Streamlit app that reads all of the above
                                   outputs and renders the interactive map

DATA INPUT SPECIFICATION
--------------------------------------------------------------------------------
The dashboard (dune_dash.py) reads the following files. All are expected
relative to the app's working directory unless noted otherwise.

--------------------------------------------------------------------------------
1) Base imagery metadata
   Path:     Base_tif/metadata.json
   Produced by: base_tif_processing.py
   Format:   JSON, keyed by "YYYY_MM"
   Required fields per entry:
     - png_path     (string, filename of PNG inside Base_tif/)
     - bounds       (dict: left, right, bottom, top - EPSG:4326)
     - date_full    (string, "YYYY-MM-DD", optional)
   Notes: if missing, the app falls back to no base imagery layer rather
   than failing.

--------------------------------------------------------------------------------
2) Crest lines (extended centerlines)
   Path:     main_data/extended_centerlines.geojson
   Produced by: crest_post_process.py
   Required columns:
     - acquisition_date (string, parseable as a date)
     - type             (string: "detected" or "connection")
     - geometry         (LineString / MultiLineString)
   Optional columns (auto-derived if absent):
     - is_gap_fill  (defaults to type == "connection")
     - length_m     (defaults to geometry length in degrees x 111,320)
   CRS: any (re-projected to EPSG:4326 on load)

--------------------------------------------------------------------------------
3) Movement / centerline points
   Path:     main_data/centerline_points.geojson
   Produced by: crest_post_process.py
   Required columns:
     - point_id           (unique per sampling point along a dune)
     - distance_along_m   (float, position along manual reference line)
     - date_YYYY_MM_DD    (one column per acquisition date, signed distance
                            in meters from the reference centerline; NaN/None
                            where unmeasured)
     - geometry           (Point)
   Notes: the app reshapes this from wide (one column per date) to long
   format internally. At least one date_* column must be present.

--------------------------------------------------------------------------------
4) Playa polygons
   Path:     main_data/merged_playa.geojson
   Produced by: playa_post_process.py (from playa_extraction.py outputs)
   Required columns:
     - acquisition_date (string, parseable as a date)
     - geometry         (Polygon / MultiPolygon)
   Optional columns (auto-derived if absent):
     - area_m2  (defaults to geometry area in degrees^2 x 111,320^2)
   CRS: any (re-projected to EPSG:4326 on load)

--------------------------------------------------------------------------------
5) Wind data
   Path:     main_data/combined_weather_with_location.csv
   Produced by: wind_data_pre.py
   Delimiter: semicolon (;)
   Required columns (station export names, note double spaces):
     - Date                          ("DD Mon YYYY", e.g. "01 May 2017")
     - Wind speed  (vc avg)          (float, m/s)
     - Wind  direction  (vc avg)     (float, degrees)
   Notes: the app renames these to datetime / speed_ms / direction on load.
   If the station export changes column spacing/naming, these columns will
   silently be dropped rather than raise an error - verify column names
   after any re-export from the station software.

--------------------------------------------------------------------------------
6) GNSS error assessment vectors
   Path:     main_data/error assessment_lines_length.geojson
   Produced by: error assessment_with_gap_csv.py
   Required columns:
     - geometry (LineString for measured points, Point for unmeasured)
     - is_measured (bool)
   Optional columns (renamed/used if present):
     - abs_error_m       -> renamed to error_m
     - signed_distance_m -> copied to gnss_val
     - left_or_right     -> copied to detected_val
   CRS: any (re-projected to EPSG:4326 on load)
   Companion file: main_data/error assessment_statistics.csv (ME, SD, RMSE, 95%
   margin of error - overall and per GNSS reference line; not read directly
   by the dashboard but used for methodology reporting).

--------------------------------------------------------------------------------
REFERENCE / MANUAL DIGITIZATION FILES (upstream inputs, not read by the app)
--------------------------------------------------------------------------------
  - star_dune_crsts.geojson
      Manual GNSS/QGIS reference centerlines, one line per dune, with an
      "id" (and optionally "dune_id", "name") field used to match detected
      crests to the correct dune during post-processing and error assessment
      comparison.

  - playa_donuts/ (folder of per-date GeoJSON files)
      Raw output of playa_extraction.py before merging.

  - crests/ (folder of per-date GeoJSON files, "crests_*.geojson")
      Raw output of crest_extraction.py before post-processing.

  - tif/ (folder of GeoTIFFs, "sossusvlei_YYYY_MM_YYYY-MM-DD.tif")
      Raw Sentinel-2 downloads from python.py, bands B2/B3/B4/B8/B11/B12
      at 10m resolution.

  - weather_station_data_csv/ (folder of monthly station CSVs)
      Raw wind station exports, semicolon-delimited, combined by
      wind_data_pre.py.

--------------------------------------------------------------------------------
KNOWN LIMITATIONS / ASSUMPTIONS
--------------------------------------------------------------------------------
  - Error visualization is only available for the Star Dune; Big Mommy
    and Inverted Y are QGIS-digitized only.
  - GNSS validation is single-epoch (March 2026) and extended across the
    full time series as a representative estimate, assuming consistent
    sensor, preprocessing, and extraction methodology over time.
  - Error distributions are right-skewed; treat any normal-distribution-based
    statistic (e.g. simple +/- SD ranges) with caution - the framework instead
    reports ME, SD, RMSE, and a 95% margin of error.
  - Wind rose percentages are normalized frequencies (normed=True), not raw
    day counts - a max value of ~10 is expected behavior, not a data error.
================================================================================