"""
============================================================
 DUNE DASHBOARD  –  Streamlit app  (4:1 column layout)
 Aeolian dune monitoring: crest lines, movement, playas,
 wind roses, and GNSS uncertainty (2017-2026, May-August)
============================================================
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, base64
from shapely.geometry import Point
from folium import plugins
from windrose import WindroseAxes
# ----------------------------Need to add to streamlit--------------------------
import os
import json
from PIL import Image

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Star Dune Dynamics Dashboard",
    page_icon=None,
    layout="wide",
)

# ------------------------------------------------------------------------------
# CUSTOM CSS – Earthy brown palette
# ------------------------------------------------------------------------------
CSS = """
<style>
/* ── Fix multiselect contrast ────────────────────────────────── */
span[data-baseweb="tag"] span {
    color: #FFFFFF !important;
}

/* ── Base variables ─────────────────────────────────────────── */
:root {
    --bg-primary: #F5F0E8;
    --bg-secondary: #EDE6D3;
    --bg-card: #FDF8F0;
    --bg-input: #FFFFFF;
    --border-color: #C9BA9B;
    --text-primary: #3B2F1E;
    --text-secondary: #5C3D1E;
    --text-muted: #8B7A6A;
    --accent: #8B5E3C;
    --accent-hover: #6B4A30;
    --shadow: rgba(59, 47, 30, 0.15);
    --brown-light: #D4C5A9;
    --brown-dark: #5C3D1E;
    --gold: #C49A6C;
    --sand: #E8DCC8;
    --cream: #FDF8F0;
}

/* ── App background ──────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    color: var(--text-primary);
}
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg-primary);
}

/* ── Block container ─────────────────────────────────────────── */
.block-container {
    padding: 0.5rem 0.5rem 0.5rem !important;
    max-width: 100% !important;
}

/* ── Panel styling for both left and right columns ────────────── */
[data-testid="column"] {
    background: var(--bg-secondary);
    padding: 0.5rem 0.8rem;
    border-radius: 4px;
    height: 100%;
    min-height: 75vh;
    font-family: 'Georgia', serif;
    font-size: 0.7rem;
}

/* Left panel gets a right border */
[data-testid="column"]:first-child {
    border-right: 1px solid var(--border-color);
}


/* Right panel gets a left border */
[data-testid="column"]:last-child {
    border-left: 1px solid var(--border-color);
}

/* ── Panel labels ────────────────────────────────────────────── */
[data-testid="column"] label {
    color: var(--text-secondary);
    font-size: .82rem !important;
}
[data-testid="column"] .stCheckbox label {
    color: var(--text-secondary);
}
[data-testid="column"] .stSelectbox label {
    color: var(--text-secondary);
}

/* ── Typography ───────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Georgia', serif;
    color: var(--brown-dark);
    letter-spacing: .02em;
}
h1 { font-size: 1.3rem; margin-bottom: 0; }
h2 { font-size: 1rem; color: var(--text-secondary); }
h3 { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .06em; }

p, span, label, div {
    font-family: 'Segoe UI', sans-serif;
    color: var(--text-primary);
}

/* ── Slider ───────────────────────────────────────────────────── */
[data-testid="stSlider"] .st-bq {
    background: var(--accent) !important;
}

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 6px 10px;
}
[data-testid="stMetricLabel"] > div {
    color: var(--text-secondary) !important;
    font-size: .75rem !important;
}
[data-testid="stMetricValue"] > div {
    color: var(--accent) !important;
    font-size: 1.2rem !important;
    font-weight: 700;
}

/* ── Dividers ─────────────────────────────────────────────────── */
hr { border-color: var(--border-color) !important; margin: 4px 0; }

/* ── Right column panel headers ──────────────────────────────── */
.right-panel-header {
    font-family: 'Georgia', serif;
    font-size: 0.9rem;
    color: var(--brown-dark);
    text-transform: uppercase;
    letter-spacing: .08em;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 3px;
    margin: 8px 0 4px 0;
}

/* ── Right column metric cards ───────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 4px 8px;
    margin-bottom: 4px;
}
.metric-label {
    color: var(--text-secondary);
    font-size: 0.7rem;
}
.metric-value {
    color: var(--accent);
    font-size: 1.1rem;
    font-weight: 700;
    display: block;
}

/* ── Select / multiselect ────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-input) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* ── Caption ──────────────────────────────────────────────────── */
.stCaption {
    color: var(--text-muted) !important;
    font-size: .78rem !important;
}

/* ── Warning boxes ────────────────────────────────────────────── */
.warn-box {
    background: #FFF3CD;
    border-left: 4px solid #D4A017;
    padding: 5px 10px;
    border-radius: 4px;
    margin: 4px 0;
    font-size: .78rem;
    color: #6B4E00;
}

/* ── Download buttons ─────────────────────────────────────────── */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-size: .78rem !important;
    padding: 5px 12px !important;
    border-radius: 5px !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: var(--accent) !important;
    color: var(--cream) !important;
}

/* ── Right column scrollable container ───────────────────────── */
.right-scroll {
    max-height: 75vh;
    overflow-y: auto;
    padding-right: 4px;
}
.right-scroll::-webkit-scrollbar { width: 4px; }
.right-scroll::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 2px;
}

/* ── Matplotlib figure backgrounds ───────────────────────────── */
[data-testid="stImage"] img { background: transparent; }

/* ── Tabs styling ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg-secondary);
    border-radius: 4px 4px 0 0;
    padding: 6px 16px;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-card);
    color: var(--accent);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--bg-card);
    color: var(--accent);
    border-bottom: 3px solid var(--accent);
}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
ALL_YEARS  = list(range(2017, 2027))

# All 12 months are now available for crest lines and playa polygons.
ALL_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTH_NAMES = list(ALL_MONTHS.keys())

def month_abbr(month_name):
    """Display helper: 'January' -> 'JAN'. Underlying value stays the full name."""
    return month_name[:3].upper()

# Wind data now covers all months, same as crest/playa data.
WIND_MONTHS = ALL_MONTHS

# Number of days per month (non-leap year), used to size the wind coverage chart.
DAYS_IN_MONTH = {
    "January": 31, "February": 28, "March": 31, "April": 30,
    "May": 31, "June": 30, "July": 31, "August": 31,
    "September": 30, "October": 31, "November": 30, "December": 31,
}

# Default focus months for the Custom preset and Monthly preset dropdown.
DEFAULT_FOCUS_MONTHS = ["May", "June", "July", "August"]

UNC_GREEN  = 2.0
UNC_YELLOW = 6.0

# Margin of Error from GNSS validation (The Star Dune, March 2026)
REPRESENTATIVE_MARGIN_OF_ERROR_M =  4.738  # 95% CI margin

WIND_WARN_PCT = 0.70
WIND_HIDE_PCT = 0.30

MAP_CENTER = [-24.76, 15.31]
MAP_ZOOM   = 14

# Brown color palette for matplotlib
MPL_BG      = "#F5F0E8"
MPL_FG      = "#3B2F1E"
MPL_GRID    = "#C9BA9B"
MPL_ACCENT  = "#8B5E3C"
MPL_ACCENT2 = "#C49A6C"

def _dark_fig(w, h):
    """Create a figure with earthy styling"""
    fig, ax = plt.subplots(figsize=(w, h), facecolor=MPL_BG)
    ax.set_facecolor(MPL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(MPL_GRID)
    ax.tick_params(colors=MPL_FG, labelsize=7)
    ax.xaxis.label.set_color(MPL_FG)
    ax.yaxis.label.set_color(MPL_FG)
    ax.title.set_color(MPL_ACCENT)
    return fig, ax

# ------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading base imagery...")
def load_base_imagery_metadata():
    """Load the metadata for processed PNG overlays"""
    metadata_path = "Base_tif/metadata.json"
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

@st.cache_data(show_spinner="Loading crest lines ...")
def load_crest_lines():
    path = "main_data/extended_centerlines.geojson"
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["date"]  = pd.to_datetime(gdf["acquisition_date"])
    gdf["year"]  = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month
    if "is_gap_fill" not in gdf.columns:
        gdf["is_gap_fill"] = gdf["type"] == "connection"
    if "length_m" not in gdf.columns:
        gdf["length_m"] = gdf.geometry.length * 111_320
    return gdf


@st.cache_data(show_spinner="Loading movement points ...")
def load_movement_points():
    path = "main_data/centerline_points.geojson"
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["point_id"] = gdf["point_id"].astype(str)
    date_cols = [c for c in gdf.columns if c.startswith("date_")]
    geom_df   = gdf[["point_id", "distance_along_m", "geometry", "orientation_deg"]].drop_duplicates("point_id")
    df_long   = gdf[["point_id", "distance_along_m"] + date_cols].melt(
        id_vars=["point_id", "distance_along_m"],
        var_name="date_col", value_name="distance_m",
    ).dropna(subset=["distance_m"])
    df_long["date"]  = pd.to_datetime(
        df_long["date_col"].str.replace("date_", "", regex=False).str.replace("_", "-"),
        format="%Y-%m-%d",
    )
    df_long["year"]  = df_long["date"].dt.year
    df_long["month"] = df_long["date"].dt.month
    df_long = df_long.merge(geom_df, on=["point_id", "distance_along_m"], how="left")
    return gpd.GeoDataFrame(df_long, geometry="geometry", crs="EPSG:4326")


@st.cache_data(show_spinner="Loading playa (Highest Purity) ...")
def load_playa_polygons():
    path = "main_data/merged_playa.geojson"
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["date"]  = pd.to_datetime(gdf["acquisition_date"])
    gdf["year"]  = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month
    if "area_m2" not in gdf.columns:
        gdf["area_m2"] = gdf.geometry.area * (111_320 ** 2)
    return gdf


@st.cache_data(show_spinner="Loading wind data ...")
def load_wind_data():
    path = "main_data/combined_weather_with_location.csv"
    df = pd.read_csv(path, sep=";").rename(columns={
        "Date": "datetime",
        "Wind speed  (vc avg)": "speed_ms",
        "Wind  direction  (vc avg)": "direction",
    })
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d %b %Y")
    df["year"]  = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df[df["month"].isin(WIND_MONTHS.values())]


@st.cache_data(show_spinner="Loading uncertainty lines ...")
def load_uncertainty_lines():
    path = "main_data/uncertainty_lines_length.geojson"
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")
    if "abs_error_m" in gdf.columns:
        gdf = gdf.rename(columns={"abs_error_m": "error_m"})
    if "signed_distance_m" in gdf.columns:
        gdf["gnss_val"] = gdf["signed_distance_m"]
    if "left_or_right" in gdf.columns:
        gdf["detected_val"] = gdf["left_or_right"]
    return gdf


# ============================================================================
# NEW: GNSS DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner="Loading GNSS points ...")
def load_gnss_points():
    """Load GNSS point data from GeoJSON"""
    path = "main_data/GNSS_all_points.geojson"
    if not os.path.exists(path):
        st.warning(f"GNSS points file not found: {path}")
        return gpd.GeoDataFrame()
    
    try:
        gdf = gpd.read_file(path).to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error loading GNSS points: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(show_spinner="Loading GNSS crest/edge/bowl lines ...")
def load_gnss_lines():
    """Load GNSS line features (crest, edge, bowl) from GeoJSON"""
    path = "main_data/GNSS_crest_edge_bowl_lines.geojson"
    if not os.path.exists(path):
        st.warning(f"GNSS lines file not found: {path}")
        return gpd.GeoDataFrame()
    
    try:
        gdf = gpd.read_file(path).to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error loading GNSS lines: {e}")
        return gpd.GeoDataFrame()

@st.cache_data(show_spinner="Loading geomorphology layers ...")
def load_geomorph_layers():
    """Load geomorphology shapefiles (line, point, polygon features)"""
    geomorph_data = {}
    
    # List of geomorphology files to load
    geomorph_files = {
        "geomorph_lines": "main_data/Geomorph-SOS1_line-features.geojson",
        "geomorph_points": "main_data/Geomorph-SOS1_point-features.geojson",
        "geomorph_polygons": "main_data/Geomorph-SOS1_polygon-features.geojson"
    }
    
    for key, path in geomorph_files.items():
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path).to_crs("EPSG:4326")
                geomorph_data[key] = gdf
            except Exception as e:
                st.warning(f"Error loading {key}: {e}")
                geomorph_data[key] = gpd.GeoDataFrame()
        else:
            geomorph_data[key] = gpd.GeoDataFrame()
    
    return geomorph_data

@st.cache_data(show_spinner="Loading SOS 1 WEST wind data...")
def load_hobo_wind_data():
    path = "main_data/Weatherstation_SOS_1_West_March_2026.csv"
    df = pd.read_csv(path, skiprows=1)
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['Datum Zeit, GMT+01:00'], format='%m.%d.%y %I:%M:%S %p')
    
    # Rename columns
    df = df.rename(columns={
        'Windrichtung, ø (LGR S/N: 22429151_duplicate, SEN S/N: 22407863, LBL: Wind direction)': 'direction',
        'Windgeschwindigkeit, m/s (LGR S/N: 22429151_duplicate, SEN S/N: 22427869, LBL: Wind speed)': 'speed_ms'
    })
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    
    return df

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def utm_to_latlon(easting, northing):
    from pyproj import Transformer
    transformer = Transformer.from_crs(f"EPSG:32733", "EPSG:4326")
    lon, lat = transformer.transform(easting, northing)
    return lon, lat

def get_base_imagery_for_date(metadata, selected_years, selected_months, date_a, date_b, preset):
    """Determine which PNG to display based on selected date range"""
    
    if not metadata:
        return None, None, None
    
    if preset == "Compare":
        # Use date_a as the reference
        target_year = date_a.year
        target_month = date_a.month
    elif preset == "Annual":
        target_year = selected_years[0] if selected_years else 2017
        target_month = 1  # Jan
    elif preset == "Monthly":
        # Use the selected month, use earliest year available
        target_year = 2017  # Or get from metadata
        month_name = selected_months[0] if selected_months else "May"
        target_month = ALL_MONTHS[month_name]
    else:  # Custom
        # Use the earliest selected year and month
        target_year = min(selected_years) if selected_years else 2017
        if selected_months:
            target_month = ALL_MONTHS[selected_months[0]]
        else:
            target_month = 5  # May
    
    key = f"{target_year}_{target_month:02d}"
    
    if key in metadata:
        png_path = os.path.join("Base_tif", metadata[key]["png_path"])
        bounds = metadata[key]["bounds"]
        date_full = metadata[key].get("date_full", None) 
        return png_path, bounds, date_full
    
    available_keys = sorted(metadata.keys())
    if available_keys:
        # Find the closest key
        closest_key = min(available_keys, key=lambda k: abs(int(k.split('_')[0]) - target_year))
        png_path = os.path.join("Base_tif", metadata[closest_key]["png_path"])
        bounds = metadata[closest_key]["bounds"]
        date_full = metadata[closest_key].get("date_full", None) 
        return png_path, bounds, date_full
    
    return None, None, None

def date_colormap(dates):
    timestamps = pd.to_datetime(dates).astype(np.int64)
    norm = plt.Normalize(timestamps.min(), timestamps.max())
    cmap = plt.cm.plasma
    return [mcolors.to_hex(cmap(norm(t))) for t in timestamps]


def diverging_color(value, vmin=-10, vmax=10):
    norm = plt.Normalize(vmin, vmax)
    r, g, b, _ = plt.cm.PiYG(norm(value))
    return mcolors.to_hex((r, g, b))


def unc_color(error_m):
    if error_m < UNC_GREEN:  return "#31A857"
    if error_m < UNC_YELLOW: return "#F7E62C"
    return "#C7400F"


def wind_completeness(wind_df, years, months):
    m_nums = [WIND_MONTHS[m] for m in months if m in WIND_MONTHS]
    sub = wind_df[wind_df["year"].isin(years) & wind_df["month"].isin(m_nums)]
    if sub.empty:
        return 0.0, sub
    expected = sum(DAYS_IN_MONTH[m] for y in years for m in months if m in DAYS_IN_MONTH)
    if expected == 0:
        return 0.0, sub
    frac = min(sub["direction"].notna().sum() / expected, 1.0)
    return frac, sub


def wind_completeness_daterange(wind_df, date_a, date_b):
    """Date-accurate version for the Compare preset: filters wind data to the
    exact calendar days between date_a and date_b (inclusive), rather than
    every day in the two dates' months/years.
    """
    d0, d1 = sorted([pd.Timestamp(date_a), pd.Timestamp(date_b)])
    sub = wind_df[(wind_df["datetime"] >= d0) & (wind_df["datetime"] <= d1)]
    expected_days = (d1 - d0).days + 1
    if expected_days <= 0:
        return 0.0, sub
    frac = min(sub["direction"].notna().sum() / expected_days, 1.0)
    return frac, sub


def build_wind_rose_image(wind_df):
    fig = plt.figure(figsize=(2.8, 2.8), facecolor=MPL_BG)
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wind_df["direction"], wind_df["speed_ms"],
           normed=True, opening=0.8, edgecolor=MPL_GRID,
           cmap=plt.cm.YlOrBr, bins=np.arange(0, 12, 2))
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_FG, labelsize=8)
    fig.patch.set_alpha(1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=MPL_BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def build_simple_wind_rose(wind_df, date_start, date_end):
    """Create a simple NSWE wind rose for a date range. Returns (base64_image, has_data)"""
    # Filter wind data for the date range
    sub = wind_df[(wind_df["datetime"] >= pd.Timestamp(date_start)) & 
                  (wind_df["datetime"] <= pd.Timestamp(date_end))]
    
    # Check if we have data
    if sub.empty or sub["direction"].notna().sum() < 5:
        return None, False
    
    # Create simple wind rose
    fig = plt.figure(figsize=(2.2, 2.2), facecolor=MPL_BG)
    ax = WindroseAxes.from_ax(fig=fig)
    
    ax.bar(sub["direction"], sub["speed_ms"],
           normed=True, opening=0.8, edgecolor=MPL_GRID,
           cmap=plt.cm.YlOrBr, bins=np.arange(0, 12, 2))
    
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_FG, labelsize=6)
    ax.set_title('')
    ax.legend().set_visible(False)
    
    fig.patch.set_alpha(1)
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=MPL_BG)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode(), True


def get_wind_rose_pairs(crest_gdf, preset, selected_years, selected_months):
    """Get consecutive date pairs for wind roses. Returns list of (date_a, date_b, label)"""
    pairs = []
    
    if crest_gdf.empty:
        return pairs
    
    dates = sorted(crest_gdf["date"].unique())
    
    if preset == "Monthly":
        # Monthly: same month, consecutive years
        month_num = ALL_MONTHS[selected_months[0]]
        month_dates = [d for d in dates if d.month == month_num]
        
        for i in range(len(month_dates) - 1):
            pairs.append((month_dates[i], month_dates[i + 1], 
                         f"{month_dates[i].year}→{month_dates[i+1].year}"))
            
    elif preset == "Annual":
        # Annual: same year, consecutive months
        year = selected_years[0]
        year_dates = [d for d in dates if d.year == year]
        
        for i in range(len(year_dates) - 1):
            pairs.append((year_dates[i], year_dates[i + 1],
                         f"{year_dates[i].strftime('%b')}→{year_dates[i+1].strftime('%b')}"))
    
    return pairs


def build_gantt_figure(wind_df, years, months):
    fig, ax = _dark_fig(5.6, max(1.8, len(years) * 0.38))
    m_nums = [WIND_MONTHS[m] for m in months if m in WIND_MONTHS]

    # Lay months out left-to-right in calendar order, each slot sized to its
    # real day count plus a small gap, so this works for any subset of months.
    ordered_months = sorted(WIND_MONTHS.items(), key=lambda item: item[1])
    x0_by_month, xticks, xlabels = {}, [], []
    cursor = 0
    for m_name, m_num in ordered_months:
        x0_by_month[m_name] = cursor
        xticks.append(cursor + DAYS_IN_MONTH[m_name] / 2)
        xlabels.append(month_abbr(m_name))
        cursor += DAYS_IN_MONTH[m_name] + 2

    for i, year in enumerate(sorted(years)):
        for m_name, m_num in ordered_months:
            if m_num not in m_nums:
                continue
            sub   = wind_df[(wind_df.year == year) & (wind_df.month == m_num) &
                            (wind_df["direction"].notna())]
            days  = DAYS_IN_MONTH[m_name]
            x0    = x0_by_month[m_name]
            valid = len(sub)
            miss  = days - valid
            pct   = (valid / days) * 100

            ax.barh(i, valid, left=x0, height=0.55,
                    color="#2D7D46", alpha=0.85)
            if valid > 0:
                ax.text(x0 + days / 2, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=6,
                        color="#FFFFFF", fontweight="bold")
            if miss > 0:
                ax.barh(i, miss, left=x0 + valid, height=0.55,
                        color="#8B2500", alpha=0.75)

    ax.set_yticks(range(len(sorted(years))))
    ax.set_yticklabels(sorted(years), fontsize=7, color=MPL_FG)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, color=MPL_FG)
    ax.set_title("Wind Coverage", fontsize=8, color=MPL_ACCENT, pad=4)
    ax.grid(axis="x", color=MPL_GRID, linewidth=0.4, linestyle=":")
    fig.tight_layout(pad=0.4)
    return fig


# ------------------------------------------------------------------------------
# MAP BUILDER
# ------------------------------------------------------------------------------

def build_map(
    crest_gdf, var_gdf, playa_gdf, unc_gdf,
    wind_b64, wind_completeness_pct,
    show_crests, show_gap_fills,
    show_movement, date_a, date_b,
    show_playa, show_wind, show_uncertainty,
    show_margin_buffer,  # NEW: separate checkbox for margin buffer
    opacity, date_min=None, date_max=None,
    show_base_imagery=True, 
    base_metadata=None,  
    selected_years=None, 
    selected_months=None,  
    preset="Custom",
    # NEW: GNSS layer parameters
    show_gnss_points=False,
    show_gnss_lines=False,
    show_geomorph_lines=False,
    show_geomorph_points=False,
    show_geomorph_polygons=False,
    gnss_points_gdf=None,
    gnss_lines_gdf=None,
    geomorph_data=None,
    hobo_df=None,
    hobo_lat=None,
    hobo_lon=None,
    show_hobo_wind=False,
):
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM,
                   tiles=None, control_scale=True)

    plugins.MeasureControl(
        position="bottomright",
        primary_length_unit="meters", secondary_length_unit="kilometers",
        primary_area_unit="sqmeters", secondary_area_unit="hectares",
        active_color="#8B5E3C", completed_color="#2D7D46",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", max_native_zoom=16, max_zoom=22,
        name="Satellite", overlay=False, control=True,
    ).add_to(m)

    if show_base_imagery and base_metadata:
        png_path, bounds, base_img_date = get_base_imagery_for_date(base_metadata, selected_years, selected_months, date_a, date_b, preset)
        
        if png_path and bounds:           
            img = Image.open(png_path)
            img_array = np.array(img)

            brightness_factor = 1.5
            img_array = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)

            folium.raster_layers.ImageOverlay(
                image=img_array,
                bounds=[[bounds["bottom"], bounds["left"]], [bounds["top"], bounds["right"]]],
                opacity=0.9,
                name="Base Imagery",
                overlay=True,
                control=True
            ).add_to(m)

    # 1. UNCERTAINTY LINES (measured error vectors)
    if show_uncertainty and not unc_gdf.empty:
        for _, row in unc_gdf.iterrows():
            err = row.get("error_m", 0)
            if pd.isna(err):
                continue
            c = unc_color(err)
            coords = list(row.geometry.coords)
            folium.PolyLine(
                locations=[[lat, lon] for lon, lat in coords],
                color=c,
                weight=5,
                opacity= opacity,
                tooltip=folium.Tooltip(
                    f"<b>GNSS Displacement</b><br>Error: {err:.2f} m"
                ),
            ).add_to(m)

    # 2. MARGIN OF ERROR BUFFER (95% CI extrapolated to all crests)
    if show_margin_buffer and not crest_gdf.empty:
        try:
            # Buffer in UTM for accurate distance
            buffered = crest_gdf.to_crs("EPSG:32733")
            buffered = buffered.assign(geometry=buffered.geometry.buffer(REPRESENTATIVE_MARGIN_OF_ERROR_M))
            dissolved = buffered.dissolve().to_crs("EPSG:4326")
            folium.GeoJson(
                dissolved.geometry.iloc[0].__geo_interface__,
                style_function=lambda f: {
                    "fillColor": "#C7400F", "color": "#C7400F",
                    "weight": 0, "fillOpacity": opacity * 0.5,
                },
                tooltip=folium.Tooltip(
                    f"<b>Margin of Error Buffer (95% CI)</b><br>±{REPRESENTATIVE_MARGIN_OF_ERROR_M:.2f} m "
                    f"(extrapolated from GNSS validation - The Star Dune, Mar 2026)"
                ),
            ).add_to(m)
        except Exception as e:
            # Silently fail if buffer can't be created
            pass

    # 3. PLAYA
    if show_playa and not playa_gdf.empty:
        dates_sorted  = sorted(playa_gdf["date"].unique())
        date_color_map = dict(zip(
            [str(d) for d in dates_sorted], date_colormap(dates_sorted)
        ))
        for _, row in playa_gdf.iterrows():
            c = date_color_map.get(str(row["date"]), "#C9BA9B")
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, col=c: {
                    "fillColor": col, "color": col,
                    "weight": 1, "fillOpacity": opacity * 0.5,
                },
                tooltip=folium.Tooltip(
                    f"<b>Playa</b><br>Date: {row['date'].date()}<br>"
                    f"Area: {row.get('area_m2', 0)/1e6:.3f} km²"
                ),
            ).add_to(m)

   

    # 5. movement POINTS — drawn as directional arrows that follow the perpendicular
    #    orientation. When diff > 0 (advance), arrow points in the perpendicular direction
    #    (orientation_deg). When diff < 0 (retreat), arrow points in the opposite direction
    #    (orientation_deg + 180).
    if show_movement and date_a and date_b and not var_gdf.empty:
        def _agg(df, dt):
            sub = df[df["date"] == pd.Timestamp(dt)]
            return sub.groupby("point_id")["geometry"].first(), \
                sub.groupby("point_id")["distance_m"].mean()

        geom_a, dist_a = _agg(var_gdf, date_a)
        geom_b, dist_b = _agg(var_gdf, date_b)
        common = dist_a.index.intersection(dist_b.index)
        
        # Get orientation for each point_id from the original data
        orientation_lookup = {}
        if 'orientation_deg' in var_gdf.columns:
            # Get unique orientation per point_id (should be constant for each point)
            for pid in common:
                orientation_values = var_gdf[var_gdf['point_id'] == pid]['orientation_deg'].unique()
                if len(orientation_values) > 0 and not pd.isna(orientation_values[0]):
                    orientation_lookup[pid] = orientation_values[0]
                else:
                    orientation_lookup[pid] = None  # Explicitly mark as missing
        else:
            # No orientation column available - skip movement arrows
            st.warning("Orientation data not available. Movement arrows require 'orientation_deg' column.")
            # Continue with empty lookup, arrows won't be drawn

        for pid in common:
            # Skip if no orientation data
            if pid not in orientation_lookup or orientation_lookup[pid] is None:
                continue
                
            val_a = float(dist_a[pid])
            val_b = float(dist_b[pid])
            geom  = geom_b[pid]
            diff  = val_b - val_a
            color = diverging_color(diff)
            magnitude = abs(diff)
            size = max(10, min(25, 10 + magnitude * 1.8))

            # Arrow direction: follow perpendicular orientation
            # orientation_deg is 0-360 where 0 = North, 90 = East, etc.
            # For advance (diff > 0): point in the perpendicular direction
            # For retreat (diff < 0): point in the opposite direction
            if diff >= 0:
                arrow_deg = orientation_lookup[pid]
            else:
                arrow_deg = (orientation_lookup[pid] + 180) % 360

            arrow_svg = f"""
            <div style="width:{size}px;height:{size}px;transform:rotate({arrow_deg-90}deg);opacity:0.7;">
            <svg width="{size}" height="{size}" viewBox="0 0 24 24">
                <line x1="0" y1="12" x2="19" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
                <polygon points="17,8 23,12 17,16" fill="{color}"/>
            </svg>
            </div>"""

            folium.Marker(
                location=[float(geom.y), float(geom.x)],
                icon=folium.DivIcon(
                    icon_size=(size, size),
                    icon_anchor=(size / 2, size / 2),
                    html=arrow_svg,
                ),
                tooltip=folium.Tooltip(
                    f"<b>Crest movement</b><br>Point: {pid}<br>"
                    f"Date A ({pd.Timestamp(date_a).date()}): {val_a:.2f} m<br>"
                    f"Date B ({pd.Timestamp(date_b).date()}): {val_b:.2f} m<br>"
                    f"Orientation: {orientation_lookup[pid]:.1f}°<br>"
                    f"<b>Change: {diff:+.2f} m</b> "
                    f"({'advance' if diff >= 0 else 'retreat'})"
                ),
            ).add_to(m)


     # 4. CREST LINES
    if show_crests and not crest_gdf.empty:
        dates_sorted  = sorted(crest_gdf["date"].unique())
        date_color_map = dict(zip(
            [str(d) for d in dates_sorted], date_colormap(dates_sorted)
        ))
        for _, row in crest_gdf.iterrows():
            if row.get("is_gap_fill", False) and not show_gap_fills:
                continue
            c    = date_color_map.get(str(row["date"]), "#C9BA9B")
            style = {
                "color": c, 
                "weight": 2, 
                "opacity": opacity*0.5 if row.get("is_gap_fill", False) else opacity, 
                "dashArray": "8 4" if row.get("is_gap_fill", False) else None}
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, s=style: s,
                tooltip=folium.Tooltip(
                    f"<b>Crest</b><br>Date: {row['date'].date()}<br>"
                    f"Length: {row.get('length_m', 0):.0f} m"
                    + (" [gap fill]" if row.get("is_gap_fill") else "")
                ),
            ).add_to(m)

    # ============================================================================
    # NEW: GNSS AND GEOMORPHOLOGY LAYERS
    # ============================================================================
    
    # 6a. GNSS Points
    if show_gnss_points and gnss_points_gdf is not None and not gnss_points_gdf.empty:
        # Color by type if available
        if 'type' in gnss_points_gdf.columns:
            type_colors = {
                'p3': '#FF6B6B',      # Red
                'p4': '#FF6B6B',      # Red
                'ccp': '#4ECDC4',    # Teal
                'crest': '#FFD93D',  # Yellow
                'edge': '#6C5CE7',   # Purple
                'bowl': '#A8E6CF'    # Light green
            }
            # Default color for unknown types
            default_color = '#95A5A6'
            
            for _, row in gnss_points_gdf.iterrows():
                point_type = row.get('type', 'Unknown')
                color = type_colors.get(point_type.lower(), default_color)
                
                # Get point name for tooltip
                name = row.get('full_name', row.get('Name', 'Unknown'))
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=2,
                    color=color,
                    fill=True,
                    stroke=False,
                    fill_color=color,
                    fill_opacity=1,
                    weight=5,
                    tooltip=folium.Tooltip(
                        f"<b>GNSS Point</b><br>"
                        f"Name: {name}<br>"
                        f"Type: {point_type}<br>"
                        f"Group: {row.get('group', 'Unknown')}<br>"
                        f"Direction: {row.get('direction', 'Unknown')}<br>"
                        f"Sequence: {row.get('sequence', '')}"
                    ),
                ).add_to(m)
        else:
            # If no type column, just plot all points in a single color
            for _, row in gnss_points_gdf.iterrows():
                name = row.get('full_name', row.get('Name', 'Unknown'))
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color='#FF6B6B',
                    fill=True,
                    fill_color='#FF6B6B',
                    fill_opacity=opacity * 0.8,
                    weight=2,
                    tooltip=folium.Tooltip(
                        f"<b>GNSS Point</b><br>"
                        f"Name: {name}"
                    ),
                ).add_to(m)

    # 6b. GNSS Lines (crest, edge, bowl)
    if show_gnss_lines and gnss_lines_gdf is not None and not gnss_lines_gdf.empty:
        if 'type' in gnss_lines_gdf.columns:
            type_colors = {
                'crest': '#FFD93D',  # Yellow
                'CREST': '#FFD93D',  # Yellow
                'edge': '#6C5CE7',   # Purple
                'bowl': '#A8E6CF'    # Light green
            }
            default_color = '#95A5A6'
        for _, row in gnss_lines_gdf.iterrows():
            coords = list(row.geometry.coords)
            # coords are [x, y, z] - we only need x and y
            # Convert from [x, y] to [lat, lon] for folium
            locations = [[coord[1], coord[0]] for coord in coords]
            
            if len(locations) >= 2:
                line_name = row.get('name', 'Unknown')
                folium.PolyLine(
                    locations=locations,
                    color=type_colors.get(row.get('type'), default_color),
                    weight=3,
                    opacity=opacity,
                    tooltip=folium.Tooltip(
                        f"<b>GNSS Line</b><br>"
                        f"Name: {line_name}"
                    ),
                ).add_to(m)

    # 6c. Geomorphology Lines
    if show_geomorph_lines and geomorph_data is not None and 'geomorph_lines' in geomorph_data:
        gdf_lines = geomorph_data['geomorph_lines']
        if not gdf_lines.empty:
            for _, row in gdf_lines.iterrows():
                coords = list(row.geometry.coords)
                folium.PolyLine(
                    locations=[[lat, lon] for lon, lat in coords],
                    color="#201C19",
                    weight=2,
                    opacity=opacity * 0.7,
                    tooltip=folium.Tooltip(
                        f"<b>Geomorph Line</b><br>"
                        f"ID: {row.get('id', 'Unknown')}"
                    ),
                ).add_to(m)

    # 6d. Geomorphology Points
    if show_geomorph_points and geomorph_data is not None and 'geomorph_points' in geomorph_data:
        gdf_points = geomorph_data['geomorph_points']
        if not gdf_points.empty:
            for _, row in gdf_points.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color="#FD6F17",
                    fill=True,
                    fill_color='#FD6F17',
                    fill_opacity=opacity * 0.7,
                    weight=1,
                    tooltip=folium.Tooltip(
                        f"<b>Geomorph Point</b><br>"
                        f"ID: {row.get('Sample', 'Unknown')}"
                    ),
                ).add_to(m)

    # 6e. Geomorphology Polygons
    if show_geomorph_polygons and geomorph_data is not None and 'geomorph_polygons' in geomorph_data:
        gdf_polygons = geomorph_data['geomorph_polygons']
        if not gdf_polygons.empty:
            if 'Feature' in gdf_polygons.columns:
                type_colors = {
                    'Fossil Dune': '#6C5CE7',   # Purple
                    'Recent Vlei': "#0CC883"    # Light green
                }
                default_color = '#95A5A6'
                for _, row in gdf_polygons.iterrows():
                    folium.GeoJson(
                        row["geometry"].__geo_interface__,
                        style={
                            "fillColor": type_colors.get(row.get('Feature'), default_color),
                            "color": type_colors.get(row.get('Feature'), default_color),
                            "weight": 1,
                            "fillOpacity": opacity * 0.3,
                        },
                        tooltip=folium.Tooltip(
                            f"<b>Geomorph Polygon</b><br>"
                            f"ID: {row.get('Feature', 'Unknown')}"
                        ),
                    ).add_to(m)

    # 7. WIND ROSE OVERLAY
    if show_wind and wind_b64 and preset == "Compare":
        badge = ""
        if wind_completeness_pct < WIND_WARN_PCT:
            badge = (f'<div style="background:#FFF3CD;color:#6B4E00;font-size:9px;'
                     f'padding:2px 5px;border-radius:3px;margin-top:3px;font-weight:700;">'
                     f'! {wind_completeness_pct*100:.0f}% coverage</div>')
        html = f"""
        <div style="position:fixed;top:10px;right:10px;z-index:9999;
                    background:rgba(253,248,240,0.92);border:1px solid #C9BA9B;
                    border-radius:8px;padding:7px 8px;text-align:center;opacity:{opacity};">
          
          <img src="data:image/png;base64,{wind_b64}" width="150"/>
          <div style="margin-top:4px;font-size:8px;color:#5C3D1E;font-weight:600;">Wind speed (m/s)</div>
          <div style="display:flex;align-items:center;margin-top:2px;gap:2px;">
            <span style="font-size:7px;color:#5C3D1E;">0</span>
            <div style="flex:1;height:6px;background:linear-gradient(to right,
              #FFFFCC,#FFEDA0,#FED976,#FEB24C,#FD8D3C,#FC4E2A,#E31A1C,#B10026);
              border-radius:2px;margin:0 2px;"></div>
            <span style="font-size:7px;color:#5C3D1E;">10+</span>
          </div>
          {badge}
        </div>"""
        m.get_root().html.add_child(folium.Element(html))

  # 7b. SOS 1 WEST WIND ROSE OVERLAY (at geographic location) - controlled by In-situ checkbox
    if show_hobo_wind and hobo_df is not None:
        
        # Filter SOS 1 WEST data for the ENTIRE dataset (no date filtering)
        hobo_sub = hobo_df
        
        if not hobo_sub.empty:
            # Build wind rose with transparent background
            fig = plt.figure(figsize=(1.5, 1.5), facecolor='none')
            ax = WindroseAxes.from_ax(fig=fig)
            ax.bar(hobo_sub["direction"], hobo_sub["speed_ms"],
                normed=True, opening=0.8, edgecolor="#050505", linewidth=0.3,
                cmap=plt.cm.YlOrBr, bins=np.arange(0, 12, 2))
            ax.set_facecolor('none')
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.legend().set_visible(False)
            ax.set_title('')
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.patch.set_alpha(0)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', transparent=True)
            plt.close(fig)
            hobo_b64 = base64.b64encode(buf.getvalue()).decode()
            
            # Use the correct coordinates
            hobo_lat, hobo_lon = utm_to_latlon(533272.70433545333799, 7262367.944419549778104)
            
            # Add marker
            folium.Marker(
                location=[hobo_lat, hobo_lon],
                icon=folium.DivIcon(
                    html=f'''
                    <div style="transform: translate(-50%, -50%);">
                        <img src="data:image/png;base64,{hobo_b64}" width="55"/>
                    </div>
                    ''',
                    icon_size=(55, 55),
                    icon_anchor=(27, 27)
                ),
                tooltip="SOS 1 WEST Weather Station (16 - 20 March 2026)"
            ).add_to(m)

    # 8. LEGEND
    sections = []

    if show_crests or show_playa:
        early = date_min.strftime("%Y-%m") if date_min else "Early"
        late  = date_max.strftime("%Y-%m") if date_max else "Late"
        playa_row = """
        <div class="lr">
            <svg width="24" height="10">
                <rect x="0" y="0" width="24" height="10" fill="#C49A6C" opacity="0.5"
                      stroke="#C49A6C" stroke-width="1" rx="2"/>
            </svg>
            <span>Playa</span>
        </div>""" if show_playa else ""
        sections.append(f"""
        <div class="ls">
          <div class="lt">Crest lines &amp; Playa</div>
          <div class="lr">
            <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4"
              stroke="#8B5E3C" stroke-width="2"/></svg>
            <span>Detected crest</span>
          </div>
          <div class="lr">
            <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4"
              stroke="#8B5E3C" stroke-width="1.5" stroke-dasharray="5,3"/></svg>
            <span>Gap fill</span>
          </div>
          {playa_row}
          <div style="display:flex;align-items:center;gap:3px;margin:4px 0 0 0;">
            <span style="font-size:8px;color:#0D0887;">{early}</span>
            <div style="flex:1;height:6px;background:linear-gradient(to right,
              #0D0887,#9C179E,#ED7953,#F0F921);border-radius:2px;"></div>
            <span style="font-size:8px;color:#ED7953;">{late}</span>
          </div>
        </div>""")

    if show_movement:
        sections.append("""
        <div class="ls">
        <div class="lt">Crest Movement</div>
        <div style="display:flex;align-items:center;gap:6px;margin:3px 0;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:3px;">
            <svg width="14" height="14" viewBox="0 0 24 24">
                <line x1="0" y1="12" x2="19" y2="12" stroke="#8B5E3C" stroke-width="2" stroke-linecap="round"/>
                <polygon points="17,8 23,12 17,16" fill="#8B5E3C"/>
            </svg>
            <span style="font-size:8px;color:#5C3D1E;"> &lt; 2 m</span>
            </div>
            <div style="display:flex;align-items:center;gap:3px;">
            <svg width="18" height="18" viewBox="0 0 24 24">
                <line x1="0" y1="12" x2="19" y2="12" stroke="#8B5E3C" stroke-width="3" stroke-linecap="round"/>
                <polygon points="17,8 23,12 17,16" fill="#8B5E3C"/>
            </svg>
            <span style="font-size:8px;color:#5C3D1E;">2 m - 6 m</span>
            </div>
            <div style="display:flex;align-items:center;gap:3px;">
            <svg width="24" height="24" viewBox="0 0 24 24">
                <line x1="0" y1="12" x2="19" y2="12" stroke="#8B5E3C" stroke-width="4" stroke-linecap="round"/>
                <polygon points="17,8 23,12 17,16" fill="#8B5E3C"/>
            </svg>
            <span style="font-size:8px;color:#5C3D1E;"> &gt; 6 m</span>
            </div>
            <div style="display:flex;align-items:center;gap:3px;margin:3px 0;width:100%;">
                <span style="font-size:8px;color:#D62646;">-10</span>
                <div style="flex:1;height:6px;background:linear-gradient(to right,
                #D62646,#F4B3C2,#FFFDE0,#D9F0D9,#008F48);border-radius:2px;"></div>
                <span style="font-size:8px;color:#008F48;">+10</span>
            </div>
        </div>
        </div>""")

   # Uncertainty section with both measured lines and margin buffer
    if show_uncertainty or show_margin_buffer:
        sections.append("""
        <div class="ls">
        <div class="lt">ERROR ASSESSMENT</div>""")
        
        if show_uncertainty:
            sections.append("""
            <div class="lr" style="margin-top:3px;"><span style="font-size:8px;font-weight:600;color:#5C3D1E;">Measured DISPLACEMENT ERROR (March 2026)</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#31A857;"></div>
                <span>&lt; 2 m</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#F7E62C;"></div>
                <span>2-6 m</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#C7400F;"></div>
                <span>&gt; 6 m</span></div>""")
        
        if show_margin_buffer:
            if show_uncertainty:
                sections.append("""
            <div style="border-top:1px dashed #C9BA9B;margin:4px 0;"></div>""")
            sections.append(f"""
            <div class="lr"><span style="font-size:8px;font-weight:600;color:#5C3D1E;">MARGIN OF ERROR (95% CI, EXTRAPOLATED)</span></div>
            <div class="lr">
                <svg width="16" height="10"><rect x="0" y="0" width="16" height="10"
                    fill="#C7400F" opacity="0.3" stroke="#C7400F" stroke-width="1"/></svg>
                <span>±{REPRESENTATIVE_MARGIN_OF_ERROR_M:.1f} m buffer (single-epoch assumption)</span>
            </div>""")
        
        sections.append("</div>")

    # NEW: GNSS Layer Legend Entries with colors
    if show_gnss_points or show_gnss_lines:
        sections.append("""
        <div class="ls">
        <div class="lt">GNSS Field Data</div>""")
        
        if show_gnss_points:
            sections.append("""
            <div class="lr">
                <div style="width:14px;height:14px;border-radius:50%;background:#FF6B6B;"></div>
                <span>GNSS Points (P3, P4)</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:14px;border-radius:50%;background:#4ECDC4;"></div>
                <span>GNSS Points (CCP)</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:14px;border-radius:50%;background:#FFD93D;"></div>
                <span>GNSS Points (Crest)</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:14px;border-radius:50%;background:#6C5CE7;"></div>
                <span>GNSS Points (Edge)</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:14px;border-radius:50%;background:#A8E6CF;"></div>
                <span>GNSS Points (Bowl)</span>
            </div>""")
        
        if show_gnss_lines:
            sections.append("""
            <div class="lr">
                <div style="width:14px;height:3px;background:#FFD93D;"></div>
                <span>GNSS Crest Line</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:3px;background:#6C5CE7;"></div>
                <span>GNSS Edge Line</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:3px;background:#A8E6CF;"></div>
                <span>GNSS Bowl Line</span>
            </div>""")
        
        sections.append("</div>")

    # Geomorphology legend with colors
    if show_geomorph_lines or show_geomorph_points or show_geomorph_polygons:
        sections.append("""
        <div class="ls">
        <div class="lt">Geomorphology</div>""")
        
        if show_geomorph_lines:
            sections.append("""
            <div class="lr">
                <div style="width:14px;height:3px;background:#201C19;"></div>
                <span>Erosion fossil dune / vlei deposits</span>
            </div>""")
        
        if show_geomorph_points:
            sections.append("""
            <div class="lr">
                <div style="width:14px;height:14px;border-radius:50%;background:#FD6F17;"></div>
                <span>Sediment Sample Points</span>
            </div>""")
        
        if show_geomorph_polygons:
            sections.append("""
            <div class="lr">
                <div style="width:14px;height:14px;background:#6C5CE7;opacity:0.5;border:1px solid #6C5CE7;"></div>
                <span>Fossil Dune</span>
            </div>
            <div class="lr">
                <div style="width:14px;height:14px;background:#A8E6CF;opacity:0.5;border:1px solid #A8E6CF;"></div>
                <span>Recent Vlei</span>
            </div>""")
        
        sections.append("</div>")

    if show_base_imagery and base_img_date:
        sections.append(f"""
        <div class="ls">
            <div class="lt">Base Imagery</div>
            <div class="lr">
                <span style="font-size:9px;color:#5C3D1E;font-weight:600;">{base_img_date}</span>
            </div>
        </div>""")

    # NEW: SOS 1 WEST Wind Rose Legend Entry
    if show_hobo_wind and hobo_df is not None:
        sections.append("""
        <div class="ls">
            <div class="lt">SOS 1 WEST Weather Station</div>
            <div class="lr">
                <svg width="24" height="10">
                    <circle cx="12" cy="5" r="4" fill="#8B5E3C" opacity="0.8"/>
                </svg>
                <span>Wind Rose (March 2026)</span>
            </div>
            <div style="display:flex;align-items:center;gap:3px;margin:4px 0 0 0;">
                <span style="font-size:7px;color:#5C3D1E;">0</span>
                <div style="flex:1;height:5px;background:linear-gradient(to right,
                    #FFFFCC,#FFEDA0,#FED976,#FEB24C,#FD8D3C,#FC4E2A,#E31A1C,#B10026);
                    border-radius:2px;"></div>
                <span style="font-size:7px;color:#5C3D1E;">10+</span>
            </div>
        </div>
        """)

    if sections:
        legend_html = f"""
        <style>
          #dl {{ position:fixed;bottom:50px;left:10px;z-index:9998;
                 font-family:'Segoe UI',sans-serif;font-size:10px;color:#3B2F1E; }}
          #db {{ background:rgba(253,248,240,0.92);border:1px solid #C9BA9B;
                 border-radius:7px;padding:7px 9px;min-width:160px;max-width:200px;
                 box-shadow:0 2px 8px rgba(0,0,0,0.15); }}
          #dt {{ cursor:pointer;user-select:none;font-weight:700;font-size:11px;
                 color:#5C3D1E;display:flex;justify-content:space-between;
                 align-items:center;font-family:Georgia,serif;letter-spacing:.04em; }}
          #dt:hover {{ color:#8B5E3C; }}
          #dc {{ margin-top:6px; }}
          .ls {{ margin-bottom:7px;padding-bottom:5px;border-bottom:1px solid #C9BA9B; }}
          .ls:last-child {{ border-bottom:none;margin-bottom:0; }}
          .lt {{ font-weight:700;font-size:9px;color:#5C3D1E;margin-bottom:3px;
                 text-transform:uppercase;letter-spacing:.06em; }}
          .lr {{ display:flex;align-items:center;gap:5px;margin-bottom:2px;font-size:9px; }}
        </style>
        <div id="dl"><div id="db">
          <div id="dt" onclick="
            var c=document.getElementById('dc');
            var a=document.getElementById('da');
            if(c.style.display==='none'){{c.style.display='block';a.textContent='▼';}}
            else{{c.style.display='none';a.textContent='►';}}">
            LEGEND <span id="da">▼</span>
          </div>
          <div id="dc">{"".join(sections)}</div>
        </div></div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ------------------------------------------------------------------------------
# CHART HELPERS
# ------------------------------------------------------------------------------

def movement_trend_fig(var_gdf, nearest_pid):
    trend = var_gdf[var_gdf["point_id"] == nearest_pid].sort_values("date")
    fig, ax = _dark_fig(3.6, 1.9)
    mask_winter = trend["date"].dt.month.isin([4, 5, 6, 7, 8, 9])   # Winter (Apr-Sep) - squares
    mask_summer = trend["date"].dt.month.isin([10, 11, 12, 1, 2, 3]) # Summer (Oct-Mar) - circles
    
    # Plot Winter (Apr-Sep) as squares
    ax.scatter(trend.loc[mask_winter, "date"], trend.loc[mask_winter, "distance_m"],
               color="#2E86C1", s=5, marker='d', zorder=5, label="Winter (Apr-Sep)")
    
    # Plot Summer (Oct-Mar) as circles
    ax.scatter(trend.loc[mask_summer, "date"], trend.loc[mask_summer, "distance_m"],
               color="#E67E22", s=5, marker='o', zorder=5, label="Summer (Oct-Mar)")
    
    # Connect points with line
    ax.plot(trend["date"], trend["distance_m"],
            marker="", color="#C9BA9B", linewidth=1, alpha=0.5)
    
    ax.axhline(0, color="#C9BA9B", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("Distance (m)", fontsize=7)
    ax.set_title(f"Point {nearest_pid}", fontsize=8)
    ax.grid(color="#C9BA9B", linewidth=0.4, linestyle=":")
    ax.legend(fontsize=6, loc="best")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(pad=0.4)
    return fig


def uncertainty_hist_fig(unc_gdf):
    fig, ax = _dark_fig(3.6, 1.9)
    vals = unc_gdf["error_m"].dropna()
    ax.hist(vals, bins=20, color="#C49A6C", edgecolor="#F5F0E8", alpha=0.9)
    ax.axvline(UNC_GREEN,  color="#31A857", linestyle="--", lw=2, label=f"<{UNC_GREEN} m")
    ax.axvline(UNC_YELLOW, color="#F7E62C", linestyle="--", lw=2, label=f"<{UNC_YELLOW} m")
    ax.set_xlabel("Error (m)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.set_title("Uncertainty Distribution", fontsize=8)
    ax.legend(fontsize=6, labelcolor="#D8E4ED",
              facecolor="#0F1923", edgecolor="#1E3448")
    ax.grid(color="#1E3448", linewidth=0.4, linestyle=":")
    fig.tight_layout(pad=0.4)
    return fig


# ------------------------------------------------------------------------------
# RENDER FUNCTIONS
# ------------------------------------------------------------------------------

       
def render_dashboard_layout_1(map_col, right_col):
    """Render the main dashboard layout with left panel, map, and right panel"""
    
    # Load data (same as layout A)
    def safe_load(fn, label):
        try:
            return fn()
        except Exception as e:
            st.error(f"Could not load {label}: {e}")
            return gpd.GeoDataFrame()

    crest_gdf = safe_load(load_crest_lines, "crest lines")
    var_gdf   = safe_load(load_movement_points, "movement points")
    playa_gdf = safe_load(load_playa_polygons, "playa (Highest Purity)")
    unc_gdf   = safe_load(load_uncertainty_lines, "uncertainty lines")
    
    # NEW: Load GNSS and geomorphology data
    gnss_points_gdf = safe_load(load_gnss_points, "GNSS points")
    gnss_lines_gdf = safe_load(load_gnss_lines, "GNSS lines")
    geomorph_data = safe_load(load_geomorph_layers, "geomorphology layers")

    try:
        wind_df = load_wind_data()
    except Exception as e:
        st.error(f"Could not load wind data: {e}")
        wind_df = pd.DataFrame()

    try:
        hobo_df = load_hobo_wind_data()
        hobo_lat, hobo_lon = utm_to_latlon(533272.70433545333799, 7262367.944419549778104)
    except Exception as e:
        hobo_df = None
        hobo_lat = None
        hobo_lon = None
        st.warning(f"Could not load SOS 1 WEST data: {e}")

    base_metadata = load_base_imagery_metadata()

    if "dune_names" not in st.session_state:
        st.session_state["dune_names"] = (
            sorted(crest_gdf["dune_name"].dropna().unique())
            if "dune_name" in crest_gdf.columns else []
        )
    date_a = None
    date_b = None

    # ── LEFT PANEL ──────────────────────────────────────────────────────────
    with st.sidebar:
        
        # ── PRESETS ──────────────────────────────────────────────────────────────
        st.markdown('<div class="right-panel-header">Presets</div>', unsafe_allow_html=True)
        
        preset = st.radio(
            "Select View Mode",
            ["Compare", "Annual", "Monthly", "Custom"],
            key="b_preset",
            label_visibility="collapsed",
            horizontal=True
        )
        
        # ── DYNAMIC DATE SELECTION BASED ON PRESET ──────────────────────────────
        
        if preset == "Annual":
            # Show specific year with selectable month range (2-6 months)
            c1, c2 = st.columns(2)
            with c1:
                selected_year = st.selectbox(
                    "Year",
                    options=sorted(crest_gdf["year"].unique()),
                    key="b_annual_year"
                )
            with c2:
                st.markdown('<p style="font-size:0.7rem;color:var(--text-secondary);margin-top:20px;">Select Months (2-6)</p>', unsafe_allow_html=True)
            
            # Month range slider for Annual preset (default May-August)
            month_range = st.select_slider(
                "Months",
                options=MONTH_NAMES,
                value=(MONTH_NAMES[4], MONTH_NAMES[7]),
                key="b_annual_month_range",
                label_visibility="collapsed"
            )
            
            start_idx = MONTH_NAMES.index(month_range[0])
            end_idx = MONTH_NAMES.index(month_range[1])
            selected_months = MONTH_NAMES[start_idx:end_idx + 1]
            selected_years = [selected_year]
            
        elif preset == "Monthly":
            # Show specific month with selectable year range (2-6 years)
            c1, c2 = st.columns(2)
            with c1:
                selected_month = st.selectbox(
                    "Month",
                    options=MONTH_NAMES,
                    index=MONTH_NAMES.index("May"),
                    format_func=month_abbr,
                    key="b_monthly_month"
                )
            with c2:
                st.markdown('<p style="font-size:0.7rem;color:var(--text-secondary);margin-top:20px;">Select Years (2-6)</p>', unsafe_allow_html=True)
            
            # Year range slider for Monthly preset (default latest 5 years)
            sorted_years = sorted(ALL_YEARS)
            year_range = st.select_slider(
                "Years",
                options=sorted_years,
                value=(sorted_years[-5], sorted_years[-1]),
                key="b_monthly_year_range",
                label_visibility="collapsed"
            )
            
            start_idx = sorted_years.index(year_range[0])
            end_idx = sorted_years.index(year_range[1])
            selected_years = sorted_years[start_idx:end_idx + 1]
            selected_months = [selected_month]
            
        elif preset == "Compare":
            date_options = sorted(crest_gdf["date"].unique())
            date_strings = [d.strftime("%Y-%m-%d") for d in date_options]
            
            c1, c2 = st.columns(2)
            with c1:
                date_a_str = st.selectbox(
                    "Date A",
                    options=date_strings,
                    index=0,
                    key="b_compare_date_a"
                )
            
            date_a = pd.to_datetime(date_a_str)
            date_a_idx = date_options.index(date_a)
            
            # Auto-set Date B to the next available date
            if date_a_idx + 1 < len(date_options):
                date_b = date_options[date_a_idx + 1]
            else:
                date_b = date_a  # Fallback
            
            with c2:
                st.markdown(
                    f'<p style="font-size:0.85rem;color:var(--text-secondary);margin-top:28px;">'
                    f'<strong>Date B:</strong> {date_b.strftime("%Y-%m-%d")}</p>',
                    unsafe_allow_html=True
                )
            
            selected_years = list(set([date_a.year, date_b.year]))
            selected_months = list(set([date_a.strftime("%B"), date_b.strftime("%B")]))
            wind_pct, f_wind = wind_completeness_daterange(wind_df, date_a, date_b)

            
        else:  # Custom
            # Full control: year selection + month selection
            selected_years = st.multiselect(
                "Years",
                ALL_YEARS,
                default=ALL_YEARS,
                key="b_custom_years"
            )
            if not selected_years:
                st.warning("Select at least one year.")
                selected_years = ALL_YEARS
            selected_months = st.multiselect(
                "Months",
                MONTH_NAMES,
                default=DEFAULT_FOCUS_MONTHS,
                format_func=month_abbr,
                key="b_custom_months"
            )
            if not selected_months:
                st.warning("Select at least one month.")
                selected_months = DEFAULT_FOCUS_MONTHS

        # ── ZOOM TO FEATURE ─────────────────────────────────────────────
        st.markdown('<div class="right-panel-header">Zoom to Feature</div>', unsafe_allow_html=True)
        dune_names = st.session_state.get("dune_names", [])
        DEFAULT_DUNE = "The Star Dune"

        # Custom zoom targets with coordinates (convert DMS to decimal)
        custom_targets = [
            {"name": "Near Dune Corridor (only insitu layers)", "lat": -24.749, "lon": 15.397, "zoom": 20},
            {"name": "North NSS (only insitu layers)", "lat": -23.790, "lon": 15.189, "zoom": 20},
        ]

        # Create zoom options list
        zoom_options = ["All Features"] + [t["name"] for t in custom_targets] + dune_names

        default_index = (
            zoom_options.index(DEFAULT_DUNE)
            if DEFAULT_DUNE in zoom_options else 0
        )

        zoom_to = st.selectbox(
            "Zoom to feature", zoom_options,
            index=default_index,
            label_visibility="collapsed", 
            key="b_zoom_select"
        )
        # Store zoom selection in session state for later use
        st.session_state["zoom_to"] = zoom_to
        
        # ── LAYERS ──────────────────────────────────────────────────────────────
        st.markdown('<div class="right-panel-header">Layers</div>', unsafe_allow_html=True)
        with st.expander("  Remote Sensing Layers", expanded=False):
            show_crests = st.checkbox("Crest lines", value=True, key="b_show_crests")
            show_gap_fills = st.checkbox("  Gap fills", value=False, disabled=not show_crests, key="b_show_gap_fills")
            if preset == "Compare":
                show_movement = st.checkbox("Crest Movement", value=True, key="b_show_movement")
            else:
                show_movement = False
            show_playa = st.checkbox("Playa (Highest Purity)", value=True, key="b_show_playa")
            show_wind = st.checkbox("Wind rose overlay", value=True, key="b_show_wind")
            show_uncertainty = st.checkbox("Displacement Error Lines (Only March 2026)", value=False, key="b_show_uncertainty")
            show_margin_buffer = st.checkbox("Margin of Error Buffer (95% CI)", value=False, key="b_show_margin_buffer")
            show_base_imagery = st.checkbox("Base Imagery (Sentinel-2)", value=True, key="b_show_base_imagery")
        
        # ── NEW: IN-SITU LAYERS ──────────────────────────────────────────────
        with st.expander("  In-situ Layers", expanded=False):
            st.markdown('<div class="right-panel-header">From MARCH 2026</div>', unsafe_allow_html=True)
            show_gnss_points = st.checkbox("GNSS Survey Points", value=False, key="b_show_gnss_points")
            show_gnss_lines = st.checkbox("GNSS Crest/Edge/Bowl Lines", value=False, key="b_show_gnss_lines")
            show_geomorph_lines = st.checkbox("Erosion fossil dune / vlei deposits", value=False, key="b_show_geomorph_lines")
            show_geomorph_points = st.checkbox("Sediment Sample Points", value=False, key="b_show_geomorph_points")
            show_geomorph_polygons = st.checkbox("Recent Vlei / Fossil Dunes", value=False, key="b_show_geomorph_polygons")
            show_hobo_wind = st.checkbox("SOS 1 WEST Weather Station Wind Rose", value=False, key="b_show_hobo_wind")

        st.markdown('<div class="right-panel-header">Opacity</div>', unsafe_allow_html=True)
        opacity = st.slider("Layer opacity", 0.2, 1.0, 0.75, 0.05,
                            label_visibility="collapsed", key="b_opacity_slider")
        

    # ── FILTER DATA ───────────────────────────────────────────────────────────


        if preset == "Compare":
            f_crest = crest_gdf[crest_gdf["date"].isin([date_a, date_b])].copy()
            f_playa = playa_gdf[playa_gdf["date"].isin([date_a, date_b])].copy()
            f_var = var_gdf[var_gdf["date"].isin([date_a, date_b])].copy()

        else:
            m_nums = [ALL_MONTHS[m] for m in selected_months]
            
            def date_filter(gdf):
                if gdf.empty or "year" not in gdf.columns:
                    return gdf
                return gdf[gdf["year"].isin(selected_years) & gdf["month"].isin(m_nums)].copy()
            
            f_crest = date_filter(crest_gdf)
            f_playa = date_filter(playa_gdf)
            f_var = var_gdf.copy()

            wind_pct, f_wind = wind_completeness(wind_df, selected_years, selected_months)

    wind_b64 = None
    if show_wind and not f_wind.empty and wind_pct >= WIND_HIDE_PCT:
        wind_b64 = build_wind_rose_image(f_wind)
    elif show_wind and wind_pct < WIND_HIDE_PCT and not f_wind.empty:
        st.warning("Wind rose hidden: data coverage below 30% for selected period.")

    date_min = f_crest["date"].min() if not f_crest.empty and "date" in f_crest.columns else None
    date_max = f_crest["date"].max() if not f_crest.empty and "date" in f_crest.columns else None

    # ── MAP ──────────────────────────────────────────────────────────────────
    with map_col:
        folium_map = build_map(
            crest_gdf=f_crest, var_gdf=f_var, playa_gdf=f_playa, unc_gdf=unc_gdf,
            wind_b64=wind_b64, wind_completeness_pct=wind_pct,
            show_crests=show_crests, show_gap_fills=show_gap_fills,
            show_movement=show_movement, date_a=date_a, date_b=date_b,
            show_playa=show_playa, show_wind=show_wind, show_uncertainty=show_uncertainty,
            show_margin_buffer=show_margin_buffer,  # NEW: pass margin buffer flag
            opacity=opacity,
            date_min=date_min, date_max=date_max,
            show_base_imagery=show_base_imagery,
            base_metadata=base_metadata,
            selected_years=selected_years,
            selected_months=selected_months,
            preset=preset,
            # NEW: GNSS layer parameters
            show_gnss_points=show_gnss_points,
            show_gnss_lines=show_gnss_lines,
            show_geomorph_lines=show_geomorph_lines,
            show_geomorph_points=show_geomorph_points,
            show_geomorph_polygons=show_geomorph_polygons,
            gnss_points_gdf=gnss_points_gdf,
            gnss_lines_gdf=gnss_lines_gdf,
            geomorph_data=geomorph_data,
            hobo_df=hobo_df, 
            hobo_lat=hobo_lat, 
            hobo_lon=hobo_lon,  
            show_hobo_wind=show_hobo_wind,
        )

        # ── ZOOM TO FEATURE (executed after map creation) ──────────────
        zoom_to = st.session_state.get("zoom_to", "All Features")
        if zoom_to != "All Features":
            found = False
            
            # Check custom targets first
            for target in custom_targets:
                if target["name"] == zoom_to:
                    folium_map.location = [target["lat"], target["lon"]]
                    folium_map.zoom_start = target["zoom"]
                    found = True
                    break
            
            # If not found in custom targets, try dune names from crest data
            if not found and not f_crest.empty and "dune_name" in f_crest.columns:
                geoms = f_crest[f_crest["dune_name"] == zoom_to].geometry
                if not geoms.empty:
                    b = geoms.total_bounds
                    folium_map.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
                    found = True

        map_data = st_folium(
            folium_map, width="100%", height=80,
            returned_objects=["last_object_clicked"],
            key="b_folium_map"
        )
        

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────
    with right_col:
        
        if preset in ["Annual", "Monthly"]:
            if f_crest.empty:
                st.caption("No crest data available for wind roses.")
            elif wind_df.empty:
                st.caption("No wind data available.")
            else:
                pairs = get_wind_rose_pairs(f_crest, preset, selected_years, selected_months)
                
                if not pairs:
                    st.caption("Not enough consecutive dates to show wind roses.")
                else:
                    st.markdown('<div class="right-panel-header">Wind Between Crests</div>', unsafe_allow_html=True)
                    
                    # Show wind roses in 2-column grid
                    cols = st.columns(2)
                    roses_shown = 0
                    
                    for idx, (date_a, date_b, label) in enumerate(pairs):
                        img_b64, has_data = build_simple_wind_rose(wind_df, date_a, date_b)
                        
                        col_idx = idx % 2
                        with cols[col_idx]:
                            st.markdown(
                                f'<div style="text-align:center;font-size:0.6rem;color:#5C3D1E;font-weight:600;">{label}</div>',
                                unsafe_allow_html=True
                            )
                            if has_data and img_b64:
                                st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
                                roses_shown += 1
                            else:
                                st.markdown(
                                    f'<div style="text-align:center;font-size:0.55rem;color:#8B7A6A;padding:10px 0;border:1px dashed #C9BA9B;border-radius:4px;">No wind data</div>',
                                    unsafe_allow_html=True
                                )
                        
                        # New row after every 2 items
                        if idx % 2 == 1:
                            cols = st.columns(2)

                    if roses_shown > 0:
                        st.markdown("""
                            <div style="font-size:7px;color:#5C3D1E;font-weight:600;text-align:center;">Wind speed (m/s)</div>
                            <div style="display:flex;align-items:center;gap:4px;padding:0 5px;">
                                <span style="font-size:6px;">0</span>
                                <div style="flex:1;height:5px;background:linear-gradient(to right,#FFFFCC,#FFEDA0,#FED976,#FEB24C,#FD8D3C,#FC4E2A,#E31A1C,#B10026);border-radius:2px;"></div>
                                <span style="font-size:6px;">10+</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    if roses_shown == 0:
                        st.caption("No wind data available for any period.")

                    
        
        # Movement trend - only for Compare preset
        elif preset == "Compare":
            # movement trend
            st.markdown('<div class="right-panel-header">Movement Trend</div>', unsafe_allow_html=True)

            if (map_data and map_data.get("last_object_clicked")
                    and show_movement and not var_gdf.empty):
                click = map_data["last_object_clicked"]
                lat, lon = click.get("lat"), click.get("lng")
                if lat and lon:
                    click_pt = Point(lon, lat)
                    var_proj = var_gdf.to_crs("EPSG:3857")
                    click_gdf = gpd.GeoDataFrame(
                        geometry=[click_pt], crs="EPSG:4326"
                    ).to_crs("EPSG:3857")
                    dists = var_proj.geometry.distance(click_gdf.geometry.iloc[0])
                    nearest_pid = var_gdf.iloc[dists.idxmin()]["point_id"]

                    fig_ts = movement_trend_fig(var_gdf, nearest_pid)
                    st.pyplot(fig_ts, use_container_width=True)
                    plt.close(fig_ts)
                else:
                    st.caption("Click a point on the map.")
            else:
                st.caption("Click a movement point on the map. (Only available in Compare preset)")

        st.markdown('<div class="right-scroll">', unsafe_allow_html=True)

        # Wind coverage
        st.markdown('<div class="right-panel-header">Wind Coverage <div style="font-size:8px;color:#5C3D1E;font-family:Georgia,serif;margin-bottom:3px;letter-spacing:.06em;">Dieprivier station </div></div>', unsafe_allow_html=True)
        if not wind_df.empty:
            try:
                # Always show full coverage (all years, all months) regardless
                # of the current preset/selection.
                fig_g = build_gantt_figure(wind_df, ALL_YEARS, MONTH_NAMES)
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)
                if wind_pct < WIND_WARN_PCT:
                    st.markdown(
                        f'<div class="warn-box">! {wind_pct*100:.0f}% coverage</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.caption("Wind coverage chart unavailable for this selection.")
        else:
            st.caption("No wind data loaded.")

        st.markdown('<div class="right-panel-header">Export</div>', unsafe_allow_html=True)

        st.download_button(
                "Download Map (HTML)",
                data=folium_map._repr_html_(),
                file_name="dune_map.html",
                mime="text/html",
                use_container_width=True,
                key="b_download_map"
            )

        if not f_crest.empty:
            st.download_button(
                "Download Crests (CSV)",
                data=f_crest.drop(columns="geometry").to_csv(index=False),
                file_name="crest_lines_filtered.csv",
                mime="text/csv",
                use_container_width=True,
                key="b_download_crests"
            )
        else:
            st.button("Download Crests (CSV)", disabled=True, use_container_width=True, key="b_download_crests_disabled")



# ------------------------------------------------------------------------------
# FEEDBACK FORM  — midterm cartographic evaluation
# ------------------------------------------------------------------------------


def render_feedback_form():

    st.markdown('<div class="right-panel-header">FEEDBACK</div>', unsafe_allow_html=True)

    with st.expander("  Cartographic Evaluation — Share Your Feedback", expanded=False):

        st.markdown(
            f"""
            <div style="background:#FDF8F0;border:1px solid #C9BA9B;border-radius:6px;
                        padding:10px 14px;margin-bottom:12px;">
                <p style="font-size:.82rem;color:#5C3D1E;margin:0 0 6px 0;">
                This is an <strong>evaluation</strong> of a cartographic monitoring
                dashboard prototype for Namib Desert star dune dynamics (for MSc Cartography thesis).
                Your feedback directly shapes the next development iteration. Google Form: 
                <a href="https://docs.google.com/forms/d/e/1FAIpQLSeR5UQci2K1d2DTloQrwQmSWXkytGCt8sVqrw3lwT35LuE0dw/viewform?usp=publish-editor" target="_blank">Feedback Form</a>)
                </p>
                </p>
                <p style="font-size:.82rem;color:#5C3D1E;margin:0 0 6px 0;">
                Responses are <strong>fully anonymous</strong>, so no name or contact
                information is collected. Takes ~3–4 minutes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------

def main():

    st.divider()

    map_col, right_col = st.columns([4, 1.3])
    render_dashboard_layout_1(map_col, right_col)
    #render_feedback_form()



if __name__ == "__main__":
    main()