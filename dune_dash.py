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
ALL_MONTHS = {"May": 5, "June": 6, "July": 7, "August": 8}
MONTH_NAMES = list(ALL_MONTHS.keys())

UNC_GREEN  = 2.0
UNC_YELLOW = 6.0

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
    geom_df   = gdf[["point_id", "distance_along_m", "geometry"]].drop_duplicates("point_id")
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


@st.cache_data(show_spinner="Loading playa polygons ...")
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
    return df[df["month"].isin(ALL_MONTHS.values())]


@st.cache_data(show_spinner="Loading uncertainty points ...")
def load_uncertainty_lines():
    path = "main_data/multiline_uncertainty_perpendicular_lines.geojson"
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")
    if "uncertainty_m" in gdf.columns:
        gdf = gdf.rename(columns={"uncertainty_m": "error_m"})
    if "signed_distance_m" in gdf.columns:
        gdf["gnss_val"] = gdf["signed_distance_m"]
    if "left_or_right" in gdf.columns:
        gdf["detected_val"] = gdf["left_or_right"]
    return gdf


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def get_base_imagery_for_date(metadata, selected_years, selected_months, date_a, date_b, preset):
    """Determine which PNG to display based on selected date range"""
    
    if not metadata:
        return None, None, None
    
    if preset == "Compare":
        # Use date_a as the reference
        target_year = date_a.year
        target_month = date_a.month
    elif preset == "Annual":
        # Use the selected year, use May as default month
        target_year = selected_years[0] if selected_years else 2017
        target_month = 5  # May
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
    m_nums = [ALL_MONTHS[m] for m in months]
    sub = wind_df[wind_df["year"].isin(years) & wind_df["month"].isin(m_nums)]
    if sub.empty:
        return 0.0, sub
    days_per_month = {"May": 31, "June": 30, "July": 31, "August": 31}
    expected = sum(days_per_month[m] for y in years for m in months)
    if expected == 0:
        return 0.0, sub
    frac = min(sub["direction"].notna().sum() / expected, 1.0)
    return frac, sub


def build_wind_rose_image(wind_df):
    fig = plt.figure(figsize=(2.8, 2.8), facecolor=MPL_BG)
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wind_df["direction"], wind_df["speed_ms"],
           normed=True, opening=0.8, edgecolor=MPL_GRID,
           cmap=plt.cm.YlOrBr, bins=np.arange(0, 12, 2))
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_FG, labelsize=6)
    fig.patch.set_alpha(1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=MPL_BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_gantt_figure(wind_df, years, months):
    fig, ax = _dark_fig(4.2, max(1.8, len(years) * 0.38))
    m_nums  = [ALL_MONTHS[m] for m in months]
    days_in = {"May": 31, "June": 30, "July": 31, "August": 31}

    for i, year in enumerate(sorted(years)):
        for m_name, m_num in ALL_MONTHS.items():
            if m_num not in m_nums:
                continue
            sub   = wind_df[(wind_df.year == year) & (wind_df.month == m_num) &
                            (wind_df["direction"].notna())]
            days  = days_in[m_name]
            x0    = (m_num - 5) * 32
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
    ax.set_xticks([15.5, 47, 78.5, 110])
    ax.set_xticklabels(["May", "Jun", "Jul", "Aug"], fontsize=7, color=MPL_FG)
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
    opacity, date_min=None, date_max=None,
    show_base_imagery=True, 
    base_metadata=None,  
    selected_years=None, 
    selected_months=None,  
    preset="Custom", 
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
                weight=2,
                opacity= 0.3,
                tooltip=folium.Tooltip(
                    f"<b>GNSS Uncertainty</b><br>Error: {err:.2f} m"
                ),
            ).add_to(m)

    # 2. PLAYA
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

    # 3. CREST LINES
    if show_crests and not crest_gdf.empty:
        dates_sorted  = sorted(crest_gdf["date"].unique())
        date_color_map = dict(zip(
            [str(d) for d in dates_sorted], date_colormap(dates_sorted)
        ))
        for _, row in crest_gdf.iterrows():
            if row.get("is_gap_fill", False) and not show_gap_fills:
                continue
            c    = date_color_map.get(str(row["date"]), "#C9BA9B")
            dash = "8 4" if row.get("is_gap_fill", False) else None
            style = {"color": c, "weight": 2, "opacity": opacity, "dashArray": dash}
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, s=style: s,
                tooltip=folium.Tooltip(
                    f"<b>Crest</b><br>Date: {row['date'].date()}<br>"
                    f"Length: {row.get('length_m', 0):.0f} m"
                    + (" [gap fill]" if row.get("is_gap_fill") else "")
                ),
            ).add_to(m)

    # 4. movement POINTS
    if show_movement and date_a and date_b and not var_gdf.empty:
        def _agg(df, dt):
            sub = df[df["date"] == pd.Timestamp(dt)]
            return sub.groupby("point_id")["geometry"].first(), \
                   sub.groupby("point_id")["distance_m"].mean()

        geom_a, dist_a = _agg(var_gdf, date_a)
        geom_b, dist_b = _agg(var_gdf, date_b)
        common = dist_a.index.intersection(dist_b.index)

        for pid in common:
            val_a = float(dist_a[pid])
            val_b = float(dist_b[pid])
            geom  = geom_b[pid]
            diff  = val_b - val_a
            color = diverging_color(diff)
            radius = max(4, min(12, abs(diff) * 1.5))
            folium.CircleMarker(
                location=[float(geom.y), float(geom.x)],
                radius=radius, stroke=False,
                color=color, fill=True, fill_color=color, fill_opacity=opacity,
                tooltip=folium.Tooltip(
                    f"<b>Crest movement</b><br>Point: {pid}<br>"
                    f"Date A ({pd.Timestamp(date_a).date()}): {val_a:.2f} m<br>"
                    f"Date B ({pd.Timestamp(date_b).date()}): {val_b:.2f} m<br>"
                    f"<b>Change: {diff:+.2f} m</b>"
                ),
            ).add_to(m)

    # 5. WIND ROSE OVERLAY
    if show_wind and wind_b64:
        badge = ""
        if wind_completeness_pct < WIND_WARN_PCT:
            badge = (f'<div style="background:#FFF3CD;color:#6B4E00;font-size:9px;'
                     f'padding:2px 5px;border-radius:3px;margin-top:3px;font-weight:700;">'
                     f'! {wind_completeness_pct*100:.0f}% coverage</div>')
        html = f"""
        <div style="position:fixed;top:10px;right:10px;z-index:9999;
                    background:rgba(253,248,240,0.92);border:1px solid #C9BA9B;
                    border-radius:8px;padding:7px 8px;text-align:center;opacity:{opacity};">
          <div style="font-size:10px;color:#5C3D1E;font-weight:700;font-family:Georgia,serif;
                      margin-bottom:3px;letter-spacing:.06em;">WIND ROSE</div>
          <div style="font-size:9px;color:#5C3D1E;font-weight:700;font-family:Georgia,serif;
                      margin-bottom:3px;letter-spacing:.06em;">Dieprivier station <br> (~ 95km towords NE)</div>
          <img src="data:image/png;base64,{wind_b64}" width="130"/>
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

    # 6. LEGEND
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
            <span>Playa polygon</span>
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
            <svg width="14" height="14">
                <circle cx="7" cy="7" r="3" fill="none" stroke="#8B5E3C" stroke-width="1.5"/>
            </svg>
            <span style="font-size:8px;color:#5C3D1E;"> &lt; 2 m</span>
            </div>
            <div style="display:flex;align-items:center;gap:3px;">
            <svg width="18" height="18">
                <circle cx="9" cy="9" r="6" fill="none" stroke="#8B5E3C" stroke-width="1.5"/>
            </svg>
            <span style="font-size:8px;color:#5C3D1E;">2 m - 6 m</span>
            </div>
            <div style="display:flex;align-items:center;gap:3px;">
            <svg width="24" height="24">
                <circle cx="12" cy="12" r="9" fill="none" stroke="#8B5E3C" stroke-width="1.5"/>
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

    if show_uncertainty:
        sections.append("""
        <div class="ls">
        <div class="lt">GNSS Uncertainty</div>
        <div class="lr"><div style="width:14px;height:3px;background:#31A857;"></div>
            <span>&lt; 2 m</span></div>
        <div class="lr"><div style="width:14px;height:3px;background:#F7E62C;"></div>
            <span>2-6 m</span></div>
        <div class="lr"><div style="width:14px;height:3px;background:#C7400F;"></div>
            <span>&gt; 6 m</span></div>
        </div>""")

    if show_base_imagery and base_img_date:
        sections.append(f"""
        <div class="ls">
            <div class="lt">Base Imagery</div>
            <div class="lr">
                <span style="font-size:9px;color:#5C3D1E;font-weight:600;">{base_img_date}</span>
            </div>
        </div>""")

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
            if(c.style.display==='none'){{c.style.display='block';a.textContent='v';}}
            else{{c.style.display='none';a.textContent='>';}}">
            LEGEND <span id="da">v</span>
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
    ax.plot(trend["date"], trend["distance_m"],
            marker="o", markersize=3, color="#8B5E3C", linewidth=1.4)
    ax.axhline(0, color="#C9BA9B", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("Distance (m)", fontsize=7)
    ax.set_title(f"Point {nearest_pid}", fontsize=8)
    ax.grid(color="#C9BA9B", linewidth=0.4, linestyle=":")
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

       
def render_dashboard_layout_1(left_col, map_col, right_col):
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
    playa_gdf = safe_load(load_playa_polygons, "playa polygons")
    unc_gdf   = safe_load(load_uncertainty_lines, "uncertainty lines")

    try:
        wind_df = load_wind_data()
    except Exception as e:
        st.error(f"Could not load wind data: {e}")
        wind_df = pd.DataFrame()

    base_metadata = load_base_imagery_metadata()

    if "dune_names" not in st.session_state:
        st.session_state["dune_names"] = (
            sorted(crest_gdf["dune_name"].dropna().unique())
            if "dune_name" in crest_gdf.columns else []
        )
    date_a = None
    date_b = None

    # ── LEFT PANEL ──────────────────────────────────────────────────────────
    with left_col:
        
        # ── PRESETS ──────────────────────────────────────────────────────────────
        st.markdown('<div class="right-panel-header">Presets</div>', unsafe_allow_html=True)
        
        preset = st.radio(
            "Select View Mode",
            ["Annual", "Monthly", "Compare", "Custom"],
            key="b_preset",
            label_visibility="collapsed",
            horizontal=True
        )
        
        # ── DYNAMIC DATE SELECTION BASED ON PRESET ──────────────────────────────
        
        if preset == "Annual":
            # Show all years, all months for selected year
            c1, c2 = st.columns(2)
            with c1:
                selected_year = st.selectbox(
                    "Year",
                    options=sorted(crest_gdf["year"].unique()),
                    key="b_annual_year"
                )
            with c2:
                st.markdown('<p style="font-size:0.7rem;color:var(--text-secondary);margin-top:20px;">All Months (May-Aug)</p>', unsafe_allow_html=True)
            
            # Filter: specific year, all months
            selected_years = [selected_year]
            selected_months = MONTH_NAMES  # All months
            
        elif preset == "Monthly":
            # Show all years, specific month
            c1, c2 = st.columns(2)
            with c1:
                selected_month = st.selectbox(
                    "Month",
                    options=MONTH_NAMES,
                    key="b_monthly_month"
                )
            with c2:
                st.markdown('<p style="font-size:0.7rem;color:var(--text-secondary);margin-top:20px;">All Years (2017-2026)</p>', unsafe_allow_html=True)
            
            # Filter: all years, specific month
            selected_years = ALL_YEARS
            selected_months = [selected_month]
            
        elif preset == "Compare":
            # Two specific dates for comparison
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
            with c2:
                date_b_str = st.selectbox(
                    "Date B",
                    options=date_strings,
                    index=len(date_strings)-1 if len(date_strings) > 1 else 0,
                    key="b_compare_date_b"
                )
            
            date_a = pd.to_datetime(date_a_str)
            date_b = pd.to_datetime(date_b_str)
            
            # For crest data: only the two specific dates
            selected_years = list(set([date_a.year, date_b.year]))
            selected_months = list(set([date_a.strftime("%B"), date_b.strftime("%B")]))

            # For Compare: use the actual years and months from both dates
            selected_years = [date_a.year, date_b.year]
            selected_months = [date_a.strftime("%B"), date_b.strftime("%B")]
            
            # For wind data
            wind_pct, f_wind = wind_completeness(wind_df, selected_years, selected_months)
            
        else:  # Custom
            # Full control: year range + month selection
            year_range = st.slider(
                "Year Range",
                2017, 2026,
                (2017, 2026),
                step=1,
                key="b_custom_years"
            )

            st.caption("Bring the circles together to select one year.")
            selected_years = list(range(year_range[0], year_range[1] + 1))
            selected_months = st.multiselect(
                "Months",
                MONTH_NAMES,
                default=MONTH_NAMES,
                key="b_custom_months"
            )
            if not selected_months:
                st.warning("Select at least one month.")
                selected_months = MONTH_NAMES
            
        
        
        # ── LAYERS ──────────────────────────────────────────────────────────────
        disable_movement = preset in ["Annual", "Monthly", "Custom"]

        st.markdown('<div class="right-panel-header">Layers</div>', unsafe_allow_html=True)
        show_crests = st.checkbox("Crest lines", value=True, key="b_show_crests")
        show_gap_fills = st.checkbox("  Gap fills", value=False, disabled=not show_crests, key="b_show_gap_fills")
        show_movement = st.checkbox("Crest Movement", value=False, disabled=disable_movement, key="b_show_movement")
        if disable_movement:
            st.caption("Crest movement only available in Compare presets.")
        show_playa = st.checkbox("Playa polygons", value=True, key="b_show_playa")
        show_wind = st.checkbox("Wind rose overlay", value=True, key="b_show_wind")
        show_uncertainty = st.checkbox("Uncertainty lines", value=True, key="b_show_uncertainty")
        show_base_imagery = st.checkbox("Base Imagery (PNG)", value=True, key="b_show_base_imagery")

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
            opacity=opacity,
            date_min=date_min, date_max=date_max,
            show_base_imagery=show_base_imagery,  # NEW
            base_metadata=base_metadata,          # NEW
            selected_years=selected_years,        # NEW
            selected_months=selected_months,      # NEW
            preset=preset,   
        )

        # Zoom-to-feature dropdown
        st.markdown('<div class="right-panel-header">Zoom to Feature</div>', unsafe_allow_html=True)
        dune_names = st.session_state.get("dune_names", [])
        zoom_options = ["-- none --"] + dune_names
        zoom_to = st.selectbox("Zoom to feature", zoom_options, label_visibility="collapsed", key="b_zoom_select")
        if zoom_to != "-- none --" and not crest_gdf.empty and "dune_name" in crest_gdf.columns:
            geoms = crest_gdf[crest_gdf["dune_name"] == zoom_to].geometry
            if not geoms.empty:
                b = geoms.total_bounds
                folium_map.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

        map_data = st_folium(
            folium_map, width="100%", height=590,
            returned_objects=["last_object_clicked"],
            key="b_folium_map"
        )
        

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="right-scroll">', unsafe_allow_html=True)

        # Wind coverage
        st.markdown('<div class="right-panel-header">Wind Coverage</div>', unsafe_allow_html=True)
        if not wind_df.empty:
            fig_g = build_gantt_figure(wind_df, selected_years, selected_months)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)
            if wind_pct < WIND_WARN_PCT:
                st.markdown(
                    f'<div class="warn-box">! {wind_pct*100:.0f}% coverage</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No wind data loaded.")
        
        # movement trend
        st.markdown('<div class="right-panel-header">movement Trend</div>', unsafe_allow_html=True)

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

        # Uncertainty histogram
        st.markdown('<div class="right-panel-header">Uncertainty</div>', unsafe_allow_html=True)
        if not unc_gdf.empty and "error_m" in unc_gdf.columns:
            fig_h = uncertainty_hist_fig(unc_gdf)
            st.pyplot(fig_h, use_container_width=True)
            plt.close(fig_h)
        else:
            st.caption("No uncertainty data.")

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

FEEDBACK_PATH = "feedback_log.csv"

LIKERT = [
    "1 – Strongly disagree",
    "2 – Disagree",
    "3 – Neutral",
    "4 – Agree",
    "5 – Strongly agree",
]

def _get_anon_id():
    """Generate or retrieve a session-scoped anonymous ID."""
    if "anon_id" not in st.session_state:
        import random, string
        st.session_state["anon_id"] = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=6)
        )
    return st.session_state["anon_id"]


def save_feedback(record: dict):
    df_new = pd.DataFrame([record])
    if os.path.exists(FEEDBACK_PATH):
        df_new.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)


def render_feedback_form():
    anon_id = _get_anon_id()

    with st.expander("  Midterm Cartographic Evaluation — Share Your Feedback", expanded=False):

        st.markdown(
            f"""
            <div style="background:#FDF8F0;border:1px solid #C9BA9B;border-radius:6px;
                        padding:10px 14px;margin-bottom:12px;">
                <p style="font-size:.82rem;color:#5C3D1E;margin:0 0 6px 0;">
                This is a <strong>midterm evaluation</strong> of a cartographic monitoring
                dashboard for Namib Desert star dune dynamics (MSc Cartography thesis).
                Your feedback directly shapes the next development iteration.
                </p>
                <p style="font-size:.82rem;color:#5C3D1E;margin:0 0 6px 0;">
                Responses are <strong>fully anonymous</strong> — no name or contact
                information is collected. Takes ~3–4 minutes.
                </p>
                <p style="font-size:.78rem;color:#8B7A6A;margin:0;">
                Your anonymous session ID: <code style="background:#EDE6D3;
                padding:1px 6px;border-radius:3px;font-weight:700;
                color:#5C3D1E;">{anon_id}</code>
                — keep this if you want to follow up.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Profile ────────────────────────────────────────────────────────
        st.markdown("**About you** *(optional — helps contextualise responses)*")
        c1, c2 = st.columns(2)
        with c1:
            expertise = st.selectbox(
                "Cartography / GIS background",
                ["Prefer not to say", "None / General Public",
                 "Geodesy", "Photogrammetry",
                 "Remote Sensing", "Geoinformatics", "Cartography", "Other Academic"],
                key="fb_expertise",
            )
        with c2:
            prior_use = st.selectbox(
                "How familiar are you with this dashboard?",
                ["First time seeing it", "Explored it briefly",
                 "Used it multiple times"],
                key="fb_prior",
            )

        st.divider()

        # ── 1. Temporal encoding ───────────────────────────────────────────
        st.markdown("**1 · Temporal Encoding**")
        st.caption(
            "Time is encoded via a plasma colormap (purple → yellow) across "
            "crest lines and playa polygons."
        )
        q1a = st.select_slider(
            "The color progression clearly communicates the sequence of observations over time.",
            LIKERT, value="3 – Neutral", key="fb_q1a",
        )
        q1b = st.select_slider(
            "I could tell earlier from more recent observations without consulting the legend.",
            LIKERT, value="3 – Neutral", key="fb_q1b",
        )

        st.divider()

        # ── 2. Multi-layer legibility ──────────────────────────────────────
        st.markdown("**2 · Multi-layer Legibility & Visual Hierarchy**")
        st.caption(
            "The map overlays crest lines, playa polygons, GNSS uncertainty lines, "
            "and optionally movement points simultaneously."
        )
        q2a = st.select_slider(
            "I can distinguish between the different map layers without confusion.",
            LIKERT, value="3 – Neutral", key="fb_q2a",
        )
        q2b = st.select_slider(
            "The visual hierarchy (what stands out first) matches what I'd expect to be most important.",
            LIKERT, value="3 – Neutral", key="fb_q2b",
        )
        q2c = st.select_slider(
            "The legend is sufficient to interpret all map symbols.",
            LIKERT, value="3 – Neutral", key="fb_q2c",
        )

        q2d = st.select_slider(
            "The gap fill in crest lines helps in understanding line continuity despite data gaps.",
            LIKERT, value="3 – Neutral", key="fb_q2d",
        )

        st.divider()

        # ── 3. Interaction & presets ───────────────────────────────────────
        st.markdown("**3 · Interaction Design**")
        st.caption(
            "Four preset modes filter data: Annual, Monthly, Compare (two specific dates), Custom."
        )
        q3a = st.select_slider(
            "The preset modes were intuitive to use without instruction.",
            LIKERT, value="3 – Neutral", key="fb_q3a",
        )
        q3b = st.select_slider(
            "The Compare preset effectively supported understanding of dune change between dates.",
            LIKERT, value="3 – Neutral", key="fb_q3b",
        )

        st.divider()

        # ── 4. Uncertainty ─────────────────────────────────────────────────
        st.markdown("**4 · Uncertainty Visualization**")
        st.caption(
            "GNSS uncertainty shown as perpendicular lines: green < 2 m, yellow 2–6 m, red > 6 m."
        )
        q4a = st.select_slider(
            "The three-tier color coding for GNSS uncertainty was immediately understandable.",
            LIKERT, value="3 – Neutral", key="fb_q4a",
        )
        q4b = st.select_slider(
            "Showing uncertainty directly on the map (not just in a table) added value.",
            LIKERT, value="3 – Neutral", key="fb_q4b",
        )

        st.divider()

        # ── 5. Wind rose & Gantt ───────────────────────────────────────────
        st.markdown("**5 · Supplementary Graphics**")
        st.caption(
            "A wind rose overlays the map top-right; a coverage Gantt chart "
            "appears in the right panel."
        )
        q5a = st.select_slider(
            "The wind rose enhanced my understanding of the environmental context.",
            LIKERT, value="3 – Neutral", key="fb_q5a",
        )
        q5b = st.select_slider(
            "The wind coverage Gantt chart was useful for understanding data gaps.",
            LIKERT, value="3 – Neutral", key="fb_q5b",
        )

        st.divider()

        # ── 6. Aesthetics ──────────────────────────────────────────────────
        st.markdown("**6 · Aesthetics & Domain Suitability**")
        st.caption(
            "Earthy brown / sand palette chosen to reflect the Namib Desert environment."
        )
        q6a = st.select_slider(
            "The color palette felt appropriate for a desert geomorphology tool.",
            LIKERT, value="3 – Neutral", key="fb_q6a",
        )
        q6b = st.select_slider(
            "The visual design did not distract from interpreting the scientific data.",
            LIKERT, value="3 – Neutral", key="fb_q6b",
        )

        st.divider()

        # ── 7. Overall ─────────────────────────────────────────────────────
        st.markdown("**7 · Overall Assessment**")
        q7a = st.select_slider(
            "The dashboard effectively communicates how star dune morphology changed over 2017–2026.",
            LIKERT, value="3 – Neutral", key="fb_q7a",
        )
        q7b = st.select_slider(
            "Compared to raw GeoJSON/CSV files, this dashboard represents a meaningful advancement in data visualization.",
            LIKERT, value="3 – Neutral", key="fb_q7b",
        )

        st.divider()

        # ── 8. What's missing / broken (midterm focus) ─────────────────────
        st.markdown("**8 · Midterm Priorities** *(most important for your feedback)*")
        st.caption(
            "This is a midterm check — your input will directly shape what gets "
            "built or changed next."
        )

        missing = st.multiselect(
            "Which of the following would most improve the dashboard? *(select all that apply)*",
            [
                "Clearer temporal colormap / legend",
                "Better layer toggling / layer order control",
                "More intuitive preset modes",
                "Stronger uncertainty communication",
                "Improved wind data integration",
                "Mobile / smaller screen support",
                "Faster load times",
                "Additional export formats",
                "More context / explanatory text on the map",
                "A time-slider / animation feature",
                "Something else (describe below)",
            ],
            key="fb_missing",
        )

        c1, c2 = st.columns(2)
        with c1:
            improvement = st.text_area(
                "What would you change or add next?",
                height=90, key="fb_improve",
                placeholder="e.g. the plasma ramp is hard to read on satellite imagery...",
            )
        with c2:
            strength = st.text_area(
                "What is working well and should be kept?",
                height=90, key="fb_strength",
                placeholder="e.g. the Compare preset made change detection very intuitive...",
            )

        st.markdown("")
        submitted = st.button("Submit feedback", type="secondary", key="fb_submit")

        if submitted:
            if "fb_submitted" in st.session_state:
                st.info("You have already submitted feedback this session. Thank you!")
            else:
                record = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "anon_id": anon_id,
                    "expertise": expertise,
                    "prior_use": prior_use,
                    "temporal_colormap_order":   q1a,
                    "temporal_no_legend_needed": q1b,
                    "layer_distinction":         q2a,
                    "visual_hierarchy":          q2b,
                    "legend_sufficiency":        q2c,
                    "gap_fill_continuity":       q2d,
                    "preset_intuitive":          q3a,
                    "compare_effective":         q3b,
                    "uncertainty_color_clear":   q4a,
                    "uncertainty_on_map_value":  q4b,
                    "windrose_contextual":       q5a,
                    "gantt_gap_useful":          q5b,
                    "palette_appropriate":       q6a,
                    "design_not_distracting":    q6b,
                    "overall_change_communication": q7a,
                    "advancement_vs_raw_data":   q7b,
                    "midterm_priorities":        "; ".join(missing),
                    "improvement_comment":       improvement,
                    "strength_comment":          strength,
                }
                save_feedback(record)
                st.session_state["fb_submitted"] = True
                st.success(f"Thank you — response recorded. Your ID: **{anon_id}**")
                st.balloons()


# ------------------------------------------------------------------------------
# ADMIN PANEL  — password-gated, bottom of page
# ------------------------------------------------------------------------------

def render_admin_panel():
    with st.expander(" Admin — View & Export Feedback", expanded=False):
        pwd = st.text_input("Password", type="password", key="admin_pwd")
        correct = st.secrets.get("ADMIN_PASSWORD", "admin")  # fallback for local dev

        if pwd == correct:
            if os.path.exists(FEEDBACK_PATH):
                df = pd.read_csv(FEEDBACK_PATH)
                st.success(f"{len(df)} response(s) collected.")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇  Download feedback_log.csv",
                    data=df.to_csv(index=False),
                    file_name="feedback_log.csv",
                    mime="text/csv",
                    key="admin_download",
                )
            else:
                st.info("No responses yet.")
        elif pwd:
            st.error("Incorrect password.")

# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------

def main():
    # Page title with padding
    st.markdown(
        '<div style="padding-top: 8px;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 style="margin:0 0 2px 0;color:#5C3D1E;">Star Dune Dynamics</h1>'
        '<p style="color:#8B7A6A;font-size:.78rem;margin:0 0 4px 0;'
        'font-family:monospace;">Aeolian crest monitoring  Namib  2017-2026  May-Aug</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    left_col, map_col, right_col = st.columns([1.2, 3.5, 1.3], gap="small")
    render_dashboard_layout_1(left_col, map_col, right_col)

    st.divider()
    render_feedback_form()
    st.divider()
    render_admin_panel()


if __name__ == "__main__":
    main()