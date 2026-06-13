"""
============================================================
 DUNE DASHBOARD  –  Streamlit app
 Aeolian dune monitoring: crest lines, variability, playas,
 wind roses, and GNSS uncertainty (2017-2026, May-August)
============================================================

REQUIRED LIBRARIES:
    pip install streamlit folium streamlit-folium geopandas
               pandas numpy matplotlib windrose

DATA PREPARATION NOTES:
  All datasets should be pre-processed (clipped, projected) before
  loading here.  This script expects WGS-84 (EPSG:4326) geometries.
  Edit every block marked  <- EDIT PATH  and  <- EDIT FIELD  below.
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

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
import io, base64, json
from datetime import date
from shapely.geometry import Point
from folium import plugins

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Star Dune Dynamics Dashboard",
    page_icon= None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CUSTOM CSS  – earthy sand/desert palette
# ------------------------------------------------------------------------------
st.markdown("""
<style>
  /* Main background */
  [data-testid="stAppViewContainer"] { background: #F5F0E8; }
  [data-testid="stSidebar"]           { background: #EDE6D3; border-right: 1px solid #C9BA9B; }

  /* Headings */
  h1, h2, h3 { font-family: 'Georgia', serif; color: #3B2F1E; }
  h1 { font-size: 1.6rem; letter-spacing: .03em; }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: #FDF8F0;
      border: 1px solid #C9BA9B;
      border-radius: 6px;
      padding: 8px;
  }

  /* Divider colour */
  hr { border-color: #C9BA9B; }

  /* Toggle labels */
  label { color: #4A3728 !important; font-size: .9rem; }

  /* Warning / info boxes */
  .warn-box {
      background: #FFF3CD; border-left: 4px solid #E6A817;
      padding: 8px 12px; border-radius: 4px; margin: 4px 0;
      font-size: .85rem; color: #6B4E00;
  }
  .info-box {
      background: #D4EDDA; border-left: 4px solid #28A745;
      padding: 8px 12px; border-radius: 4px; margin: 4px 0;
      font-size: .85rem; color: #155724;
  }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
ALL_YEARS  = list(range(2017, 2027))          # 2017 – 2026 inclusive
ALL_MONTHS = {"May": 5, "June": 6, "July": 7, "August": 8}
MONTH_NAMES = list(ALL_MONTHS.keys())

# Uncertainty colour thresholds (metres)
UNC_GREEN  = 2.0
UNC_YELLOW = 6.0

# Wind rose completeness thresholds
WIND_WARN_PCT  = 0.70   # show warning badge below this
WIND_HIDE_PCT  = 0.30   # hide wind rose entirely below this

# Default map centre  <- EDIT: set to your dune field coordinates
MAP_CENTER = [-24.76, 15.31]   # <- EDIT: [latitude, longitude]
MAP_ZOOM   = 14

# ------------------------------------------------------------------------------
# -- DATA LOADING  (cached so Streamlit doesn't reload on every interaction) ---
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading crest lines …")
def load_crest_lines():
    """
    GeoDataFrame of dune crest lines, one row per acquisition date.
    """
    # --- <- EDIT PATH ----------------------------------------------------------
    path = "main_data/extended_centerlines.geojson"
    # --------------------------------------------------------------------------
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")

    # <- EDIT FIELD: replace "acquisition_date" with your actual date column name
    gdf["date"] = pd.to_datetime(gdf["acquisition_date"])
    gdf["year"]  = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month

    # Ensure required columns exist; set defaults if absent
    if "is_gap_fill" not in gdf.columns:
        gdf["is_gap_fill"] = gdf["type"] == "connection"
    if "length_m" not in gdf.columns:
        gdf["length_m"] = gdf.geometry.length * 111_320  # rough degrees->metres

    return gdf


@st.cache_data(show_spinner="Loading variability points …")
def load_variability_points():
    """
    Loads centerline_points.geojson (wide format) and melts to long format.
    """
    # --- <- EDIT PATH ----------------------------------------------------------
    path = "main_data/centerline_points.geojson"
    # --------------------------------------------------------------------------
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")

    # Build a truly unique point_id from point_id + distance_along_m
    gdf["point_id"] = gdf["point_id"].astype(str)

    # Identify all wide-format date columns (date_YYYY_MM_DD)
    date_cols = [c for c in gdf.columns if c.startswith("date_")]

    # Keep geometry separate before melt (melt drops it)
    geom_df = gdf[["point_id", "distance_along_m", "geometry"]].drop_duplicates("point_id")

    # Melt wide -> long
    df_long = gdf[["point_id", "distance_along_m"] + date_cols].melt(
        id_vars=["point_id", "distance_along_m"],
        var_name="date_col",
        value_name="distance_m",
    )

    # Drop rows where no measurement exists for that date
    df_long = df_long.dropna(subset=["distance_m"])

    # Convert column name "date_2017_05_20" -> datetime 2017-05-20
    df_long["date"] = pd.to_datetime(
        df_long["date_col"].str.replace("date_", "", regex=False).str.replace("_", "-"),
        format="%Y-%m-%d",
    )
    df_long["year"]  = df_long["date"].dt.year
    df_long["month"] = df_long["date"].dt.month

    # Re-attach geometry
    df_long = df_long.merge(geom_df, on=["point_id", "distance_along_m"], how="left")
    result = gpd.GeoDataFrame(df_long, geometry="geometry", crs="EPSG:4326")

    return result


@st.cache_data(show_spinner="Loading playa polygons …")
def load_playa_polygons():
    """
    GeoDataFrame of playa (dry lake) outline polygons per date.
    """
    # --- <- EDIT PATH ----------------------------------------------------------
    path = "main_data/merged_playa.geojson"
    # --------------------------------------------------------------------------
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")

    gdf["date"] = pd.to_datetime(gdf["acquisition_date"])
    gdf["year"]  = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month

    if "area_m2" not in gdf.columns:
        gdf["area_m2"] = gdf.geometry.area * (111_320 ** 2)  # rough conversion

    return gdf


@st.cache_data(show_spinner="Loading wind data …")
def load_wind_data():
    """
    DataFrame of wind speed & direction observations.
    """
    # --- <- EDIT PATH ----------------------------------------------------------
    path = "main_data/combined_weather_with_location.csv"
    # --------------------------------------------------------------------------
    df = pd.read_csv(path, sep=";")

    # <- EDIT FIELD: rename your columns to match expected names
    df = df.rename(columns={
        "Date": "datetime",
        "Wind speed  (vc avg)": "speed_ms",
        "Wind  direction  (vc avg)": "direction",
    })

    df["datetime"] = pd.to_datetime(df["datetime"], format="%d %b %Y")
    df["year"]  = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df = df[df["month"].isin(ALL_MONTHS.values())]

    return df


@st.cache_data(show_spinner="Loading uncertainty points …")
def load_uncertainty_points():
    """
    Static GeoDataFrame of GNSS vs. detected crest comparison points.
    """
    # --- <- EDIT PATH ----------------------------------------------------------
    path = "main_data/multiline_uncertainty_uncertainty_points.geojson"
    # --------------------------------------------------------------------------
    gdf = gpd.read_file(path)
    
    # Remove Z values from coordinates (convert 3D to 2D)
    gdf.geometry = gdf.geometry.apply(lambda geom: Point(geom.x, geom.y))
    
    gdf = gdf.to_crs("EPSG:4326")

    # <- EDIT FIELD: rename your error column to "error_m"
    if "uncertainty_m" in gdf.columns:
        gdf = gdf.rename(columns={"uncertainty_m": "error_m"})
    
    # Add required fields if they exist in your data
    if "signed_distance_m" in gdf.columns:
        gdf["gnss_val"] = gdf["signed_distance_m"]
    if "left_or_right" in gdf.columns:
        gdf["detected_val"] = gdf["left_or_right"]

    return gdf


# ------------------------------------------------------------------------------
# HELPER UTILITIES
# ------------------------------------------------------------------------------

def date_colormap(dates):
    """Map a sequence of dates to hex colours (viridis over time)."""
    timestamps = pd.to_datetime(dates).astype(np.int64)
    norm = plt.Normalize(timestamps.min(), timestamps.max())
    cmap = plt.cm.plasma
    return [mcolors.to_hex(cmap(norm(t))) for t in timestamps]


def diverging_color(value, vmin=-10, vmax=10):
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.PiYG 
    r, g, b, a = cmap(norm(value))
    return mcolors.to_hex((r, g, b))



def unc_color(error_m):
    if error_m < UNC_GREEN:  return "#28A745"
    if error_m < UNC_YELLOW: return "#FFC107"
    return "#DC3545"


def wind_completeness(wind_df, years, months):
    """Return (fraction_complete, filtered_df) for the given date filter."""
    m_nums = [ALL_MONTHS[m] for m in months]
    sub = wind_df[wind_df["year"].isin(years) & wind_df["month"].isin(m_nums)]
    if sub.empty:
        return 0.0, sub
    # Expected daily rows for each month-year combination
    days_per_month = {"May": 31, "June": 30, "July": 31, "August": 31}
    expected = sum(days_per_month[m] for y in years for m in months)
    if expected == 0:
        return 0.0, sub
    # Use direction column completeness (not just date presence)
    valid_direction = sub["direction"].notna().sum()
    frac = min(valid_direction / expected, 1.0)
    return frac, sub


def build_wind_rose_image(wind_df):
    """Return a base64 PNG of a wind rose from the given wind DataFrame."""
    try:
        from windrose import WindroseAxes
        fig = plt.figure(figsize=(3, 3), facecolor="none")
        ax  = WindroseAxes.from_ax(fig=fig)
        ax.bar(wind_df["direction"], wind_df["speed_ms"],
               normed=True, opening=0.8, edgecolor="white",
               cmap=plt.cm.YlOrRd, bins=np.arange(0, 12, 2))
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
    except ImportError:
        # Fallback: simple polar bar chart
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                               figsize=(3, 3), facecolor="none")
        bins  = np.linspace(0, 2*np.pi, 17)
        dirs  = np.deg2rad(wind_df["direction"].dropna())
        counts, _ = np.histogram(dirs, bins=bins)
        theta = (bins[:-1] + bins[1:]) / 2
        ax.bar(theta, counts, width=np.diff(bins), color="#E6A817",
               alpha=0.8, edgecolor="white")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_gantt_figure(wind_df, years, months):
    """Horizontal Gantt chart: green = data present, red = gap."""
    fig, ax = plt.subplots(figsize=(9, max(2, len(years)*0.45)),
                           facecolor="#F5F0E8")
    m_nums = [ALL_MONTHS[m] for m in months]

    for i, year in enumerate(sorted(years)):
        for m_name, m_num in ALL_MONTHS.items():
            if m_num not in m_nums:
                continue
            sub = wind_df[(wind_df.year==year) & (wind_df.month==m_num) & (wind_df["direction"].notna())]
            days = {"May":31,"June":30,"July":31,"August":31}[m_name]
            x_start = (m_num - 5) * 31                         # rough x offset
            color = "#28A745" if len(sub) > 0 else "#DC3545"
            pct = (len(sub) / days) * 100
            total_days = days
            valid_days = len(sub)
            missing_days = total_days - valid_days

           # Green segment
            ax.barh(i, valid_days, left=x_start, height=0.6, color="#28A745", alpha=0.8)

            if valid_days > 0:
                ax.text(x_start + total_days/2, i, f"{pct:.0f}%",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")

            # Red segment
            if missing_days > 0:
                ax.barh(i, missing_days, left=x_start + valid_days, height=0.6, color="#DC3545", alpha=0.8)

    ax.set_yticks(range(len(sorted(years))))
    ax.set_yticklabels(sorted(years), fontsize=9)
    # Calculate center positions
    bar_centers = [15.5, 46, 77.5, 108.5]  # or compute dynamically
    ax.set_xticks(bar_centers)
    ax.set_xticklabels(["May", "Jun", "Jul", "Aug"],fontsize=9)
    ax.set_title("Wind Data Coverage (green = present, red = gap)",
                 fontsize=10, color="#3B2F1E", pad=6)
    ax.set_facecolor("#F5F0E8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#C9BA9B")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# MAP BUILDING
# ------------------------------------------------------------------------------

def build_map(
    crest_gdf, var_gdf, playa_gdf, unc_gdf,
    wind_b64, wind_completeness_pct,
    show_crests, show_gap_fills,
    show_variability, date_a, date_b,
    show_playa, show_wind, show_uncertainty,
    opacity
):
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles=None,
        control_scale=True,
    )

    plugins.MeasureControl(
        position='bottomright',    # <- change this
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares',
        active_color='#FF0000',
        completed_color='#00FF00'
    ).add_to(m)

    # Add a tile layer that supports zoom beyond native resolution
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        max_native_zoom=16,
        max_zoom=22,
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # -- 1. UNCERTAINTY POINTS (bottom) ----------------------------------------
    if show_uncertainty and not unc_gdf.empty:
        for _, row in unc_gdf.iterrows():
            err = row.get("error_m", 0)
            if pd.isna(err):
                continue
            c   = unc_color(err)
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=6,
                color=c, fill=True, fill_color=c, fill_opacity=1.0,
                tooltip=folium.Tooltip(
                    f"<b>GNSS Uncertainty</b><br>"
                    f"Error: {err:.2f} m<br>"
                    f"GNSS: {row.get('gnss_val','—')}<br>"
                    f"Detected: {row.get('detected_val','—')}"
                ),
            ).add_to(m)

    # -- 2. PLAYA POLYGONS ----------------------------------------------------
    if show_playa and not playa_gdf.empty:
        dates_sorted = sorted(playa_gdf["date"].unique())
        colors = date_colormap(dates_sorted)
        date_color_map = dict(zip([str(d) for d in dates_sorted], colors))

        for _, row in playa_gdf.iterrows():
            c = date_color_map.get(str(row["date"]), "#AAAAAA")
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, col=c: {
                    "fillColor": col, "color": col,
                    "weight": 1, "fillOpacity": opacity * 0.55,
                },
                tooltip=folium.Tooltip(
                    f"<b>Playa</b><br>Date: {row['date'].date()}<br>"
                    f"Area: {row.get('area_m2', 0)/1e6:.3f} km²"
                ),
            ).add_to(m)

    # -- 3. CREST LINES -------------------------------------------------------
    if show_crests and not crest_gdf.empty:
        dates_sorted = sorted(crest_gdf["date"].unique())
        colors = date_colormap(dates_sorted)
        date_color_map = dict(zip([str(d) for d in dates_sorted], colors))

        for _, row in crest_gdf.iterrows():
            if row.get("is_gap_fill", False) and not show_gap_fills:
                continue
            c    = date_color_map.get(str(row["date"]), "#FFFFFF")
            dash = "8 4" if row.get("is_gap_fill", False) else None
            style = {
                "color": c,
                "weight": 2,
                "opacity": opacity,
                "dashArray": dash,
            }
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, s=style: s,
                tooltip=folium.Tooltip(
                    f"<b>Crest</b><br>Date: {row['date'].date()}<br>"
                    f"Length: {row.get('length_m', 0):.0f} m"
                    + (" [gap fill]" if row.get("is_gap_fill") else "")
                ),
            ).add_to(m)

    # -- 4. VARIABILITY POINTS ------------------------------------------------
    if show_variability and date_a and date_b and not var_gdf.empty:
        # Aggregate to one row per point_id per date (mean if somehow duplicated)
        def _agg_date(df, dt):
            sub = df[df["date"] == pd.Timestamp(dt)].copy()
            # Take the first geometry and mean distance per unique point_id
            geom_map = sub.groupby("point_id")["geometry"].first()
            dist_map = sub.groupby("point_id")["distance_m"].mean()
            return geom_map, dist_map

        geom_a, dist_a = _agg_date(var_gdf, date_a)
        geom_b, dist_b = _agg_date(var_gdf, date_b)
        common = dist_a.index.intersection(dist_b.index)

        for pid in common:
            val_a  = float(dist_a[pid])
            val_b  = float(dist_b[pid])
            geom   = geom_b[pid]          # use Date B geometry for placement
            diff   = val_b - val_a
            color  = diverging_color(diff)
            radius = max(4, min(12, abs(diff) * 1.5))

            folium.CircleMarker(
                location=[float(geom.y), float(geom.x)],
                radius=radius,
                stroke=False, 
                color=color,
                fill=True, fill_color=color,
                fill_opacity=opacity,
                tooltip=folium.Tooltip(
                    f"<b>Crest variability</b><br>"
                    f"Point: {pid}<br>"
                    f"Date A ({pd.Timestamp(date_a).date()}): {val_a:.2f} m<br>"
                    f"Date B ({pd.Timestamp(date_b).date()}): {val_b:.2f} m<br>"
                    f"<b>Δ: {diff:+.2f} m</b>"
                ),
            ).add_to(m)

    # -- 5. WIND ROSE (floating HTML overlay) ---------------------------------
    if show_wind and wind_b64:
        badge = ""
        if wind_completeness_pct < WIND_WARN_PCT:
            badge = (f'<div style="background:#FFC107;color:#000;font-size:10px;'
                     f'padding:2px 5px;border-radius:3px;margin-top:3px;">'
                     f'⚠ {wind_completeness_pct*100:.0f}% data coverage</div>')
        html = f"""
        <div style="position:fixed;top:10px;right:10px;z-index:9999;
                    background:rgba(253,248,240,.85);border:1px solid #C9BA9B;
                    border-radius:8px;padding:6px;text-align:center;
                    opacity:{opacity};">
          <div style="font-size:11px;color:#3B2F1E;font-weight:bold;
                      margin-bottom:3px;">Wind Rose</div>
          <img src="data:image/png;base64,{wind_b64}" width="130" />
          <!-- Wind speed colour scale -->
          <div style="margin-top:5px;font-size:9px;color:#3B2F1E;font-weight:bold;">
            Wind speed (m/s)
          </div>
          <div style="display:flex;align-items:center;margin-top:2px;gap:1px;">
            <span style="font-size:8px;color:#555;">0</span>
            <div style="flex:1;height:8px;background:linear-gradient(to right,
              #FFFFCC,#FFEDA0,#FED976,#FEB24C,#FD8D3C,#FC4E2A,#E31A1C,#B10026);
              border-radius:2px;margin:0 2px;"></div>
            <span style="font-size:8px;color:#555;">10+</span>
          </div>
          {badge}
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))

    # -- 6. COLLAPSIBLE AUTO-UPDATING LEGEND ----------------------------------
    # Build legend sections only for active layers
    legend_sections = []

    if show_crests:
        legend_sections.append("""
        <div class="leg-section">
            <div class="leg-title">Crest Lines</div>
            <div class="leg-row">
            <svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5"
                stroke="#F0C060" stroke-width="2.5"/></svg>
            <span>Detected crest</span>
            </div>
            <div class="leg-row">
            <svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5"
                stroke="#F0C060" stroke-width="2" stroke-dasharray="5,3"/></svg>
            <span>Gap fill (automated)</span>
            </div>
            <div class="leg-row" style="font-size:9px;color:#888;margin-top:2px;">
            Colour = acquisition date
            </div>
            <div style="display:flex;align-items:center;gap:3px;margin:3px 0;">
            <span style="font-size:9px;color:#0D0887;">Early</span>
            <div style="flex:1;height:8px;background:linear-gradient(to right,
                #0D0887,#9C179E,#ED7953,#F0F921);
                border-radius:2px;"></div>
            <span style="font-size:9px;color:#ED7953;">Late</span>
            </div>
        </div>""")

    if show_variability:
        legend_sections.append("""
          <div class="leg-section">
            <div class="leg-title">Crest Variability Δ (m)</div>
            <div style="display:flex;align-items:center;gap:3px;margin:3px 0;">
              <span style="font-size:9px;color:#D62646;">−10</span>
              <div style="flex:1;height:8px;background:linear-gradient(to right,
                #D62646,#F4B3C2,#FFFDE0,#D9F0D9,#008F48);
                border-radius:2px;"></div>
              <span style="font-size:9px;color:#008F48;">+10</span>
            </div>
            <div class="leg-row" style="font-size:9px;color:#888;">
              Pink = retreat &nbsp;|&nbsp; Green = advance
            </div>
            <div class="leg-row" style="font-size:9px;color:#888;">
              Circle size ∝ magnitude of change
            </div>
          </div>""")

    if show_playa:
        legend_sections.append("""
          <div class="leg-section">
            <div class="leg-title">Playa Polygons</div>
            <div class="leg-row">
              <div style="width:14px;height:14px;background:linear-gradient(135deg,
                #0D0887,#9C179E,#ED7953,#F0F921);border-radius:2px;opacity:0.7;"></div>
              <span>Fill colour = date (plasma scale)</span>
            </div>
          </div>""")

    if show_uncertainty:
        legend_sections.append("""
          <div class="leg-section">
            <div class="leg-title">GNSS Uncertainty</div>
            <div class="leg-row">
              <div style="width:12px;height:12px;border-radius:50%;
                background:#28A745;border:1px solid #fff;"></div>
              <span>&lt; 2 m &nbsp;(good)</span>
            </div>
            <div class="leg-row">
              <div style="width:12px;height:12px;border-radius:50%;
                background:#FFC107;border:1px solid #fff;"></div>
              <span>2 – 6 m &nbsp;(moderate)</span>
            </div>
            <div class="leg-row">
              <div style="width:12px;height:12px;border-radius:50%;
                background:#DC3545;border:1px solid #fff;"></div>
              <span>&gt; 6 m &nbsp;(poor)</span>
            </div>
          </div>""")

    if show_wind:
        legend_sections.append("""
          <div class="leg-section">
            <div class="leg-title">Wind Speed Scale</div>
            <div style="display:flex;align-items:center;gap:3px;margin:3px 0;">
              <span style="font-size:9px;color:#555;">Calm</span>
              <div style="flex:1;height:8px;background:linear-gradient(to right,
                #FFFFCC,#FED976,#FD8D3C,#E31A1C,#800026);
                border-radius:2px;"></div>
              <span style="font-size:9px;color:#555;">Strong</span>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-size:8px;color:#888;margin-top:1px;">
              <span>0 m/s</span><span>2</span><span>4</span>
              <span>6</span><span>8</span><span>10+ m/s</span>
            </div>
          </div>""")

    # Only render legend if at least one layer is active
    if legend_sections:
        sections_html = "".join(legend_sections)
        legend_html = f"""
        <style>
          #dune-legend {{ position:fixed; bottom:50px; left:10px; z-index:9998;
            font-family:Arial,sans-serif; font-size:11px; color:#3B2F1E; }}
          #legend-box {{ background:rgba(253,248,240,.92);
            border:1px solid #C9BA9B; border-radius:8px;
            padding:8px 10px; min-width:170px; max-width:210px;
            box-shadow:0 1px 4px rgba(0,0,0,.15); }}
          #legend-toggle {{ cursor:pointer; user-select:none;
            font-weight:bold; font-size:12px; color:#3B2F1E;
            display:flex; justify-content:space-between; align-items:center; }}
          #legend-toggle:hover {{ color:#8B5E3C; }}
          #legend-content {{ margin-top:6px; }}
          .leg-section {{ margin-bottom:8px; padding-bottom:6px;
            border-bottom:1px solid #E0D5C0; }}
          .leg-section:last-child {{ border-bottom:none; margin-bottom:0; }}
          .leg-title {{ font-weight:bold; font-size:10px;
            color:#5C3D1E; margin-bottom:4px; text-transform:uppercase;
            letter-spacing:.04em; }}
          .leg-row {{ display:flex; align-items:center; gap:6px;
            margin-bottom:3px; font-size:10px; }}
        </style>
        <div id="dune-legend">
          <div id="legend-box">
            <div id="legend-toggle" onclick="
              var c=document.getElementById('legend-content');
              var a=document.getElementById('leg-arrow');
              if(c.style.display==='none'){{c.style.display='block';a.textContent='▾';}}
              else{{c.style.display='none';a.textContent='▸';}}">
              Legend <span id="leg-arrow">▾</span>
            </div>
            <div id="legend-content">
              {sections_html}
            </div>
          </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    # -- LAYER CONTROL --------------------------------------------------------
    return m


# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------

def main():
    st.markdown("## Star Dune Dynamics Dashboard")
    st.markdown("Aeolian crest monitoring · 2017–2026 · May–August")
    st.divider()

    # -- Load data -------------------------------------------------------------
    try:
        crest_gdf = load_crest_lines()
        
        if "dune_name" in crest_gdf.columns:
            st.session_state["dune_names"] = sorted(crest_gdf["dune_name"].dropna().unique())
        
    except Exception as e:
        st.error(f"Could not load crest lines: {e}")
        crest_gdf = gpd.GeoDataFrame()
        st.session_state["dune_names"] = []

    try:
        var_gdf = load_variability_points()
    except Exception as e:
        st.error(f"Could not load variability points: {e}")
        var_gdf = gpd.GeoDataFrame()

    try:
        playa_gdf = load_playa_polygons()
    except Exception as e:
        st.error(f"Could not load playa polygons: {e}")
        playa_gdf = gpd.GeoDataFrame()

    try:
        wind_df = load_wind_data()
    except Exception as e:
        st.error(f"Could not load wind data: {e}")
        wind_df = pd.DataFrame()

    try:
        unc_gdf = load_uncertainty_points()
    except Exception as e:
        st.error(f"Could not load uncertainty points: {e}")
        unc_gdf = gpd.GeoDataFrame()

    # -- SIDEBAR ---------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Date Range")

        year_range = st.slider(
            "Years", min_value=2017, max_value=2026,
            value=(2017, 2026), step=1
        )
        selected_years = list(range(year_range[0], year_range[1]+1))

        selected_months = st.multiselect(
            "Months", options=MONTH_NAMES,
            default=MONTH_NAMES
        )
        if not selected_months:
            st.warning("Select at least one month.")
            selected_months = MONTH_NAMES

        st.divider()
        st.markdown("### Layer Toggles")

        show_crests      = st.checkbox("Crest lines",       value=True)
        show_gap_fills   = st.checkbox("  ↳ Show gap fills", value=True,
                                        disabled=not show_crests)

        show_variability = st.checkbox("Crest variability", value=True)

        # Variability date comparison
        var_dates = []
        if show_variability and not var_gdf.empty:
            var_dates = sorted(var_gdf["date"].unique())
            if len(var_dates) >= 2:
                date_a_idx = st.select_slider(
                    "Date A", options=range(len(var_dates)),
                    value=0,
                    format_func=lambda i: str(var_dates[i].date())
                )
                date_b_idx = st.select_slider(
                    "Date B", options=range(len(var_dates)),
                    value=len(var_dates)-1,
                    format_func=lambda i: str(var_dates[i].date())
                )
                date_a = var_dates[date_a_idx]
                date_b = var_dates[date_b_idx]
            else:
                st.caption("Need ≥2 variability dates.")
                date_a = date_b = None
        else:
            date_a = date_b = None

        show_playa   = st.checkbox("Playa polygons",   value=True)
        show_wind    = st.checkbox("Wind rose overlay", value=True)
        show_uncertainty = st.checkbox("Uncertainty points", value=True)

        st.divider()
        st.markdown("### Global Opacity")
        opacity = st.slider("Opacity", 0.3, 1.0, 0.7, 0.05)

        st.divider()
        st.markdown("### Zoom to Feature")

        # Get dune names from session state
        dune_names = st.session_state.get("dune_names", [])
        if dune_names:
            zoom_options = ["— none —"] + dune_names
            zoom_to = st.selectbox("Feature", zoom_options)
        else:
            zoom_to = "— none —"
            st.caption("No dune names found in data")

    # -- Filter datasets by date range -----------------------------------------
    m_nums = [ALL_MONTHS[m] for m in selected_months]

    def date_filter(gdf):
        if gdf.empty or "year" not in gdf.columns:
            return gdf
        return gdf[gdf["year"].isin(selected_years) &
                   gdf["month"].isin(m_nums)].copy()

    f_crest = date_filter(crest_gdf)
    f_var   = var_gdf.copy()           # variability filtered by date_a/b in map builder
    f_playa = date_filter(playa_gdf)

    # Wind completeness
    wind_pct, f_wind = wind_completeness(wind_df, selected_years, selected_months)

    # Wind rose image
    wind_b64 = None
    if show_wind and not f_wind.empty and wind_pct >= WIND_HIDE_PCT:
        wind_b64 = build_wind_rose_image(f_wind)
    elif show_wind and wind_pct < WIND_HIDE_PCT and not f_wind.empty:
        st.warning("Wind rose hidden: data coverage < 30 % for selected period.")

    # -- SIDEBAR METRICS -------------------------------------------------------
    with st.sidebar:
        st.divider()
        st.markdown("### Summary")

        crest_km = f_crest["length_m"].sum() / 1000 if not f_crest.empty and "length_m" in f_crest else 0
        st.metric("Visible crest length", f"{crest_km:.1f} km")

        if not unc_gdf.empty and "error_m" in unc_gdf.columns:
            mean_unc = unc_gdf["error_m"].mean()
            st.metric("Mean uncertainty", f"±{mean_unc:.2f} m")

        st.metric("Wind data coverage", f"{wind_pct*100:.0f} %")

        if wind_pct < WIND_WARN_PCT and wind_pct >= WIND_HIDE_PCT:
            st.markdown(
                '<div class="warn-box">⚠️ Wind rose based on incomplete data</div>',
                unsafe_allow_html=True
            )

    # -- MAIN MAP --------------------------------------------------------------
    folium_map = build_map(
        crest_gdf=f_crest,
        var_gdf=f_var,
        playa_gdf=f_playa,
        unc_gdf=unc_gdf,
        wind_b64=wind_b64,
        wind_completeness_pct=wind_pct,
        show_crests=show_crests,
        show_gap_fills=show_gap_fills,
        show_variability=show_variability,
        date_a=date_a,
        date_b=date_b,
        show_playa=show_playa,
        show_wind=show_wind,
        show_uncertainty=show_uncertainty,
        opacity=opacity,
    )

    if zoom_to != "— none —" and not crest_gdf.empty:
        if "dune_name" in crest_gdf.columns:
            dune_geoms = crest_gdf[crest_gdf["dune_name"] == zoom_to].geometry
            if not dune_geoms.empty:
                bounds = dune_geoms.total_bounds
                folium_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    map_data = st_folium(folium_map, width="100%", height=620,
                     returned_objects=["last_object_clicked"])

    # -- SECONDARY PANELS ------------------------------------------------------
    st.divider()
    ts_col, gantt_col = st.columns([1, 1])

    # Time-series panel
    with ts_col:
        st.markdown("#### Crest Variability Trend")
        if (map_data and map_data.get("last_object_clicked") and
                show_variability and not var_gdf.empty):
            click = map_data["last_object_clicked"]
            lat, lon = click.get("lat"), click.get("lng")
            if lat and lon:
                from shapely.geometry import Point
                click_pt = Point(lon, lat)
                var_proj = var_gdf.to_crs("EPSG:3857")
                click_proj = gpd.GeoDataFrame(
                    geometry=[click_pt], crs="EPSG:4326"
                ).to_crs("EPSG:3857")
                distances = var_proj.geometry.distance(click_proj.geometry.iloc[0])
                nearest_pid = var_gdf.iloc[distances.idxmin()]["point_id"]
                trend = var_gdf[var_gdf["point_id"]==nearest_pid].sort_values("date")

                fig_ts, ax_ts = plt.subplots(figsize=(5, 2.5), facecolor="#F5F0E8")
                ax_ts.plot(trend["date"], trend["distance_m"],
                           marker="o", color="#8B5E3C", linewidth=1.5)
                ax_ts.axhline(0, color="#C9BA9B", linestyle="--", linewidth=0.8)
                ax_ts.set_xlabel("Date", fontsize=8)
                ax_ts.set_ylabel("Distance (m)", fontsize=8)
                ax_ts.set_title(f"Point ID: {nearest_pid}", fontsize=9)
                ax_ts.set_facecolor("#F5F0E8")
                for sp in ax_ts.spines.values():
                    sp.set_edgecolor("#C9BA9B")
                fig_ts.tight_layout()
                st.pyplot(fig_ts)
                plt.close(fig_ts)
            else:
                st.caption("Click a variability point on the map to see its trend.")
        else:
            st.caption("Click a variability point on the map to populate this chart.")

        # Uncertainty histogram
        st.markdown("#### Uncertainty Distribution")
        if not unc_gdf.empty and "error_m" in unc_gdf.columns:
            fig_h, ax_h = plt.subplots(figsize=(5, 2.2), facecolor="#F5F0E8")
            vals = unc_gdf["error_m"].dropna()
            ax_h.hist(vals, bins=20, color="#8B5E3C", edgecolor="#F5F0E8", alpha=0.9)
            ax_h.axvline(UNC_GREEN,  color="#28A745", linestyle="--", lw=1,
                         label=f"<{UNC_GREEN} m (good)")
            ax_h.axvline(UNC_YELLOW, color="#FFC107", linestyle="--", lw=1,
                         label=f"<{UNC_YELLOW} m (moderate)")
            ax_h.set_xlabel("Error (m)", fontsize=8)
            ax_h.set_ylabel("Count",     fontsize=8)
            ax_h.set_facecolor("#F5F0E8")
            ax_h.legend(fontsize=7)
            for sp in ax_h.spines.values():
                sp.set_edgecolor("#C9BA9B")
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
        else:
            st.caption("No uncertainty data loaded.")

    # Gantt chart panel
    with gantt_col:
        st.markdown("#### Wind Data Coverage")
        if not wind_df.empty:
            fig_g = build_gantt_figure(wind_df, selected_years, selected_months)
            st.pyplot(fig_g)
            plt.close(fig_g)
            st.caption(
                "Green = wind data present.  Red = data gap.  "
                "Percentage shows daily completeness per month."
            )
        else:
            st.caption("No wind data loaded.")

    # -- EXPORT CONTROLS -------------------------------------------------------
    st.divider()
    st.markdown("#### Export")
    exp1, exp2 = st.columns(2)

    with exp1:
        # HTML map export
        map_html = folium_map._repr_html_()
        st.download_button(
            "Download map (HTML)",
            data=map_html,
            file_name="dune_map.html",
            mime="text/html",
        )

    with exp2:
        # CSV of currently visible crest lines
        if not f_crest.empty:
            csv_data = f_crest.drop(columns="geometry").to_csv(index=False)
            st.download_button(
                "Export crest data (CSV)",
                data=csv_data,
                file_name="crest_lines_filtered.csv",
                mime="text/csv",
            )
        else:
            st.button("Export crest data (CSV)", disabled=True)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()