"""
Streamlit dashboard for the Flash Flood Prediction System.

Run with:  streamlit run app.py

This simulates a "live" rainfall event hitting the region — in a real
deployment, current_rainfall would come from an IMD/weather API
instead of a slider.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
from streamlit_folium import st_folium

from flood_risk import compute_risk

st.set_page_config(page_title="Flash Flood Early Warning", layout="wide")

# --- Custom header banner ---
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #1B4965 0%, #2E86AB 50%, #5FA8D3 100%);
        padding: 22px 28px;
        border-radius: 10px;
        margin-bottom: 18px;
    ">
        <h1 style="color: white; margin: 0; font-size: 28px;">
            🌧️ Flash Flood Early Warning System
        </h1>
        <p style="color: #DDEEFF; margin: 4px 0 0 0; font-size: 15px;">
            Hilly Region Multi-Source Risk Dashboard — SIH Prototype
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load village data ---
df = pd.read_csv("villages.csv")

# Backward compatibility: older villages.csv files (generated before the
# multi-region update) won't have a 'region' column.
if "region" not in df.columns:
    df["region"] = "Unnamed region"

# --- Sidebar: region filter (only shown if more than one region present) ---
available_regions = sorted(df["region"].unique())
if len(available_regions) > 1:
    st.sidebar.header("Region")
    selected_regions = st.sidebar.multiselect(
        "Show villages from",
        options=available_regions,
        default=available_regions,
    )
    df = df[df["region"].isin(selected_regions)].reset_index(drop=True)

if df.empty:
    st.warning("No villages match the selected region(s). Pick at least one region in the sidebar.")
    st.stop()

# --- Sidebar controls: simulate an incoming rainfall event ---
st.sidebar.header("Simulate Current Rainfall Event")
st.sidebar.write(
    "In production, this would be pulled automatically from IMD's "
    "real-time rainfall API instead of set manually."
)

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Light rain", "Moderate rain", "Heavy rain (flash flood risk)", "Custom"],
)

if scenario == "Light rain":
    base_rainfall = 5
    spread = 5
elif scenario == "Moderate rain":
    base_rainfall = 20
    spread = 10
elif scenario == "Heavy rain (flash flood risk)":
    base_rainfall = 50
    spread = 15
else:
    base_rainfall = st.sidebar.slider("Average rainfall (mm/hr)", 0, 80, 30)
    spread = st.sidebar.slider("Spread across villages", 0, 30, 10)

np.random.seed(42)
current_rainfall = pd.Series(
    np.clip(np.random.normal(base_rainfall, spread, len(df)), 0, 100)
)

# --- Compute risk ---
result = compute_risk(df, current_rainfall)

# --- Optional: ML model prediction (if trained) ---
MODEL_PATH = "flood_model.joblib"
model_available = os.path.exists(MODEL_PATH)

if model_available:
    import joblib
    model = joblib.load(MODEL_PATH)
    ml_features = [
        "slope_deg",
        "recent_rainfall_3day_mm",
        "distance_to_stream_m",
        "current_rainfall_mm_per_hr",
    ]
    ml_probs = model.predict_proba(result[ml_features])[:, 1]  # P(flooded)
    result["ml_flood_probability"] = (ml_probs * 100).round(1)

    st.sidebar.success("✅ ML model loaded (flood_model.joblib)")
    st.sidebar.caption(
        "Trained on synthetic labels — shown as a demo of the ML pipeline, "
        "not a validated real-world prediction."
    )
else:
    st.sidebar.info(
        "ℹ️ No trained ML model found. Run `python train_model.py` to "
        "generate flood_model.joblib and see ML predictions here too."
    )

# --- Top summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("High Risk Villages", (result["risk_level"] == "High").sum())
col2.metric("Moderate Risk Villages", (result["risk_level"] == "Moderate").sum())
col3.metric("Low Risk Villages", (result["risk_level"] == "Low").sum())

most_urgent = result.loc[result["time_to_impact_min"].idxmin()]
col4.metric(
    "Most Urgent",
    most_urgent["village"],
    f"{most_urgent['time_to_impact_min']:.0f} min",
    delta_color="inverse",
)

# --- Map ---
st.subheader("Risk Map")

color_map = {"High": "red", "Moderate": "orange", "Low": "green"}

center_lat = result["lat"].mean()
center_lon = result["lon"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

for _, row in result.iterrows():
    ml_line = ""
    if model_available:
        ml_line = f"<b>ML flood probability: {row['ml_flood_probability']:.1f}%</b><br>"

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8,
        color=color_map[row["risk_level"]],
        fill=True,
        fill_color=color_map[row["risk_level"]],
        fill_opacity=0.8,
        popup=(
            f"<b>{row['village']}</b> ({row['region']})<br>"
            f"Risk: {row['risk_level']} ({row['risk_score']})<br>"
            f"{ml_line}"
            f"<b>Est. time to impact: {row['time_to_impact_label']}</b><br>"
            f"({row['time_to_impact_min']:.1f} min at ~{row['est_velocity_mps']} m/s runoff)<br>"
            f"Current rainfall: {row['current_rainfall_mm_per_hr']:.1f} mm/hr<br>"
            f"Slope: {row['slope_deg']}°<br>"
            f"Distance to stream: {row['distance_to_stream_m']:.0f} m"
        ),
    ).add_to(m)

st_folium(m, width=1000, height=500)

# --- Alert list ---
st.subheader("🚨 Active Alerts (High Risk)")
st.caption(
    "Sorted by urgency (time to impact), not just risk score — "
    "a village with less warning time needs attention first."
)
high_risk = result[result["risk_level"] == "High"].sort_values(
    "time_to_impact_min", ascending=True
)

if high_risk.empty:
    st.success("No villages currently at high flash flood risk.")
else:
    for _, row in high_risk.iterrows():
        ml_text = ""
        if model_available:
            ml_text = f" | ML probability {row['ml_flood_probability']:.0f}%"
        st.error(
            f"**{row['village']}** — ⏱️ **{row['time_to_impact_label']}** "
            f"(~{row['time_to_impact_min']:.0f} min) | "
            f"Risk score {row['risk_score']}/100{ml_text} | "
            f"Rainfall {row['current_rainfall_mm_per_hr']:.1f} mm/hr | "
            f"Slope {row['slope_deg']}° | "
            f"{row['distance_to_stream_m']:.0f}m from stream"
        )

# --- Full data table ---
with st.expander("View full village data table"):
    table_cols = [
        "village",
        "region",
        "risk_level",
        "risk_score",
        "time_to_impact_label",
        "time_to_impact_min",
        "est_velocity_mps",
        "current_rainfall_mm_per_hr",
        "recent_rainfall_3day_mm",
        "slope_deg",
        "distance_to_stream_m",
    ]
    if model_available:
        table_cols.insert(4, "ml_flood_probability")
    table_df = result[table_cols].sort_values("risk_score", ascending=False)

    row_colors = {
        "High": "background-color: #FFCDD2; color: #7A0000;",
        "Moderate": "background-color: #FFE0B2; color: #7A4B00;",
        "Low": "background-color: #C8E6C9; color: #0D4F0D;",
    }

    def color_by_risk(row):
        style = row_colors.get(row["risk_level"], "")
        return [style] * len(row)

    styled_table = table_df.style.apply(color_by_risk, axis=1)
    st.dataframe(styled_table, use_container_width=True)