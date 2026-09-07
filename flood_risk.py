"""
Rule-based flash flood risk scoring.

This is intentionally NOT a black-box ML model to start with — flash
flood early warning systems in the real world (like India's own
Central Water Commission flood forecasting) lean heavily on
threshold/rule-based logic because it's explainable and doesn't need
huge historical flood-event datasets to work reasonably well.

The scoring combines four factors that all genuinely matter for
flash floods in hilly terrain:
  1. Current rainfall intensity (mm/hour) — the immediate trigger
  2. Recent 3-day rainfall (mm) — proxy for soil saturation
  3. Slope (degrees) — steeper slope = faster runoff = less warning time
  4. Distance to stream (m) — closer villages flood faster and harder

Each factor is normalized to 0-1 and combined with weights that
reflect real hydrological priority (rainfall matters most).
"""

import pandas as pd
import numpy as np


def normalize(series, low, high):
    """Clip and scale a series to the 0-1 range given expected bounds."""
    clipped = series.clip(lower=low, upper=high)
    return (clipped - low) / (high - low)


def estimate_time_to_impact(df: pd.DataFrame, current_rainfall_mm_per_hr: pd.Series) -> pd.DataFrame:
    """
    Rough physics-based estimate of how long a village has before
    floodwater from upstream/uphill runoff reaches it.

    This is a SIMPLIFIED approximation, not a calibrated hydrological
    model. Real overland/channel flow velocity depends on Manning's
    equation (roughness, hydraulic radius, channel shape), which needs
    field survey data we don't have. The goal here is a directionally
    correct, explainable estimate — steeper slopes and heavier rainfall
    mean faster runoff, which is hydrologically true — not a
    precise forecast.

    Steps:
      1. Estimate runoff velocity (m/s) from slope: steeper slope
         accelerates runoff, based on the intuition behind the
         Manning/kinematic-wave relationship (velocity scales with
         the square root of slope).
      2. Boost velocity when current rainfall is heavy — more water
         volume overwhelms soil infiltration and moves faster
         overland instead of soaking in.
      3. Time to impact = distance to nearest stream / velocity.

    Returns df with three new columns: est_velocity_mps,
    time_to_impact_min, time_to_impact_label.
    """
    df = df.copy()

    # 1. Base velocity from slope (m/s).
    #    slope_deg=0  -> ~0.4 m/s (near-flat, slow sheet flow)
    #    slope_deg=45 -> ~5.0 m/s (steep hillside, fast runoff)
    slope_fraction = np.tan(np.radians(df["slope_deg"]))
    base_velocity = 0.4 + 4.6 * normalize(pd.Series(slope_fraction), 0, np.tan(np.radians(45)))

    # 2. Rainfall intensity multiplier: heavy rainfall saturates soil
    #    faster, so more water runs off instead of infiltrating,
    #    increasing effective flow speed. Capped so it doesn't run away.
    rainfall_multiplier = 1 + 0.5 * normalize(current_rainfall_mm_per_hr, 0, 60)

    velocity_mps = (base_velocity * rainfall_multiplier).clip(lower=0.3, upper=8.0)
    df["est_velocity_mps"] = velocity_mps.round(2)

    # 3. Time = distance / velocity, converted to minutes.
    time_to_impact_sec = df["distance_to_stream_m"] / df["est_velocity_mps"]
    df["time_to_impact_min"] = (time_to_impact_sec / 60).round(1)

    def label(minutes):
        if minutes < 15:
            return "Immediate (<15 min)"
        elif minutes < 60:
            return "Short (15-60 min)"
        elif minutes < 180:
            return "Moderate (1-3 hr)"
        else:
            return "Extended (>3 hr)"

    df["time_to_impact_label"] = df["time_to_impact_min"].apply(label)
    return df


def compute_risk(df: pd.DataFrame, current_rainfall_mm_per_hr: pd.Series) -> pd.DataFrame:
    """
    df: dataframe with columns slope_deg, recent_rainfall_3day_mm,
        distance_to_stream_m (from villages.csv)
    current_rainfall_mm_per_hr: a Series of the SAME length as df,
        representing simulated/real-time current rainfall per village

    Returns df with two new columns: risk_score (0-100) and risk_level.
    """
    df = df.copy()
    df["current_rainfall_mm_per_hr"] = current_rainfall_mm_per_hr

    # Normalize each factor to 0-1
    rain_now_norm = normalize(df["current_rainfall_mm_per_hr"], 0, 60)       # 60mm/hr = extreme
    rain_recent_norm = normalize(df["recent_rainfall_3day_mm"], 0, 150)      # saturation proxy
    slope_norm = normalize(df["slope_deg"], 0, 45)                          # steeper = riskier
    # Closer to stream = higher risk, so invert distance
    stream_norm = 1 - normalize(df["distance_to_stream_m"], 0, 3000)

    # Weights: current rainfall matters most, then soil saturation,
    # then terrain factors. Tune these based on domain expert input
    # or historical flood event correlation if you get real data.
    weights = {
        "rain_now": 0.40,
        "rain_recent": 0.25,
        "slope": 0.20,
        "stream": 0.15,
    }

    risk_score = (
        weights["rain_now"] * rain_now_norm
        + weights["rain_recent"] * rain_recent_norm
        + weights["slope"] * slope_norm
        + weights["stream"] * stream_norm
    ) * 100

    df["risk_score"] = risk_score.round(1)

    def classify(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Moderate"
        else:
            return "Low"

    df["risk_level"] = df["risk_score"].apply(classify)

    # Add time-to-impact estimate alongside the risk score
    df = estimate_time_to_impact(df, df["current_rainfall_mm_per_hr"])

    return df


if __name__ == "__main__":
    # Quick standalone test
    df = pd.read_csv("villages.csv")
    # Simulate a heavy rainfall event hitting the region unevenly
    np.random.seed(1)
    simulated_rainfall = pd.Series(np.random.uniform(5, 70, len(df)))

    result = compute_risk(df, simulated_rainfall)
    print(result[[
        "village", "current_rainfall_mm_per_hr", "risk_score", "risk_level",
        "time_to_impact_min", "time_to_impact_label",
    ]]
          .sort_values("risk_score", ascending=False)
          .to_string(index=False))