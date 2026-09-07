"""
Generates a synthetic dataset of villages in a hilly test region.

In a real SIH submission, you'd replace this with:
  - Village locations: Census/OpenStreetMap data
  - Elevation & slope: DEM data from Bhuvan (ISRO) or USGS SRTM
  - Historical rainfall: IMD (India) or DHM (Nepal) open data portals
This script exists so you have something to build/test the pipeline
against immediately, without waiting on real data access approvals.

Usage:
    python generate_data.py                  # uses DEFAULT_REGION below
    python generate_data.py kullu             # generate for a specific region
    python generate_data.py --list            # list available regions
"""

import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Region registry: real flood-prone hilly districts across Himachal Pradesh
# and Nepal. Add more entries here as needed — just give a center lat/lon.
# ---------------------------------------------------------------------------
REGIONS = {
    "kullu": {
        "label": "Kullu, Himachal Pradesh",
        "lat": 31.9576,
        "lon": 77.1095,
        "note": "Frequent cloudburst/flash flood events along the Beas river",
    },
    "mandi": {
        "label": "Mandi, Himachal Pradesh",
        "lat": 31.7084,
        "lon": 76.9319,
        "note": "Steep terrain, multiple flood incidents in recent years",
    },
    "kangra": {
        "label": "Kangra, Himachal Pradesh",
        "lat": 32.0998,
        "lon": 76.2691,
        "note": "High rainfall zone, hilly with dense stream network",
    },
    "shimla": {
        "label": "Shimla, Himachal Pradesh",
        "lat": 31.1048,
        "lon": 77.1734,
        "note": "Urban-hill mix, useful contrast case",
    },
    "sindhupalchok": {
        "label": "Sindhupalchok, Nepal",
        "lat": 27.9500,
        "lon": 85.6833,
        "note": "Site of the devastating 2021 Melamchi flash flood",
    },
    "kaski": {
        "label": "Kaski (Pokhara), Nepal",
        "lat": 28.2096,
        "lon": 83.9856,
        "note": "Steep Himalayan foothill terrain",
    },
    "kathmandu": {
        "label": "Kathmandu Valley, Nepal",
        "lat": 27.7172,
        "lon": 85.3240,
        "note": "Less steep, more populated — good comparison case",
    },
}

# Default region used when no command-line argument is given.
# Sindhupalchok is a strong pick for a SIH pitch: it lets you reference
# a real, well-documented disaster (2021 Melamchi flash flood) to justify
# necessity.
DEFAULT_REGION = "sindhupalchok"

N_VILLAGES = 25

# Spread around the center point, in degrees. ~0.15 degrees is roughly a
# 33km x 33km box — tight enough to stay within one district's varied
# terrain rather than spanning into unrelated valleys/ridges.
COORD_SPREAD = 0.15


def generate(region_key: str):
    if region_key not in REGIONS:
        available = ", ".join(REGIONS.keys())
        raise ValueError(f"Unknown region '{region_key}'. Available: {available}")

    region = REGIONS[region_key]
    base_lat, base_lon = region["lat"], region["lon"]

    np.random.seed(42)

    villages = []
    for i in range(N_VILLAGES):
        name = f"{region['label'].split(',')[0].replace(' ', '_')}_Village_{i+1}"
        lat = base_lat + np.random.uniform(-COORD_SPREAD, COORD_SPREAD)
        lon = base_lon + np.random.uniform(-COORD_SPREAD, COORD_SPREAD)
        # Elevation in meters — hilly regions vary a lot over short distances
        elevation = np.random.uniform(300, 2200)
        # Slope in degrees — steeper slope = faster water runoff = higher flash flood risk
        slope = np.random.uniform(2, 45)
        # Proxy for soil saturation: average rainfall over the last 3 days (mm)
        recent_rainfall_3day_mm = np.random.uniform(0, 150)
        # Distance to nearest stream/river in meters — closer streams flood faster
        distance_to_stream_m = np.random.uniform(50, 3000)

        villages.append({
            "village": name,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "elevation_m": round(elevation, 1),
            "slope_deg": round(slope, 1),
            "recent_rainfall_3day_mm": round(recent_rainfall_3day_mm, 1),
            "distance_to_stream_m": round(distance_to_stream_m, 1),
        })

    df = pd.DataFrame(villages)
    df.to_csv("villages.csv", index=False)
    print(f"Region: {region['label']} ({region['note']})")
    print(f"Center: {base_lat}, {base_lon}")
    print(f"Generated villages.csv with {len(df)} villages.")
    print(df.head())
    return df


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("Available regions:")
        for key, r in REGIONS.items():
            print(f"  {key:15s} -> {r['label']} ({r['note']})")
        sys.exit(0)

    region_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REGION
    generate(region_key)