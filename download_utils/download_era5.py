"""
ERA5 hourly single-level reanalysis batch download demo.

Requires: pip install cdsapi
CDS API setup: https://cds.climate.copernicus.eu/how-to-api
  - Create account at https://cds.climate.copernicus.eu
  - Copy UID:API_KEY from https://cds.climate.copernicus.eu/how-to-api
  - Save to ~/.cdsapirc (default) or pass via yaml config

Common single-level variables for ocean forcing:
  10m_u_component_of_wind
  10m_v_component_of_wind
  2m_temperature
  2m_dewpoint_temperature
  mean_sea_level_pressure
  sea_surface_temperature
  significant_height_of_combined_wind_waves_and_swell
  surface_solar_radiation_downwards
  surface_thermal_radiation_downwards
  total_precipitation
  skin_temperature
"""

import cdsapi
import os
import time
import yaml
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_credentials(yaml_path=CONFIG_PATH):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["url"], config["key"]


CDS_URL, CDS_KEY = load_credentials()

# ---- 可修改区域 ----
# Dataset: ERA5 hourly data on single levels
DATASET = "reanalysis-era5-single-levels"

# Variables to download
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Spatial domain (same as project GLORYS domain: NW Pacific)
AREA = [49.99, 100.0, 0.0, 159.99]  # [North, West, South, East]

# Time range
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# Time selection (hourly steps 0..23)
HOURS = ["12:00"]

# Output
OUTPUT_DIR = r"G:\datasets\ERA5_10m_uv_wind"
# ---- 可修改区域 ----

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_era5(start_date_str, end_date_str):
    """Download ERA5 data by year, one file per year, skip existing files."""
    client = cdsapi.Client(url=CDS_URL, key=CDS_KEY, quiet=False)

    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    years = list(range(start_dt.year, end_dt.year + 1))
    total = len(years)
    success_count = 0
    fail_count = 0

    ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
    ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]

    MAX_RETRIES = 10
    RETRY_DELAY = 180  # seconds between retries on rate-limit

    for i, year in enumerate(years, 1):
        output_file = os.path.join(OUTPUT_DIR, f"era5_single_level_{year}.nc")

        if os.path.exists(output_file):
            print(f"[SKIP] {year}: File already exists")
            success_count += 1
            continue

        print(f"\n[{i}/{total}] Downloading {year} (full year, {len(HOURS)} step(s)/day)...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  Submitting to CDS (attempt {attempt}/{MAX_RETRIES})...")
                client.retrieve(
                    DATASET,
                    {
                        "product_type": "reanalysis",
                        "variable": VARIABLES,
                        "year": str(year),
                        "month": ALL_MONTHS,
                        "day": ALL_DAYS,
                        "time": HOURS,
                        "area": AREA,
                        "format": "netcdf",
                    },
                    output_file,
                )
                print(f"[SUCCESS] Saved to {output_file}")
                success_count += 1
                break

            except Exception as e:
                err = str(e)
                if "rejected" in err.lower() or "limited" in err.lower():
                    if attempt < MAX_RETRIES:
                        print(f"  Rate-limited, waiting {RETRY_DELAY}s before retry...")
                        time.sleep(RETRY_DELAY)
                        continue
                print(f"[ERROR] Failed to download {year}: {e}")
                fail_count += 1
                break

        # Cooldown between years to avoid hammering CDS
        if i < total:
            print(f"  Cooling down 30s before next year...")
            time.sleep(30)

    return success_count, fail_count


if __name__ == "__main__":
    print("=" * 60)
    print("ERA5 Hourly Single-Level Data Download Tool")
    print(f"Dataset: {DATASET}")
    print(f"Variables: {VARIABLES}")
    print(f"Area (N,W,S,E): {AREA}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Hours: {HOURS}")
    print("=" * 60)

    success, failed = download_era5(START_DATE, END_DATE)

    print("\n" + "=" * 60)
    print(f"Complete: {success} success, {failed} failed")
    print("=" * 60)
