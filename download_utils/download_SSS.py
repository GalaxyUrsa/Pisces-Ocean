import copernicusmarine
import os
from datetime import datetime, timedelta

# Copernicus credentials
USERNAME = "ghuang12"
PASSWORD = "!Hjh123456789"

# Dataset configuration
dataset_id = "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D"
variables = ['sos']  # Salinity, Temperature, U/V velocity

# Spatial domain (Northwestern Pacific)
minimum_longitude = 100
maximum_longitude = 160
minimum_latitude = 0
maximum_latitude = 50

# Depth range
minimum_depth = 0.49402499198913574  # Surface
maximum_depth = 651  # ~650m depth

# Time range
start_date = "2026-01-01"
end_date = "2026-02-28"

# Output directory
output_dir = "./downloaded_data"
os.makedirs(output_dir, exist_ok=True)


def download_daily(start_date_str, end_date_str):
    """Download SLA data for each day."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    current_day = start_dt
    while current_day <= end_dt:
        day_str = current_day.strftime("%Y-%m-%d")
        day_start = f"{day_str}T00:00:00"
        day_end = f"{day_str}T23:59:59"

        day_str_compact = current_day.strftime("%Y%m%d")
        output_file = os.path.join(output_dir, f"dataset-sss-ssd-nrt-daily_{day_str_compact}.nc")

        print(f"\nProcessing day: {day_str}")

        try:
            print("Downloading SLA data...")
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=minimum_longitude,
                maximum_longitude=maximum_longitude,
                minimum_latitude=minimum_latitude,
                maximum_latitude=maximum_latitude,
                start_datetime=day_start,
                end_datetime=day_end,
                # minimum_depth=minimum_depth,
                # maximum_depth=maximum_depth,
                output_filename=output_file,
                username=USERNAME,
                password=PASSWORD
            )
            print(f"✅ Downloaded: {output_file}")

        except Exception as e:
            print(f"❌ Error processing {day_str}: {e}")

        current_day += timedelta(days=1)


if __name__ == "__main__":
    print("=" * 80)
    print("SLA Daily Data Download Tool")
    print("=" * 80)
    print(f"Time Range: {start_date} to {end_date}")
    print()

    download_daily(start_date, end_date)

    print("\n" + "=" * 80)
    print("✅ All days processed.")
    print("=" * 80)