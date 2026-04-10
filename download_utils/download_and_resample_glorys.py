import copernicusmarine
import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta

# Copernicus credentials
USERNAME = "ghuang12"
PASSWORD = "!Hjh123456789"

# Dataset configuration
dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
variables = ["so", "thetao"]  # Salinity, Temperature, U/V velocity

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
output_dir = "./downloaded_data/Glorys"
os.makedirs(output_dir, exist_ok=True)


def download_and_resample_daily(start_date_str, end_date_str):
    """Download and resample GLORYS data for a single day."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    current_day = start_dt
    while current_day <= end_dt:
        day_str = current_day.strftime("%Y%m%d")
        day_start = f"{day_str}T00:00:00"
        day_end = f"{day_str}T23:59:59"

        output_file_raw = os.path.join(output_dir, f"glorys_raw_0.083deg_{day_str}.nc")
        output_file_resampled = os.path.join(output_dir, f"glory_resample_1_8_{day_str}.nc")

        print(f"\nProcessing day: {day_str}")

        try:
            # Step 1: Download one day of data
            print("Downloading GLORYS data...")
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=minimum_longitude,
                maximum_longitude=maximum_longitude,
                minimum_latitude=minimum_latitude,
                maximum_latitude=maximum_latitude,
                start_datetime=day_start,
                end_datetime=day_end,
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
                output_filename=output_file_raw,
                username=USERNAME,
                password=PASSWORD
            )
            print(f"✅ Downloaded: {output_file_raw}")

            # Step 2: Resample to 1/8° resolution
            print("Resampling to 1/8° resolution...")
            ds = xr.open_dataset(output_file_raw)

            # Target grid
            target_lon = np.arange(100, 159.875 + 0.125, 0.125)
            target_lat = np.arange(0, 49.875 + 0.125, 0.125)

            # Interpolate
            ds_resampled = ds.interp(
                longitude=target_lon,
                latitude=target_lat,
                method='linear'
            )

            # Save resampled data
            ds_resampled.to_netcdf(output_file_resampled)
            print(f"✅ Saved resampled file: {output_file_resampled}")

            # Cleanup raw file
            os.remove(output_file_raw)
            print(f"✅ Deleted raw file: {output_file_raw}")

            # Close datasets
            ds.close()
            ds_resampled.close()

        except Exception as e:
            print(f"❌ Error processing {day_str}: {e}")

        current_day += timedelta(days=1)


if __name__ == "__main__":
    print("=" * 80)
    print("GLORYS Daily Data Download & Resampling Tool")
    print("=" * 80)
    print(f"Time Range: {start_date} to {end_date}")
    print()

    download_and_resample_daily(start_date, end_date)

    print("\n" + "=" * 80)
    print("✅ All days processed.")
    print("=" * 80)