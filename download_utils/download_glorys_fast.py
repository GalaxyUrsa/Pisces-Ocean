import copernicusmarine
import xarray as xr
import numpy as np
import os
from datetime import datetime

# Copernicus credentials
USERNAME = "ghuang12"
PASSWORD = "!Hjh123456789"

# Dataset configuration
dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
variables = ["so", "thetao"]

# Spatial domain (Northwestern Pacific)
minimum_longitude = 100
maximum_longitude = 160
minimum_latitude = 0
maximum_latitude = 50

# Depth range
minimum_depth = 0.49402499198913574
maximum_depth = 651

# Time range
start_date = "2024-07-01"
end_date = "2024-07-10"

# Directories
raw_file = "./downloaded_data/glorys_raw_bulk.nc"
output_dir = "./downloaded_data/Glorys"
os.makedirs(output_dir, exist_ok=True)


def download_bulk():
    """Download entire time range in one request."""
    print("Downloading GLORYS data in bulk...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        output_filename=raw_file,
        username=USERNAME,
        password=PASSWORD
    )
    print(f"✅ Bulk download complete: {raw_file}")


def split_and_resample():
    """Split bulk file by day and resample to 1/8°."""
    print("\nSplitting and resampling...")

    # Target grid
    target_lon = np.arange(100, 159.875 + 0.125, 0.125)
    target_lat = np.arange(0, 49.875 + 0.125, 0.125)

    ds = xr.open_dataset(raw_file, chunks={'time': 1})

    for i in range(len(ds.time)):
        ds_day = ds.isel(time=i)
        day_str = str(ds_day.time.values)[:10].replace('-', '')
        output_file = os.path.join(output_dir, f"glory_resample_1_8_{day_str}.nc")

        if os.path.exists(output_file):
            print(f"⏭️  Already exists: {day_str}")
            continue

        ds_resampled = ds_day.interp(
            longitude=target_lon,
            latitude=target_lat,
            method='linear'
        ).compute()
        ds_resampled.to_netcdf(output_file)
        print(f"✅ {day_str}")

    ds.close()
    print("\n✅ All days split and resampled.")


if __name__ == "__main__":
    print("=" * 80)
    print("GLORYS Fast Bulk Download & Resample Tool")
    print("=" * 80)
    print(f"Time Range: {start_date} to {end_date}")
    print()

    # Step 1: One-shot download
    if not os.path.exists(raw_file):
        download_bulk()
    else:
        print(f"⏭️  Bulk file already exists, skipping download: {raw_file}")

    # Step 2: Split by day + resample
    split_and_resample()

    # Step 3: Clean up bulk file
    if os.path.exists(raw_file):
        os.remove(raw_file)
        print(f"✅ Deleted bulk file: {raw_file}")

    print("\n" + "=" * 80)
    print("✅ All done.")
    print("=" * 80)
