"""
快速拆分 2015_2025_wave_dir_height_1200Z.nc 为每日 mwd/swh 单文件，多进程加速。

输出结构：
  temp/mwd/era5_wave_mwd_yyyymmdd.nc
  temp/swh/era5_wave_swh_yyyymmdd.nc
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool, cpu_count

INPUT_FILE = os.path.join(os.path.dirname(__file__), "temp", "2015_2025_wave_dir_height_1200Z.nc")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "temp")
MWD_DIR = os.path.join(OUTPUT_ROOT, "mwd")
SWH_DIR = os.path.join(OUTPUT_ROOT, "swh")


def write_day(args):
    t, mwd_data, swh_data, lat, lon = args
    day = pd.Timestamp(t).strftime("%Y%m%d")
    coords = {"valid_time": [t], "latitude": lat, "longitude": lon}

    mwd_path = os.path.join(MWD_DIR, f"era5_wave_mwd_{day}.nc")
    swh_path = os.path.join(SWH_DIR, f"era5_wave_swh_{day}.nc")

    if not os.path.exists(mwd_path):
        xr.Dataset(
            {"mwd": (["valid_time", "latitude", "longitude"], mwd_data[np.newaxis])},
            coords=coords,
        ).to_netcdf(mwd_path)

    if not os.path.exists(swh_path):
        xr.Dataset(
            {"swh": (["valid_time", "latitude", "longitude"], swh_data[np.newaxis])},
            coords=coords,
        ).to_netcdf(swh_path)

    print(f"[DONE] {day}")


if __name__ == "__main__":
    os.makedirs(MWD_DIR, exist_ok=True)
    os.makedirs(SWH_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = xr.open_dataset(INPUT_FILE)
    times = ds.valid_time.values
    lat = ds.latitude.values
    lon = ds.longitude.values
    mwd_all = ds["mwd"].values
    swh_all = ds["swh"].values
    ds.close()

    args = [(times[i], mwd_all[i], swh_all[i], lat, lon) for i in range(len(times))]

    workers = min(cpu_count(), 8)
    print(f"Writing {len(args)} days with {workers} workers...")
    with Pool(workers) as pool:
        pool.map(write_day, args)

    print("All done.")
