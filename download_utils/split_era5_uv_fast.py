"""
快速拆分 2015_2025_wind_uv_10_1200Z.nc 为每日 u10/v10 单文件，多进程加速。
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool, cpu_count

INPUT_FILE = os.path.join(os.path.dirname(__file__), "temp", "2015_2025_wind_uv_10_1200Z.nc")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "temp")
U_DIR = os.path.join(OUTPUT_ROOT, "u10_fast")
V_DIR = os.path.join(OUTPUT_ROOT, "v10_fast")

KEEP_COORDS = ("valid_time", "latitude", "longitude")


def write_day(args):
    idx, t, u_data, v_data, lat, lon = args
    day = pd.Timestamp(t).strftime("%Y%m%d")

    u_path = os.path.join(U_DIR, f"era5_wind_u_10_{day}.nc")
    v_path = os.path.join(V_DIR, f"era5_wind_v_10_{day}.nc")

    coords = {"valid_time": [t], "latitude": lat, "longitude": lon}

    if not os.path.exists(u_path):
        xr.Dataset(
            {"u10": (["valid_time", "latitude", "longitude"], u_data[np.newaxis])},
            coords=coords,
        ).to_netcdf(u_path)

    if not os.path.exists(v_path):
        xr.Dataset(
            {"v10": (["valid_time", "latitude", "longitude"], v_data[np.newaxis])},
            coords=coords,
        ).to_netcdf(v_path)

    print(f"[DONE] {day}")


if __name__ == "__main__":
    os.makedirs(U_DIR, exist_ok=True)
    os.makedirs(V_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = xr.open_dataset(INPUT_FILE)
    times = ds.valid_time.values
    lat = ds.latitude.values
    lon = ds.longitude.values
    u_all = ds["u10"].values   # (T, lat, lon)
    v_all = ds["v10"].values
    ds.close()

    args = [(i, times[i], u_all[i], v_all[i], lat, lon) for i in range(len(times))]

    workers = min(cpu_count(), 8)
    print(f"Writing {len(args)} days with {workers} workers...")
    with Pool(workers) as pool:
        pool.map(write_day, args)

    print("All done.")
