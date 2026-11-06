"""
从原始 mwd nc 文件批量处理：
  1. 插值到 0.083deg
  2. 分解为东向/北向分量（气象学约定：来向，顺时针从北）
     mwd_u = -sin(mwd_rad)
     mwd_v = -cos(mwd_rad)

输出：
  F:/datasets/mwd_u_0.083deg/era5_wave_mwd_u_yyyymmdd.nc
  F:/datasets/mwd_v_0.083deg/era5_wave_mwd_v_yyyymmdd.nc
"""

import os
import glob
import numpy as np
import xarray as xr
from multiprocessing import Pool, cpu_count

INPUT_DIR = os.path.join(os.path.dirname(__file__), "temp", "mwd_0.083deg")
U_DIR = os.path.join(os.path.dirname(__file__), "temp", "mwd_u_0.083deg")
V_DIR = os.path.join(os.path.dirname(__file__), "temp", "mwd_v_0.083deg")

LAT_NEW = np.linspace(49.917, 0.0, 600)
LON_NEW = np.linspace(100.0, 159.917, 720)


def process(fpath):
    fname = os.path.basename(fpath)
    day = fname.replace("era5_wave_mwd_", "").replace(".nc", "")

    u_path = os.path.join(U_DIR, f"era5_wave_mwd_u_{day}.nc")
    v_path = os.path.join(V_DIR, f"era5_wave_mwd_v_{day}.nc")

    if os.path.exists(u_path) and os.path.exists(v_path):
        print(f"[SKIP] {day}")
        return

    ds = xr.open_dataset(fpath)
    ds_interp = ds.interp(latitude=LAT_NEW, longitude=LON_NEW, method="linear")
    mwd_rad = np.deg2rad(ds_interp["mwd"].values)
    t = ds_interp.valid_time.values
    coords = {"valid_time": t, "latitude": LAT_NEW, "longitude": LON_NEW}

    if not os.path.exists(u_path):
        xr.Dataset(
            {"mwd_u": (["valid_time", "latitude", "longitude"], -np.sin(mwd_rad))},
            coords=coords,
        ).to_netcdf(u_path)

    if not os.path.exists(v_path):
        xr.Dataset(
            {"mwd_v": (["valid_time", "latitude", "longitude"], -np.cos(mwd_rad))},
            coords=coords,
        ).to_netcdf(v_path)

    ds.close()
    print(f"[DONE] {day}")


if __name__ == "__main__":
    os.makedirs(U_DIR, exist_ok=True)
    os.makedirs(V_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nc")))
    print(f"Found {len(files)} files.")

    workers = min(cpu_count(), 8)
    with Pool(workers) as pool:
        pool.map(process, files)

    print("All done.")
