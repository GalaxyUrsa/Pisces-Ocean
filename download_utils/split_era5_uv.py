"""
将 2015_2025_wind_uv_10_1200Z.nc 按天拆分为 u10/v10 各自的单日文件。

输出结构：
  <output_root>/u10/era5_wind_u_10_yyyymmdd.nc
  <output_root>/v10/era5_wind_v_10_yyyymmdd.nc
"""

import os
import xarray as xr

INPUT_FILE = os.path.join(os.path.dirname(__file__), "temp", "2015_2025_wind_uv_10_1200Z.nc")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "temp")

ds = xr.open_dataset(INPUT_FILE)

u_dir = os.path.join(OUTPUT_ROOT, "u10")
v_dir = os.path.join(OUTPUT_ROOT, "v10")
os.makedirs(u_dir, exist_ok=True)
os.makedirs(v_dir, exist_ok=True)

for t in ds.valid_time.values:
    day = xr.DataArray(t).dt.strftime("%Y%m%d").item()
    ds_day = ds.sel(valid_time=[t])

    u_path = os.path.join(u_dir, f"era5_wind_u_10_{day}.nc")
    v_path = os.path.join(v_dir, f"era5_wind_v_10_{day}.nc")

    drop = [c for c in ds_day.coords if c not in ("valid_time", "latitude", "longitude")]
    if not os.path.exists(u_path):
        ds_day[["u10"]].drop_vars(drop).to_netcdf(u_path)
    if not os.path.exists(v_path):
        ds_day[["v10"]].drop_vars(drop).to_netcdf(v_path)

    print(f"[DONE] {day}")

ds.close()
print("All done.")
