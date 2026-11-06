"""
批量将 u10_origin/ 下所有 nc 文件插值到 0.083deg，输出到 u10_0.083deg/。
"""

import os
import glob
import numpy as np
import xarray as xr

BASE = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE, "temp", "mwd")
OUTPUT_DIR = os.path.join(BASE, "temp", "mwd_0.083deg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nc")))
print(f"Found {len(files)} files.")

for fpath in files:
    fname = os.path.basename(fpath)
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"[SKIP] {fname}")
        continue

    lat_new = np.linspace(49.917, 0.0, 600)
    lon_new = np.linspace(100.0, 159.917, 720)
    ds = xr.open_dataset(fpath)
    ds.interp(latitude=lat_new, longitude=lon_new, method="linear").to_netcdf(out_path)
    ds.close()
    print(f"[DONE] {fname}")

print("All done.")