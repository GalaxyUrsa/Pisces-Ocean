import xarray as xr
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"F:\PythonWorkspace\Pisces-Ocean\download_utils\temp\u10_fast\era5_wind_u_10_20160630.nc"

ds = xr.open_dataset(path)
print(ds)
print("\n--- 坐标值 ---")
for name, coord in ds.coords.items():
    print(f"{name}: {coord.values}")
