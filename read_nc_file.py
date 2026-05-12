"""
Simple script to read NetCDF file and extract data
"""
import numpy as np
import xarray as xr

# File path
file_path = r"F:\PythonWorkspace\Pisces-Ocean\resize_sample\glorys_0.25deg_20240101.nc"
# "C:\Users\user\Downloads\cmems_mod_glo_phy_anfc_0.083deg_P1D-m_1776734444479.nc"


# Open the NetCDF file
print(f"Opening file: {file_path}")
ds = xr.open_dataset(file_path)

# Display file structure
print("\n" + "="*60)
print("File Structure")
print("="*60)
print(f"\nDimensions: {dict(ds.dims)}")
print(f"\nCoordinates: {list(ds.coords.keys())}")
print(f"\nData variables: {list(ds.data_vars.keys())}")

# Display each variable's shape
print("\n" + "="*60)
print("Variable Shapes")
print("="*60)
for var_name in ds.data_vars:
    var_data = ds[var_name]
    print(f"{var_name}: {var_data.shape} | dtype: {var_data.dtype}")

# ==================== 新增：经纬度坐标读取 ====================
print("\n" + "="*60)
print("Coordinate Information (Longitude & Latitude)")
print("="*60)

lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

# 读取经度
if lon_name in ds.coords:
    lon = ds[lon_name].values
    print(f"\nLongitude ('{lon_name}'):")
    print(f"  Shape    : {lon.shape}")
    print(f"  Range    : [{lon.min():.4f}, {lon.max():.4f}]")
    print(f"  Count    : {len(lon)}")
    if len(lon) > 1:
        print(f"  Step     : {(lon[1] - lon[0]):.6f} (uniform: {np.allclose(np.diff(lon), lon[1]-lon[0])})")
    # 显示目标区域对应的索引范围
    lon_idx = np.where((lon >= 100) & (lon <= 159.999))[0]
    if len(lon_idx) > 0:
        print(f"  Target   : indices [{lon_idx[0]}:{lon_idx[-1]+1}] -> values [{lon[lon_idx[0]]:.3f}, {lon[lon_idx[-1]]:.3f}]")
else:
    print(f"Warning: Longitude coordinate '{lon_name}' not found!")

# 读取纬度
if lat_name in ds.coords:
    lat = ds[lat_name].values
    print(f"\nLatitude ('{lat_name}'):")
    print(f"  Shape    : {lat.shape}")
    print(f"  Range    : [{lat.min():.4f}, {lat.max():.4f}]")
    print(f"  Count    : {len(lat)}")
    if len(lat) > 1:
        print(f"  Step     : {(lat[1] - lat[0]):.6f} (uniform: {np.allclose(np.diff(lat), lat[1]-lat[0])})")
    # 显示目标区域对应的索引范围
    lat_idx = np.where((lat >= 0) & (lat <= 49.999))[0]
    if len(lat_idx) > 0:
        print(f"  Target   : indices [{lat_idx[0]}:{lat_idx[-1]+1}] -> values [{lat[lat_idx[0]]:.3f}, {lat[lat_idx[-1]]:.3f}]")
else:
    print(f"Warning: Latitude coordinate '{lat_name}' not found!")
# ==============================================================

# Read thetao and so variables (as specified for Background folder)
print("\n" + "="*60)
print("Reading thetao and so variables")
print("="*60)

if 'thetao' in ds:
    thetao = ds['thetao'].values
    print(f"\nthetao:")
    print(f"  Shape: {thetao.shape}")
    print(f"  Dtype: {thetao.dtype}")
    print(f"  Range: [{np.nanmin(thetao):.4f}, {np.nanmax(thetao):.4f}]")
    print(f"  NaN count: {np.isnan(thetao).sum()}")
else:
    print("Warning: 'thetao' not found in file")

if 'so' in ds:
    so = ds['so'].values
    print(f"\nso:")
    print(f"  Shape: {so.shape}")
    print(f"  Dtype: {so.dtype}")
    print(f"  Range: [{np.nanmin(so):.4f}, {np.nanmax(so):.4f}]")
    print(f"  NaN count: {np.isnan(so).sum()}")
else:
    print("Warning: 'so' not found in file")

# If you need to select specific spatial region (like in dataloader.py)
print("\n" + "="*60)
print("Reading with spatial selection (lon: 100-159.875, lat: 0-49.875)")
print("="*60)

if 'thetao' in ds:
    thetao_subset = ds['thetao'].sel(
        {lon_name: slice(100, 159.99),
         lat_name: slice(0, 49.99)}
    ).values

    # Remove time dimension if it exists (squeeze removes dimensions of size 1)
    if thetao_subset.shape[0] == 1:
        thetao_subset = thetao_subset.squeeze(axis=0)

    print(f"\nthetao (subset):")
    print(f"  Shape: {thetao_subset.shape}")
    print(f"  Expected shape: (33, 400, 480)")
    print(f"  Match: {thetao_subset.shape == (33, 400, 480)}")

if 'so' in ds:
    so_subset = ds['so'].sel(
        {lon_name: slice(100, 159.999),
         lat_name: slice(0, 49.999)}
    ).values

    # Remove time dimension if it exists
    if so_subset.shape[0] == 1:
        so_subset = so_subset.squeeze(axis=0)

    print(f"\nso (subset):")
    print(f"  Shape: {so_subset.shape}")
    print(f"  Expected shape: (33, 400, 480)")
    print(f"  Match: {so_subset.shape == (33, 400, 480)}")

# Close the dataset
ds.close()

print("\n" + "="*60)
print("Done!")
print("="*60)