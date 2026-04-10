"""
Simple script to read NetCDF file and extract data
"""
import numpy as np
import xarray as xr

# File path
file_path = r"F:\PythonWorkspace\predict_ts\datasets\Background\glory_resample_1_8_20250501.nc"

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

lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

if 'thetao' in ds:
    thetao_subset = ds['thetao'].sel(
        {lon_name: slice(100, 159.875),
         lat_name: slice(0, 49.875)}
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
        {lon_name: slice(100, 159.875),
         lat_name: slice(0, 49.875)}
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
