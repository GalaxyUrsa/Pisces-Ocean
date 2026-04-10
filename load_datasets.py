"""
Dataset Loader for Multiple Ocean Data Sources

This script loads oceanographic data from different sources:
- SSS: Sea Surface Salinity (sos)
- SLA: Sea Level Anomaly (sla)
- SST: Sea Surface Temperature (sst)
- Glorys: Temperature (thetao) and Salinity (so)
- Background: Temperature (thetao) and Salinity (so)
"""

import os
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from pathlib import Path


class OceanDatasetLoader:
    """Load ocean datasets from multiple sources"""

    def __init__(self, base_path: str = r"F:\PythonWorkspace\predict_ts\datasets"):
        """
        Initialize the dataset loader

        Args:
            base_path: Base directory containing dataset folders
        """
        self.base_path = Path(base_path)
        self.folders = {
            'SSS': 'SSS',
            'SLA': 'SLA',
            'SST': 'SST',
            'Glorys': 'Glorys',
            'Background': 'Background'
        }

        # Define which variables to load from each folder
        self.variable_mapping = {
            'SSS': ['sos'],
            'SLA': ['sla', 'ugos', 'vgos'],
            'SST': ['sst'],
            'Glorys': ['thetao', 'so'],
            'Background': ['thetao', 'so']
        }

        # Depth indices to select for 3D variables (20 levels out of 33)
        self.depth_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]

    def get_files_in_folder(self, folder_name: str) -> List[Path]:
        """
        Get all NetCDF files in a specific folder, sorted by date

        Args:
            folder_name: Name of the folder (SSS, SLA, SST, Glorys, Background)

        Returns:
            List of file paths sorted by filename
        """
        folder_path = self.base_path / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = sorted(folder_path.glob("*.nc"))
        return files

    def load_single_file(self, file_path: Path, variables: List[str],
                        lon_slice: Tuple[float, float] = (100, 160),
                        lat_slice: Tuple[float, float] = (0, 50),
                        select_depth: bool = False) -> Dict[str, np.ndarray]:
        """
        Load specific variables from a single NetCDF file

        Args:
            file_path: Path to the NetCDF file
            variables: List of variable names to load
            lon_slice: Longitude range (min, max)
            lat_slice: Latitude range (min, max)
            select_depth: Whether to select specific depth levels (for 3D variables)

        Returns:
            Dictionary mapping variable names to numpy arrays
        """
        ds = xr.open_dataset(file_path)

        # Determine coordinate names (handle different naming conventions)
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

        result = {}
        for var in variables:
            if var not in ds:
                print(f"Warning: Variable '{var}' not found in {file_path.name}")
                continue

            # Select spatial subset
            var_data = ds[var].sel(
                {lon_name: slice(lon_slice[0], lon_slice[1]),
                 lat_name: slice(lat_slice[0], lat_slice[1])}
            )

            # Select specific depth levels if variable has depth dimension
            if select_depth and 'depth' in var_data.dims:
                var_data = var_data.isel(depth=self.depth_indices)

            data = var_data.values

            # Remove all dimensions of size 1 (including time dimension)
            data = np.squeeze(data)

            result[var] = data

        ds.close()
        return result

    def load_single_date(self, date: str,
                        lon_slice: Tuple[float, float] = (100, 160),
                        lat_slice: Tuple[float, float] = (0, 50),
                        isLog: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load data from all folders for a specific date

        Args:
            date: Date string in format 'YYYYMMDD' (e.g., '20250501')
            lon_slice: Longitude range (min, max)
            lat_slice: Latitude range (min, max)
            isLog: Whether to print loading progress (default: False)

        Returns:
            Nested dictionary: {folder_name: {variable_name: numpy_array}}
        """
        result = {}

        if isLog:
            print(f"\n{'='*60}")
            print(f"Loading data for date: {date}")
            print(f"{'='*60}")

        for folder_name in self.folders.values():
            try:
                # Background folder reads data from 7 days earlier
                if folder_name == 'Background':
                    lookup_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')
                else:
                    lookup_date = date

                # Get all files in the folder
                folder_path = self.base_path / folder_name
                files = sorted(folder_path.glob("*.nc"))

                # Find the file matching the date
                matching_file = None
                for file in files:
                    if lookup_date in file.name:
                        matching_file = file
                        break

                if matching_file is None:
                    if isLog:
                        print(f"\nWarning: No file found for date {lookup_date} in {folder_name}")
                    continue

                if isLog:
                    print(f"\n{folder_name}: {matching_file.name}")

                # Determine if we need to select depth levels
                select_depth = folder_name in ['Glorys', 'Background']
                variables = self.variable_mapping[folder_name]

                # Load the file
                file_data = self.load_single_file(matching_file, variables, lon_slice, lat_slice, select_depth)

                # Print shapes
                if isLog:
                    for var_name, var_data in file_data.items():
                        print(f"  {var_name}: shape {var_data.shape}")

                result[folder_name] = file_data

            except Exception as e:
                if isLog:
                    print(f"Error loading {folder_name} for date {date}: {e}")

        return result

    def load_folder_data(self, folder_name: str,
                        lon_slice: Tuple[float, float] = (100, 160),
                        lat_slice: Tuple[float, float] = (0, 50)) -> Dict[str, List[np.ndarray]]:
        """
        Load all data from a specific folder

        Args:
            folder_name: Name of the folder (SSS, SLA, SST, Glorys, Background)
            lon_slice: Longitude range (min, max)
            lat_slice: Latitude range (min, max)

        Returns:
            Dictionary mapping variable names to lists of numpy arrays (one per file)
        """
        files = self.get_files_in_folder(folder_name)
        variables = self.variable_mapping[folder_name]

        # Determine if we need to select depth levels (for Glorys and Background)
        select_depth = folder_name in ['Glorys', 'Background']

        # Initialize result dictionary
        result = {var: [] for var in variables}

        print(f"\nLoading {folder_name} data from {len(files)} files...")
        if select_depth:
            print(f"  Selecting {len(self.depth_indices)} depth levels: {self.depth_indices}")

        for file_path in files:
            file_data = self.load_single_file(file_path, variables, lon_slice, lat_slice, select_depth)

            for var in variables:
                if var in file_data:
                    result[var].append(file_data[var])

        # Convert lists to numpy arrays (stack along time dimension)
        for var in variables:
            if result[var]:
                result[var] = np.stack(result[var], axis=0)
                print(f"  {var}: shape {result[var].shape}")

        return result

    def load_all_datasets(self,
                         lon_slice: Tuple[float, float] = (100, 160),
                         lat_slice: Tuple[float, float] = (0, 50)) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load all datasets from all folders

        Args:
            lon_slice: Longitude range (min, max)
            lat_slice: Latitude range (min, max)

        Returns:
            Nested dictionary: {folder_name: {variable_name: numpy_array}}
        """
        all_data = {}

        for folder_name in self.folders.values():
            try:
                folder_data = self.load_folder_data(folder_name, lon_slice, lat_slice)
                all_data[folder_name] = folder_data
            except Exception as e:
                print(f"Error loading {folder_name}: {e}")

        return all_data


def main():
    """Example usage"""
    # Initialize loader
    loader = OceanDatasetLoader()

    # Option 1: Load data for a single date
    print("=" * 60)
    print("Loading data for a single date")
    print("=" * 60)

    date = '20250501'  # Change this to load different dates
    data = loader.load_single_date(date)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Data Summary for {date}")
    print("=" * 60)

    for folder_name, folder_data in data.items():
        print(f"\n{folder_name}:")
        for var_name, var_data in folder_data.items():
            print(f"  {var_name}:")
            print(f"    Shape: {var_data.shape}")
            print(f"    Dtype: {var_data.dtype}")
            print(f"    Range: [{np.nanmin(var_data):.4f}, {np.nanmax(var_data):.4f}]")
            print(f"    NaN count: {np.isnan(var_data).sum()}")

    # Option 2: Load data for multiple dates
    print("\n\n" + "=" * 60)
    print("Loading data for multiple dates")
    print("=" * 60)

    dates = ['20250501', '20250502', '20250503']
    all_dates_data = {}

    for date in dates:
        all_dates_data[date] = loader.load_single_date(date)

    print("\n" + "=" * 60)
    print(f"Loaded data for {len(all_dates_data)} dates")
    print("=" * 60)

    return data


if __name__ == "__main__":
    data = main()