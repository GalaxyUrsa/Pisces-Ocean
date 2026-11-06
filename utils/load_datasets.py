"""
Dataset Loader for Ocean Data Sources

Loads NetCDF data driven by a data_index defined in train.py.
Each entry in data_index is a triple: [folder, variable, output_name].
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from pathlib import Path

from Data_Config import DEPTH_INDICES


_DATE_RE = re.compile(r'(\d{8})')


class OceanDatasetLoader:
    """Load ocean datasets driven by an external data_index."""

    def __init__(self, base_path: str = r"D:\datasets"):
        self.base_path = Path(base_path)

        self.depth_indices = DEPTH_INDICES

        # 每个 folder 第一次访问时建立 {YYYYMMDD: Path} 索引，后续 O(1) 查询。
        self._folder_index: Dict[str, Dict[str, Path]] = {}

    def _get_folder_index(self, folder: str) -> Dict[str, Path]:
        idx = self._folder_index.get(folder)
        if idx is not None:
            return idx
        folder_path = self.base_path / folder
        idx = {}
        if folder_path.exists():
            for f in folder_path.glob("*.nc"):
                m = _DATE_RE.search(f.name)
                if m:
                    idx[m.group(1)] = f
        self._folder_index[folder] = idx
        return idx

    def load_single_file(self, file_path: Path, variables: List[str],
                         lon_slice: Tuple[float, float] = (105, 125),
                         lat_slice: Tuple[float, float] = (5, 21.7),
                         select_depth: bool = False) -> Dict[str, np.ndarray]:
        """
        Load specific variables from a single NetCDF file.

        Returns:
            Dict mapping variable name to numpy array.
        """
        ds = xr.open_dataset(file_path)

        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

        result = {}
        for var in variables:
            if var not in ds:
                print(f"Warning: Variable '{var}' not found in {file_path.name}")
                continue

            lat_vals = ds[lat_name].values
            lat_min, lat_max = lat_slice
            if lat_vals[0] > lat_vals[-1]:  # 降序
                lat_sel = slice(lat_max, lat_min)
            else:
                lat_sel = slice(lat_min, lat_max)

            var_data = ds[var].sel(
                {lon_name: slice(lon_slice[0], lon_slice[1]),
                 lat_name: lat_sel}
            )

            if select_depth and 'depth' in var_data.dims:
                var_data = var_data.isel(depth=self.depth_indices)

            arr = np.squeeze(var_data.values)
            # 统一为升序 lat（低纬 → 高纬），ERA5 等降序数据需要翻转
            sel_lat = var_data[lat_name].values
            if sel_lat[0] > sel_lat[-1]:
                arr = arr[..., ::-1, :].copy()
            result[var] = arr

        ds.close()
        return result

    def load_single_date(self, date: str,
                         data_index: List[List[str]],
                         lon_slice: Tuple[float, float] = (100, 160),
                         lat_slice: Tuple[float, float] = (0, 50),
                         isLog: bool = False) -> Dict[str, np.ndarray]:
        """
        Load all data required by data_index for a given date.

        Args:
            date:       Date string 'YYYYMMDD'.
            data_index: List of [folder, variable, output_name, opts?] entries.
            lon_slice:  Longitude range (min, max).
            lat_slice:  Latitude range (min, max).
            isLog:      Print loading progress.

        Returns:
            Flat dict: {output_name: numpy_array}
        """
        if isLog:
            print(f"\n{'='*60}")
            print(f"Loading data for date: {date}")
            print(f"{'='*60}")

        # Pass 1: collect all variables needed per (folder, lookup_date, select_depth)
        key_vars: Dict[tuple, List[str]] = {}
        entry_keys = []
        for entry in data_index:
            folder, var, output_name = entry[0], entry[1], entry[2]
            opts = entry[3] if len(entry) > 3 else {}
            offset_days = opts.get('bg_offset_days', 0)
            if offset_days:
                lookup_date = (datetime.strptime(date, '%Y%m%d') + timedelta(days=offset_days)).strftime('%Y%m%d')
            else:
                lookup_date = date
            select_depth = opts.get('select_depth', False)
            cache_key = (folder, lookup_date, select_depth)
            if var not in key_vars.setdefault(cache_key, []):
                key_vars[cache_key].append(var)
            entry_keys.append((cache_key, var, output_name))

        # Pass 2: load each file once with all needed variables
        file_cache: Dict[tuple, Dict[str, np.ndarray]] = {}
        for cache_key, vars_needed in key_vars.items():
            folder, lookup_date, select_depth = cache_key
            try:
                folder_idx = self._get_folder_index(folder)
                matching_file = folder_idx.get(lookup_date)

                if matching_file is None:
                    if isLog:
                        print(f"Warning: No file found for date {lookup_date} in {folder}")
                    file_cache[cache_key] = {}
                    continue

                if isLog:
                    print(f"\n{folder} [{lookup_date}]: {matching_file.name}")

                file_cache[cache_key] = self.load_single_file(
                    matching_file, vars_needed, lon_slice, lat_slice, select_depth
                )

            except Exception as e:
                if isLog:
                    print(f"Error loading {folder} for date {date}: {e}")
                file_cache[cache_key] = {}

        # Pass 3: assemble result in data_index order
        result: Dict[str, np.ndarray] = {}
        for cache_key, var, output_name in entry_keys:
            cached = file_cache.get(cache_key, {})
            if var in cached:
                if isLog:
                    print(f"  {output_name} ({var}): shape {cached[var].shape}")
                result[output_name] = cached[var]

        return result