"""
Ocean Data Dataset Classes for Training

This module provides PyTorch Dataset classes for loading and preprocessing
oceanographic data for reconstruction and forecast model training.
"""

import os
import re
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, date
from typing import Tuple, List, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset


def date_series(last_n: int, length: int, ref_date: date = None) -> List[str]:
    """
    Generate a date series relative to a reference date.

    Args:
        last_n (int): Number of days from the reference date to the last date
        length (int): Length of the date series
        ref_date (date): Reference date, defaults to today

    Returns:
        List[str]: List of date strings in YYYYMMDD format
    """
    if ref_date is None:
        ref_date = date.today()
    start = ref_date - timedelta(days=last_n + length - 1)
    return [
        (start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(length)
    ]


def find_files_by_keyword_and_dates(root_folder: str, keyword_list: List[str], date_list: List[str]) -> Dict[str, List[str]]:
    """
    Find files in a directory tree that match specific keywords and dates.

    Args:
        root_folder (str): Root directory to search
        keyword_list (List[str]): Keywords to match in filenames
        date_list (List[str]): List of dates in YYYYMMDD format

    Returns:
        Dict[str, List[str]]: Dictionary mapping dates to file paths
    """
    result_dict = {}
    date_set = set(date_list)

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            match = re.search(r"(\d{8})", filename)
            if match:
                date_str = match.group(1)
                if date_str in date_set and any(k.lower() in filename.lower() for k in keyword_list):
                    full_path = os.path.join(dirpath, filename)
                    result_dict.setdefault(date_str, []).append(full_path)

    return result_dict


class OceanReconstructionDataset(Dataset):
    """
    Dataset for Ocean Reconstruction Model Training

    Input: Surface observations (5 channels) + Background subsurface (80 channels)
    Target: Current subsurface state (80 channels)

    Args:
        date_list (List[str]): List of dates in YYYYMMDD format
        surface_dir (str): Directory containing SLA data
        sst_dir (str): Directory containing SST data
        sss_dir (str): Directory containing SSS data
        background_dir (str): Directory containing background subsurface data
        target_dir (str): Directory containing target (GLORYS) data
        mean (np.ndarray): Mean values for normalization (85 channels)
        std (np.ndarray): Standard deviation for normalization (85 channels)
        background_offset (int): Days offset for background data (default: -7)
    """

    def __init__(
        self,
        date_list: List[str],
        surface_dir: str,
        sst_dir: str,
        sss_dir: str,
        background_dir: str,
        target_dir: str,
        mean: np.ndarray,
        std: np.ndarray,
        background_offset: int = -7
    ):
        self.surface_dir = surface_dir
        self.sst_dir = sst_dir
        self.sss_dir = sss_dir
        self.background_dir = background_dir
        self.target_dir = target_dir
        self.mean = mean
        self.std = std
        self.background_offset = background_offset

        # Depth indices for subsurface data
        self.depth_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]
        self.deep_vars = ['thetao', 'so', 'uo', 'vo']

        # Filter valid dates (only keep dates with all required files)
        print(f"Filtering valid dates from {len(date_list)} candidates...")
        self.date_list = self._filter_valid_dates(date_list)
        print(f"Found {len(self.date_list)} valid dates with complete data")

        if len(self.date_list) == 0:
            raise ValueError("No valid training samples found! Please check your data directories.")

    def _filter_valid_dates(self, date_list: List[str]) -> List[str]:
        """Filter dates that have all required data files"""
        valid_dates = []

        for date_str in date_list:
            # Calculate background date
            current_date = datetime.strptime(date_str, "%Y%m%d")
            background_date = current_date + timedelta(days=self.background_offset)
            background_date_str = background_date.strftime("%Y%m%d")

            try:
                # Check SLA file
                sla_files = find_files_by_keyword_and_dates(self.surface_dir, ["allsat"], [date_str])
                if date_str not in sla_files or len(sla_files[date_str]) == 0:
                    continue

                # Check SST file
                sst_files = find_files_by_keyword_and_dates(self.sst_dir, ["oisst"], [date_str])
                if date_str not in sst_files or len(sst_files[date_str]) == 0:
                    continue

                # Check SSS file
                sss_file = f"{self.sss_dir}/model_output_sss_{date_str}.nc"
                if not os.path.exists(sss_file):
                    continue

                # Check background file
                background_files = find_files_by_keyword_and_dates(
                    self.background_dir, ["ocean", "glorys", "cmems", "recon"], [background_date_str]
                )
                if background_date_str not in background_files or len(background_files[background_date_str]) == 0:
                    continue

                # Check target file
                target_files = find_files_by_keyword_and_dates(
                    self.target_dir, ["ocean", "glorys", "cmems", "recon"], [date_str]
                )
                if date_str not in target_files or len(target_files[date_str]) == 0:
                    continue

                # All files exist, add to valid dates
                valid_dates.append(date_str)

            except Exception:
                continue

        return valid_dates

    def __len__(self) -> int:
        return len(self.date_list)

    def _load_surface_data(self, date_str: str) -> np.ndarray:
        """Load surface observation data (SLA, SST, SSS, u, v)"""

        # Load SLA and velocity
        sla_files = find_files_by_keyword_and_dates(self.surface_dir, ["allsat"], [date_str])
        if date_str not in sla_files or len(sla_files[date_str]) == 0:
            raise FileNotFoundError(f"SLA file not found for date {date_str}")

        sla_file = sla_files[date_str][-1]
        ds_sla = xr.open_dataset(sla_file)

        sur_sla = ds_sla[['sla']].sel(
            longitude=slice(100, 160), latitude=slice(0, 50)
        ).to_array().values.reshape(-1, 400, 480)

        sur_var = ds_sla[['ugos', 'vgos']].sel(
            longitude=slice(100, 160), latitude=slice(0, 50)
        ).to_array().values.reshape(-1, 400, 480)

        # Load SST
        sst_files = find_files_by_keyword_and_dates(self.sst_dir, ["oisst"], [date_str])
        if date_str not in sst_files or len(sst_files[date_str]) == 0:
            raise FileNotFoundError(f"SST file not found for date {date_str}")

        sst_file = sst_files[date_str][0]
        sur_sst = xr.open_dataset(sst_file)[["sst"]].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values.reshape(-1, 400, 480)

        # Load SSS
        sss_file = f"{self.sss_dir}/model_output_sss_{date_str}.nc"
        if not os.path.exists(sss_file):
            raise FileNotFoundError(f"SSS file not found: {sss_file}")

        sur_so = xr.open_dataset(sss_file)[['sss_output']].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values.reshape(-1, 400, 480)

        # Concatenate: SLA(1) + SST(1) + SSS(1) + velocity(2) = 5 channels
        surface_data = np.concatenate((sur_sla, sur_sst, sur_so, sur_var), axis=0)

        return surface_data.astype(np.float32)

    def _load_subsurface_data(self, date_str: str, data_dir: str) -> np.ndarray:
        """Load subsurface data (temperature, salinity, u, v at 20 depth levels)"""

        # Find subsurface data file
        files = find_files_by_keyword_and_dates(data_dir, ["ocean", "glorys", "cmems", "recon"], [date_str])
        if date_str not in files or len(files[date_str]) == 0:
            raise FileNotFoundError(f"Subsurface file not found for date {date_str}")

        subsurface_file = files[date_str][0]

        # Load data
        ds = xr.open_dataset(subsurface_file)

        # Check if depth dimension exists and get available depth levels
        if 'depth' in ds.dims:
            available_depth_levels = len(ds['depth'])
            # Use only the depth indices that are available
            valid_depth_idx = [idx for idx in self.depth_idx if idx < available_depth_levels]

            if len(valid_depth_idx) < 20:
                # If file has fewer than 20 depth levels, use all available
                depth_var = ds[self.deep_vars].sel(
                    longitude=slice(100, 159.875), latitude=slice(0, 49.875)
                ).to_array().values
            else:
                # Use the specified depth indices
                depth_var = ds[self.deep_vars].sel(
                    longitude=slice(100, 159.875), latitude=slice(0, 49.875)
                ).isel(depth=valid_depth_idx).to_array().values
        else:
            # No depth dimension, load all data
            depth_var = ds[self.deep_vars].sel(
                longitude=slice(100, 159.875), latitude=slice(0, 49.875)
            ).to_array().values

        # Reshape to (80, 400, 480): 4 variables × 20 depth levels
        depth_var = depth_var.reshape(-1, 400, 480)

        return depth_var.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample

        Returns:
            input_data (torch.Tensor): Input tensor (85, 400, 480)
            target_data (torch.Tensor): Target tensor (80, 400, 480)
            mask (torch.Tensor): Valid data mask (80, 400, 480)
        """
        date_str = self.date_list[idx]

        # Calculate background date
        current_date = datetime.strptime(date_str, "%Y%m%d")
        background_date = current_date + timedelta(days=self.background_offset)
        background_date_str = background_date.strftime("%Y%m%d")

        try:
            # Load surface data (5 channels)
            surface_data = self._load_surface_data(date_str)

            # Load background subsurface data (80 channels)
            background_data = self._load_subsurface_data(background_date_str, self.background_dir)

            # Load target subsurface data (80 channels)
            target_data = self._load_subsurface_data(date_str, self.target_dir)

            # Concatenate input: surface + background = 85 channels
            input_data = np.concatenate([surface_data, background_data], axis=0)

            # Create mask for valid data
            mask = ~np.isnan(target_data)

            # Normalize input
            input_data = (input_data - self.mean.reshape(85, 1, 1)) / self.std.reshape(85, 1, 1)

            # Handle NaN values
            input_data = np.nan_to_num(input_data, nan=0.0)
            target_data = np.nan_to_num(target_data, nan=0.0)

            return (
                torch.from_numpy(input_data).float(),
                torch.from_numpy(target_data).float(),
                torch.from_numpy(mask.astype(np.float32))
            )

        except Exception as e:
            print(f"Error loading data for date {date_str}: {e}")
            raise


class OceanForecastDataset(Dataset):
    """
    Dataset for Ocean Forecast Model Training

    Input: Surface observations (5 channels) + Current subsurface state (80 channels)
    Target: Future ocean state (81 channels: SLA + 80 subsurface channels)

    Args:
        date_list (List[str]): List of dates in YYYYMMDD format
        surface_dir (str): Directory containing SLA data
        sst_dir (str): Directory containing SST data
        sss_dir (str): Directory containing SSS data
        current_dir (str): Directory containing current subsurface data
        target_dir (str): Directory containing target (future) data
        mean (np.ndarray): Mean values for normalization (85 channels)
        std (np.ndarray): Standard deviation for normalization (85 channels)
        lead_days (int): Forecast lead time in days (1-10)
    """

    def __init__(
        self,
        date_list: List[str],
        surface_dir: str,
        sst_dir: str,
        sss_dir: str,
        current_dir: str,
        target_dir: str,
        mean: np.ndarray,
        std: np.ndarray,
        lead_days: int = 1
    ):
        self.date_list = date_list
        self.surface_dir = surface_dir
        self.sst_dir = sst_dir
        self.sss_dir = sss_dir
        self.current_dir = current_dir
        self.target_dir = target_dir
        self.mean = mean
        self.std = std
        self.lead_days = lead_days

        # Depth indices for subsurface data
        self.depth_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]
        self.deep_vars = ['thetao', 'so', 'uo', 'vo']

    def __len__(self) -> int:
        return len(self.date_list)

    def _load_surface_data(self, date_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load surface observation data and return both surface data and SLA separately"""

        # Load SLA and velocity
        sla_files = find_files_by_keyword_and_dates(self.surface_dir, ["allsat"], [date_str])
        if date_str not in sla_files or len(sla_files[date_str]) == 0:
            raise FileNotFoundError(f"SLA file not found for date {date_str}")

        sla_file = sla_files[date_str][-1]
        ds_sla = xr.open_dataset(sla_file)

        sur_sla = ds_sla[['sla']].sel(
            longitude=slice(100, 160), latitude=slice(0, 50)
        ).to_array().values.reshape(-1, 400, 480)

        sur_var = ds_sla[['ugos', 'vgos']].sel(
            longitude=slice(100, 160), latitude=slice(0, 50)
        ).to_array().values.reshape(-1, 400, 480)

        # Load SST
        sst_files = find_files_by_keyword_and_dates(self.sst_dir, ["oisst"], [date_str])
        if date_str not in sst_files or len(sst_files[date_str]) == 0:
            raise FileNotFoundError(f"SST file not found for date {date_str}")

        sst_file = sst_files[date_str][0]
        sur_sst = xr.open_dataset(sst_file)[["sst"]].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values.reshape(-1, 400, 480)

        # Load SSS
        sss_file = f"{self.sss_dir}/model_output_sss_{date_str}.nc"
        if not os.path.exists(sss_file):
            raise FileNotFoundError(f"SSS file not found: {sss_file}")

        sur_so = xr.open_dataset(sss_file)[['sss_output']].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values.reshape(-1, 400, 480)

        # Concatenate: SLA(1) + SST(1) + SSS(1) + velocity(2) = 5 channels
        surface_data = np.concatenate((sur_sla, sur_sst, sur_so, sur_var), axis=0)

        return surface_data.astype(np.float32), sur_sla.astype(np.float32)

    def _load_subsurface_data(self, date_str: str, data_dir: str) -> np.ndarray:
        """Load subsurface data (temperature, salinity, u, v at 20 depth levels)"""

        # Find subsurface data file
        files = find_files_by_keyword_and_dates(data_dir, ["ocean", "glorys", "cmems", "recon"], [date_str])
        if date_str not in files or len(files[date_str]) == 0:
            raise FileNotFoundError(f"Subsurface file not found for date {date_str}")

        subsurface_file = files[date_str][0]

        # Load data
        ds = xr.open_dataset(subsurface_file)

        # Check if depth dimension exists and get available depth levels
        if 'depth' in ds.dims:
            available_depth_levels = len(ds['depth'])
            # Use only the depth indices that are available
            valid_depth_idx = [idx for idx in self.depth_idx if idx < available_depth_levels]

            if len(valid_depth_idx) < 20:
                # If file has fewer than 20 depth levels, use all available
                depth_var = ds[self.deep_vars].sel(
                    longitude=slice(100, 159.875), latitude=slice(0, 49.875)
                ).to_array().values
            else:
                # Use the specified depth indices
                depth_var = ds[self.deep_vars].sel(
                    longitude=slice(100, 159.875), latitude=slice(0, 49.875)
                ).isel(depth=valid_depth_idx).to_array().values
        else:
            # No depth dimension, load all data
            depth_var = ds[self.deep_vars].sel(
                longitude=slice(100, 159.875), latitude=slice(0, 49.875)
            ).to_array().values

        # Reshape to (80, 400, 480): 4 variables × 20 depth levels
        depth_var = depth_var.reshape(-1, 400, 480)

        return depth_var.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample

        Returns:
            input_data (torch.Tensor): Input tensor (85, 400, 480)
            target_data (torch.Tensor): Target tensor (81, 400, 480)
            mask (torch.Tensor): Valid data mask (81, 400, 480)
        """
        date_str = self.date_list[idx]

        # Calculate future date
        current_date = datetime.strptime(date_str, "%Y%m%d")
        future_date = current_date + timedelta(days=self.lead_days)
        future_date_str = future_date.strftime("%Y%m%d")

        try:
            # Load current surface data (5 channels)
            surface_data, _ = self._load_surface_data(date_str)

            # Load current subsurface data (80 channels)
            current_data = self._load_subsurface_data(date_str, self.current_dir)

            # Load future surface data (for SLA target)
            _, future_sla = self._load_surface_data(future_date_str)

            # Load future subsurface data (80 channels)
            future_subsurface = self._load_subsurface_data(future_date_str, self.target_dir)

            # Concatenate input: surface + current subsurface = 85 channels
            input_data = np.concatenate([surface_data, current_data], axis=0)

            # Concatenate target: SLA + future subsurface = 81 channels
            target_data = np.concatenate([future_sla, future_subsurface], axis=0)

            # Create mask for valid data
            mask = ~np.isnan(target_data)

            # Normalize input
            input_data = (input_data - self.mean.reshape(85, 1, 1)) / self.std.reshape(85, 1, 1)

            # Handle NaN values
            input_data = np.nan_to_num(input_data, nan=0.0)
            target_data = np.nan_to_num(target_data, nan=0.0)

            return (
                torch.from_numpy(input_data).float(),
                torch.from_numpy(target_data).float(),
                torch.from_numpy(mask.astype(np.float32))
            )

        except Exception as e:
            print(f"Error loading data for date {date_str}: {e}")
            raise


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset classes...")

    # Generate sample date list
    test_dates = date_series(0, 10, ref_date=date(2025, 7, 1))
    print(f"Test dates: {test_dates}")
