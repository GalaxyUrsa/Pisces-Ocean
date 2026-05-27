"""
Ocean Inference Utilities

Shared helpers used by inference.py and autoregressive_forecast.py.
"""

import os
import numpy as np
import torch
import xarray as xr

from Data_Config import (
    SURFACE_VARS, _SURFACE_INDEX, RAW_DATASET_PATH,
    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END,
    NAN_FILL_VALUE,
)
from models.simple_convnext_net import ConvNeXtUNet as mymodel

IN_CHANNELS = len(SURFACE_VARS) + 40

DEFAULT_PROFILE_COORDS = [
    (12.5, 115.0),
    (25.0, 130.0),
    (37.5, 145.0),
]


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    model = mymodel(in_channels=IN_CHANNELS, out_channels=40).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    norm_stats = checkpoint.get('norm_stats', None)
    if norm_stats is not None:
        print("✓ Normalisation statistics loaded from checkpoint")
    else:
        print("⚠ Warning: No normalisation statistics found in checkpoint")
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    return model, norm_stats


def normalize_data(data, var_name: str, norm_stats):
    """Z-score normalisation."""
    if norm_stats is None:
        return data
    mean = norm_stats[var_name]['mean']
    std  = norm_stats[var_name]['std']
    return (data - mean) / std


def denormalize_data(data, var_name: str, norm_stats):
    """Reverse Z-score normalisation."""
    if norm_stats is None:
        return data
    mean = norm_stats[var_name]['mean']
    std  = norm_stats[var_name]['std']
    if isinstance(data, torch.Tensor):
        return data * std + mean
    return data * std + mean


def _flatten(raw_data) -> dict:
    return raw_data


def prepare_input(raw_data, norm_stats=None) -> torch.Tensor:
    """
    Build the model input tensor.
    Shape: (IN_CHANNELS, H, W)
    """
    data = _flatten(raw_data)
    channels = []

    for v in SURFACE_VARS:
        name = _SURFACE_INDEX[v][2]
        arr = data[name]
        if norm_stats is not None:
            arr = normalize_data(arr, name, norm_stats)
        channels.append(np.expand_dims(arr, axis=0))

    bg_t = data['bg_t_3d']
    bg_s = data['bg_s_3d']
    if norm_stats is not None:
        bg_t = normalize_data(bg_t, 'bg_t_3d', norm_stats)
        bg_s = normalize_data(bg_s, 'bg_s_3d', norm_stats)
    channels.append(bg_t)
    channels.append(bg_s)

    inputs = np.concatenate(channels, axis=0).astype(np.float32)
    inputs = inputs[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
    inputs = np.nan_to_num(inputs, nan=NAN_FILL_VALUE)
    return torch.from_numpy(inputs)


def prepare_target(raw_data) -> np.ndarray:
    """Return target array (40, H, W) = [label_t_3d, label_s_3d]."""
    data = _flatten(raw_data)
    arr = np.concatenate(
        [data['label_t_3d'], data['label_s_3d']], axis=0
    ).astype(np.float32)
    return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]


def prepare_background(raw_data) -> np.ndarray:
    """Return background array (40, H, W) = [bg_t_3d, bg_s_3d]."""
    data = _flatten(raw_data)
    arr = np.concatenate(
        [data['bg_t_3d'], data['bg_s_3d']], axis=0
    ).astype(np.float32)
    return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]


def save_to_netcdf(prediction, target, background, date_str: str, save_dir: str):
    """Save prediction, target (optional), and background arrays to NetCDF files."""
    os.makedirs(save_dir, exist_ok=True)

    n_depth, n_lat, n_lon = prediction[:20].shape
    lon = np.linspace(100, 159.875, n_lon)
    lat = np.linspace(0,  49.875,  n_lat)
    depth_values = np.array([
        0.49, 2.65, 5.08, 7.93, 11.41, 15.81, 21.60, 29.44, 40.34, 55.76,
        77.85, 92.32, 109.73, 130.67, 155.85, 186.13, 222.48, 318.13, 453.94, 643.57
    ])[:n_depth]

    y, m, d = date_str[:4], date_str[4:6], date_str[6:]
    time_coord = np.array([np.datetime64(f"{y}-{m}-{d}", 'ns')])

    def make_ds(temp, salt, description):
        return xr.Dataset(
            {
                'thetao': (['time', 'depth', 'latitude', 'longitude'],
                           temp[np.newaxis, :, :, :]),
                'so':     (['time', 'depth', 'latitude', 'longitude'],
                           salt[np.newaxis, :, :, :]),
            },
            coords={
                'time':      time_coord,
                'depth':     depth_values,
                'latitude':  lat,
                'longitude': lon,
            },
            attrs={'description': description, 'date': date_str}
        )

    pred_ds = make_ds(prediction[:20], prediction[20:], 'Predicted ocean reconstruction')
    bg_ds   = make_ds(background[:20], background[20:], 'Background field')

    pred_path = os.path.join(save_dir, f'prediction_{date_str}.nc')
    bg_path   = os.path.join(save_dir, f'background_{date_str}.nc')

    pred_ds.to_netcdf(pred_path)
    bg_ds.to_netcdf(bg_path)

    saved = [pred_path, bg_path]
    print(f"✅ Saved prediction:  {pred_path}")
    print(f"✅ Saved background:  {bg_path}")

    if target is not None:
        target_ds = make_ds(target[:20], target[20:], 'Ground truth ocean data')
        target_path = os.path.join(save_dir, f'target_{date_str}.nc')
        target_ds.to_netcdf(target_path)
        saved.append(target_path)
        print(f"✅ Saved target:      {target_path}")

    return tuple(saved)


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Compute RMSE, MAE, and Pearson correlation over valid (non-NaN) points."""
    mask = ~np.isnan(target) & ~np.isnan(prediction)
    if mask.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'corr': np.nan}
    p = prediction[mask]
    t = target[mask]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(np.abs(p - t)))
    corr = float(np.corrcoef(p, t)[0, 1])
    return {'rmse': rmse, 'mae': mae, 'corr': corr}


def compute_layer_rmse(prediction: np.ndarray, target: np.ndarray) -> list:
    """Compute per-depth-layer RMSE."""
    rmse_list = []
    for i in range(prediction.shape[0]):
        mask = ~np.isnan(target[i]) & ~np.isnan(prediction[i])
        if mask.sum() == 0:
            rmse_list.append(np.nan)
        else:
            diff = prediction[i][mask] - target[i][mask]
            rmse_list.append(float(np.sqrt(np.mean(diff ** 2))))
    return rmse_list


def print_metrics_table(metrics, temp_metrics, salt_metrics,
                         bg_metrics, bg_temp_metrics, bg_salt_metrics):
    """Pretty-print a comparison table to stdout."""
    header = (f"{'Metric':<12} {'Pred-Overall':>14} {'Pred-Temp':>12} {'Pred-Salt':>12}"
              f" {'BG-Overall':>12} {'BG-Temp':>10} {'BG-Salt':>10}")
    sep = "-" * len(header)

    def row(name, key):
        def fmt(d): return f"{d[key]:.6f}" if d is not None else "   N/A  "
        return (f"{name:<12} {fmt(metrics):>14} {fmt(temp_metrics):>12}"
                f" {fmt(salt_metrics):>12} {fmt(bg_metrics):>12}"
                f" {fmt(bg_temp_metrics):>10} {fmt(bg_salt_metrics):>10}")

    print(f"\n{sep}")
    print(header)
    print(sep)
    print(row("RMSE",        "rmse"))
    print(row("MAE",         "mae"))
    print(row("Correlation", "corr"))
    print(sep)
