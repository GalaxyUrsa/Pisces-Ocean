"""
Ocean Inference Utilities

Shared helpers used by inference.py and autoregressive_forecast.py.
"""

import os
import numpy as np
import torch
import xarray as xr

from Data_Config import (
    INPUT_VARS, OUTPUT_VARS, RAW_DATASET_PATH,
    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END,
    NAN_FILL_VALUE, N_DEPTH, DEPTH_VALUES,
    MODEL_CLASS as mymodel, IN_CHANNELS, OUT_CHANNELS,
)

# Mapping from nc_var name → channel slice in the stacked output array
_offset = 0
OUT_VAR_SLICES: dict = {}
OUT_VAR_IS_3D: dict = {}
for _e in OUTPUT_VARS:
    _is3d = bool(_e[3].get('select_depth'))
    _n = N_DEPTH if _is3d else 1
    OUT_VAR_SLICES[_e[1]] = slice(_offset, _offset + _n)
    OUT_VAR_IS_3D[_e[1]] = _is3d
    _offset += _n

DEFAULT_PROFILE_COORDS = [
    (12.5, 115.0),
    (25.0, 130.0),
    (37.5, 145.0),
]


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    model = mymodel(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    norm_stats = checkpoint.get('norm_stats', None)
    if norm_stats is not None:
        print("✓ Normalisation statistics loaded from checkpoint")
    else:
        print("⚠ Warning: No normalisation statistics found in checkpoint")
    print("Model loaded successfully")
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

    for entry in INPUT_VARS:
        key = entry[2]
        arr = data[key]
        if norm_stats is not None:
            arr = normalize_data(arr, key, norm_stats)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        channels.append(arr)

    inputs = np.concatenate(channels, axis=0).astype(np.float32)
    inputs = inputs[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
    inputs = np.nan_to_num(inputs, nan=NAN_FILL_VALUE)
    return torch.from_numpy(inputs)


def _collect_channels(data, keys) -> np.ndarray:
    channels = []
    for k in keys:
        a = data[k]
        if a.ndim == 2:
            a = np.expand_dims(a, axis=0)
        channels.append(a)
    return np.concatenate(channels, axis=0).astype(np.float32)


def prepare_target(raw_data) -> np.ndarray:
    """Return target array (40, H, W) from OUTPUT_VARS."""
    data = _flatten(raw_data)
    arr = _collect_channels(data, [e[2] for e in OUTPUT_VARS])
    return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]


def prepare_background(raw_data) -> np.ndarray:
    """Return background array (40, H, W) from OUTPUT_VARS (label_ → bg_)."""
    data = _flatten(raw_data)
    bg_keys = [e[2].replace('label_', 'bg_') for e in OUTPUT_VARS]
    arr = _collect_channels(data, bg_keys)
    return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]


def save_to_netcdf(prediction, target, background, date_str: str, save_dir: str):
    """Save prediction, target (optional), and background arrays to NetCDF files."""
    os.makedirs(save_dir, exist_ok=True)

    n_lat, n_lon = prediction.shape[-2], prediction.shape[-1]
    lon = np.linspace(100, 159.875, n_lon)
    lat = np.linspace(0,  49.875,  n_lat)
    depth_values = DEPTH_VALUES

    y, m, d = date_str[:4], date_str[4:6], date_str[6:]
    time_coord = np.array([np.datetime64(f"{y}-{m}-{d}", 'ns')])

    def make_ds(arr, description):
        data_vars = {}
        for nc_var, sl in OUT_VAR_SLICES.items():
            chunk = arr[sl]  # (N_DEPTH, H, W) or (1, H, W)
            if OUT_VAR_IS_3D[nc_var]:
                data_vars[nc_var] = (['time', 'depth', 'latitude', 'longitude'], chunk[np.newaxis])
            else:
                data_vars[nc_var] = (['time', 'latitude', 'longitude'], chunk[0][np.newaxis])
        coords = {'time': time_coord, 'depth': depth_values, 'latitude': lat, 'longitude': lon}
        return xr.Dataset(data_vars, coords=coords, attrs={'description': description, 'date': date_str})

    pred_ds = make_ds(prediction, 'Predicted ocean reconstruction')
    bg_ds   = make_ds(background, 'Background field')

    pred_path = os.path.join(save_dir, f'prediction_{date_str}.nc')
    bg_path   = os.path.join(save_dir, f'background_{date_str}.nc')

    pred_ds.to_netcdf(pred_path)
    bg_ds.to_netcdf(bg_path)

    saved = [pred_path, bg_path]
    print(f"✅ Saved prediction:  {pred_path}")
    print(f"✅ Saved background:  {bg_path}")

    if target is not None:
        target_ds = make_ds(target, 'Ground truth ocean data')
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


def print_metrics_table(metrics, var_metrics: dict, bg_metrics, bg_var_metrics: dict):
    """Pretty-print a comparison table to stdout. Columns are driven by var_metrics keys."""
    from Data_Config import VAR_VIZ_CONFIG, VAR_VIZ_DEFAULT

    def fmt(d, key): return f"{d[key]:.6f}" if d is not None else "   N/A  "

    var_names = list(var_metrics.keys())
    col_w = 12

    # Header
    header = f"{'Metric':<12} {'Pred-Overall':>{col_w}}"
    for v in var_names:
        label = VAR_VIZ_CONFIG.get(v, VAR_VIZ_DEFAULT)['label'].split(' ')[0]  # e.g. 'Temperature'
        header += f" {'Pred-' + label[:4]:>{col_w}}"
    header += f" {'BG-Overall':>{col_w}}"
    for v in var_names:
        label = VAR_VIZ_CONFIG.get(v, VAR_VIZ_DEFAULT)['label'].split(' ')[0]
        header += f" {'BG-' + label[:4]:>{col_w}}"

    sep = "-" * len(header)

    def row(name, key):
        s = f"{name:<12} {fmt(metrics, key):>{col_w}}"
        for v in var_names:
            s += f" {fmt(var_metrics.get(v), key):>{col_w}}"
        s += f" {fmt(bg_metrics, key):>{col_w}}"
        for v in var_names:
            s += f" {fmt(bg_var_metrics.get(v), key):>{col_w}}"
        return s

    print(f"\n{sep}")
    print(header)
    print(sep)
    print(row("RMSE",        "rmse"))
    print(row("MAE",         "mae"))
    print(row("Correlation", "corr"))
    print(sep)
