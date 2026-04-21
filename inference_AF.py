"""
Ocean Reconstruction Model Inference Script (AF mode)

使用 AF 数据集进行推理：
  - 背景场：D:\\datasets\\AF_so  /  D:\\datasets\\AF_thetao（目标日期 -7 天）
  - 当天数据：同上（目标日期当天）
  - SSS = AF_so  第 0 层（表面）
  - SST = AF_thetao 第 0 层（表面）

仿照 inference_glory.py，使用相同的模型和归一化参数。

Usage:
    python inference_AF.py --date 20260202 --model_path best_model.pth
"""

import os
import argparse
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from datetime import datetime

from load_datasets_AF import AFDatasetLoader
from visualize_results_glory import visualize_glory_results

# ── Model import ──────────────────────────────────────────────────────────────
from models.simple_convnext_net import ConvNeXtUNet as mymodel

# =============================================================================
# Data-index configuration
# =============================================================================
# 使用 SSS + SST 两个表面变量
SURFACE_VARS = [
    'sss',   # Sea Surface Salinity  ← AF_so 第 0 层
    'sst',   # Sea Surface Temperature ← AF_thetao 第 0 层
]

# 与 OceanDatasetLoader 的键名映射对应：
#   _SURFACE_INDEX[v] = [logical_folder, nc_variable, norm_key]
_SURFACE_INDEX = {
    'sss': ['SSS', 'so',     'sss'],
    'sst': ['SST', 'thetao', 'sst'],
}

# data_index 列表（用于 _flatten）
data_index = (
    [_SURFACE_INDEX[v] for v in SURFACE_VARS] +
    [
        ['Glorys',     'thetao', 'label_t_3d'],
        ['Glorys',     'so',     'label_s_3d'],
        ['Background', 'thetao', 'bg_t_3d'],
        ['Background', 'so',     'bg_s_3d'],
    ]
)

# IN_CHANNELS = len(SURFACE_VARS) + 20(bg_t) + 20(bg_s)
IN_CHANNELS = len(SURFACE_VARS) + 40   # = 42

# Default profile coordinates (lat, lon) for vertical profile analysis
DEFAULT_PROFILE_COORDS = [
    (12.5, 115.0),
    (25.0, 130.0),
    (37.5, 145.0),
]


# =============================================================================
# Model helpers
# =============================================================================

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


# =============================================================================
# Data preparation
# =============================================================================

def _flatten(raw_data) -> dict:
    """Extract fields from the nested raw_data dict using data_index."""
    out = {}
    for folder, var, name in data_index:
        out[name] = raw_data[folder][var]
    return out


def prepare_input(raw_data, norm_stats=None) -> torch.Tensor:
    """
    Build the model input tensor.

    Shape: (IN_CHANNELS, H, W) = (42, H, W)
    Channels: [sss(1), sst(1), bg_t_3d(20), bg_s_3d(20)]
    """
    data = _flatten(raw_data)

    channels = []

    # 1. 表面变量
    for v in SURFACE_VARS:
        name = _SURFACE_INDEX[v][2]   # 'sss' 或 'sst'
        arr = data[name]
        if norm_stats is not None:
            arr = normalize_data(arr, name, norm_stats)
        channels.append(np.expand_dims(arr, axis=0))   # (1, H, W)

    # 2. 背景场
    bg_t = data['bg_t_3d']   # (20, H, W)
    bg_s = data['bg_s_3d']   # (20, H, W)
    if norm_stats is not None:
        bg_t = normalize_data(bg_t, 'bg_t_3d', norm_stats)
        bg_s = normalize_data(bg_s, 'bg_s_3d', norm_stats)
    channels.append(bg_t)
    channels.append(bg_s)

    inputs = np.concatenate(channels, axis=0).astype(np.float32)   # (42, H, W)
    inputs = np.nan_to_num(inputs, nan=0.0)
    return torch.from_numpy(inputs)


def prepare_target(raw_data) -> np.ndarray:
    """Return target array (40, H, W) = [label_t_3d, label_s_3d]."""
    data = _flatten(raw_data)
    return np.concatenate(
        [data['label_t_3d'], data['label_s_3d']], axis=0
    ).astype(np.float32)


def prepare_background(raw_data) -> np.ndarray:
    """Return background array (40, H, W) = [bg_t_3d, bg_s_3d]."""
    data = _flatten(raw_data)
    return np.concatenate(
        [data['bg_t_3d'], data['bg_s_3d']], axis=0
    ).astype(np.float32)


# =============================================================================
# NetCDF I/O
# =============================================================================

def save_to_netcdf(prediction, target, background, date_str: str, save_dir: str):
    """Save prediction, target, and background arrays to NetCDF files."""
    os.makedirs(save_dir, exist_ok=True)

    n_depth, n_lat, n_lon = prediction[:20].shape
    lon = np.linspace(100, 159.875, n_lon)
    lat = np.linspace(0,   49.875,  n_lat)
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

    pred_ds   = make_ds(prediction[:20],  prediction[20:],  'Predicted ocean reconstruction (AF)')
    target_ds = make_ds(target[:20],      target[20:],      'Ground truth ocean data (AF)')
    bg_ds     = make_ds(background[:20],  background[20:],  'Background field (AF -7 days)')

    pred_path   = os.path.join(save_dir, f'prediction_{date_str}.nc')
    target_path = os.path.join(save_dir, f'target_{date_str}.nc')
    bg_path     = os.path.join(save_dir, f'background_{date_str}.nc')

    pred_ds.to_netcdf(pred_path,   engine='scipy')
    target_ds.to_netcdf(target_path, engine='scipy')
    bg_ds.to_netcdf(bg_path,     engine='scipy')

    print(f"✅ Saved prediction:  {pred_path}")
    print(f"✅ Saved target:      {target_path}")
    print(f"✅ Saved background:  {bg_path}")
    return pred_path, target_path, bg_path


# =============================================================================
# Metrics
# =============================================================================

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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ocean Reconstruction Inference (AF mode — SST/SSS from AF data)')
    parser.add_argument('--model_path', type=str,
                        default='./logs/20260420_122400/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--date', type=str, default='20260202',
                        help='Inference date (YYYYMMDD)')
    parser.add_argument('--save_dir', type=str, default='./inference_AF_results',
                        help='Root directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--profile_coords', type=str,
                        default="12.5,115; 25.0,130.0; 37.5,145; 20.0,120.0; 30.0,135.0",
                        help='Semicolon-separated lat,lon pairs for vertical profiles')
    args = parser.parse_args()

    # ── Profile coordinates ────────────────────────────────────────────────
    if args.profile_coords:
        profile_coords = []
        for pair in args.profile_coords.split(';'):
            lat_s, lon_s = pair.strip().split(',')
            profile_coords.append((float(lat_s), float(lon_s)))
        print(f"Profile coordinates: {profile_coords}")
    else:
        profile_coords = DEFAULT_PROFILE_COORDS

    # ── Output directory ───────────────────────────────────────────────────
    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results directory: {save_dir}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model, norm_stats = load_model(args.model_path, device)

    # ── Load raw data ─────────────────────────────────────────────────────
    print("\nLoading data...")
    # SSS / SST / Background ← AF 数据；Glorys(target label) ← D:\datasets\Glorys
    dataloader = AFDatasetLoader()
    raw_data = dataloader.load_single_date(args.date, isLog=True)

    # ── Prepare tensors ───────────────────────────────────────────────────
    print("\nPreparing input / target / background data...")
    inputs     = prepare_input(raw_data, norm_stats).unsqueeze(0).to(device)
    target     = prepare_target(raw_data)       # (40, H, W)
    background = prepare_background(raw_data)   # (40, H, W)

    print(f"  Input shape:      {inputs.shape}")
    print(f"  Target shape:     {target.shape}")
    print(f"  Background shape: {background.shape}")
    print(f"  Target NaN ratio:     {np.isnan(target).mean():.2%}")
    print(f"  Background NaN ratio: {np.isnan(background).mean():.2%}")

    # ── Inference ─────────────────────────────────────────────────────────
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(inputs)

    pred_norm = outputs.squeeze(0).cpu().numpy()   # (40, H, W)

    # ── Denormalise ───────────────────────────────────────────────────────
    if norm_stats is not None:
        print("Denormalising predictions...")
        pred_temp = denormalize_data(
            torch.from_numpy(pred_norm[:20]), 'label_t_3d', norm_stats).numpy()
        pred_salt = denormalize_data(
            torch.from_numpy(pred_norm[20:]), 'label_s_3d', norm_stats).numpy()
        prediction = np.concatenate([pred_temp, pred_salt], axis=0)
    else:
        prediction = pred_norm

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\nComputing metrics...")

    metrics      = compute_metrics(prediction,      target)
    temp_metrics = compute_metrics(prediction[:20], target[:20])
    salt_metrics = compute_metrics(prediction[20:], target[20:])

    temp_layer_rmse = compute_layer_rmse(prediction[:20], target[:20])
    salt_layer_rmse = compute_layer_rmse(prediction[20:], target[20:])

    bg_metrics      = compute_metrics(background,      target)
    bg_temp_metrics = compute_metrics(background[:20], target[:20])
    bg_salt_metrics = compute_metrics(background[20:], target[20:])

    bg_temp_layer_rmse = compute_layer_rmse(background[:20], target[:20])
    bg_salt_layer_rmse = compute_layer_rmse(background[20:], target[20:])

    print_metrics_table(metrics, temp_metrics, salt_metrics,
                        bg_metrics, bg_temp_metrics, bg_salt_metrics)

    # ── Save NetCDF ───────────────────────────────────────────────────────
    print("\nSaving NetCDF files...")
    pred_path, target_path, bg_path = save_to_netcdf(
        prediction, target, background, args.date, save_dir)

    # ── Visualise ─────────────────────────────────────────────────────────
    html_path = visualize_glory_results(
        pred_path, target_path, bg_path,
        metrics, temp_metrics, salt_metrics,
        save_dir, args.date,
        profile_coords=profile_coords,
        temp_layer_rmse=temp_layer_rmse,
        salt_layer_rmse=salt_layer_rmse,
        bg_metrics=bg_metrics,
        bg_temp_metrics=bg_temp_metrics,
        bg_salt_metrics=bg_salt_metrics,
        bg_temp_layer_rmse=bg_temp_layer_rmse,
        bg_salt_layer_rmse=bg_salt_layer_rmse,
    )

    print("\n" + "=" * 60)
    print("Inference (AF mode) completed successfully!")
    print(f"Results saved to: {save_dir}")
    print(f"HTML report:      {html_path}")
    print("=" * 60)

    return pred_path, target_path, bg_path, html_path


if __name__ == '__main__':
    main()