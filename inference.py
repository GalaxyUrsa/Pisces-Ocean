"""
Ocean Reconstruction Model Inference Script

The new model takes only the background subsurface fields (bg_t_3d + bg_s_3d) and surface observation field (sst + sss)
as input (40 channels total) and predicts the full 3D temperature + salinity.

Usage:
    python inference_glory.py --date 20260202 --model_path best_model.pth
"""

import os
import argparse
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from datetime import datetime

from load_datasets import OceanDatasetLoader
from visualize_results_glory import visualize_glory_results

from Data_Config import SURFACE_VARS, _SURFACE_INDEX, data_index, RAW_DATASET_PATH, CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END

# =============================================================================
# Model Import
# =============================================================================
# Swap to whichever architecture you trained with:
from models.simple_convnext_net import ConvNeXtUNet as mymodel
# from models.unet import UNet as mymodel
# from models.unet3d import UNet3D as mymodel
# from models.HCANet import HCANet as mymodel


# 40 input channels = bg_t_3d (20) + bg_s_3d (20)
IN_CHANNELS = len(SURFACE_VARS) + 40   # surface vars + bg_t_3d(20) + bg_s_3d(20) → 42

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
    """raw_data is already a flat {output_name: array} dict from load_single_date."""
    return raw_data

# =============================================================================
# Build The Input Tensor
# =============================================================================
def prepare_input(raw_data, norm_stats=None) -> torch.Tensor:
    """
    Build the model input tensor.

    Shape: (IN_CHANNELS, H, W) = (len(SURFACE_VARS)+40, H, W)
    Channels: [surface_vars × len(SURFACE_VARS), bg_t_3d × 20, bg_s_3d × 20]
    与 train.py 的 OceanDataset.__getitem__ 保持完全一致。
    """
    data = _flatten(raw_data)

    channels = []

    # 1. 表面变量（与训练时一致）
    for v in SURFACE_VARS:
        name = _SURFACE_INDEX[v][2]   # e.g. 'sss', 'sst'
        arr = data[name]
        if norm_stats is not None:
            arr = normalize_data(arr, name, norm_stats)
        channels.append(np.expand_dims(arr, axis=0))  # (1, H, W)

    # 2. 背景场
    bg_t = data['bg_t_3d']
    bg_s = data['bg_s_3d']
    if norm_stats is not None:
        bg_t = normalize_data(bg_t, 'bg_t_3d', norm_stats)
        bg_s = normalize_data(bg_s, 'bg_s_3d', norm_stats)
    channels.append(bg_t)   # (20, H, W)
    channels.append(bg_s)   # (20, H, W)

    inputs = np.concatenate(channels, axis=0).astype(np.float32)  # (IN_CHANNELS, H, W)
    inputs = inputs[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
    inputs = np.nan_to_num(inputs, nan=0.0)
    return torch.from_numpy(inputs)

# =============================================================================
# Build The Target Tensor
# =============================================================================
def prepare_target(raw_data) -> np.ndarray:
    """Return target array (40, H, W) = [label_t_3d, label_s_3d]."""
    data = _flatten(raw_data)
    arr = np.concatenate(
        [data['label_t_3d'], data['label_s_3d']], axis=0
    ).astype(np.float32)
    return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

# =============================================================================
# Build The Background Tensor
# =============================================================================
def prepare_background(raw_data) -> np.ndarray:
    """Return background array (40, H, W) = [bg_t_3d, bg_s_3d]."""
    data = _flatten(raw_data)
    arr = np.concatenate(
        [data['bg_t_3d'], data['bg_s_3d']], axis=0
    ).astype(np.float32)
    return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]


# =============================================================================
# NetCDF I/O
# =============================================================================

def save_to_netcdf(prediction, target, background, date_str: str, save_dir: str):
    """Save prediction, target, and background arrays to NetCDF files."""
    os.makedirs(save_dir, exist_ok=True)

    # Dynamically derive grid size from actual data shape (depth, lat, lon)
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

    pred_ds   = make_ds(prediction[:20],  prediction[20:],  'Predicted ocean reconstruction')
    target_ds = make_ds(target[:20],      target[20:],      'Ground truth ocean data')
    bg_ds     = make_ds(background[:20],  background[20:],  'Background field (Glorys -7 days)')

    pred_path   = os.path.join(save_dir, f'prediction_{date_str}.nc')
    target_path = os.path.join(save_dir, f'target_{date_str}.nc')
    bg_path     = os.path.join(save_dir, f'background_{date_str}.nc')

    pred_ds.to_netcdf(pred_path)
    target_ds.to_netcdf(target_path)
    bg_ds.to_netcdf(bg_path)

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
    """Compute per-depth-layer RMSE.

    Args:
        prediction : (N_layers, H, W)
        target     : (N_layers, H, W)

    Returns:
        list of float, length N_layers
    """
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
    header = f"{'Metric':<12} {'Pred-Overall':>14} {'Pred-Temp':>12} {'Pred-Salt':>12}" \
             f" {'BG-Overall':>12} {'BG-Temp':>10} {'BG-Salt':>10}"
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
    parser = argparse.ArgumentParser(description='Ocean Reconstruction Inference')
    parser.add_argument('--model_path', type=str, default='./logs/20260511_112905/best_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--date', type=str, default='20251220', help='Inference date (YYYYMMDD)')
    parser.add_argument('--save_dir', type=str, default='./inference_glory_results', help='Root directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--profile_coords', type=str, default="12.5,115; 25.0,130.0; 37.5,145; 20.0,120.0; 30.0,135.0", 
                        help='Semicolon-separated lat,lon pairs for vertical profiles')
    args = parser.parse_args()

    # Decode Profile Coordinates
    if args.profile_coords:
        profile_coords = []
        for pair in args.profile_coords.split(';'):
            lat_s, lon_s = pair.strip().split(',')
            profile_coords.append((float(lat_s), float(lon_s)))
        print(f"Profile coordinates: {profile_coords}")
    else:
        profile_coords = DEFAULT_PROFILE_COORDS

    # Make Output Dictionary
    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results directory: {save_dir}")

    # Choose Device For Inference
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model and Normalization States
    model, norm_stats = load_model(args.model_path, device)

    # Load Ocean Data
    print("\nLoading data...")
    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)
    raw_data = dataloader.load_single_date(args.date, data_index, isLog=True)

    # Transform Data to Tensor
    print("\nPreparing input / target / background data...")
    inputs     = prepare_input(raw_data, norm_stats).unsqueeze(0).to(device)
    target     = prepare_target(raw_data)      # (40, H, W)  – raw physical units
    background = prepare_background(raw_data)  # (40, H, W)  – raw physical units

    # Inference...
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(inputs)

    # Transform Tensor to Numpy
    pred_norm = outputs.squeeze(0).cpu().numpy()  # (40, H, W)

    # Denormalise
    if norm_stats is not None:
        print("\nDenormalising predictions...")
        # Gary
        # the normalization states is strange
        pred_temp = denormalize_data(torch.from_numpy(pred_norm[:20]), 'label_t_3d', norm_stats).numpy()
        pred_salt = denormalize_data(torch.from_numpy(pred_norm[20:]), 'label_s_3d', norm_stats).numpy()
        prediction = np.concatenate([pred_temp, pred_salt], axis=0)
    else:
        prediction = pred_norm

    # # ---------------------------------------------------------------------------
    # # 温度逐格点异常值过滤：
    # # 若某格点的预测温度与背景温度之差的绝对值 > 3（单位与数据一致，通常为 °C），
    # # 认为该格点预测异常（多发生在边缘/陆地边界），用背景值替换预测值。
    # # 替换同时作用于后续的 metrics 计算和 NetCDF 输出，盐度不做此过滤。
    # # ---------------------------------------------------------------------------
    # temp_outlier_mask = np.abs(prediction[:20] - background[:20]) > 3  # (20, H, W) bool
    # prediction[:20] = np.where(temp_outlier_mask, background[:20], prediction[:20])

    # Compute Metrics
    print("\nComputing metrics...")
    metrics      = compute_metrics(prediction,     target)
    temp_metrics = compute_metrics(prediction[:20], target[:20])
    salt_metrics = compute_metrics(prediction[20:], target[20:])
    temp_layer_rmse = compute_layer_rmse(prediction[:20], target[:20])
    salt_layer_rmse = compute_layer_rmse(prediction[20:], target[20:])

    bg_metrics      = compute_metrics(background,     target)
    bg_temp_metrics = compute_metrics(background[:20], target[:20])
    bg_salt_metrics = compute_metrics(background[20:], target[20:])
    bg_temp_layer_rmse = compute_layer_rmse(background[:20], target[:20])
    bg_salt_layer_rmse = compute_layer_rmse(background[20:], target[20:])

    print_metrics_table(metrics, temp_metrics, salt_metrics, bg_metrics, bg_temp_metrics, bg_salt_metrics)

    # Save Predict Result as NetCDF
    print("\nSaving NetCDF files...")
    pred_path, target_path, bg_path = save_to_netcdf(prediction, target, background, args.date, save_dir)

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
    print("Inference (Glory mode) completed successfully!")
    print(f"Results saved to: {save_dir}")
    print(f"HTML report:      {html_path}")
    print("=" * 60)

    return pred_path, target_path, bg_path, html_path


if __name__ == '__main__':
    main()
