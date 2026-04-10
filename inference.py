"""
Ocean Reconstruction Model Inference Script

This script performs inference using the trained model to reconstruct 3D ocean fields
from surface observations and background subsurface data.

Usage:
    python inference.py --date 20250508 --model_path best_model.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from load_datasets import OceanDatasetLoader
from visualize_results import visualize_inference_results

# from models.mymodel import SimpleModel as mymodel
# from models.unet import UNet as mymodel
# from models.unet3d import UNet3D as mymodel
from models.simple_convnext_net import ConvNeXtUNet as mymodel


# # Simple model definition (must match training model)
# class SimpleModel(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
#         self.conv4 = nn.Conv2d(64, out_channels, 3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.conv4(x)
#         return x


# =============================================================================
# 消融实验配置 — 与 train.py 保持一致
# =============================================================================
SURFACE_VARS = [
    'sss',  # Sea Surface Salinity
    'sst',  # Sea Surface Temperature
    'sla',  # Sea Level Anomaly
    'ugos',
    'vgos',
]

# 数据索引（自动推导，无需手动修改）
_SURFACE_INDEX = {
    'sss':  ['SSS', 'sos',  'sss'],
    'sst':  ['SST', 'sst',  'sst'],
    'sla':  ['SLA', 'sla',  'sla'],
    'ugos': ['SLA', 'ugos', 'ugos'],
    'vgos': ['SLA', 'vgos', 'vgos'],
}

data_index = (
    [_SURFACE_INDEX[v] for v in SURFACE_VARS] +
    [
        ['Glorys',     'thetao', 'label_t_3d'],
        ['Glorys',     'so',     'label_s_3d'],
        ['Background', 'thetao', 'bg_t_3d'],
        ['Background', 'so',     'bg_s_3d'],
    ]
)
IN_CHANNELS = len(SURFACE_VARS) + 40

# Default (lat, lon) coordinates for vertical profile analysis
# Covers South China Sea, East China Sea, and Northwest Pacific
DEFAULT_PROFILE_COORDS = [
    (12.5, 115.0),
    (25.0, 130.0),
    (37.5, 145.0),
]


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {model_path}")

    model = mymodel(in_channels=IN_CHANNELS, out_channels=40).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load normalization statistics if available
    norm_stats = checkpoint.get('norm_stats', None)
    if norm_stats is not None:
        print("✓ Normalization statistics loaded from checkpoint")
    else:
        print("⚠ Warning: No normalization statistics found in checkpoint")

    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    return model, norm_stats


def normalize_data(data, var_name, norm_stats):
    """Apply Z-score normalization to data"""
    if norm_stats is None:
        return data

    mean = norm_stats[var_name]['mean']
    std = norm_stats[var_name]['std']

    return (data - mean) / std


def denormalize_data(data, var_name, norm_stats):
    """Reverse Z-score normalization"""
    if norm_stats is None:
        return data

    mean = norm_stats[var_name]['mean']
    std = norm_stats[var_name]['std']

    if isinstance(data, torch.Tensor):
        return data * std + mean
    else:
        return data * std + mean


def prepare_input(raw_data, norm_stats=None):
    """Prepare input data from raw data dictionary"""
    # Flatten data
    data = {}
    for folder, var, name in data_index:
        data[name] = raw_data[folder][var]

    # Apply normalization if statistics are provided
    if norm_stats is not None:
        for v in SURFACE_VARS:
            data[v] = normalize_data(data[v], v, norm_stats)
        data['bg_t_3d'] = normalize_data(data['bg_t_3d'], 'bg_t_3d', norm_stats)
        data['bg_s_3d'] = normalize_data(data['bg_s_3d'], 'bg_s_3d', norm_stats)

    # Build input: surface vars + bg_t_3d(20) + bg_s_3d(20)
    inputs = np.concatenate(
        [np.expand_dims(data[v], axis=0) for v in SURFACE_VARS] +
        [data['bg_t_3d'], data['bg_s_3d']],
        axis=0
    ).astype(np.float32)  # (IN_CHANNELS, 400, 480)

    # Replace NaN with 0
    inputs = np.nan_to_num(inputs, nan=0.0)

    return torch.from_numpy(inputs)


def prepare_target(raw_data):
    """Prepare target data from raw data dictionary"""
    data = {}
    for folder, var, name in data_index:
        data[name] = raw_data[folder][var]

    targets = np.concatenate([
        data['label_t_3d'],                       # (20, 400, 480)
        data['label_s_3d']                        # (20, 400, 480)
    ], axis=0).astype(np.float32)                 # (40, 400, 480)

    return targets


def prepare_background(raw_data):
    """Extract background (Glorys -7 days) data"""
    data = {}
    for folder, var, name in data_index:
        data[name] = raw_data[folder][var]

    background = np.concatenate([
        data['bg_t_3d'],                          # (20, 400, 480)
        data['bg_s_3d']                           # (20, 400, 480)
    ], axis=0).astype(np.float32)                 # (40, 400, 480)

    return background


def save_to_netcdf(prediction, target, background, date_str, save_dir):
    """Save prediction, target, and background to NetCDF files"""
    os.makedirs(save_dir, exist_ok=True)

    lon = np.linspace(100, 159.875, 480)
    lat = np.linspace(0, 49.875, 400)
    depth_values = np.array([0.49, 2.65, 5.08, 7.93, 11.41, 15.81, 21.60, 29.44, 40.34, 55.76,
                             77.85, 92.32, 109.73, 130.67, 155.85, 186.13, 222.48, 318.13, 453.94, 643.57])

    year, month, day = date_str[:4], date_str[4:6], date_str[6:]
    time_coord = np.array([np.datetime64(f"{year}-{month}-{day}")])

    def make_ds(temp, salt, description):
        ds = xr.Dataset(
            {
                'thetao': (['time', 'depth', 'latitude', 'longitude'], temp[np.newaxis, :, :, :]),
                'so':     (['time', 'depth', 'latitude', 'longitude'], salt[np.newaxis, :, :, :])
            },
            coords={'time': time_coord, 'depth': depth_values, 'latitude': lat, 'longitude': lon}
        )
        ds.attrs['description'] = description
        ds.attrs['date'] = date_str
        return ds

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


def compute_metrics(prediction, target):
    """Compute evaluation metrics"""
    # Create mask for valid values
    mask = ~np.isnan(target)

    if mask.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'corr': np.nan}

    pred_valid = prediction[mask]
    target_valid = target[mask]

    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))

    # MAE
    mae = np.mean(np.abs(pred_valid - target_valid))

    # Correlation
    corr = np.corrcoef(pred_valid, target_valid)[0, 1]

    return {'rmse': rmse, 'mae': mae, 'corr': corr}


def compute_layer_rmse(prediction, target):
    """Compute RMSE for each depth layer separately.

    Args:
        prediction: numpy array of shape (N_layers, H, W)
        target:     numpy array of shape (N_layers, H, W)

    Returns:
        list of float, length N_layers
    """
    n_layers = prediction.shape[0]
    rmse_per_layer = []
    for i in range(n_layers):
        mask = ~np.isnan(target[i])
        if mask.sum() == 0:
            rmse_per_layer.append(np.nan)
        else:
            diff = prediction[i][mask] - target[i][mask]
            rmse_per_layer.append(float(np.sqrt(np.mean(diff ** 2))))
    return rmse_per_layer


def main():
    parser = argparse.ArgumentParser(description='Ocean Reconstruction Model Inference')
    parser.add_argument('--model_path', type=str, default='./logs/20260408_160455/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--date', type=str, default='20251220',
                       help='Date to run inference on (YYYYMMDD format)')
    parser.add_argument('--save_dir', type=str, default='./inference_results',
                       help='Directory to save inference results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--profile_coords', type=str,
                        default="12.5,115; 25.0,130.0; 37.5,145; 20.0,120.0; 30.0,135.0",
                        help='Semicolon-separated lat,lon pairs for vertical profiles, '
                            'e.g. "12.5,115.0;25.0,130.0;37.5,145.0"')

    args = parser.parse_args()

    # Parse profile coordinates
    if args.profile_coords:
        profile_coords = []
        for pair in args.profile_coords.split(';'):
            lat_s, lon_s = pair.strip().split(',')
            profile_coords.append((float(lat_s), float(lon_s)))
        print(f"Using custom profile coordinates: {profile_coords}")
    else:
        profile_coords = DEFAULT_PROFILE_COORDS
        print(f"Using default profile coordinates: {profile_coords}")

    # Create timestamped output directory
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results directory: {save_dir}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and normalization statistics
    model, norm_stats = load_model(args.model_path, device)

    # Initialize data loader
    print("\nLoading data...")
    dataloader = OceanDatasetLoader(r"F:\PythonWorkspace\predict_ts\datasets")

    # Load data for the specified date
    raw_data = dataloader.load_single_date(args.date, isLog=True)

    # Prepare input (with normalization)
    print("\nPreparing input data...")
    inputs = prepare_input(raw_data, norm_stats)
    inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension

    # Prepare target (ground truth)
    target = prepare_target(raw_data)

    # Prepare background (Glorys -7 days)
    background = prepare_background(raw_data)

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(inputs)

    # Convert to numpy
    prediction_normalized = outputs.squeeze(0).cpu().numpy()  # (40, 400, 480)

    # Denormalize prediction to get real physical values
    if norm_stats is not None:
        print("\nDenormalizing predictions...")
        # Split into temperature and salinity
        pred_temp_norm = torch.from_numpy(prediction_normalized[:20])
        pred_salt_norm = torch.from_numpy(prediction_normalized[20:])

        # Denormalize
        pred_temp = denormalize_data(pred_temp_norm, 'label_t_3d', norm_stats).numpy()
        pred_salt = denormalize_data(pred_salt_norm, 'label_s_3d', norm_stats).numpy()

        # Combine back
        prediction = np.concatenate([pred_temp, pred_salt], axis=0)
    else:
        prediction = prediction_normalized

    # Compute metrics (now both are in original scale)
    print("\nComputing metrics...")
    metrics = compute_metrics(prediction, target)
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Correlation: {metrics['corr']:.6f}")

    # Compute metrics for temperature and salinity separately
    temp_metrics = compute_metrics(prediction[:20], target[:20])
    salt_metrics = compute_metrics(prediction[20:], target[20:])

    # Compute per-layer RMSE
    temp_layer_rmse = compute_layer_rmse(prediction[:20], target[:20])
    salt_layer_rmse = compute_layer_rmse(prediction[20:], target[20:])

    # Compute background (Glorys -7 days) metrics vs target
    bg_metrics      = compute_metrics(background, target)
    bg_temp_metrics = compute_metrics(background[:20], target[:20])
    bg_salt_metrics = compute_metrics(background[20:], target[20:])
    bg_temp_layer_rmse = compute_layer_rmse(background[:20], target[:20])
    bg_salt_layer_rmse = compute_layer_rmse(background[20:], target[20:])

    print(f"\nBackground metrics (vs target):")
    print(f"  RMSE: {bg_metrics['rmse']:.6f}")
    print(f"  MAE:  {bg_metrics['mae']:.6f}")
    print(f"  Corr: {bg_metrics['corr']:.6f}")

    print(f"\nTemperature metrics:")
    print(f"  RMSE: {temp_metrics['rmse']:.6f}")
    print(f"  MAE: {temp_metrics['mae']:.6f}")
    print(f"  Correlation: {temp_metrics['corr']:.6f}")

    print(f"\nSalinity metrics:")
    print(f"  RMSE: {salt_metrics['rmse']:.6f}")
    print(f"  MAE: {salt_metrics['mae']:.6f}")
    print(f"  Correlation: {salt_metrics['corr']:.6f}")

    # Save results to NetCDF
    print("\nSaving results...")
    pred_path, target_path, bg_path = save_to_netcdf(prediction, target, background, args.date, save_dir)

    # Generate visualizations and HTML report
    html_path = visualize_inference_results(
        pred_path, target_path, bg_path,
        metrics, temp_metrics, salt_metrics,
        save_dir, args.date, profile_coords,
        temp_layer_rmse=temp_layer_rmse,
        salt_layer_rmse=salt_layer_rmse,
        bg_metrics=bg_metrics,
        bg_temp_metrics=bg_temp_metrics,
        bg_salt_metrics=bg_salt_metrics,
        bg_temp_layer_rmse=bg_temp_layer_rmse,
        bg_salt_layer_rmse=bg_salt_layer_rmse,
    )

    print("\n" + "="*60)
    print("Inference completed successfully!")
    print(f"Results saved to: {save_dir}")
    print(f"HTML report:      {html_path}")
    print(f"Interactive:      python serve.py --result_dir {save_dir}")
    print("="*60)

    return pred_path, target_path, bg_path, html_path


if __name__ == '__main__':
    main()
