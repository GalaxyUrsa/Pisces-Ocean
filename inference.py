"""
Ocean Reconstruction Model Inference Script

Usage:
    python inference.py --date 20260202 --model_path best_model.pth
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from load_datasets import OceanDatasetLoader
from visualize_results_glory import visualize_glory_results
from inference_utils import (
    load_model, prepare_input, prepare_target, prepare_background,
    save_to_netcdf, compute_metrics, compute_layer_rmse, print_metrics_table,
    DEFAULT_PROFILE_COORDS,
)
from Data_Config import (
    data_index, RAW_DATASET_PATH,
    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END,
)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ocean Reconstruction Inference')
    parser.add_argument('--model_path', type=str, default='./logs/20260520_151940_finetune/best_model.pth', help='Path to trained model checkpoint')
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

    # 模型输出物理量纲的残差，pred = bg + residual
    residual = outputs.squeeze(0).cpu().numpy()  # (40, H, W)
    prediction = background + residual

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
