"""
7-Day Independent Ocean Forecast

For a given start_date, runs 7 independent inferences:
  Day 1: bg from (start_date - 1d), model finetune_1, predicts (start_date + 1d)
  Day 2: bg from (start_date - 2d), model finetune_2, predicts (start_date + 2d)
  ...
  Day 7: bg from (start_date - 7d), model finetune_7, predicts (start_date + 7d)

Each step loads data independently from disk — no autoregressive rollout.

Usage:
    python inference_7day.py --start_date 20251220 --save_dir ./7day_results
"""

import argparse
import numpy as np
import torch
from datetime import datetime, timedelta

from load_datasets import OceanDatasetLoader
from inference_utils import (
    load_model, prepare_input, prepare_background,
    save_to_netcdf, compute_metrics,
)

# 预加载7个模型，避免批量评估时重复加载
_MODELS = None

def get_models(device: torch.device):
    global _MODELS
    if _MODELS is None:
        _MODELS = [load_model(p, device) for p in MODEL_PATHS]
    return _MODELS
from viz.visualize_autoregressive import visualize_autoregressive_results
from Data_Config_7day import (
    make_data_index, MODEL_PATHS, RAW_DATASET_PATH,
    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END,
    SURFACE_VARS, _SURFACE_INDEX, NAN_FILL_VALUE,
)

_LABEL_INDEX = [
    ['Glorys_thetao_0.083deg', 'thetao', 'label_t_3d', {'select_depth': True, 'bg_offset_days': -1}],
    ['Glorys_so_0.083deg',     'so',     'label_s_3d', {'select_depth': True, 'bg_offset_days': -1}],
]


def _try_load_target(dataloader, date_str: str):
    """Load Glorys ground-truth for date_str. Returns (40, H, W) array or None."""
    try:
        raw = dataloader.load_single_date(date_str, _LABEL_INDEX, isLog=False)
        if 'label_t_3d' not in raw or 'label_s_3d' not in raw:
            return None
        arr = np.concatenate([raw['label_t_3d'], raw['label_s_3d']], axis=0).astype(np.float32)
        return arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
    except Exception:
        return None


def run_7day_single(start_date: str, dataloader, models, device: torch.device) -> list:
    """
    Run 7-day forecast for one start_date. Returns:
        list of (target_date, temp_rmse_or_None, salt_rmse_or_None)
    No NetCDF files are written.
    """
    results = []
    for day in range(1, 8):
        target_date = (
            datetime.strptime(start_date, '%Y%m%d') + timedelta(days=day)
        ).strftime('%Y%m%d')

        model, norm_stats = models[day - 1]
        data_index = make_data_index(day)
        try:
            raw_data = dataloader.load_single_date(start_date, data_index, isLog=True)
        except Exception as e:
            print(f"  [WARN] Day {day}: failed to load input for {start_date}: {e}")
            results.append((target_date, None, None))
            continue

        inputs     = prepare_input(raw_data, norm_stats).unsqueeze(0).to(device)
        background = prepare_background(raw_data)

        with torch.no_grad():
            outputs = model(inputs)

        prediction = background + outputs.squeeze(0).cpu().numpy()
        target     = _try_load_target(dataloader, target_date)

        if target is not None:
            temp_rmse = compute_metrics(prediction[:20], target[:20])['rmse']
            salt_rmse = compute_metrics(prediction[20:], target[20:])['rmse']
            results.append((target_date, temp_rmse, salt_rmse))
        else:
            results.append((target_date, None, None))

    return results


def run_7day_forecast(start_date: str, save_dir: str, device: torch.device):
    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)
    summary_rows = []  # [(target_date, temp_rmse_or_None, salt_rmse_or_None)]

    for day in range(1, 8):
        target_date = (
            datetime.strptime(start_date, '%Y%m%d') + timedelta(days=day)
        ).strftime('%Y%m%d')
        bg_date = (
            datetime.strptime(start_date, '%Y%m%d') - timedelta(days=day - 1)
        ).strftime('%Y%m%d')

        print(f"\n{'='*60}")
        print(f"Day {day}/7  →  target: {target_date}  (bg offset: -{day}d, bg_date ref: {start_date})")
        print(f"{'='*60}")

        model, norm_stats = load_model(MODEL_PATHS[day - 1], device)

        data_index = make_data_index(day)
        print(f"Loading data for {start_date} with bg_offset_days=-{day}...")
        raw_data = dataloader.load_single_date(start_date, data_index, isLog=True)

        inputs     = prepare_input(raw_data, norm_stats).unsqueeze(0).to(device)
        background = prepare_background(raw_data)  # (40, H, W)

        with torch.no_grad():
            outputs = model(inputs)

        residual   = outputs.squeeze(0).cpu().numpy()
        prediction = background + residual

        target = _try_load_target(dataloader, target_date)

        save_to_netcdf(prediction, target, background, target_date, save_dir)

        if target is not None:
            temp_metrics = compute_metrics(prediction[:20], target[:20])
            salt_metrics = compute_metrics(prediction[20:], target[20:])
            print(f"  Temp RMSE: {temp_metrics['rmse']:.4f}  Salt RMSE: {salt_metrics['rmse']:.4f}")
            summary_rows.append((target_date, temp_metrics['rmse'], salt_metrics['rmse']))
        else:
            print("  (No Glorys ground truth found — metrics skipped)")
            summary_rows.append((target_date, None, None))

    print(f"\n{'='*60}")
    print("7-Day Forecast Summary")
    print(f"{'='*60}")
    print(f"{'Day':<6} {'Target Date':<14} {'Temp RMSE':>12} {'Salt RMSE':>12}")
    print("-" * 46)
    for i, (date_str, t_rmse, s_rmse) in enumerate(summary_rows):
        t_str = f"{t_rmse:.4f}" if t_rmse is not None else "    N/A"
        s_str = f"{s_rmse:.4f}" if s_rmse is not None else "    N/A"
        print(f"{i+1:<6} {date_str:<14} {t_str:>12} {s_str:>12}")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}")

    html_path = visualize_autoregressive_results(
        summary_rows=summary_rows,
        save_dir=save_dir,
        start_date=start_date,
        n_days=7,
    )
    print(f"HTML report:      {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description='7-Day Independent Ocean Forecast')
    parser.add_argument('--start_date', type=str, default='20251220', help='Start date YYYYMMDD')
    parser.add_argument('--save_dir',   type=str, default='./results/7day_results', help='Directory to save results')
    parser.add_argument('--device',     type=str, default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_7day_forecast(
        start_date=args.start_date,
        save_dir=args.save_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
