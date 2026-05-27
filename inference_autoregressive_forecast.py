"""
Autoregressive Multi-Step Ocean Forecast

Rolls the model forward N days starting from a given date.
Each step's prediction (bg + residual) becomes the next step's background field.
No disk reads after step 0 — only bg_t_3d / bg_s_3d are replaced each step.

Usage:
    python autoregressive_forecast.py \
        --start_date 20251220 \
        --n_days 10 \
        --model_path ./logs/best_model.pth
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
from visualize_autoregressive import visualize_autoregressive_results
from Data_Config import data_index, RAW_DATASET_PATH, CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END

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


def run_autoregressive(start_date: str, n_days: int, model_path: str,
                       save_dir: str, device: torch.device):
    model, norm_stats = load_model(model_path, device)

    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)
    print(f"\nLoading initial data for {start_date}...")
    raw_data = dataloader.load_single_date(start_date, data_index, isLog=True)

    summary_rows = []  # [(date_str, temp_rmse, salt_rmse)]

    for step in range(n_days):
        current_date = (
            datetime.strptime(start_date, '%Y%m%d') + timedelta(days=step)
        ).strftime('%Y%m%d')

        print(f"\n{'='*60}")
        print(f"Step {step+1}/{n_days}  →  {current_date}")
        print(f"{'='*60}")

        inputs     = prepare_input(raw_data, norm_stats).unsqueeze(0).to(device)
        background = prepare_background(raw_data)  # (40, H, W)

        with torch.no_grad():
            outputs = model(inputs)

        residual   = outputs.squeeze(0).cpu().numpy()  # (40, H, W)
        prediction = background + residual              # (40, H, W)

        target = _try_load_target(dataloader, current_date)

        save_to_netcdf(prediction, target, background, current_date, save_dir)

        if target is not None:
            temp_metrics = compute_metrics(prediction[:20], target[:20])
            salt_metrics = compute_metrics(prediction[20:], target[20:])
            print(f"  Temp RMSE: {temp_metrics['rmse']:.4f}  Salt RMSE: {salt_metrics['rmse']:.4f}")
            summary_rows.append((current_date, temp_metrics['rmse'], salt_metrics['rmse']))
        else:
            print("  (No Glorys ground truth found — metrics skipped)")
            summary_rows.append((current_date, None, None))

        # Autoregressive update: replace background with this step's prediction
        raw_data['bg_t_3d'] = prediction[:20].copy()  # (20, H, W)
        raw_data['bg_s_3d'] = prediction[20:].copy()  # (20, H, W)

    print(f"\n{'='*60}")
    print("Autoregressive Forecast Summary")
    print(f"{'='*60}")
    print(f"{'Step':<6} {'Date':<12} {'Temp RMSE':>12} {'Salt RMSE':>12}")
    print("-" * 44)
    for i, (date_str, t_rmse, s_rmse) in enumerate(summary_rows):
        t_str = f"{t_rmse:.4f}" if t_rmse is not None else "    N/A"
        s_str = f"{s_rmse:.4f}" if s_rmse is not None else "    N/A"
        print(f"{i+1:<6} {date_str:<12} {t_str:>12} {s_str:>12}")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}")

    html_path = visualize_autoregressive_results(
        summary_rows=summary_rows,
        save_dir=save_dir,
        start_date=start_date,
        n_days=n_days,
    )
    print(f"HTML report:      {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description='Autoregressive Ocean Forecast')
    parser.add_argument('--start_date', type=str, default='20251220', help='Start date YYYYMMDD')
    parser.add_argument('--n_days',     type=int, default=10,         help='Number of forecast days')
    parser.add_argument('--model_path', type=str, default='./logs/20260520_190103_finetune/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir',   type=str, default='./autoregressive_results',
                        help='Root directory to save results')
    parser.add_argument('--device',     type=str, default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_autoregressive(
        start_date=args.start_date,
        n_days=args.n_days,
        model_path=args.model_path,
        save_dir=args.save_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
