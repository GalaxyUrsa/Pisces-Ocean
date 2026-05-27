"""
Batch 7-Day Forecast Evaluation

Iterates over a date range, runs 7-day independent inference for each start_date,
and generates an HTML report showing RMSE trends across the evaluation period.

Usage:
    python batch_eval_7day.py --start 20250101 --end 20251231 --save_dir ./batch_eval_results
    python batch_eval_7day.py --start 20250101 --end 20251231 --step 7  # every 7 days
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch

from load_datasets import OceanDatasetLoader
from inference_7day import run_7day_single, get_models, _try_load_target
from Data_Config_7day import RAW_DATASET_PATH


# =============================================================================
# Date range helpers
# =============================================================================

def date_range(start: str, end: str, step: int = 1):
    """Yield YYYYMMDD strings from start to end (inclusive) with given step."""
    cur = datetime.strptime(start, '%Y%m%d')
    end_dt = datetime.strptime(end, '%Y%m%d')
    while cur <= end_dt:
        yield cur.strftime('%Y%m%d')
        cur += timedelta(days=step)


# =============================================================================
# Plotting
# =============================================================================

def plot_rmse_over_time(all_results: dict, save_dir: str):
    """
    all_results: {start_date: [(target_date, t_rmse, s_rmse), ...]}

    Produces two figures:
      1. Per-lead-day RMSE averaged over all start_dates (bar chart)
      2. Each lead day's RMSE as a time series over start_dates
    """
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_files = []

    start_dates = sorted(all_results.keys())
    n_days = 7

    # Shape: (n_start_dates, 7) — NaN where target was unavailable
    temp_matrix = np.full((len(start_dates), n_days), np.nan)
    salt_matrix = np.full((len(start_dates), n_days), np.nan)

    for i, sd in enumerate(start_dates):
        for j, (_, t_rmse, s_rmse) in enumerate(all_results[sd]):
            if t_rmse is not None:
                temp_matrix[i, j] = t_rmse
                salt_matrix[i, j] = s_rmse

    # --- Figure 1: mean RMSE per lead day ---
    mean_temp = np.nanmean(temp_matrix, axis=0)
    mean_salt = np.nanmean(salt_matrix, axis=0)
    days = np.arange(1, n_days + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(days, mean_temp, color='#e74c3c', alpha=0.8)
    axes[0].plot(days, mean_temp, '-o', color='#c0392b', linewidth=1.5, markersize=5)
    axes[0].set_xlabel('Lead Day')
    axes[0].set_ylabel('Mean RMSE (°C)')
    axes[0].set_title('Temperature — Mean RMSE per Lead Day')
    axes[0].set_xticks(days)
    axes[0].grid(True, alpha=0.3, axis='y')
    for x, y in zip(days, mean_temp):
        axes[0].annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                         xytext=(0, 6), fontsize=8, ha='center')

    axes[1].bar(days, mean_salt, color='#3498db', alpha=0.8)
    axes[1].plot(days, mean_salt, '-o', color='#2980b9', linewidth=1.5, markersize=5)
    axes[1].set_xlabel('Lead Day')
    axes[1].set_ylabel('Mean RMSE (PSU)')
    axes[1].set_title('Salinity — Mean RMSE per Lead Day')
    axes[1].set_xticks(days)
    axes[1].grid(True, alpha=0.3, axis='y')
    for x, y in zip(days, mean_salt):
        axes[1].annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                         xytext=(0, 6), fontsize=8, ha='center')

    plt.suptitle(f'7-Day Forecast — Mean RMSE by Lead Day\n({start_dates[0]} to {start_dates[-1]})')
    plt.tight_layout()
    p = os.path.join(plots_dir, 'mean_rmse_per_leadday.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(p)

    # --- Figure 2: RMSE time series per lead day ---
    x_labels = start_dates
    x = np.arange(len(x_labels))
    tick_step = max(1, len(x_labels) // 20)  # at most ~20 ticks

    colors = plt.cm.tab10(np.linspace(0, 1, n_days))

    for matrix, unit, var_short, title in [
        (temp_matrix, '°C',  'temp', 'Temperature RMSE over Time'),
        (salt_matrix, 'PSU', 'salt', 'Salinity RMSE over Time'),
    ]:
        fig, ax = plt.subplots(figsize=(16, 5))
        for d in range(n_days):
            ax.plot(x, matrix[:, d], '-', color=colors[d],
                    linewidth=1.2, label=f'Day {d+1}', alpha=0.85)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(x_labels[::tick_step], rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('Start Date')
        ax.set_ylabel(f'RMSE ({unit})')
        ax.set_title(title)
        ax.legend(loc='upper right', ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(plots_dir, f'rmse_timeseries_{var_short}.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(p)

    return plot_files, mean_temp, mean_salt


# =============================================================================
# HTML report
# =============================================================================

def generate_batch_html(all_results: dict, plot_files: list,
                        mean_temp: np.ndarray, mean_salt: np.ndarray,
                        start: str, end: str, step: int, save_dir: str):
    start_dates = sorted(all_results.keys())
    n_days = 7

    # Summary stats table (mean RMSE per lead day)
    lead_rows = ''
    for d in range(n_days):
        t = mean_temp[d]
        s = mean_salt[d]
        t_str = f'{t:.6f}' if not np.isnan(t) else 'N/A'
        s_str = f'{s:.6f}' if not np.isnan(s) else 'N/A'
        row_bg = 'style="background-color:#f9f9f9;"' if d % 2 == 0 else ''
        lead_rows += (f'<tr {row_bg}><td>{d+1}</td>'
                      f'<td>{t_str}</td><td>{s_str}</td></tr>\n')

    # Per-start-date detail table
    detail_rows = ''
    for i, sd in enumerate(start_dates):
        row_bg = 'style="background-color:#f9f9f9;"' if i % 2 == 0 else ''
        cells = f'<td>{sd}</td>'
        for _, t_rmse, s_rmse in all_results[sd]:
            if t_rmse is not None:
                cells += f'<td>{t_rmse:.4f} / {s_rmse:.4f}</td>'
            else:
                cells += '<td>N/A</td>'
        detail_rows += f'<tr {row_bg}>{cells}</tr>\n'

    day_headers = ''.join(f'<th>Day {d+1}<br><small>T/S RMSE</small></th>' for d in range(n_days))

    # Plot images
    plots_html = ''
    for p in plot_files:
        rel = os.path.relpath(p, save_dir)
        plots_html += (f'<div class="plot-container">'
                       f'<img src="{rel}" alt="{os.path.basename(p)}"></div>\n')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>7-Day Batch Forecast Evaluation — {start} to {end}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a6b8a 0%, #0d3b52 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0; font-size: 2em; }}
        .subtitle {{ margin-top: 8px; opacity: 0.9; font-size: 1em; }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{ color: #1a6b8a; border-bottom: 3px solid #1a6b8a; padding-bottom: 8px; margin-top: 0; }}
        .plot-container {{ text-align: center; margin-bottom: 20px; }}
        .plot-container img {{
            max-width: 100%; height: auto;
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.85em; }}
        th, td {{ padding: 8px 10px; text-align: center; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #1a6b8a; color: white; }}
        tr:hover {{ background-color: #f0f7fa; }}
        td:first-child {{ text-align: left; font-weight: bold; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 2px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 7-Day Batch Forecast Evaluation</h1>
        <div class="subtitle">Period: {start} → {end} &nbsp;|&nbsp; Step: every {step} day(s) &nbsp;|&nbsp; {len(start_dates)} start dates</div>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="section">
        <h2>📊 Mean RMSE by Lead Day</h2>
        <table>
            <tr><th>Lead Day</th><th>Mean Temp RMSE (°C)</th><th>Mean Salt RMSE (PSU)</th></tr>
            {lead_rows}
        </table>
    </div>

    <div class="section">
        <h2>📈 RMSE Plots</h2>
        {plots_html}
    </div>

    <div class="section">
        <h2>📋 Per-Start-Date Detail (Temp RMSE / Salt RMSE)</h2>
        <div style="overflow-x:auto;">
        <table>
            <tr><th>Start Date</th>{day_headers}</tr>
            {detail_rows}
        </table>
        </div>
    </div>

    <div class="footer">
        <p>7-Day Independent Forecast Batch Evaluation</p>
    </div>
</body>
</html>
"""

    html_path = os.path.join(save_dir, 'batch_eval_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ HTML report: {html_path}")
    return html_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Batch 7-Day Forecast Evaluation')
    parser.add_argument('--start',    type=str, required=True, help='Eval start date YYYYMMDD')
    parser.add_argument('--end',      type=str, required=True, help='Eval end date YYYYMMDD')
    parser.add_argument('--step',     type=int, default=1,     help='Step between start dates (days)')
    parser.add_argument('--save_dir', type=str, default='./results/batch_eval_results', help='Output directory')
    parser.add_argument('--device',   type=str, default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading models...")
    models = get_models(device)

    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)
    dates = list(date_range(args.start, args.end, args.step))
    print(f"Evaluating {len(dates)} start dates from {args.start} to {args.end} (step={args.step})")

    all_results = {}
    for i, sd in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] start_date={sd}")
        all_results[sd] = run_7day_single(sd, dataloader, models, device)

    print("\nGenerating plots...")
    plot_files, mean_temp, mean_salt = plot_rmse_over_time(all_results, args.save_dir)

    generate_batch_html(all_results, plot_files, mean_temp, mean_salt,
                        args.start, args.end, args.step, args.save_dir)


if __name__ == '__main__':
    main()
