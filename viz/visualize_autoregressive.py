"""
Visualization module for autoregressive multi-step ocean forecast results.

Generates:
  - RMSE vs forecast lead time plot (temperature + salinity)
  - Spatial comparison plots per step (Prediction / Error vs target if available)
  - Vertical profile plots per step
  - HTML report summarizing all steps
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime


# =============================================================================
# RMSE vs lead time
# =============================================================================

def create_rmse_leadtime_plot(summary_rows, save_dir):
    """
    Plot RMSE as a function of forecast lead time (days).

    summary_rows: list of (date_str, temp_rmse_or_None, salt_rmse_or_None)
    """
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    steps      = [i + 1 for i, (_, t, s) in enumerate(summary_rows) if t is not None]
    temp_rmses = [t for (_, t, s) in summary_rows if t is not None]
    salt_rmses = [s for (_, t, s) in summary_rows if s is not None]

    if not steps:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, temp_rmses, '-o', color='#e74c3c', linewidth=2, markersize=6)
    axes[0].set_xlabel('Forecast Lead Time (days)')
    axes[0].set_ylabel('RMSE (°C)')
    axes[0].set_title('Temperature RMSE vs Lead Time')
    axes[0].grid(True, alpha=0.3)
    for x, y in zip(steps, temp_rmses):
        axes[0].annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), fontsize=8, ha='center')

    axes[1].plot(steps, salt_rmses, '-o', color='#3498db', linewidth=2, markersize=6)
    axes[1].set_xlabel('Forecast Lead Time (days)')
    axes[1].set_ylabel('RMSE (PSU)')
    axes[1].set_title('Salinity RMSE vs Lead Time')
    axes[1].grid(True, alpha=0.3)
    for x, y in zip(steps, salt_rmses):
        axes[1].annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), fontsize=8, ha='center')

    plt.suptitle('Autoregressive Forecast — RMSE vs Lead Time')
    plt.tight_layout()

    filepath = os.path.join(plots_dir, 'rmse_leadtime.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: rmse_leadtime.png")
    return filepath


# =============================================================================
# Spatial comparison per step
# =============================================================================

def create_step_comparison_plot(pred_ds, target_ds, bg_ds, save_dir, date_str,
                                 depth_idx=0):
    """
    Single-depth spatial comparison for one forecast step.
    Columns: Background / Prediction / Target (if available) / Error
    """
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_files = []
    depth_val  = pred_ds.depth.values[depth_idx]

    for var_name, var_label, vmin, vmax, err_lim in [
        ('thetao', 'Temperature (°C)', 0,  40, 10),
        ('so',     'Salinity (PSU)',   30, 40,  3),
    ]:
        pred_data = pred_ds[var_name].isel(time=0, depth=depth_idx).values
        bg_data   = bg_ds[var_name].isel(time=0, depth=depth_idx).values

        has_target = target_ds is not None
        ncols = 4 if has_target else 2
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

        def _imshow(ax, data, title, cmap='RdYlBu_r', v0=vmin, v1=vmax):
            im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1,
                           aspect='auto', origin='lower')
            ax.set_title(title)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax)

        _imshow(axes[0], bg_data,   f'Background\n{depth_val:.1f}m')
        _imshow(axes[1], pred_data, f'Prediction\n{depth_val:.1f}m')

        if has_target:
            target_data = target_ds[var_name].isel(time=0, depth=depth_idx).values
            error       = pred_data - target_data
            _imshow(axes[2], target_data, f'Target\n{depth_val:.1f}m')
            _imshow(axes[3], error, f'Error (Pred-Target)\n{depth_val:.1f}m',
                    cmap='RdBu_r', v0=-err_lim, v1=err_lim)

        var_short = 'temp' if var_name == 'thetao' else 'salt'
        plt.suptitle(f'{var_label} — {date_str}')
        plt.tight_layout()

        filename = f'{var_short}_step_{date_str}.png'
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(filepath)
        print(f"  Created: {filename}")

    return plot_files


# =============================================================================
# HTML report
# =============================================================================

def generate_autoregressive_html(summary_rows, step_plots, rmse_plot,
                                  save_dir, start_date, n_days):
    """
    Generate HTML report for the full autoregressive forecast run.

    summary_rows : list of (date_str, temp_rmse_or_None, salt_rmse_or_None)
    step_plots   : dict {date_str: [plot_path, ...]}
    rmse_plot    : path to RMSE-vs-leadtime plot (or None)
    """
    has_metrics = any(t is not None for (_, t, _) in summary_rows)

    # Summary table rows
    table_rows = ''
    for i, (date_str, t_rmse, s_rmse) in enumerate(summary_rows):
        row_bg = 'style="background-color:#f9f9f9;"' if i % 2 == 0 else ''
        t_str  = f'{t_rmse:.6f}' if t_rmse is not None else 'N/A'
        s_str  = f'{s_rmse:.6f}' if s_rmse is not None else 'N/A'
        table_rows += (
            f'<tr {row_bg}><td>{i+1}</td><td>{date_str}</td>'
            f'<td>{t_str}</td><td>{s_str}</td></tr>\n'
        )

    # Step sections
    step_sections = ''
    for i, (date_str, t_rmse, s_rmse) in enumerate(summary_rows):
        plots = step_plots.get(date_str, [])
        metric_line = ''
        if t_rmse is not None:
            metric_line = (f'<p><strong>Temp RMSE:</strong> {t_rmse:.6f} &nbsp;|&nbsp; '
                           f'<strong>Salt RMSE:</strong> {s_rmse:.6f}</p>')
        plot_html = ''.join(
            f'<div class="plot-container">'
            f'<img src="{os.path.relpath(p, save_dir)}" alt="{os.path.basename(p)}">'
            f'</div>'
            for p in plots
        )
        step_sections += f"""
        <div class="section">
            <h2>Step {i+1} — {date_str}</h2>
            {metric_line}
            <div class="plot-grid">{plot_html}</div>
        </div>
"""

    rmse_section = ''
    if rmse_plot:
        rmse_rel = os.path.relpath(rmse_plot, save_dir)
        rmse_section = f"""
    <div class="section">
        <h2>📈 RMSE vs Forecast Lead Time</h2>
        <div class="plot-container">
            <img src="{rmse_rel}" alt="RMSE vs Lead Time">
        </div>
    </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autoregressive Ocean Forecast — {start_date} +{n_days}d</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
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
        h1 {{ margin: 0; font-size: 2.2em; }}
        .subtitle {{ margin-top: 10px; opacity: 0.9; font-size: 1.1em; }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #1a6b8a;
            border-bottom: 3px solid #1a6b8a;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .plot-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 15px; }}
        .plot-container {{ text-align: center; }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #1a6b8a; color: white; }}
        tr:hover {{ background-color: #f0f7fa; }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 Autoregressive Ocean Forecast</h1>
        <div class="subtitle">Start Date: {start_date} &nbsp;|&nbsp; Forecast Days: {n_days}</div>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="section">
        <h2>📊 Forecast Summary</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Date</th>
                <th>Temp RMSE (°C)</th>
                <th>Salt RMSE (PSU)</th>
            </tr>
            {table_rows}
        </table>
    </div>

    {rmse_section}

    {step_sections}

    <div class="footer">
        <p>Generated by Autoregressive Ocean Forecast System</p>
        <p>© 2026 — Powered by PyTorch &amp; xarray</p>
    </div>
</body>
</html>
"""

    html_path = os.path.join(save_dir, 'autoregressive_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HTML report generated: {html_path}")
    return html_path


# =============================================================================
# Main entry point
# =============================================================================

def visualize_autoregressive_results(summary_rows, save_dir, start_date, n_days,
                                      depth_idx=0):
    """
    Build all plots and HTML report for an autoregressive forecast run.

    summary_rows : list of (date_str, temp_rmse_or_None, salt_rmse_or_None)
                   in step order.
    save_dir     : directory where prediction_*.nc / background_*.nc / target_*.nc live.
    depth_idx    : which depth layer to use for spatial comparison plots.

    Returns:
        html_path
    """
    print("\n" + "=" * 60)
    print("Creating autoregressive forecast visualizations...")
    print("=" * 60)

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    step_plots = {}

    for i, (date_str, t_rmse, s_rmse) in enumerate(summary_rows):
        pred_path = os.path.join(save_dir, f'prediction_{date_str}.nc')
        bg_path   = os.path.join(save_dir, f'background_{date_str}.nc')
        tgt_path  = os.path.join(save_dir, f'target_{date_str}.nc')

        if not os.path.exists(pred_path) or not os.path.exists(bg_path):
            print(f"  Step {i+1} ({date_str}): NetCDF not found, skipping plots")
            step_plots[date_str] = []
            continue

        print(f"\nStep {i+1}/{len(summary_rows)} — {date_str}")
        pred_ds = xr.open_dataset(pred_path)
        bg_ds   = xr.open_dataset(bg_path)
        tgt_ds  = xr.open_dataset(tgt_path) if os.path.exists(tgt_path) else None

        plots = create_step_comparison_plot(pred_ds, tgt_ds, bg_ds,
                                            save_dir, date_str, depth_idx)
        step_plots[date_str] = plots

        pred_ds.close()
        bg_ds.close()
        if tgt_ds is not None:
            tgt_ds.close()

    print("\nCreating RMSE vs lead time plot...")
    rmse_plot = create_rmse_leadtime_plot(summary_rows, save_dir)

    print("\nGenerating HTML report...")
    html_path = generate_autoregressive_html(
        summary_rows, step_plots, rmse_plot, save_dir, start_date, n_days
    )

    print("\n" + "=" * 60)
    print(f"Visualization complete. HTML report: {html_path}")
    print("=" * 60)

    return html_path
