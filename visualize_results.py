"""
Visualization module for ocean reconstruction inference results

This module creates visualizations and HTML reports comparing predictions with ground truth.
"""

import os
import shutil
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime


def create_comparison_plots(pred_ds, target_ds, bg_ds, save_dir, date_str, depth_indices=None):
    """Create comparison plots for temperature and salinity at different depths"""

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_files = []

    if depth_indices is None:
        depth_indices = list(range(20))

    for var_name, var_label in [('thetao', 'Temperature (°C)'), ('so', 'Salinity (PSU)')]:
        for depth_idx in depth_indices:
            depth_val = pred_ds.depth.values[depth_idx]

            pred_data   = pred_ds[var_name].isel(time=0, depth=depth_idx).values
            target_data = target_ds[var_name].isel(time=0, depth=depth_idx).values
            bg_data     = bg_ds[var_name].isel(time=0, depth=depth_idx).values

            error_pred = pred_data - target_data
            error_bg   = bg_data   - target_data

            fig, axes = plt.subplots(1, 5, figsize=(30, 5))

            if var_name == 'thetao':
                vmin, vmax = 0, 40
                err_min, err_max = -10, 10
            else:
                vmin, vmax = 30, 40
                err_min, err_max = -3, 3

            for ax, data, title in [
                (axes[0], bg_data,     f'Background\nDepth: {depth_val:.1f}m'),
                (axes[1], target_data, f'Target\nDepth: {depth_val:.1f}m'),
                (axes[2], pred_data,   f'Prediction\nDepth: {depth_val:.1f}m'),
            ]:
                im = ax.imshow(data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                               aspect='auto', origin='lower')
                ax.set_title(f'{title}\n{var_label}')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax)

            im_err = axes[3].imshow(error_pred, cmap='RdBu_r', vmin=err_min, vmax=err_max,
                                    aspect='auto', origin='lower')
            axes[3].set_title(f'Error (Pred - Target)\nDepth: {depth_val:.1f}m')
            axes[3].set_xlabel('Longitude')
            axes[3].set_ylabel('Latitude')
            plt.colorbar(im_err, ax=axes[3])

            im_bg_err = axes[4].imshow(error_bg, cmap='RdBu_r', vmin=err_min, vmax=err_max,
                                       aspect='auto', origin='lower')
            axes[4].set_title(f'Error (BG - Target)\nDepth: {depth_val:.1f}m')
            axes[4].set_xlabel('Longitude')
            axes[4].set_ylabel('Latitude')
            plt.colorbar(im_bg_err, ax=axes[4])

            plt.tight_layout()

            var_short = 'temp' if var_name == 'thetao' else 'salt'
            filename = f'{var_short}_depth_{depth_val:.1f}m_{date_str}.png'
            filepath = os.path.join(plots_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            plot_files.append(filepath)
            print(f"  Created: {filename}")

    return plot_files


def create_vertical_profile_plots(pred_ds, target_ds, bg_ds, save_dir, date_str, profile_coords=None, depth_indices=None):
    """Create vertical profile plots at user-specified (lat, lon) locations with a map overview"""

    if profile_coords is None:
        profile_coords = [(12.5, 115.0), (25.0, 130.0), (37.5, 145.0)]
    if depth_indices is None:
        depth_indices = list(range(len(pred_ds.depth.values)))

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_files = []

    lats = pred_ds.latitude.values
    lons = pred_ds.longitude.values
    depths = pred_ds.depth.values[depth_indices]

    # Find nearest grid indices and filter out points with no valid data
    valid_points = []
    for lat_val, lon_val in profile_coords:
        lat_idx = int(np.argmin(np.abs(lats - lat_val)))
        lon_idx = int(np.argmin(np.abs(lons - lon_val)))
        profile = pred_ds['thetao'].isel(time=0, latitude=lat_idx, longitude=lon_idx).values
        if not np.all(np.isnan(profile)):
            valid_points.append((lat_val, lon_val, lat_idx, lon_idx))
        else:
            print(f"  Skipping point ({lat_val}, {lon_val}): no valid data")

    if not valid_points:
        print("  No valid profile points found, skipping vertical profile plots")
        return plot_files

    n = len(valid_points)
    ncols = min(n + 1, 4)
    nrows = (n + 1 + ncols - 1) // ncols  # ceiling division

    for var_name, var_label in [('thetao', 'Temperature (°C)'), ('so', 'Salinity (PSU)')]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
        # Flatten to 1D list for easy indexing
        axes_flat = np.array(axes).flatten().tolist()
        # Hide unused axes
        for i in range(n + 1, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Map panel: surface field with marked locations
        ax_map = axes_flat[0]
        surface = pred_ds[var_name].isel(time=0, depth=0).values
        if var_name == 'thetao':
            vmin, vmax = 0, 40
        else:
            vmin, vmax = 30, 40
        im = ax_map.imshow(surface, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                           aspect='auto', origin='lower',
                           extent=[lons[0], lons[-1], lats[0], lats[-1]])
        plt.colorbar(im, ax=ax_map, shrink=0.8)
        for i, (lat_val, lon_val, _, _) in enumerate(valid_points):
            ax_map.plot(lon_val, lat_val, 'k*', markersize=12)
            ax_map.annotate(f'P{i + 1}', (lon_val, lat_val),
                            textcoords='offset points', xytext=(5, 5),
                            fontsize=10, fontweight='bold', color='black')
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.set_title(f'Profile Locations\n(surface {var_label})')
        ax_map.grid(True, alpha=0.3)

        # Profile panels
        for i, (lat_val, lon_val, lat_idx, lon_idx) in enumerate(valid_points):
            ax = axes_flat[i + 1]
            pred_profile   = pred_ds[var_name].isel(time=0, latitude=lat_idx, longitude=lon_idx).values[depth_indices]
            target_profile = target_ds[var_name].isel(time=0, latitude=lat_idx, longitude=lon_idx).values[depth_indices]
            bg_profile     = bg_ds[var_name].isel(time=0, latitude=lat_idx, longitude=lon_idx).values[depth_indices]

            ax.plot(bg_profile,     depths, 'g:^',  label='Background', linewidth=2, markersize=4)
            ax.plot(pred_profile,   depths, 'b-o',  label='Prediction', linewidth=2, markersize=4)
            ax.plot(target_profile, depths, 'r--s', label='Target',     linewidth=2, markersize=4)
            ax.invert_yaxis()
            ax.set_xlabel(var_label)
            ax.set_ylabel('Depth (m)')
            ax.set_title(f'P{i + 1}: {lat_val:.2f}°N, {lon_val:.2f}°E')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Vertical Profiles - {var_label} - {date_str}')
        plt.tight_layout()

        var_short = 'temp' if var_name == 'thetao' else 'salt'
        filename = f'{var_short}_vertical_profiles_{date_str}.png'
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        plot_files.append(filepath)
        print(f"  Created: {filename}")

    return plot_files


def _draw_layer_rmse(ax, rmse_vals, depth_values, label, color, title_prefix):
    ax.plot(rmse_vals, depth_values, '-o', color=color, linewidth=2, markersize=5)
    ax.invert_yaxis()
    ax.set_xlabel(f'RMSE ({label})')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'{title_prefix} — {label}')
    ax.grid(True, alpha=0.3)
    for r, d in zip(rmse_vals, depth_values):
        ax.annotate(f'{r:.4f}', (r, d), textcoords='offset points',
                    xytext=(6, 0), fontsize=7, color=color)


def create_layer_rmse_plot(temp_layer_rmse, salt_layer_rmse, depth_values, save_dir, date_str,
                           bg_temp_layer_rmse=None, bg_salt_layer_rmse=None):
    """Create two separate per-layer RMSE plots: Prediction-Target and Background-Target."""

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: Prediction - Target ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _draw_layer_rmse(axes[0], temp_layer_rmse, depth_values, 'Temperature (°C)', '#e74c3c', 'Pred-Target RMSE')
    _draw_layer_rmse(axes[1], salt_layer_rmse, depth_values, 'Salinity (PSU)',   '#3498db', 'Pred-Target RMSE')
    plt.suptitle(f'Per-Layer RMSE (Prediction - Target) — {date_str}')
    plt.tight_layout()
    pred_filename = f'layer_rmse_pred_{date_str}.png'
    pred_filepath = os.path.join(plots_dir, pred_filename)
    plt.savefig(pred_filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: {pred_filename}")

    # --- Plot 2: Background - Target ---
    bg_filepath = None
    if bg_temp_layer_rmse is not None and bg_salt_layer_rmse is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _draw_layer_rmse(axes[0], bg_temp_layer_rmse, depth_values, 'Temperature (°C)', '#27ae60', 'BG-Target RMSE')
        _draw_layer_rmse(axes[1], bg_salt_layer_rmse, depth_values, 'Salinity (PSU)',   '#f39c12', 'BG-Target RMSE')
        plt.suptitle(f'Per-Layer RMSE (Background - Target) — {date_str}')
        plt.tight_layout()
        bg_filename = f'layer_rmse_bg_{date_str}.png'
        bg_filepath = os.path.join(plots_dir, bg_filename)
        plt.savefig(bg_filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Created: {bg_filename}")

    return pred_filepath, bg_filepath


def create_metrics_plot(metrics, temp_metrics, salt_metrics, save_dir, date_str):
    """Create a bar plot showietrics"""

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ['Overall', 'Temperature', 'Salinity']

    # RMSE
    rmse_values = [metrics['rmse'], temp_metrics['rmse'], salt_metrics['rmse']]
    axes[0].bar(categories, rmse_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Square Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE
    mae_values = [metrics['mae'], temp_metrics['mae'], salt_metrics['mae']]
    axes[1].bar(categories, mae_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Correlation
    corr_values = [metrics['corr'], temp_metrics['corr'], salt_metrics['corr']]
    axes[2].bar(categories, corr_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('Correlation Coefficient')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Evaluation Metrics - {date_str}')
    plt.tight_layout()

    filename = f'metrics_{date_str}.png'
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {filename}")
    return filepath


def generate_html_report(pred_path, target_path, metrics, temp_metrics, salt_metrics,
                        plot_files, save_dir, date_str,
                        temp_layer_rmse=None, salt_layer_rmse=None, depth_values=None,
                        bg_metrics=None, bg_temp_metrics=None, bg_salt_metrics=None,
                        bg_temp_layer_rmse=None, bg_salt_layer_rmse=None):
    """Generate an HTML report with all visualizations and metrics"""

    # Parse date
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:]
    date_formatted = f"{year}-{month}-{day}"

    # Create relative paths for plots
    plot_rel_paths = [os.path.relpath(p, save_dir) for p in plot_files]

    # Pre-format background metric values to avoid f-string format specifier issues
    def _fmt(d, key):
        return f'{d[key]:.6f}' if d is not None else 'N/A'

    bg_rmse  = _fmt(bg_metrics,      'rmse');  bg_mae  = _fmt(bg_metrics,      'mae');  bg_corr  = _fmt(bg_metrics,      'corr')
    bgt_rmse = _fmt(bg_temp_metrics, 'rmse');  bgt_mae = _fmt(bg_temp_metrics, 'mae');  bgt_corr = _fmt(bg_temp_metrics, 'corr')
    bgs_rmse = _fmt(bg_salt_metrics, 'rmse');  bgs_mae = _fmt(bg_salt_metrics, 'mae');  bgs_corr = _fmt(bg_salt_metrics, 'corr')

    # Build per-layer RMSE table rows
    layer_rmse_rows = ''
    if temp_layer_rmse is not None and salt_layer_rmse is not None and depth_values is not None:
        for i, (d, tr, sr) in enumerate(zip(depth_values, temp_layer_rmse, salt_layer_rmse)):
            row_class = 'style="background-color:#f9f9f9;"' if i % 2 == 0 else ''
            bg_tr = f'{bg_temp_layer_rmse[i]:.6f}' if bg_temp_layer_rmse is not None else 'N/A'
            bg_sr = f'{bg_salt_layer_rmse[i]:.6f}' if bg_salt_layer_rmse is not None else 'N/A'
            layer_rmse_rows += (
                f'<tr {row_class}><td>{i+1}</td><td>{d:.2f}</td>'
                f'<td>{tr:.6f}</td><td>{bg_tr}</td>'
                f'<td>{sr:.6f}</td><td>{bg_sr}</td></tr>\n'
            )

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ocean Reconstruction Results - {date_formatted}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 15px 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .metric-value {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 1.1em;
        }}
        .metric-label {{
            font-weight: 500;
        }}
        .metric-number {{
            font-weight: bold;
            font-size: 1.2em;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-top: 20px;
        }}
        .plot-container {{
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .plot-caption {{
            margin-top: 10px;
            color: #666;
            font-style: italic;
        }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #ddd;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 Ocean Reconstruction Results</h1>
        <div class="subtitle">Inference Date: {date_formatted}</div>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="section">
        <h2>📊 Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Overall Performance</h3>
                <div class="metric-value">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-number">{metrics['rmse']:.6f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-number">{metrics['mae']:.6f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">Correlation:</span>
                    <span class="metric-number">{metrics['corr']:.6f}</span>
                </div>
            </div>

            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>Temperature (θ)</h3>
                <div class="metric-value">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-number">{temp_metrics['rmse']:.6f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-number">{temp_metrics['mae']:.6f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">Correlation:</span>
                    <span class="metric-number">{temp_metrics['corr']:.6f}</span>
                </div>
            </div>

            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>Salinity (S)</h3>
                <div class="metric-value">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-number">{salt_metrics['rmse']:.6f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-number">{salt_metrics['mae']:.6f}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">Correlation:</span>
                    <span class="metric-number">{salt_metrics['corr']:.6f}</span>
                </div>
            </div>

            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>Background (Overall)</h3>
                <div class="metric-value">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-number">{bg_rmse}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-number">{bg_mae}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">Correlation:</span>
                    <span class="metric-number">{bg_corr}</span>
                </div>
            </div>

            <div class="metric-card" style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);">
                <h3>Background Temperature</h3>
                <div class="metric-value">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-number">{bgt_rmse}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-number">{bgt_mae}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">Correlation:</span>
                    <span class="metric-number">{bgt_corr}</span>
                </div>
            </div>

            <div class="metric-card" style="background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);">
                <h3>Background Salinity</h3>
                <div class="metric-value">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-number">{bgs_rmse}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-number">{bgs_mae}</span>
                </div>
                <div class="metric-value">
                    <span class="metric-label">Correlation:</span>
                    <span class="metric-number">{bgs_corr}</span>
                </div>
            </div>
        </div>

        <div class="plot-container" style="margin-top: 30px;">
            <img src="{plot_rel_paths[-1]}" alt="Metrics Comparison">
        </div>

        <h3 style="color:#667eea; margin-top:30px;">Per-Layer RMSE — Prediction vs Target</h3>
        <div class="plot-container">
            <img src="{plot_rel_paths[-3]}" alt="Per-Layer RMSE Prediction">
        </div>

        <h3 style="color:#27ae60; margin-top:30px;">Per-Layer RMSE — Background vs Target</h3>
        <div class="plot-container">
            <img src="{plot_rel_paths[-2]}" alt="Per-Layer RMSE Background">
        </div>
        <table>
            <tr>
                <th>Layer</th>
                <th>Depth (m)</th>
                <th>Temp RMSE (Pred)</th>
                <th>Temp RMSE (BG)</th>
                <th>Salt RMSE (Pred)</th>
                <th>Salt RMSE (BG)</th>
            </tr>
            {layer_rmse_rows}
        </table>
    </div>

    <div class="section">
        <h2>🗺️ Spatial Comparisons</h2>
        <div class="info-box">
            <strong>Note:</strong> Each row shows Background (Glorys -7 days), Target, Prediction, and Error (Pred - Target) at a given depth.
        </div>
        <div class="plot-grid">
"""

    # Add spatial comparison plots (excluding the last one which is metrics)
    for plot_path in plot_rel_paths[:-5]:  # Exclude vertical profiles, layer_rmse and metrics
        plot_name = os.path.basename(plot_path)
        html_content += f"""
            <div class="plot-container">
                <img src="{plot_path}" alt="{plot_name}">
                <div class="plot-caption">{plot_name}</div>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>📈 Vertical Profiles</h2>
        <div class="info-box">
            <strong>Note:</strong> Vertical profiles show how temperature and salinity vary with depth at selected locations.
            Green dotted lines represent background (Glorys -7 days), blue lines represent predictions, red dashed lines represent ground truth.
        </div>
        <div class="plot-grid">
"""

    # Add vertical profile plots
    for plot_path in plot_rel_paths[-5:-3]:  # The two vertical profile plots
        plot_name = os.path.basename(plot_path)
        html_content += f"""
            <div class="plot-container">
                <img src="{plot_path}" alt="{plot_name}">
                <div class="plot-caption">{plot_name}</div>
            </div>
"""

    html_content += f"""
        </div>
    </div>

    <div class="section">
        <h2>📁 Output Files</h2>
        <table>
            <tr>
                <th>File Type</th>
                <th>File Path</th>
            </tr>
            <tr>
                <td>Prediction NetCDF</td>
                <td>{os.path.basename(pred_path)}</td>
            </tr>
            <tr>
                <td>Target NetCDF</td>
                <td>{os.path.basename(target_path)}</td>
            </tr>
            <tr>
                <td>HTML Report</td>
                <td>report_{date_str}.html</td>
            </tr>
        </table>
    </div>

    <div class="footer">
        <p>Generated by Ocean Reconstruction Model Inference System</p>
        <p>© 2026 - Powered by PyTorch & xarray</p>
    </div>
</body>
</html>
"""

    # Save HTML file
    html_path = os.path.join(save_dir, f'report_{date_str}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ HTML report generated: {html_path}")
    return html_path


def generate_interactive_report(save_dir):
    """Copy the interactive_report.html template into save_dir.

    The template expects a Flask server (serve.py) running at the same origin.
    Returns the path to the copied file.
    """
    template_path = os.path.join(os.path.dirname(__file__), 'interactive_report.html')
    if not os.path.exists(template_path):
        print("⚠ interactive_report.html template not found, skipping interactive report")
        return None

    dest_path = os.path.join(save_dir, 'interactive_report.html')
    shutil.copy2(template_path, dest_path)
    print(f"✅ Interactive report copied to: {dest_path}")
    return dest_path


def visualize_inference_results(pred_path, target_path, bg_path, metrics, temp_metrics, salt_metrics,
                                save_dir, date_str, profile_coords=None,
                                temp_layer_rmse=None, salt_layer_rmse=None,
                                bg_metrics=None, bg_temp_metrics=None, bg_salt_metrics=None,
                                bg_temp_layer_rmse=None, bg_salt_layer_rmse=None):
    """
    Main function to create all visualizations and HTML report

    Args:
        pred_path: Path to prediction NetCDF file
        target_path: Path to target NetCDF file
        bg_path: Path to background NetCDF file
        metrics: Dictionary with overall metrics
        temp_metrics: Dictionary with temperature metrics
        salt_metrics: Dictionary with salinity metrics
        save_dir: Directory to save visualizations
        date_str: Date string (YYYYMMDD format)

    Returns:
        html_path: Path to generated HTML report
    """

    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    pred_ds   = xr.open_dataset(pred_path)
    target_ds = xr.open_dataset(target_path)
    bg_ds     = xr.open_dataset(bg_path)

    depth_indices = list(range(20))

    print("\n1. Creating spatial comparison plots...")
    comparison_plots = create_comparison_plots(pred_ds, target_ds, bg_ds, save_dir, date_str, depth_indices)

    print("\n2. Creating vertical profile plots...")
    profile_plots = create_vertical_profile_plots(pred_ds, target_ds, bg_ds, save_dir, date_str, profile_coords, depth_indices)

    print("\n3. Creating per-layer RMSE plots...")
    depth_values = pred_ds.depth.values
    layer_rmse_pred_plot, layer_rmse_bg_plot = create_layer_rmse_plot(
        temp_layer_rmse, salt_layer_rmse, depth_values, save_dir, date_str,
        bg_temp_layer_rmse=bg_temp_layer_rmse,
        bg_salt_layer_rmse=bg_salt_layer_rmse,
    )

    print("\n4. Creating metrics plot...")
    metrics_plot = create_metrics_plot(metrics, temp_metrics, salt_metrics, save_dir, date_str)

    layer_rmse_plots = [layer_rmse_pred_plot]
    if layer_rmse_bg_plot is not None:
        layer_rmse_plots.append(layer_rmse_bg_plot)

    all_plots = comparison_plots + profile_plots + layer_rmse_plots + [metrics_plot]

    print("\n5. Generating HTML report...")
    html_path = generate_html_report(pred_path, target_path, metrics, temp_metrics, salt_metrics,
                                     all_plots, save_dir, date_str,
                                     temp_layer_rmse=temp_layer_rmse,
                                     salt_layer_rmse=salt_layer_rmse,
                                     depth_values=depth_values,
                                     bg_metrics=bg_metrics,
                                     bg_temp_metrics=bg_temp_metrics,
                                     bg_salt_metrics=bg_salt_metrics,
                                     bg_temp_layer_rmse=bg_temp_layer_rmse,
                                     bg_salt_layer_rmse=bg_salt_layer_rmse)

    print("\n6. Copying interactive report...")
    interactive_path = generate_interactive_report(save_dir)

    print("\n" + "="*60)
    print("Visualization completed!")
    print(f"Total plots created: {len(all_plots)}")
    print(f"HTML report:         {html_path}")
    if interactive_path:
        print(f"Interactive report:  {interactive_path}")
        print(f"  → Run: python serve.py --result_dir {save_dir}")
    print("="*60)

    return html_path