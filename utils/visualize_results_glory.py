"""
Visualization module for Glory-mode (no SST/SSS/SLA) inference results.

Generates:
  - Spatial comparison plots (Background / Target / Prediction / Error)
  - Vertical profile plots
  - Per-layer RMSE plots
  - Metrics bar chart
  - HTML report
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

from Data_Config import OUTPUT_VARS, VAR_VIZ_CONFIG, VAR_VIZ_DEFAULT

# 仅含 depth 维的输出变量，供 viz 函数过滤
_DEPTH_VARS = {e[1] for e in OUTPUT_VARS if e[3].get('select_depth')}


# =============================================================================
# Spatial comparison plots
# =============================================================================

def create_comparison_plots(pred_ds, target_ds, bg_ds, save_dir, date_str,
                             depth_indices=None):
    """Create comparison plots for temperature and salinity at different depths."""

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_files = []

    if depth_indices is None:
        depth_indices = list(range(20))

    for entry in OUTPUT_VARS:
        var_name = entry[1]
        is_3d    = var_name in _DEPTH_VARS
        cfg = VAR_VIZ_CONFIG.get(var_name, VAR_VIZ_DEFAULT)
        var_label = cfg['label']
        vmin, vmax = cfg['vmin'], cfg['vmax']
        err_min, err_max = cfg['err_min'], cfg['err_max']
        cmap = cfg['cmap']

        if is_3d:
            iter_depths = depth_indices
        else:
            iter_depths = [None]  # single pass, no depth dim

        for depth_idx in iter_depths:
            if depth_idx is not None:
                depth_val   = pred_ds.depth.values[depth_idx]
                depth_label = f'Depth: {depth_val:.1f}m'
                pred_data   = pred_ds[var_name].isel(time=0, depth=depth_idx).values
                target_data = target_ds[var_name].isel(time=0, depth=depth_idx).values
                bg_data     = bg_ds[var_name].isel(time=0, depth=depth_idx).values
                fname_suffix = f'depth_{depth_val:.1f}m_{date_str}'
            else:
                depth_label = 'Surface'
                pred_data   = pred_ds[var_name].isel(time=0).values
                target_data = target_ds[var_name].isel(time=0).values
                bg_data     = bg_ds[var_name].isel(time=0).values
                fname_suffix = f'surface_{date_str}'

            error_pred = pred_data - target_data
            error_bg   = bg_data   - target_data

            fig, axes = plt.subplots(1, 5, figsize=(30, 5))

            for ax, data, title in [
                (axes[0], bg_data,     f'Background\n{depth_label}'),
                (axes[1], target_data, f'Target\n{depth_label}'),
                (axes[2], pred_data,   f'Prediction\n{depth_label}'),
            ]:
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect='auto', origin='lower')
                ax.set_title(f'{title}\n{var_label}')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax)

            im_err = axes[3].imshow(error_pred, cmap='RdBu_r',
                                    vmin=err_min, vmax=err_max,
                                    aspect='auto', origin='lower')
            axes[3].set_title(f'Error (Pred - Target)\n{depth_label}')
            axes[3].set_xlabel('Longitude')
            axes[3].set_ylabel('Latitude')
            plt.colorbar(im_err, ax=axes[3])

            im_bg_err = axes[4].imshow(error_bg, cmap='RdBu_r',
                                       vmin=err_min, vmax=err_max,
                                       aspect='auto', origin='lower')
            axes[4].set_title(f'Error (BG - Target)\n{depth_label}')
            axes[4].set_xlabel('Longitude')
            axes[4].set_ylabel('Latitude')
            plt.colorbar(im_bg_err, ax=axes[4])

            plt.tight_layout()

            filename = f'{var_name}_{fname_suffix}.png'
            filepath = os.path.join(plots_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            plot_files.append(filepath)
            print(f"  Created: {filename}")

    return plot_files


# =============================================================================
# Vertical profile plots
# =============================================================================

def create_vertical_profile_plots(pred_ds, target_ds, bg_ds, save_dir, date_str,
                                   profile_coords=None, depth_indices=None):
    """Create vertical profile plots at user-specified (lat, lon) locations."""

    if profile_coords is None:
        profile_coords = [(12.5, 115.0), (25.0, 130.0), (37.5, 145.0)]
    if depth_indices is None:
        depth_indices = list(range(len(pred_ds.depth.values)))

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_files = []

    lats   = pred_ds.latitude.values
    lons   = pred_ds.longitude.values
    depths = pred_ds.depth.values[depth_indices]

    # Find nearest grid indices; skip points outside domain or with no valid data
    lat_min, lat_max = lats[0], lats[-1]
    lon_min, lon_max = lons[0], lons[-1]
    valid_points = []
    for lat_val, lon_val in profile_coords:
        if not (lat_min <= lat_val <= lat_max and lon_min <= lon_val <= lon_max):
            print(f"  Skipping point ({lat_val}, {lon_val}): outside domain "
                  f"[{lat_min:.2f}–{lat_max:.2f}°N, {lon_min:.2f}–{lon_max:.2f}°E]")
            continue
        lat_idx = int(np.argmin(np.abs(lats - lat_val)))
        lon_idx = int(np.argmin(np.abs(lons - lon_val)))
        profile = pred_ds['thetao'].isel(time=0, latitude=lat_idx,
                                         longitude=lon_idx).values
        if not np.all(np.isnan(profile)):
            valid_points.append((lat_val, lon_val, lat_idx, lon_idx))
        else:
            print(f"  Skipping point ({lat_val}, {lon_val}): no valid data")

    if not valid_points:
        print("  No valid profile points found, skipping vertical profile plots")
        return plot_files

    n     = len(valid_points)
    ncols = min(n + 1, 4)
    nrows = (n + 1 + ncols - 1) // ncols

    for entry in OUTPUT_VARS:
        var_name  = entry[1]
        if var_name not in _DEPTH_VARS:
            continue
        cfg       = VAR_VIZ_CONFIG.get(var_name, VAR_VIZ_DEFAULT)
        var_label = cfg['label']
        vmin, vmax = cfg['vmin'], cfg['vmax']

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
        axes_flat = np.array(axes).flatten().tolist()
        for i in range(n + 1, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Map panel
        ax_map  = axes_flat[0]
        surface = pred_ds[var_name].isel(time=0, depth=0).values
        im = ax_map.imshow(surface, cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                           aspect='auto', origin='lower',
                           extent=[lons[0], lons[-1], lats[0], lats[-1]])
        plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.08, shrink=0.9)
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
            pred_profile   = pred_ds[var_name].isel(
                time=0, latitude=lat_idx, longitude=lon_idx).values[depth_indices]
            target_profile = target_ds[var_name].isel(
                time=0, latitude=lat_idx, longitude=lon_idx).values[depth_indices]
            bg_profile     = bg_ds[var_name].isel(
                time=0, latitude=lat_idx, longitude=lon_idx).values[depth_indices]

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

        filename = f'{var_name}_vertical_profiles_{date_str}.png'
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        plot_files.append(filepath)
        print(f"  Created: {filename}")

    return plot_files


# =============================================================================
# Per-layer RMSE
# =============================================================================

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


def create_layer_rmse_plot(var_layer_rmse: dict, depth_values, save_dir, date_str,
                            bg_var_layer_rmse: dict = None):
    """Create per-layer RMSE plots for all OUTPUT_VARS variables."""

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    var_names = [v for v in var_layer_rmse if v in _DEPTH_VARS]
    n = len(var_names)
    colors_pred = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22']
    colors_bg   = ['#27ae60', '#f39c12', '#1abc9c', '#d35400']

    def _make_plot(rmse_dict, colors, title_prefix, filename):
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]
        for i, v in enumerate(var_names):
            cfg = VAR_VIZ_CONFIG.get(v, VAR_VIZ_DEFAULT)
            _draw_layer_rmse(axes[i], rmse_dict[v], depth_values,
                             cfg['label'], colors[i % len(colors)], title_prefix)
        plt.suptitle(f'Per-Layer RMSE ({title_prefix}) — {date_str}')
        plt.tight_layout()
        fp = os.path.join(plots_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Created: {filename}")
        return fp

    pred_filepath = _make_plot(var_layer_rmse, colors_pred,
                               'Prediction - Target', f'layer_rmse_pred_{date_str}.png')

    bg_filepath = None
    if bg_var_layer_rmse is not None:
        bg_filepath = _make_plot(bg_var_layer_rmse, colors_bg,
                                 'Background - Target', f'layer_rmse_bg_{date_str}.png')

    return pred_filepath, bg_filepath


# =============================================================================
# Metrics bar chart
# =============================================================================

def create_metrics_plot(metrics, var_metrics: dict, save_dir, date_str):
    """Create a bar plot showing metrics for all OUTPUT_VARS variables."""

    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    categories = ['Overall'] + [VAR_VIZ_CONFIG.get(v, VAR_VIZ_DEFAULT)['label'].split(' ')[0]
                                 for v in var_metrics]
    bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22']

    def _vals(key):
        return [metrics[key]] + [m[key] for m in var_metrics.values()]

    fig, axes = plt.subplots(1, 3, figsize=(5 * len(categories), 5))

    colors = bar_colors[:len(categories)]
    axes[0].bar(categories, _vals('rmse'), color=colors)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Square Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(categories, _vals('mae'), color=colors)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(categories, _vals('corr'), color=colors)
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


# =============================================================================
# HTML report
# =============================================================================

def generate_html_report(pred_path, target_path,
                          metrics, var_metrics: dict,
                          plot_files, save_dir, date_str,
                          var_layer_rmse: dict = None, depth_values=None,
                          bg_metrics=None, bg_var_metrics: dict = None,
                          bg_var_layer_rmse: dict = None):
    """Generate an HTML report with all visualizations and metrics."""

    year  = date_str[:4]
    month = date_str[4:6]
    day   = date_str[6:]
    date_formatted = f"{year}-{month}-{day}"

    plot_rel_paths = [os.path.relpath(p, save_dir) for p in plot_files]

    def _fmt(d, key):
        return f'{d[key]:.6f}' if d is not None else 'N/A'

    n_vars = len(var_metrics)

    # ── Dynamic metric cards ──────────────────────────────────────────────────
    def _metric_card(title, m, color, is_bg=False):
        if m is None:
            return ''
        overlay = 'rgba(0,0,0,0.18)' if is_bg else 'rgba(0,0,0,0)'
        badge = '<span style="font-size:0.7em;opacity:0.7;margin-left:6px;">(BG)</span>' if is_bg else ''
        return f"""<div class="metric-card" style="background:{color};position:relative;overflow:hidden;">
                <div style="position:absolute;inset:0;background:{overlay};pointer-events:none;"></div>
                <div class="card-title">{title}{badge}</div>
                <div class="card-row"><span>RMSE</span><span class="card-val">{_fmt(m, 'rmse')}</span></div>
                <div class="card-row"><span>MAE</span><span class="card-val">{_fmt(m, 'mae')}</span></div>
                <div class="card-row"><span>Corr</span><span class="card-val">{_fmt(m, 'corr')}</span></div>
            </div>"""

    # Overall row
    overall_row = f"""
        <div class="metrics-row">
            <div class="row-label">Overall</div>
            {_metric_card('Prediction', metrics,    'linear-gradient(135deg,#667eea,#764ba2)')}
            {_metric_card('Background', bg_metrics, 'linear-gradient(135deg,#667eea,#764ba2)', is_bg=True)}
        </div>"""

    # Per-variable rows (Pred vs BG side by side, same color per row)
    var_rows = ''
    row_colors = ['#e74c3c','#3498db','#9b59b6','#e67e22','#1abc9c','#e91e63','#00bcd4','#ff9800','#8bc34a']
    for i, (v, m) in enumerate(var_metrics.items()):
        label  = VAR_VIZ_CONFIG.get(v, VAR_VIZ_DEFAULT)['label']
        bg_m   = (bg_var_metrics or {}).get(v)
        color  = row_colors[i % len(row_colors)]
        var_rows += f"""
        <div class="metrics-row">
            <div class="row-label">{label}</div>
            {_metric_card('Prediction', m,    color)}
            {_metric_card('Background', bg_m, color, is_bg=True)}
        </div>"""

    metric_section = overall_row + var_rows

    # ── Dynamic layer RMSE table ──────────────────────────────────────────────
    var_names = [v for v in var_metrics if v in _DEPTH_VARS]
    layer_rmse_header = '<tr><th>Layer</th><th>Depth (m)</th>'
    for v in var_names:
        label = VAR_VIZ_CONFIG.get(v, VAR_VIZ_DEFAULT)['label']
        layer_rmse_header += f'<th>{label} RMSE (Pred)</th><th>{label} RMSE (BG)</th>'
    layer_rmse_header += '</tr>'

    layer_rmse_rows = ''
    if var_layer_rmse is not None and depth_values is not None:
        for i, d in enumerate(depth_values):
            row_style = 'style="background-color:#f9f9f9;"' if i % 2 == 0 else ''
            row = f'<tr {row_style}><td>{i+1}</td><td>{d:.2f}</td>'
            for v in var_names:
                pred_r = f'{var_layer_rmse[v][i]:.6f}' if var_layer_rmse and v in var_layer_rmse else 'N/A'
                bg_r   = f'{bg_var_layer_rmse[v][i]:.6f}' if bg_var_layer_rmse and v in bg_var_layer_rmse else 'N/A'
                row += f'<td>{pred_r}</td><td>{bg_r}</td>'
            row += '</tr>\n'
            layer_rmse_rows += row

    # ── Plot index splits (tail: n_depth_vars profiles + 2 rmse + 1 metrics) ──
    n_depth_vars = len(_DEPTH_VARS)
    n_tail = n_depth_vars + 3  # profile plots + 2 rmse + 1 metrics
    comparison_rel  = plot_rel_paths[:-n_tail]
    profile_rel     = plot_rel_paths[-n_tail:-3]
    rmse_pred_rel   = plot_rel_paths[-3]
    rmse_bg_rel     = plot_rel_paths[-2]
    metrics_rel     = plot_rel_paths[-1]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ocean Reconstruction Results (Glory) - {date_formatted}</title>
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
        h1 {{ margin: 0; font-size: 2.5em; }}
        .subtitle {{ margin-top: 10px; opacity: 0.9; font-size: 1.1em; }}
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
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 20px;
        }}
        .metrics-row {{
            display: grid;
            grid-template-columns: 160px 1fr 1fr;
            gap: 10px;
            align-items: stretch;
        }}
        .row-label {{
            display: flex;
            align-items: center;
            font-weight: 600;
            font-size: 0.95em;
            color: #333;
            padding: 8px 12px;
            background: #f0f0f0;
            border-radius: 8px;
        }}
        .metric-card {{
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }}
        .card-title {{ font-weight: 600; font-size: 0.85em; opacity: 0.85; margin-bottom: 6px; }}
        .card-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            margin: 3px 0;
        }}
        .card-val {{ font-weight: bold; }}
        .plot-grid {{ display: grid; grid-template-columns: 1fr; gap: 30px; margin-top: 20px; }}
        .plot-container {{ text-align: center; }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .plot-caption {{ margin-top: 10px; color: #666; font-style: italic; }}
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
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #667eea; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 Ocean Reconstruction Results (Glory)</h1>
        <div class="subtitle">Inference Date: {date_formatted}</div>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="section">
        <h2>📊 Performance Metrics</h2>
        <div class="metrics-grid">
            {metric_section}
        </div>

        <div class="plot-container" style="margin-top: 30px;">
            <img src="{metrics_rel}" alt="Metrics Comparison">
        </div>

        <h3 style="color:#667eea; margin-top:30px;">Per-Layer RMSE — Prediction vs Target</h3>
        <div class="plot-container">
            <img src="{rmse_pred_rel}" alt="Per-Layer RMSE Prediction">
        </div>

        <h3 style="color:#27ae60; margin-top:30px;">Per-Layer RMSE — Background vs Target</h3>
        <div class="plot-container">
            <img src="{rmse_bg_rel}" alt="Per-Layer RMSE Background">
        </div>
        <table>
            {layer_rmse_header}
            {layer_rmse_rows}
        </table>
    </div>

    <div class="section">
        <h2>🗺️ Spatial Comparisons</h2>
        <div class="info-box">
            <strong>Note:</strong> Each row shows Background, Target, Prediction,
            Error (Pred - Target), and Error (BG - Target) at a given depth.
        </div>
        <div class="plot-grid">
"""

    for plot_path in comparison_rel:
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
            <strong>Note:</strong> Green dotted = background, blue = prediction, red dashed = target.
        </div>
        <div class="plot-grid">
"""

    for plot_path in profile_rel:
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
            <tr><th>File Type</th><th>File Path</th></tr>
            <tr><td>Prediction NetCDF</td><td>{os.path.basename(pred_path)}</td></tr>
            <tr><td>Target NetCDF</td><td>{os.path.basename(target_path)}</td></tr>
            <tr><td>HTML Report</td><td>report_{date_str}.html</td></tr>
        </table>
    </div>

    <div class="footer">
        <p>Generated by Ocean Reconstruction Model Inference System (Glory mode)</p>
        <p>© 2026 - Powered by PyTorch &amp; xarray</p>
    </div>
</body>
</html>
"""

    html_path = os.path.join(save_dir, f'report_{date_str}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ HTML report generated: {html_path}")
    return html_path


# =============================================================================
# Main entry point
# =============================================================================

def visualize_glory_results(pred_path, target_path, bg_path,
                             metrics, var_metrics: dict,
                             save_dir, date_str, profile_coords=None,
                             var_layer_rmse: dict = None,
                             bg_metrics=None, bg_var_metrics: dict = None,
                             bg_var_layer_rmse: dict = None):
    """
    Main function to create all visualizations and HTML report.

    Returns:
        html_path : path to the generated HTML report
    """
    print("\n" + "=" * 60)
    print("Creating Glory-mode visualizations...")
    print("=" * 60)

    pred_ds   = xr.open_dataset(pred_path)
    target_ds = xr.open_dataset(target_path)
    bg_ds     = xr.open_dataset(bg_path)

    depth_indices = list(range(len(pred_ds.depth.values)))
    depth_values  = pred_ds.depth.values

    print("\n1. Creating spatial comparison plots...")
    comparison_plots = create_comparison_plots(
        pred_ds, target_ds, bg_ds, save_dir, date_str, depth_indices)

    print("\n2. Creating vertical profile plots...")
    profile_plots = create_vertical_profile_plots(
        pred_ds, target_ds, bg_ds, save_dir, date_str, profile_coords, depth_indices)

    print("\n3. Creating per-layer RMSE plots...")
    layer_rmse_pred_plot, layer_rmse_bg_plot = create_layer_rmse_plot(
        var_layer_rmse, depth_values, save_dir, date_str,
        bg_var_layer_rmse=bg_var_layer_rmse,
    )

    print("\n4. Creating metrics plot...")
    metrics_plot = create_metrics_plot(metrics, var_metrics, save_dir, date_str)

    layer_rmse_plots = [layer_rmse_pred_plot]
    if layer_rmse_bg_plot is not None:
        layer_rmse_plots.append(layer_rmse_bg_plot)

    all_plots = comparison_plots + profile_plots + layer_rmse_plots + [metrics_plot]

    print("\n5. Generating HTML report...")
    html_path = generate_html_report(
        pred_path, target_path,
        metrics, var_metrics,
        all_plots, save_dir, date_str,
        var_layer_rmse=var_layer_rmse,
        depth_values=depth_values,
        bg_metrics=bg_metrics,
        bg_var_metrics=bg_var_metrics,
        bg_var_layer_rmse=bg_var_layer_rmse,
    )

    pred_ds.close()
    target_ds.close()
    bg_ds.close()

    print("\n" + "=" * 60)
    print("Visualization completed!")
    print(f"Total plots created: {len(all_plots)}")
    print(f"HTML report:         {html_path}")
    print("=" * 60)

    return html_path
