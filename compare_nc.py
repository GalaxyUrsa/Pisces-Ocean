"""
Compare two NetCDF files visually.

Usage:
    # Same variable name in both files
    python compare_nc.py --file_a file_a.nc --file_b file_b.nc --var thetao

    # Different variable names (e.g. thetao vs sst)
    python compare_nc.py --file_a file_a.nc --file_b file_b.nc --var_a thetao --var_b sst

    python compare_nc.py --file_a file_a.nc --file_b file_b.nc --var so --depth_idx 0 3 5
    python compare_nc.py --file_a file_a.nc --file_b file_b.nc --var thetao --label_a Prediction --label_b Target

    # Run with defaults (file_a, file_b, var_a, var_b all have defaults)
    python compare_nc.py
"""

import argparse
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime


def load_var(path: str, var: str):
    """Load a variable from a NetCDF file, squeezed to (depth, lat, lon) or (lat, lon)."""
    ds = xr.open_dataset(path)
    if var not in ds:
        available = list(ds.data_vars)
        raise KeyError(f"Variable '{var}' not found in {path}. Available: {available}")
    data = np.squeeze(ds[var].values)
    return data, ds


def get_depth_values(ds, var: str):
    """Return depth coordinate values if the variable has a depth dimension."""
    da = ds[var]
    for dim in da.dims:
        if 'depth' in dim.lower() or dim == 'lev':
            if dim in ds.coords:
                return ds.coords[dim].values
    return None


def plot_comparison(data_a, data_b, depth_indices, depth_values,
                    label_a, label_b, var, save_dir):
    """For each depth index, plot: A | B | (B - A)."""
    os.makedirs(save_dir, exist_ok=True)
    plot_files = []

    is_3d = data_a.ndim == 3
    indices = depth_indices if is_3d else [None]

    for idx in indices:
        if is_3d:
            a = data_a[idx]
            b = data_b[idx]
            depth_label = f"depth_idx={idx}" if depth_values is None else f"{depth_values[idx]:.1f}m"
            title_suffix = f" | {depth_label}"
            fname = f"{var}_depth{idx:02d}.png"
        else:
            a = data_a
            b = data_b
            title_suffix = ""
            fname = f"{var}.png"

        diff = b - a

        valid = np.concatenate([a[~np.isnan(a)], b[~np.isnan(b)]])
        vmin = float(np.nanpercentile(valid, 2))
        vmax = float(np.nanpercentile(valid, 98))

        valid_diff = diff[~np.isnan(diff)]
        abs_max = float(np.nanpercentile(np.abs(valid_diff), 98)) if len(valid_diff) else 1.0

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{var}{title_suffix}", fontsize=13)

        im0 = axes[0].imshow(a, origin='lower', vmin=vmin, vmax=vmax, cmap='RdYlBu_r', aspect='auto')
        axes[0].set_title(label_a)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(b, origin='lower', vmin=vmin, vmax=vmax, cmap='RdYlBu_r', aspect='auto')
        axes[1].set_title(label_b)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(diff, origin='lower', vmin=-abs_max, vmax=abs_max,
                              cmap='bwr', aspect='auto')
        axes[2].set_title(f"{label_b} - {label_a}")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        if len(valid_diff) > 0:
            rmse = float(np.sqrt(np.mean(valid_diff ** 2)))
            mae  = float(np.mean(np.abs(valid_diff)))
            bias = float(np.mean(valid_diff))
            fig.text(0.5, 0.01,
                     f"RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:.4f}",
                     ha='center', fontsize=10, color='gray')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(out_path)
        print(f"  Saved: {out_path}")

    return plot_files


def print_stats(data_a, data_b, label_a, label_b, depth_indices, depth_values):
    """Print per-depth RMSE/MAE/Bias table to stdout."""
    is_3d = data_a.ndim == 3
    header = f"{'Depth':>12}  {'RMSE':>10}  {'MAE':>10}  {'Bias':>10}  {'Max|diff|':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  {label_b} - {label_a}  |  shape: {data_a.shape}")
    print(sep)
    print(header)
    print(sep)

    if not is_3d:
        diff = (data_b - data_a).ravel()
        diff = diff[~np.isnan(diff)]
        if len(diff):
            print(f"{'(2D)':>12}  {np.sqrt(np.mean(diff**2)):>10.4f}  "
                  f"{np.mean(np.abs(diff)):>10.4f}  {np.mean(diff):>10.4f}  "
                  f"{np.max(np.abs(diff)):>10.4f}")
    else:
        for idx in depth_indices:
            dlabel = f"idx={idx}" if depth_values is None else f"{depth_values[idx]:.1f}m"
            diff = (data_b[idx] - data_a[idx]).ravel()
            diff = diff[~np.isnan(diff)]
            if len(diff):
                print(f"{dlabel:>12}  {np.sqrt(np.mean(diff**2)):>10.4f}  "
                      f"{np.mean(np.abs(diff)):>10.4f}  {np.mean(diff):>10.4f}  "
                      f"{np.max(np.abs(diff)):>10.4f}")
            else:
                print(f"{dlabel:>12}  {'all NaN':>10}")
    print(sep)


def interactive_compare(data_a, data_b, depth_indices, depth_values,
                        label_a, label_b, var_title):
    """
    Show an interactive plot. Click anywhere on any panel to print the
    pixel value of A, B, and (B - A) at that grid point.
    Press left/right arrow keys to step through depth levels.
    """
    is_3d = data_a.ndim == 3
    indices = depth_indices if is_3d else [None]
    state = {'idx_pos': 0}  # mutable so the key handler can update it
    cbars = []  # track colorbars so we can remove them on redraw

    def draw(idx):
        if is_3d:
            a = data_a[idx]
            b = data_b[idx]
            depth_label = f"depth_idx={idx}" if depth_values is None else f"{depth_values[idx]:.1f}m"
            title_suffix = f" | {depth_label}"
        else:
            a = data_a
            b = data_b
            title_suffix = ""

        diff = b - a
        valid = np.concatenate([a[~np.isnan(a)], b[~np.isnan(b)]])
        vmin = float(np.nanpercentile(valid, 2))
        vmax = float(np.nanpercentile(valid, 98))
        valid_diff = diff[~np.isnan(diff)]
        abs_max = float(np.nanpercentile(np.abs(valid_diff), 98)) if len(valid_diff) else 1.0

        # Remove old colorbars before clearing axes
        for cb in cbars:
            cb.remove()
        cbars.clear()

        for ax in axes:
            ax.cla()

        im0 = axes[0].imshow(a,    origin='lower', vmin=vmin,     vmax=vmax,     cmap='RdYlBu_r', aspect='auto')
        im1 = axes[1].imshow(b,    origin='lower', vmin=vmin,     vmax=vmax,     cmap='RdYlBu_r', aspect='auto')
        im2 = axes[2].imshow(diff, origin='lower', vmin=-abs_max, vmax=abs_max,  cmap='bwr',      aspect='auto')
        cbars.append(fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04))
        cbars.append(fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04))
        cbars.append(fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04))
        axes[0].set_title(label_a)
        axes[1].set_title(label_b)
        axes[2].set_title(f"{label_b} − {label_a}")
        fig.suptitle(f"{var_title}{title_suffix}  |  click to inspect  |  ←/→ to change depth", fontsize=11)
        fig.canvas.draw_idle()
        return a, b, diff

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    current = {'a': None, 'b': None, 'diff': None}
    current['a'], current['b'], current['diff'] = draw(indices[state['idx_pos']])

    def on_click(event):
        if event.inaxes not in axes:
            return
        col = int(round(event.xdata)) if event.xdata is not None else None
        row = int(round(event.ydata)) if event.ydata is not None else None
        if col is None or row is None:
            return
        h, w = current['a'].shape
        if not (0 <= row < h and 0 <= col < w):
            return
        va = current['a'][row, col]
        vb = current['b'][row, col]
        vd = current['diff'][row, col]
        depth_label = ""
        if is_3d:
            idx = indices[state['idx_pos']]
            depth_label = f"  depth={'idx='+str(idx) if depth_values is None else f'{depth_values[idx]:.1f}m'}"
        print(f"[row={row}, col={col}{depth_label}]  "
              f"{label_a}={va:.4f}  {label_b}={vb:.4f}  diff={vd:.4f}")

    def on_key(event):
        if not is_3d:
            return
        if event.key == 'right':
            state['idx_pos'] = min(state['idx_pos'] + 1, len(indices) - 1)
        elif event.key == 'left':
            state['idx_pos'] = max(state['idx_pos'] - 1, 0)
        else:
            return
        current['a'], current['b'], current['diff'] = draw(indices[state['idx_pos']])

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare two NetCDF files visually.')
    # parser.add_argument('--file_a', type=str, default=r"D:\datasets\AF_thetao_surface_0.25deg\AF_thetao_surface_0.083deg_20251220.nc",
    #                     help='First NetCDF file (reference / A)')
    # parser.add_argument('--file_b', type=str, default=r"D:\datasets\METOFFICE_SST_0.25deg\multiobs_sss_0.125deg_20251220.nc",
    #                     help='Second NetCDF file (comparison / B)')
    # parser.add_argument('--var', type=str, default=None,
    #                     help='Variable name for both files (e.g. thetao). Use --var_a/--var_b if they differ.')
    # parser.add_argument('--var_a', type=str, default="thetao",
    #                     help='Variable name in file_a (overrides --var)')
    # parser.add_argument('--var_b', type=str, default="analysed_sst",
    #                     help='Variable name in file_b (overrides --var)')
    parser.add_argument('--file_a', type=str, default=r"F:\PythonWorkspace\Pisces-Ocean\inference_glory_results\20260509_131708\prediction_20251220.nc",
                        help='First NetCDF file (reference / A)')
    parser.add_argument('--file_b', type=str, default=r"F:\PythonWorkspace\Pisces-Ocean\inference_glory_results\20260509_131708\target_20251220.nc",
                        help='Second NetCDF file (comparison / B)')
    parser.add_argument('--var', type=str, default=None,
                        help='Variable name for both files (e.g. thetao). Use --var_a/--var_b if they differ.')
    parser.add_argument('--var_a', type=str, default="thetao",
                        help='Variable name in file_a (overrides --var)')
    parser.add_argument('--var_b', type=str, default="thetao",
                        help='Variable name in file_b (overrides --var)')
    parser.add_argument('--depth_idx', type=int, nargs='+', default=None,
                        help='Depth indices to plot (default: all). E.g. --depth_idx 0 5 10')
    parser.add_argument('--label_a', type=str, default=None,
                        help='Label for file_a (default: filename stem)')
    parser.add_argument('--label_b', type=str, default=None,
                        help='Label for file_b (default: filename stem)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory (default: compare_<timestamp>)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Only print stats, skip saving plots')
    parser.add_argument('--interactive', action='store_true',
                        help='Open an interactive window: click to inspect values, ←/→ to step through depths')
    args = parser.parse_args()

    var_a = args.var_a or args.var
    var_b = args.var_b or args.var
    if not var_a or not var_b:
        parser.error('Specify --var (same for both) or --var_a and --var_b separately.')

    # Title shown in plots uses "var_a" or "var_a/var_b" if they differ
    var_title = var_a if var_a == var_b else f"{var_a}_vs_{var_b}"

    label_a = args.label_a or os.path.splitext(os.path.basename(args.file_a))[0]
    label_b = args.label_b or os.path.splitext(os.path.basename(args.file_b))[0]

    save_dir = args.save_dir or f"./compare_result/compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Loading from:")
    print(f"  A: {args.file_a}  [{var_a}]")
    print(f"  B: {args.file_b}  [{var_b}]")

    data_a, ds_a = load_var(args.file_a, var_a)
    data_b, ds_b = load_var(args.file_b, var_b)

    print(f"  A shape: {data_a.shape}")
    print(f"  B shape: {data_b.shape}")

    if data_a.shape != data_b.shape:
        raise ValueError(f"Shape mismatch: A={data_a.shape}, B={data_b.shape}")

    depth_values = get_depth_values(ds_a, var_a)

    is_3d = data_a.ndim == 3
    if is_3d:
        n_depth = data_a.shape[0]
        depth_indices = args.depth_idx if args.depth_idx is not None else list(range(n_depth))
        depth_indices = [i for i in depth_indices if 0 <= i < n_depth]
        print(f"  Depth levels to process: {depth_indices}")
    else:
        depth_indices = []

    print_stats(data_a, data_b, label_a, label_b, depth_indices, depth_values)

    if not args.no_plot:
        print(f"\nSaving plots to: {save_dir}")
        plot_comparison(data_a, data_b, depth_indices, depth_values,
                        label_a, label_b, var_title, save_dir)
        print(f"\nDone. {len(depth_indices) if is_3d else 1} plot(s) saved to {save_dir}/")

    if args.interactive:
        print("\nOpening interactive window (click to inspect, ←/→ to change depth)...")
        interactive_compare(data_a, data_b, depth_indices, depth_values,
                            label_a, label_b, var_title)


if __name__ == '__main__':
    main()
