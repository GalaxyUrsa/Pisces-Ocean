"""
Interactive Ocean Reconstruction Viewer - Flask Backend

Usage:
    python serve.py --pred prediction_20251220.nc --target target_20251220.nc --bg background_20251220.nc
    python serve.py --result_dir ./inference_results/20260408_151035
"""

import os
import argparse
import numpy as np
import xarray as xr
from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path

app = Flask(__name__)

# Global dataset handles (loaded once at startup)
_pred_ds = None
_target_ds = None
_bg_ds = None
_report_dir = None


def load_datasets(pred_path, target_path, bg_path):
    global _pred_ds, _target_ds, _bg_ds
    _pred_ds   = xr.open_dataset(pred_path)
    _target_ds = xr.open_dataset(target_path)
    _bg_ds     = xr.open_dataset(bg_path)
    print(f"Loaded prediction:  {pred_path}")
    print(f"Loaded target:      {target_path}")
    print(f"Loaded background:  {bg_path}")


@app.route('/')
def index():
    """Serve the interactive HTML report"""
    return send_from_directory(_report_dir, 'interactive_report.html')


@app.route('/surface')
def surface():
    """Return surface field (depth index 0) for both variables as 2-D arrays.

    Query params:
        var: 'thetao' or 'so' (default: 'thetao')
    """
    var = request.args.get('var', 'thetao')
    if var not in ('thetao', 'so'):
        return jsonify({'error': 'var must be thetao or so'}), 400

    pred_surf   = _pred_ds[var].isel(time=0, depth=0).values
    target_surf = _target_ds[var].isel(time=0, depth=0).values

    lats = _pred_ds.latitude.values.tolist()
    lons = _pred_ds.longitude.values.tolist()

    # Replace NaN with null for JSON
    def to_list(arr):
        return np.where(np.isnan(arr), None, arr).tolist()

    return jsonify({
        'var': var,
        'lats': lats,
        'lons': lons,
        'pred':   to_list(pred_surf),
        'target': to_list(target_surf),
    })


@app.route('/profile')
def profile():
    """Return vertical profiles at the nearest grid point.

    Query params:
        lat: latitude (float)
        lon: longitude (float)
    """
    try:
        lat_q = float(request.args.get('lat'))
        lon_q = float(request.args.get('lon'))
    except (TypeError, ValueError):
        return jsonify({'error': 'lat and lon must be numeric'}), 400

    lats = _pred_ds.latitude.values
    lons = _pred_ds.longitude.values
    depths = _pred_ds.depth.values

    lat_idx = int(np.argmin(np.abs(lats - lat_q)))
    lon_idx = int(np.argmin(np.abs(lons - lon_q)))

    actual_lat = float(lats[lat_idx])
    actual_lon = float(lons[lon_idx])

    result = {
        'query_lat': lat_q,
        'query_lon': lon_q,
        'actual_lat': actual_lat,
        'actual_lon': actual_lon,
        'depths': depths.tolist(),
        'thetao': {},
        'so': {},
    }

    for var in ('thetao', 'so'):
        for src, ds in [('pred', _pred_ds), ('target', _target_ds), ('bg', _bg_ds)]:
            vals = ds[var].isel(time=0, latitude=lat_idx, longitude=lon_idx).values
            result[var][src] = [None if np.isnan(v) else float(v) for v in vals]

    return jsonify(result)


@app.route('/meta')
def meta():
    """Return grid metadata"""
    return jsonify({
        'lats':   _pred_ds.latitude.values.tolist(),
        'lons':   _pred_ds.longitude.values.tolist(),
        'depths': _pred_ds.depth.values.tolist(),
        'date':   str(_pred_ds.time.values[0])[:10],
    })


def main():
    parser = argparse.ArgumentParser(description='Interactive Ocean Reconstruction Viewer')
    parser.add_argument('--pred',       type=str, help='Path to prediction NetCDF')
    parser.add_argument('--target',     type=str, help='Path to target NetCDF')
    parser.add_argument('--bg',         type=str, help='Path to background NetCDF')
    parser.add_argument('--result_dir', type=str, help='Inference result directory (auto-finds NC files)')
    parser.add_argument('--port',       type=int, default=5000)
    parser.add_argument('--date',       type=str, default=None, help='Date string YYYYMMDD (used with --result_dir)')
    args = parser.parse_args()

    global _report_dir

    if args.result_dir:
        result_dir = args.result_dir
        # Auto-find NC files
        nc_files = list(Path(result_dir).glob('*.nc'))
        pred_path   = next((str(f) for f in nc_files if 'prediction' in f.name), None)
        target_path = next((str(f) for f in nc_files if 'target'     in f.name), None)
        bg_path     = next((str(f) for f in nc_files if 'background' in f.name), None)
        if not all([pred_path, target_path, bg_path]):
            raise FileNotFoundError(f"Could not find prediction/target/background NC files in {result_dir}")
        _report_dir = result_dir
    else:
        if not all([args.pred, args.target, args.bg]):
            parser.error('Provide either --result_dir or all of --pred --target --bg')
        pred_path   = args.pred
        target_path = args.target
        bg_path     = args.bg
        _report_dir = str(Path(pred_path).parent)

    load_datasets(pred_path, target_path, bg_path)

    print(f"\nServing interactive report at http://localhost:{args.port}/")
    print("Press Ctrl+C to stop.\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
