"""
对文件夹内多个 nc 文件两两计算 RMSE，输出矩阵，并生成空间分布图。

直接修改下方 CONFIG 区域，然后运行：
    python rmse_matrix.py
"""

import itertools
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG：修改这里
# =============================================================================

FOLDER = r"./surface_rmse"   # nc 文件所在文件夹

# 每一项：[文件名（不含.nc）, 变量名]
FILES = [
    ['glorys',    'so'],
    ['AF',        'so'],
    ['multiobs', 'sos'],
]

NO_PLOT = False   # True = 只打印表格，不生成任何图

# =============================================================================


def load_data_with_coords(nc_path: Path, var: str):
    """加载变量数据及经纬度坐标，返回 (data, lons, lats)。"""
    ds = xr.open_dataset(nc_path)
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {nc_path.name}. Available: {list(ds.data_vars)}")
    data = np.squeeze(ds[var].values).astype(np.float32)

    # 尝试读取经纬度
    lon_names = ['longitude', 'lon', 'x']
    lat_names = ['latitude',  'lat', 'y']
    lons = lats = None
    for name in lon_names:
        if name in ds.coords:
            lons = ds.coords[name].values
            break
    for name in lat_names:
        if name in ds.coords:
            lats = ds.coords[name].values
            break

    ds.close()
    return data, lons, lats


def load_data(nc_path: Path, var: str) -> np.ndarray:
    data, _, _ = load_data_with_coords(nc_path, var)
    return data


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def print_matrix(names, matrix):
    col_w = max(len(n) for n in names) + 2
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row = f"{row_name:>{col_w}}"
        for j in range(len(names)):
            row += f"{'—':>{col_w}}" if i == j else f"{matrix[i][j]:>{col_w}.6f}"
        print(row)


def plot_matrix(names, matrix, save_path):
    n = len(names)
    mat_data = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                mat_data[i, j] = matrix[i][j]

    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.2)))
    im = ax.imshow(mat_data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='RMSE')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i][j]:.4f}", ha='center', va='center', fontsize=9)
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')

    ax.set_title('Pairwise RMSE Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RMSE matrix plot saved to: {save_path}")


def _pcolormesh_or_imshow(ax, arr, lons, lats, cmap, vmin, vmax):
    if lons is not None and lats is not None:
        im = ax.pcolormesh(lons, lats, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel('Lon', fontsize=8)
        ax.set_ylabel('Lat', fontsize=8)
        ax.tick_params(labelsize=7)
    else:
        im = ax.imshow(arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    return im


def plot_pairwise_spatial(names, data_dict, coords_dict, save_path):
    """
    所有两两组合，每行三列：A | B | A-B（差值，colorbar 与 A/B 一致）
    """
    pairs = list(itertools.combinations(names, 2))
    n_rows = len(pairs)

    # 统一场值 colorbar 范围（所有数据 2~98 百分位）
    all_vals = np.concatenate([d[~np.isnan(d)] for d in data_dict.values()])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    # 差值范围：以最大绝对差的 98 百分位做对称范围
    all_diffs = []
    for na, nb in pairs:
        d = data_dict[na] - data_dict[nb]
        all_diffs.append(d[~np.isnan(d)])
    diff_abs_max = float(np.percentile(np.abs(np.concatenate(all_diffs)), 98))
    diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows), squeeze=False)

    for row, (na, nb) in enumerate(pairs):
        a = data_dict[na]
        b = data_dict[nb]
        diff = a - b
        lons_a, lats_a = coords_dict[na]
        lons_b, lats_b = coords_dict[nb]

        # 列 0：A
        im0 = _pcolormesh_or_imshow(axes[row, 0], a, lons_a, lats_a, 'RdYlBu_r', vmin, vmax)
        axes[row, 0].set_title(na, fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], orientation='horizontal', pad=0.08, fraction=0.046)

        # 列 1：B
        im1 = _pcolormesh_or_imshow(axes[row, 1], b, lons_b, lats_b, 'RdYlBu_r', vmin, vmax)
        axes[row, 1].set_title(nb, fontsize=10)
        plt.colorbar(im1, ax=axes[row, 1], orientation='horizontal', pad=0.08, fraction=0.046)

        # 列 2：A - B（与 A/B 共用 colorbar 范围）
        im2 = _pcolormesh_or_imshow(axes[row, 2], diff, lons_a, lats_a, 'RdBu_r', diff_vmin, diff_vmax)
        axes[row, 2].set_title(f'{na} - {nb}', fontsize=10)
        plt.colorbar(im2, ax=axes[row, 2], orientation='horizontal', pad=0.08, fraction=0.046)

    plt.suptitle('Pairwise Spatial Comparison', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pairwise spatial plot saved to: {save_path}")


def main():
    folder = Path(FOLDER)

    names = [entry[0] for entry in FILES]
    var_map = {entry[0]: entry[1] for entry in FILES}

    print("Loading data...")
    data = {}
    coords = {}
    for name, var in var_map.items():
        nc_path = folder / f"{name}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(f"File not found: {nc_path}")
        print(f"  {nc_path.name}  →  ds['{var}']")
        arr, lons, lats = load_data_with_coords(nc_path, var)
        print(f"    shape: {arr.shape}")
        data[name] = arr
        coords[name] = (lons, lats)

    shapes = {name: arr.shape for name, arr in data.items()}
    if len(set(shapes.values())) > 1:
        print("\nWarning: shape mismatch:")
        for name, shape in shapes.items():
            print(f"  {name}: {shape}")

    print("\nComputing pairwise RMSE...")
    n = len(names)
    matrix = [[None] * n for _ in range(n)]
    for i, j in itertools.product(range(n), range(n)):
        if i == j:
            continue
        matrix[i][j] = compute_rmse(data[names[i]], data[names[j]])

    print("\nRMSE Matrix (row vs col):\n")
    print_matrix(names, matrix)

    if not NO_PLOT:
        plot_matrix(names, matrix, str(folder / 'rmse_matrix.png'))
        plot_pairwise_spatial(names, data, coords, str(folder / 'spatial_comparison.png'))


if __name__ == '__main__':
    main()