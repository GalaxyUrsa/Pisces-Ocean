"""
GLORYS vs AF 数据 RMSE 分析脚本（支持自定义空间范围 + 0.49m 表层可视化）
===================================

功能：
    对比 GLORYS 与 AF 数据在 thetao（温度）和 so（盐度）
    33 个深度层的逐层 RMSE，支持自定义经纬度裁剪范围。
    新增 0.49 m 表层空间分布对比图（GLORYS 场 / AF 场 / 差异场）。

用法示例：
    # 1) 基础分析（默认范围）
    python analyze_glorys_af_rmse.py --dates 20260202

    # 2) 分析南海并输出 0.49m 表层空间分布图
    python analyze_glorys_af_rmse.py --dates 20260202 --region scs --plot_surface --save_plot

    # 3) 自定义小范围 + 表层可视化
    python analyze_glorys_af_rmse.py --dates 20260202 --lon_range 118 122 --lat_range 18 22 --plot_surface
"""

import os
import argparse
import numpy as np
import h5py
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# ── 可选依赖 ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib 未安装，跳过绘图功能。")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas 未安装，跳过 CSV 保存功能。")

# ===== 中文字体修复 =====
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

# =============================================================================
# 常量配置
# =============================================================================

# GLORYS 33 个标准深度层（单位：m）
GLORYS_33_DEPTHS = np.array([
      0.49,   1.54,   2.65,   3.82,   5.08,   6.44,   7.93,   9.57,
     11.41,  13.47,  15.81,  18.50,  21.60,  25.21,  29.44,  34.43,
     40.34,  47.37,  55.76,  65.81,  77.85,  92.33, 109.73, 130.67,
    155.85, 186.13, 222.48, 266.04, 318.13, 380.21, 453.94, 541.09,
    643.57
], dtype=np.float32)

assert len(GLORYS_33_DEPTHS) == 33, "深度层数量应为 33"

# 预设海域范围（min_lon, max_lon, min_lat, max_lat）
PRESET_REGIONS = {
    "nwpac": (100.0, 160.0, 0.0, 50.0),   # 西北太平洋（默认）
    "scs":   (105.0, 125.0, 5.0, 25.0),   # 南海
    "ecs":   (120.0, 130.0, 20.0, 42.0),  # 东海
    "global": (0.0, 360.0, -90.0, 90.0),  # 全球（不裁剪）
}


# =============================================================================
# 数据加载工具
# =============================================================================

def _find_file(folder: Path, date_str: str) -> Path:
    """在 folder 中查找包含 date_str 的 .nc 文件（取第一个匹配）。"""
    matches = sorted(folder.glob(f"*{date_str}*.nc"))
    if not matches:
        raise FileNotFoundError(f"在 {folder} 中找不到日期 {date_str} 的文件")
    return matches[0]


def _load_af_nc_all33(file_path: Path, variable: str,
                      lon_range: Tuple[float, float],
                      lat_range: Tuple[float, float]) -> np.ndarray:
    """
    用 h5py 读取 AF NetCDF4 文件中指定变量的全部 33 个深度层。
    返回 shape: (33, H, W)，dtype float32，陆地/无效像素置为 NaN。
    """
    with h5py.File(file_path, "r") as ds:
        lat = ds["latitude"][:]
        lon = ds["longitude"][:]

        lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])

        if not lat_mask.any() or not lon_mask.any():
            raise ValueError(
                f"AF 数据在指定范围内无有效格点: lon={lon_range}, lat={lat_range}. "
                f"数据实际范围: lon[{lon.min():.2f}, {lon.max():.2f}], "
                f"lat[{lat.min():.2f}, {lat.max():.2f}]"
            )

        lat_sl = slice(np.where(lat_mask)[0][0], np.where(lat_mask)[0][-1] + 1)
        lon_sl = slice(np.where(lon_mask)[0][0], np.where(lon_mask)[0][-1] + 1)

        raw = ds[variable]
        if raw.ndim == 4:
            data = raw[0, :, lat_sl, lon_sl]
        elif raw.ndim == 3:
            data = raw[:, lat_sl, lon_sl]
        else:
            raise ValueError(f"意外的数据维度: {raw.shape}")

    arr = np.array(data, dtype=np.float64)
    arr[arr > 1e10]  = np.nan
    arr[arr < -1e10] = np.nan
    return arr.astype(np.float32)


def _load_glorys_nc_all33(file_path: Path, variable: str,
                          lon_range: Tuple[float, float],
                          lat_range: Tuple[float, float]) -> np.ndarray:
    """
    读取 GLORYS NetCDF 文件中指定变量的全部 33 个深度层。
    返回 shape: (33, H, W)，dtype float32，无效像素置为 NaN。
    """
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_name = "latitude"  if "latitude"  in ds.coords else "lat"

        lon_vals = ds[lon_name].values
        lat_vals = ds[lat_name].values
        if (lon_range[0] < lon_vals.min() or lon_range[1] > lon_vals.max() or
            lat_range[0] < lat_vals.min() or lat_range[1] > lat_vals.max()):
            raise ValueError(
                f"GLORYS 数据未覆盖目标区域: lon={lon_range}, lat={lat_range}. "
                f"数据实际范围: lon[{lon_vals.min():.2f}, {lon_vals.max():.2f}], "
                f"lat[{lat_vals.min():.2f}, {lat_vals.max():.2f}]"
            )

        var_data = ds[variable].sel(
            {lon_name: slice(lon_range[0], lon_range[1]),
             lat_name: slice(lat_range[0], lat_range[1])}
        )

        if "time" in var_data.dims:
            var_data = var_data.isel(time=0)

        arr = var_data.values

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    return arr.astype(np.float32)


def _get_2d_coords(file_path: Path,
                   lon_range: Tuple[float, float],
                   lat_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 GLORYS 文件中提取裁剪后的 2D 经纬度网格，用于 pcolormesh。
    返回: lon2d, lat2d，shape 均为 (H, W)。
    """
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_name = "latitude"  if "latitude"  in ds.coords else "lat"

        lon = ds[lon_name].sel({lon_name: slice(lon_range[0], lon_range[1])}).values
        lat = ds[lat_name].sel({lat_name: slice(lat_range[0], lat_range[1])}).values

        lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d, lat2d


# =============================================================================
# 逐层 RMSE 计算（增加空间加权选项）
# =============================================================================

def compute_layer_rmse_all33(
        glorys_3d: np.ndarray,
        af_3d: np.ndarray,
        latitudes: Optional[np.ndarray] = None,
        weight_by_coslat: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算两个 (33, H, W) 数组在每个深度层的 RMSE、MAE 和有效像素数。

    参数:
        weight_by_coslat: 是否按 cos(lat) 进行纬度加权（默认 False）
        latitudes: 纬度数组 (H,)，仅在 weight_by_coslat=True 时需要

    返回:
        rmse_arr : (33,)
        mae_arr  : (33,)
        count_arr: (33,) 每层有效像素数
    """
    assert glorys_3d.shape == af_3d.shape, (
        f"形状不匹配: glorys={glorys_3d.shape}, af={af_3d.shape}"
    )
    assert glorys_3d.shape[0] == 33, (
        f"期望 33 个深度层，实际得到 {glorys_3d.shape[0]} 层"
    )

    rmse_arr = np.full(33, np.nan, dtype=np.float64)
    mae_arr  = np.full(33, np.nan, dtype=np.float64)
    count_arr = np.zeros(33, dtype=np.int64)

    _, H, W = glorys_3d.shape

    for i in range(33):
        g = glorys_3d[i].astype(np.float64)
        a = af_3d[i].astype(np.float64)
        valid = ~np.isnan(g) & ~np.isnan(a)
        n_valid = valid.sum()

        if n_valid == 0:
            continue

        count_arr[i] = n_valid
        diff = g[valid] - a[valid]

        if weight_by_coslat and latitudes is not None:
            cos_weights = np.cos(np.deg2rad(latitudes))[:, np.newaxis]  # (H, 1)
            weights_2d = np.broadcast_to(cos_weights, (H, W))
            w = weights_2d[valid]
            w = w / w.sum()  # 归一化
            rmse_arr[i] = np.sqrt(np.sum((diff ** 2) * w))
            mae_arr[i]  = np.sum(np.abs(diff) * w)
        else:
            rmse_arr[i] = np.sqrt(np.mean(diff ** 2))
            mae_arr[i]  = np.mean(np.abs(diff))

    return rmse_arr, mae_arr, count_arr


# =============================================================================
# 表层（0.49 m）空间分布可视化
# =============================================================================

def plot_surface_comparison(
        glorys_thetao_surf: np.ndarray,
        af_thetao_surf: np.ndarray,
        glorys_so_surf: np.ndarray,
        af_so_surf: np.ndarray,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        date_str: str,
        save_dir: str
) -> None:
    """
    绘制 0.49 m 表层 GLORYS vs AF 空间分布对比图。
    2 行 × 3 列子图：温度（GLORYS / AF / Diff）、盐度（GLORYS / AF / Diff）。
    """
    if not HAS_MATPLOTLIB:
        return

    diff_thetao = glorys_thetao_surf - af_thetao_surf
    diff_so = glorys_so_surf - af_so_surf

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    region_str = f"{lon2d.min():.0f}-{lon2d.max():.0f}°E, {lat2d.min():.0f}-{lat2d.max():.0f}°N"
    fig.suptitle(
        f"GLORYS vs AF — 0.49 m 表层空间分布对比 ({date_str})\n区域: {region_str}",
        fontsize=14, fontweight="bold"
    )

    land_color = "#cccccc"  # 陆地背景色

    # 计算统一色标范围
    t_min = min(np.nanmin(glorys_thetao_surf), np.nanmin(af_thetao_surf))
    t_max = max(np.nanmax(glorys_thetao_surf), np.nanmax(af_thetao_surf))
    t_diff_vmax = max(abs(np.nanmin(diff_thetao)), abs(np.nanmax(diff_thetao)))

    s_min = min(np.nanmin(glorys_so_surf), np.nanmin(af_so_surf))
    s_max = max(np.nanmax(glorys_so_surf), np.nanmax(af_so_surf))
    s_diff_vmax = max(abs(np.nanmin(diff_so)), abs(np.nanmax(diff_so)))

    plot_configs = [
        # (数据, vmin, vmax, 标题, cmap, 是否对称色标)
        (glorys_thetao_surf, t_min, t_max, "GLORYS 温度 (°C)", "turbo", False),
        (af_thetao_surf,     t_min, t_max, "AF 温度 (°C)",     "turbo", False),
        (diff_thetao,        -t_diff_vmax, t_diff_vmax, "差异 (GLORYS − AF) °C", "RdBu_r", True),
        (glorys_so_surf,     s_min, s_max, "GLORYS 盐度 (PSU)", "turbo", False),
        (af_so_surf,         s_min, s_max, "AF 盐度 (PSU)",     "turbo", False),
        (diff_so,            -s_diff_vmax, s_diff_vmax, "差异 (GLORYS − AF) PSU", "RdBu_r", True),
    ]

    for ax, (data, vmin, vmax, title, cmap, _) in zip(axes.flat, plot_configs):
        # 将 NaN 区域显式置为灰色（陆地）
        data_plot = np.where(np.isnan(data), np.nan, data)
        im = ax.pcolormesh(lon2d, lat2d, data_plot, shading="auto",
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("经度 (°E)", fontsize=10)
        ax.set_ylabel("纬度 (°N)", fontsize=10)
        ax.set_facecolor(land_color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"surface_049m_{date_str}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"🗺️  表层空间分布图已保存: {save_path}")
    plt.close(fig)


# =============================================================================
# 单日期分析（支持动态范围 + 表层绘图）
# =============================================================================

def analyze_single_date(
        date_str: str,
        glorys_dir: Path,
        af_thetao_dir: Path,
        af_so_dir: Path,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        weight_by_coslat: bool = False,
        plot_surface: bool = False,
        save_dir: str = "./rmse_analysis_results",
        verbose: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    对单个日期计算 thetao 和 so 在 33 层上的 RMSE/MAE。
    若 plot_surface=True，额外输出 0.49 m 表层空间分布对比图。
    返回 None 表示数据加载失败。
    """
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  日期: {date_str}  |  范围: {lon_range[0]:.1f}-{lon_range[1]:.1f}°E, {lat_range[0]:.1f}-{lat_range[1]:.1f}°N")
        print(f"{'─'*55}")

    try:
        glorys_file    = _find_file(glorys_dir,    date_str)
        af_thetao_file = _find_file(af_thetao_dir, date_str)
        af_so_file     = _find_file(af_so_dir,     date_str)

        if verbose:
            print(f"  GLORYS   : {glorys_file.name}")
            print(f"  AF thetao: {af_thetao_file.name}")
            print(f"  AF so    : {af_so_file.name}")

        # ── 加载数据 ───────────────────────────────────────────────────────
        glorys_thetao = _load_glorys_nc_all33(glorys_file,    "thetao", lon_range, lat_range)
        af_thetao     = _load_af_nc_all33    (af_thetao_file, "thetao", lon_range, lat_range)
        glorys_so     = _load_glorys_nc_all33(glorys_file, "so", lon_range, lat_range)
        af_so         = _load_af_nc_all33    (af_so_file,  "so", lon_range, lat_range)

        # 提取纬度用于加权
        latitudes = None
        if weight_by_coslat:
            with xr.open_dataset(glorys_file, mask_and_scale=True) as ds:
                lat_name = "latitude" if "latitude" in ds.coords else "lat"
                lat_vals = ds[lat_name].sel({lat_name: slice(lat_range[0], lat_range[1])}).values
                latitudes = lat_vals

        if verbose:
            print(f"  GLORYS thetao shape: {glorys_thetao.shape}")
            print(f"  AF     thetao shape: {af_thetao.shape}")

        # ── 逐层 RMSE ─────────────────────────────────────────────────────
        thetao_rmse, thetao_mae, thetao_cnt = compute_layer_rmse_all33(
            glorys_thetao, af_thetao, latitudes, weight_by_coslat
        )
        so_rmse, so_mae, so_cnt = compute_layer_rmse_all33(
            glorys_so, af_so, latitudes, weight_by_coslat
        )

        # ── 0.49 m 表层空间分布图 ─────────────────────────────────────────
        if plot_surface and HAS_MATPLOTLIB:
            try:
                os.makedirs(save_dir, exist_ok=True)
                lon2d, lat2d = _get_2d_coords(glorys_file, lon_range, lat_range)
                plot_surface_comparison(
                    glorys_thetao[0], af_thetao[0],
                    glorys_so[0], af_so[0],
                    lon2d, lat2d, date_str, save_dir
                )
            except Exception as e:
                if verbose:
                    print(f"  ⚠ 表层绘图失败: {e}")

        return {
            "date":        date_str,
            "thetao_rmse": thetao_rmse,
            "thetao_mae":  thetao_mae,
            "thetao_cnt":  thetao_cnt,
            "so_rmse":     so_rmse,
            "so_mae":      so_mae,
            "so_cnt":      so_cnt,
        }

    except FileNotFoundError as e:
        print(f"  ⚠ 文件缺失，跳过: {e}")
        return None
    except ValueError as e:
        print(f"  ⚠ 数据范围错误，跳过: {e}")
        return None
    except Exception as e:
        print(f"  ✗ 分析失败: {e}")
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# 结果打印（增强版，含 MAE 和像素数）
# =============================================================================

def print_rmse_table(results: List[Dict], mean_only: bool = False):
    """
    打印逐层 RMSE/MAE 对比表，多日期时打印均值。
    """
    n_dates = len(results)

    thetao_stack = np.stack([r["thetao_rmse"] for r in results], axis=0)
    so_stack     = np.stack([r["so_rmse"]     for r in results], axis=0)
    mean_thetao  = np.nanmean(thetao_stack, axis=0)
    mean_so      = np.nanmean(so_stack,     axis=0)

    print("\n" + "=" * 85)
    print("  GLORYS vs AF  逐层 RMSE 分析结果")
    print(f"  日期数量: {n_dates}   |   深度层数: 33")
    print("=" * 85)

    if not mean_only and n_dates == 1:
        r = results[0]
        header = (f"{'层':<4} {'深度(m)':<10} "
                  f"{'thetao RMSE':<14} {'thetao MAE':<14} {'thetao 像素':<10} "
                  f"{'so RMSE':<14} {'so MAE':<14}")
        print(header)
        print("─" * len(header))
        for i in range(33):
            print(f"{i:<4} {GLORYS_33_DEPTHS[i]:<10.2f} "
                  f"{r['thetao_rmse'][i]:<14.6f} {r['thetao_mae'][i]:<14.6f} {r['thetao_cnt'][i]:<10} "
                  f"{r['so_rmse'][i]:<14.6f} {r['so_mae'][i]:<14.6f}")

    else:
        header = (f"{'层':<4} {'深度(m)':<10} {'thetao RMSE 均值':<18} {'so RMSE 均值':<18}")
        print(header)
        print("─" * len(header))
        for i in range(33):
            t_str = f"{mean_thetao[i]:.6f}" if not np.isnan(mean_thetao[i]) else "NaN"
            s_str = f"{mean_so[i]:.6f}" if not np.isnan(mean_so[i]) else "NaN"
            print(f"{i:<4} {GLORYS_33_DEPTHS[i]:<10.2f} {t_str:<18} {s_str:<18}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  汇总统计（所有层、所有日期均值）")
    print("=" * 85)
    print(f"  thetao 整体均值 RMSE : {np.nanmean(mean_thetao):.6f} °C")
    print(f"  thetao 最小层   RMSE : {np.nanmin(mean_thetao):.6f} °C  "
          f"(层 {int(np.nanargmin(mean_thetao))}, "
          f"深度 {GLORYS_33_DEPTHS[int(np.nanargmin(mean_thetao))]:.2f} m)")
    print(f"  thetao 最大层   RMSE : {np.nanmax(mean_thetao):.6f} °C  "
          f"(层 {int(np.nanargmax(mean_thetao))}, "
          f"深度 {GLORYS_33_DEPTHS[int(np.nanargmax(mean_thetao))]:.2f} m)")
    print()
    print(f"  so     整体均值 RMSE : {np.nanmean(mean_so):.6f} PSU")
    print(f"  so     最小层   RMSE : {np.nanmin(mean_so):.6f} PSU  "
          f"(层 {int(np.nanargmin(mean_so))}, "
          f"深度 {GLORYS_33_DEPTHS[int(np.nanargmin(mean_so))]:.2f} m)")
    print(f"  so     最大层   RMSE : {np.nanmax(mean_so):.6f} PSU  "
          f"(层 {int(np.nanargmax(mean_so))}, "
          f"深度 {GLORYS_33_DEPTHS[int(np.nanargmax(mean_so))]:.2f} m)")
    print("=" * 85)

    return mean_thetao, mean_so


# =============================================================================
# 绘图（垂直廓线 + 0.49m 高亮标记）
# =============================================================================

def plot_rmse_profile(mean_thetao: np.ndarray, mean_so: np.ndarray,
                      results: List[Dict], save_dir: str,
                      lon_range: Tuple[float, float],
                      lat_range: Tuple[float, float]):
    """绘制 GLORYS vs AF 逐层 RMSE 垂直廓线图，并高亮 0.49 m 表层。"""
    if not HAS_MATPLOTLIB:
        print("⚠ matplotlib 不可用，跳过绘图。")
        return

    depths = GLORYS_33_DEPTHS
    n_dates = len(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
    region_str = f"{lon_range[0]:.0f}-{lon_range[1]:.0f}°E, {lat_range[0]:.0f}-{lat_range[1]:.0f}°N"
    fig.suptitle(
        f"GLORYS vs AF — 逐层 RMSE 垂直廓线\n"
        f"区域: {region_str}  |  "
        f"日期: {', '.join(r['date'] for r in results[:3])}"
        + ("..." if n_dates > 3 else ""),
        fontsize=13, fontweight="bold"
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_dates, 1)))

    # ── thetao ──────────────────────────────────────────────────────────
    ax = axes[0]
    for idx, r in enumerate(results):
        label = r["date"] if n_dates <= 10 else None
        ax.plot(r["thetao_rmse"], depths, color=colors[idx],
                alpha=0.4, linewidth=1.0, label=label)
    ax.plot(mean_thetao, depths, color="black", linewidth=2.5,
            label="均值" if n_dates > 1 else results[0]["date"])

    # 高亮 0.49 m 表层（索引 0）
    ax.scatter(mean_thetao[0], depths[0], c="red", s=100,
               zorder=5, edgecolors="black", linewidths=1.2)
    ax.axhline(y=depths[0], color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("RMSE (°C)", fontsize=12)
    ax.set_ylabel("深度 (m)", fontsize=12)
    ax.set_title("温度 thetao", fontsize=13)
    ax.invert_yaxis()
    ax.set_yscale("symlog", linthresh=50)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_yticks(depths[::2])
    ax.set_yticklabels([f"{d:.1f}" for d in depths[::2]], fontsize=7)

    # ── so ──────────────────────────────────────────────────────────────
    ax = axes[1]
    for idx, r in enumerate(results):
        label = r["date"] if n_dates <= 10 else None
        ax.plot(r["so_rmse"], depths, color=colors[idx],
                alpha=0.4, linewidth=1.0, label=label)
    ax.plot(mean_so, depths, color="black", linewidth=2.5,
            label="均值" if n_dates > 1 else results[0]["date"])

    # 高亮 0.49 m 表层（索引 0）
    ax.scatter(mean_so[0], depths[0], c="red", s=100,
               zorder=5, edgecolors="black", linewidths=1.2)
    ax.axhline(y=depths[0], color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("RMSE (PSU)", fontsize=12)
    ax.set_title("盐度 so", fontsize=13)
    ax.invert_yaxis()
    ax.set_yscale("symlog", linthresh=50)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_yticks(depths[::2])
    ax.set_yticklabels([f"{d:.1f}" for d in depths[::2]], fontsize=7)

    plt.tight_layout()

    date_tag = results[0]["date"] if n_dates == 1 else f"{results[0]['date']}_{results[-1]['date']}"
    region_tag = f"{lon_range[0]:.0f}_{lon_range[1]:.0f}_{lat_range[0]:.0f}_{lat_range[1]:.0f}"
    save_path = os.path.join(save_dir, f"rmse_profile_{region_tag}_{date_tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 垂直廓线图已保存: {save_path}")
    plt.close(fig)


# =============================================================================
# CSV 保存（增加区域信息）
# =============================================================================

def save_csv(mean_thetao: np.ndarray, mean_so: np.ndarray,
             results: List[Dict], save_dir: str,
             lon_range: Tuple[float, float],
             lat_range: Tuple[float, float]):
    """将逐层 RMSE 结果保存为 CSV 文件。"""
    if not HAS_PANDAS:
        print("⚠ pandas 不可用，跳过 CSV 保存。")
        return

    rows = []
    for i in range(33):
        row = {
            "layer_idx": i,
            "depth_m": float(GLORYS_33_DEPTHS[i]),
            "lon_min": lon_range[0], "lon_max": lon_range[1],
            "lat_min": lat_range[0], "lat_max": lat_range[1],
            "mean_thetao_rmse": float(mean_thetao[i]),
            "mean_so_rmse": float(mean_so[i]),
        }
        for r in results:
            row[f"thetao_rmse_{r['date']}"] = float(r["thetao_rmse"][i])
            row[f"so_rmse_{r['date']}"]     = float(r["so_rmse"][i])
        rows.append(row)

    df = pd.DataFrame(rows)
    n_dates = len(results)
    date_tag = results[0]["date"] if n_dates == 1 else f"{results[0]['date']}_{results[-1]['date']}"
    region_tag = f"{lon_range[0]:.0f}_{lon_range[1]:.0f}_{lat_range[0]:.0f}_{lat_range[1]:.0f}"
    save_path = os.path.join(save_dir, f"rmse_glorys_af_{region_tag}_{date_tag}.csv")
    df.to_csv(save_path, index=False, float_format="%.8f")
    print(f"📄 CSV 已保存: {save_path}")


# =============================================================================
# 日期范围生成
# =============================================================================

def date_range(start: str, end: str) -> List[str]:
    """生成从 start 到 end（含）的日期字符串列表，格式 YYYYMMDD。"""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end,   "%Y%m%d")
    dates = []
    cur = s
    while cur <= e:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="计算 GLORYS 与 AF 数据在 33 个深度层的逐层 RMSE（支持自定义空间范围与表层可视化）",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── 日期参数 ──────────────────────────────────────────────────────────
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--dates", nargs="+", metavar="YYYYMMDD",
                            help="要分析的日期（一个或多个，空格分隔）")
    date_group.add_argument("--start_date", metavar="YYYYMMDD",
                            help="日期范围起始（需同时指定 --end_date）")
    parser.add_argument("--end_date", metavar="YYYYMMDD",
                        help="日期范围结束")

    # ── 空间范围参数 ─────────────────────────────────────────────────────
    region_group = parser.add_mutually_exclusive_group()
    region_group.add_argument("--region", type=str, choices=list(PRESET_REGIONS.keys()),
                              default="nwpac",
                              help=f"预设海域范围: {', '.join(PRESET_REGIONS.keys())} (默认: nwpac)")
    region_group.add_argument("--lon_range", nargs=2, type=float, metavar=("MIN", "MAX"),
                              help="自定义经度范围，例如: --lon_range 105 125")
    parser.add_argument("--lat_range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="自定义纬度范围（需同时指定 --lon_range），例如: --lat_range 5 25")

    # ── 计算与可视化选项 ──────────────────────────────────────────────────
    parser.add_argument("--coslat_weight", action="store_true",
                        help="是否按 cos(纬度) 进行面积加权（默认不加权）")
    parser.add_argument("--plot_surface", action="store_true",
                        help="是否绘制 0.49 m 表层空间分布对比图（GLORYS vs AF）")

    # ── 路径参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--glorys_dir", type=str, default=r"D:\datasets\Glorys")
    parser.add_argument("--af_thetao_dir", type=str, default=r"D:\datasets\AF_thetao")
    parser.add_argument("--af_so_dir", type=str, default=r"D:\datasets\AF_so")

    # ── 输出参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default="./rmse_analysis_results")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # ── 解析空间范围 ──────────────────────────────────────────────────────
    if args.lon_range:
        if not args.lat_range:
            parser.error("使用 --lon_range 时必须同时指定 --lat_range")
        lon_range = tuple(args.lon_range)
        lat_range = tuple(args.lat_range)
    else:
        lon_range = (PRESET_REGIONS[args.region][0], PRESET_REGIONS[args.region][1])
        lat_range = (PRESET_REGIONS[args.region][2], PRESET_REGIONS[args.region][3])

    # ── 日期列表 ──────────────────────────────────────────────────────────
    if args.dates:
        dates = args.dates
    else:
        if args.end_date is None:
            parser.error("使用 --start_date 时必须同时指定 --end_date")
        dates = date_range(args.start_date, args.end_date)

    print(f"\n{'='*60}")
    print(f"  GLORYS vs AF  逐层 RMSE 分析")
    print(f"{'='*60}")
    print(f"  待分析日期: {len(dates)} 个")
    print(f"  空间范围  : {lon_range[0]:.1f}-{lon_range[1]:.1f}°E, {lat_range[0]:.1f}-{lat_range[1]:.1f}°N")
    if args.coslat_weight:
        print(f"  加权方式  : cos(纬度) 面积加权")
    if args.plot_surface:
        print(f"  表层可视化: 启用 0.49 m 空间分布对比图")
    print(f"  GLORYS 路径   : {args.glorys_dir}")
    print(f"  AF thetao 路径: {args.af_thetao_dir}")
    print(f"  AF so 路径    : {args.af_so_dir}")
    print(f"{'='*60}")

    glorys_dir    = Path(args.glorys_dir)
    af_thetao_dir = Path(args.af_thetao_dir)
    af_so_dir     = Path(args.af_so_dir)

    for name, p in [("GLORYS", glorys_dir), ("AF_thetao", af_thetao_dir), ("AF_so", af_so_dir)]:
        if not p.exists():
            print(f"  ✗ {name} 目录不存在: {p}")
            raise SystemExit(1)

    # 提前创建输出目录（表层图可能在单日期分析时就需要写入）
    if args.save_csv or args.save_plot or args.plot_surface:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── 逐日期分析 ────────────────────────────────────────────────────────
    results = []
    for date_str in dates:
        r = analyze_single_date(
            date_str, glorys_dir, af_thetao_dir, af_so_dir,
            lon_range, lat_range,
            weight_by_coslat=args.coslat_weight,
            plot_surface=args.plot_surface,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
        if r is not None:
            results.append(r)

    if not results:
        print("\n✗ 没有成功分析任何日期，请检查数据路径和文件名。")
        raise SystemExit(1)

    print(f"\n✓ 成功分析 {len(results)}/{len(dates)} 个日期")

    # ── 打印结果 ──────────────────────────────────────────────────────────
    mean_thetao, mean_so = print_rmse_table(results)

    # ── 保存输出 ──────────────────────────────────────────────────────────
    if args.save_csv:
        save_csv(mean_thetao, mean_so, results, args.save_dir, lon_range, lat_range)

    if args.save_plot:
        plot_rmse_profile(mean_thetao, mean_so, results, args.save_dir, lon_range, lat_range)

    return results, mean_thetao, mean_so


if __name__ == "__main__":
    main()