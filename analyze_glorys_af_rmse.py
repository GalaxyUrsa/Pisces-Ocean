"""
GLORYS vs AF 数据 RMSE 分析脚本
===================================

功能：
    对比 GLORYS 与 AF (Analysis & Forecast) 数据在 thetao（温度）和 so（盐度）
    两个变量上，在全部 33 个深度层的逐层 RMSE（均方根误差）。

    - GLORYS 数据路径: D:\\datasets\\Glorys\\
    - AF_thetao 数据路径: D:\\datasets\\AF_thetao\\
    - AF_so     数据路径: D:\\datasets\\AF_so\\

用法示例：
    # 分析单个日期
    python analyze_glorys_af_rmse.py --dates 20260202

    # 分析多个日期（空格分隔）
    python analyze_glorys_af_rmse.py --dates 20260202 20260203 20260204

    # 分析日期范围
    python analyze_glorys_af_rmse.py --start_date 20260101 --end_date 20260131

    # 输出 CSV 和图像
    python analyze_glorys_af_rmse.py --dates 20260202 --save_csv --save_plot
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

# ── 可选依赖（绘图/CSV）────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # 非交互式后端，服务器环境适用
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib 未安装，跳过绘图功能。可通过 pip install matplotlib 安装。")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas 未安装，跳过 CSV 保存功能。可通过 pip install pandas 安装。")


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

# 经纬度裁剪范围（与训练时保持一致）
LON_RANGE = (100.0, 160.0)
LAT_RANGE = (0.0,   50.0)


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
                      lon_range: Tuple[float, float] = LON_RANGE,
                      lat_range: Tuple[float, float] = LAT_RANGE) -> np.ndarray:
    """
    用 h5py 读取 AF NetCDF4 文件中指定变量的全部 33 个深度层。

    AF 数据原始 shape: (1, 33, H, W)，未打包（float32），
    陆地填充值为 > 1e10 或 < -1e10。

    返回 shape: (33, H, W)，dtype float32，陆地/无效像素置为 NaN。
    """
    with h5py.File(file_path, "r") as ds:
        lat = ds["latitude"][:]
        lon = ds["longitude"][:]

        lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])

        lat_sl = slice(np.where(lat_mask)[0][0], np.where(lat_mask)[0][-1] + 1)
        lon_sl = slice(np.where(lon_mask)[0][0], np.where(lon_mask)[0][-1] + 1)

        raw = ds[variable]             # (1, 33, H, W) or (33, H, W)
        # 兼容有无 time 维度的情况
        if raw.ndim == 4:
            data = raw[0, :, lat_sl, lon_sl]    # (33, H, W)
        elif raw.ndim == 3:
            data = raw[:, lat_sl, lon_sl]        # (33, H, W)
        else:
            raise ValueError(f"意外的数据维度: {raw.shape}")

    arr = np.array(data, dtype=np.float64)
    # AF 数据陆地填充
    arr[arr > 1e10]  = np.nan
    arr[arr < -1e10] = np.nan
    return arr.astype(np.float32)


def _load_glorys_nc_all33(file_path: Path, variable: str,
                           lon_range: Tuple[float, float] = LON_RANGE,
                           lat_range: Tuple[float, float] = LAT_RANGE) -> np.ndarray:
    """
    读取 GLORYS NetCDF 文件中指定变量的全部 33 个深度层。

    GLORYS 数据可能以 int16 打包存储，需要使用 xarray 自动解包
    (scale_factor / add_offset / _FillValue)。

    返回 shape: (33, H, W)，dtype float32，无效像素置为 NaN。
    """
    ds = xr.open_dataset(file_path, mask_and_scale=True)

    # 兼容经纬度坐标名称
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude"  if "latitude"  in ds.coords else "lat"

    var_data = ds[variable].sel(
        {lon_name: slice(lon_range[0], lon_range[1]),
         lat_name: slice(lat_range[0], lat_range[1])}
    )

    # 去掉 time 维度（如果有）
    if "time" in var_data.dims:
        var_data = var_data.isel(time=0)

    arr = var_data.values  # (33, H, W) or (depth, lat, lon)
    ds.close()

    if arr.ndim == 2:
        # 如果数据没有深度维（不应出现），扩展为 (1, H, W)
        arr = arr[np.newaxis, ...]

    return arr.astype(np.float32)


# =============================================================================
# 逐层 RMSE 计算
# =============================================================================

def compute_layer_rmse_all33(
        glorys_3d: np.ndarray,
        af_3d: np.ndarray
) -> np.ndarray:
    """
    计算两个 (33, H, W) 数组在每个深度层的 RMSE。

    只对两者均不为 NaN 的像素计算。

    Returns:
        rmse_arr : shape (33,)，每层的 RMSE；若某层有效像素为 0 则返回 NaN。
    """
    assert glorys_3d.shape == af_3d.shape, (
        f"形状不匹配: glorys={glorys_3d.shape}, af={af_3d.shape}"
    )
    assert glorys_3d.shape[0] == 33, (
        f"期望 33 个深度层，实际得到 {glorys_3d.shape[0]} 层"
    )

    rmse_arr = np.full(33, np.nan, dtype=np.float64)
    for i in range(33):
        g = glorys_3d[i].astype(np.float64)
        a = af_3d[i].astype(np.float64)
        valid = ~np.isnan(g) & ~np.isnan(a)
        if valid.sum() == 0:
            continue
        diff = g[valid] - a[valid]
        rmse_arr[i] = np.sqrt(np.mean(diff ** 2))

    return rmse_arr


# =============================================================================
# 单日期分析
# =============================================================================

def analyze_single_date(
        date_str: str,
        glorys_dir: Path,
        af_thetao_dir: Path,
        af_so_dir: Path,
        verbose: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    对单个日期计算 thetao 和 so 在 33 层上的 RMSE。

    Returns:
        dict with keys:
            'thetao_rmse' : np.ndarray (33,)
            'so_rmse'     : np.ndarray (33,)
            'date'        : str
        返回 None 表示数据加载失败。
    """
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  日期: {date_str}")
        print(f"{'─'*55}")

    try:
        # ── 查找文件 ──────────────────────────────────────────────────────
        glorys_file    = _find_file(glorys_dir,    date_str)
        af_thetao_file = _find_file(af_thetao_dir, date_str)
        af_so_file     = _find_file(af_so_dir,     date_str)

        if verbose:
            print(f"  GLORYS   : {glorys_file.name}")
            print(f"  AF thetao: {af_thetao_file.name}")
            print(f"  AF so    : {af_so_file.name}")

        # ── 加载 thetao ───────────────────────────────────────────────────
        glorys_thetao = _load_glorys_nc_all33(glorys_file,    "thetao")
        af_thetao     = _load_af_nc_all33    (af_thetao_file, "thetao")

        # ── 加载 so ───────────────────────────────────────────────────────
        glorys_so     = _load_glorys_nc_all33(glorys_file, "so")
        af_so         = _load_af_nc_all33    (af_so_file,  "so")

        if verbose:
            print(f"  GLORYS thetao shape: {glorys_thetao.shape}")
            print(f"  AF     thetao shape: {af_thetao.shape}")
            print(f"  GLORYS so     shape: {glorys_so.shape}")
            print(f"  AF     so     shape: {af_so.shape}")

        # ── 逐层 RMSE ─────────────────────────────────────────────────────
        thetao_rmse = compute_layer_rmse_all33(glorys_thetao, af_thetao)
        so_rmse     = compute_layer_rmse_all33(glorys_so,     af_so)

        return {
            "date":        date_str,
            "thetao_rmse": thetao_rmse,   # (33,)
            "so_rmse":     so_rmse,        # (33,)
        }

    except FileNotFoundError as e:
        print(f"  ⚠ 文件缺失，跳过: {e}")
        return None
    except Exception as e:
        print(f"  ✗ 分析失败: {e}")
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# 结果打印
# =============================================================================

def print_rmse_table(results: List[Dict], mean_only: bool = False):
    """
    打印逐层 RMSE 对比表。

    若 results 包含多个日期，还会打印多日期均值。
    """
    n_dates = len(results)

    # 计算多日期均值（忽略 NaN）
    thetao_stack = np.stack([r["thetao_rmse"] for r in results], axis=0)  # (n, 33)
    so_stack     = np.stack([r["so_rmse"]     for r in results], axis=0)  # (n, 33)
    mean_thetao  = np.nanmean(thetao_stack, axis=0)  # (33,)
    mean_so      = np.nanmean(so_stack,     axis=0)  # (33,)

    # ── 表头 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  GLORYS vs AF  逐层 RMSE 分析结果")
    print(f"  日期数量: {n_dates}   |   深度层数: 33")
    print("=" * 78)

    if not mean_only and n_dates == 1:
        # 单日期：直接打印该日期的 RMSE
        r = results[0]
        header = (f"{'层':<4} {'深度(m)':<10} "
                  f"{'thetao RMSE (°C)':<20} {'so RMSE (PSU)':<20}")
        print(header)
        print("─" * len(header))
        for i in range(33):
            t_val = r["thetao_rmse"][i]
            s_val = r["so_rmse"][i]
            t_str = f"{t_val:.6f}" if not np.isnan(t_val) else "   NaN  "
            s_str = f"{s_val:.6f}" if not np.isnan(s_val) else "   NaN  "
            print(f"{i:<4} {GLORYS_33_DEPTHS[i]:<10.2f} {t_str:<20} {s_str:<20}")

    else:
        # 多日期：打印每个日期 + 均值
        date_cols = "  ".join(f"{r['date'][:8]:>12}" for r in results)
        thetao_header = (f"\n{'─'*20} thetao RMSE (°C) {'─'*20}\n"
                         f"{'层':<4} {'深度(m)':<10} {date_cols}  {'均值':>12}")
        print(thetao_header)
        print("─" * (28 + 14 * n_dates))
        for i in range(33):
            row_vals = "  ".join(
                f"{r['thetao_rmse'][i]:>12.6f}" if not np.isnan(r['thetao_rmse'][i])
                else f"{'NaN':>12}"
                for r in results
            )
            mean_str = (f"{mean_thetao[i]:>12.6f}" if not np.isnan(mean_thetao[i])
                        else f"{'NaN':>12}")
            print(f"{i:<4} {GLORYS_33_DEPTHS[i]:<10.2f} {row_vals}  {mean_str}")

        so_header = (f"\n{'─'*20} so RMSE (PSU) {'─'*20}\n"
                     f"{'层':<4} {'深度(m)':<10} {date_cols}  {'均值':>12}")
        print(so_header)
        print("─" * (28 + 14 * n_dates))
        for i in range(33):
            row_vals = "  ".join(
                f"{r['so_rmse'][i]:>12.6f}" if not np.isnan(r['so_rmse'][i])
                else f"{'NaN':>12}"
                for r in results
            )
            mean_str = (f"{mean_so[i]:>12.6f}" if not np.isnan(mean_so[i])
                        else f"{'NaN':>12}")
            print(f"{i:<4} {GLORYS_33_DEPTHS[i]:<10.2f} {row_vals}  {mean_str}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  汇总统计（所有层、所有日期均值）")
    print("=" * 78)
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
    print("=" * 78)

    return mean_thetao, mean_so


# =============================================================================
# 绘图
# =============================================================================

def plot_rmse_profile(mean_thetao: np.ndarray, mean_so: np.ndarray,
                      results: List[Dict], save_dir: str):
    """
    绘制 GLORYS vs AF 逐层 RMSE 垂直廓线图（深度随 y 轴递增，向下）。
    """
    if not HAS_MATPLOTLIB:
        print("⚠ matplotlib 不可用，跳过绘图。")
        return

    depths = GLORYS_33_DEPTHS
    n_dates = len(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
    fig.suptitle(
        f"GLORYS vs AF — 逐层 RMSE 垂直廓线\n"
        f"日期: {', '.join(r['date'] for r in results[:5])}"
        + ("..." if n_dates > 5 else ""),
        fontsize=14, fontweight="bold"
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
    ax.set_xlabel("RMSE (°C)", fontsize=12)
    ax.set_ylabel("深度 (m)", fontsize=12)
    ax.set_title("温度 thetao", fontsize=13)
    ax.invert_yaxis()
    ax.set_yscale("symlog", linthresh=50)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # 标注各层深度刻度
    ax.set_yticks(depths)
    ax.set_yticklabels([f"{d:.1f}" for d in depths], fontsize=6)

    # ── so ──────────────────────────────────────────────────────────────
    ax = axes[1]
    for idx, r in enumerate(results):
        label = r["date"] if n_dates <= 10 else None
        ax.plot(r["so_rmse"], depths, color=colors[idx],
                alpha=0.4, linewidth=1.0, label=label)

    ax.plot(mean_so, depths, color="black", linewidth=2.5,
            label="均值" if n_dates > 1 else results[0]["date"])
    ax.set_xlabel("RMSE (PSU)", fontsize=12)
    ax.set_title("盐度 so", fontsize=13)
    ax.invert_yaxis()
    ax.set_yscale("symlog", linthresh=50)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax.set_yticks(depths)
    ax.set_yticklabels([f"{d:.1f}" for d in depths], fontsize=6)

    plt.tight_layout()

    date_tag = results[0]["date"] if n_dates == 1 else f"{results[0]['date']}_{results[-1]['date']}"
    save_path = os.path.join(save_dir, f"rmse_profile_{date_tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 图像已保存: {save_path}")
    plt.close(fig)


# =============================================================================
# CSV 保存
# =============================================================================

def save_csv(mean_thetao: np.ndarray, mean_so: np.ndarray,
             results: List[Dict], save_dir: str):
    """将逐层 RMSE 结果保存为 CSV 文件。"""
    if not HAS_PANDAS:
        print("⚠ pandas 不可用，跳过 CSV 保存。")
        return

    rows = []
    for i in range(33):
        row = {
            "layer_idx": i,
            "depth_m": float(GLORYS_33_DEPTHS[i]),
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
    save_path = os.path.join(save_dir, f"rmse_glorys_af_{date_tag}.csv")
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
        description="计算 GLORYS 与 AF 数据在 33 个深度层的逐层 RMSE",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── 日期参数 ──────────────────────────────────────────────────────────
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--dates", nargs="+", metavar="YYYYMMDD",
        help="要分析的日期（一个或多个，空格分隔）\n  示例: --dates 20260202 20260203"
    )
    date_group.add_argument(
        "--start_date", metavar="YYYYMMDD",
        help="日期范围起始（需同时指定 --end_date）"
    )

    parser.add_argument(
        "--end_date", metavar="YYYYMMDD",
        help="日期范围结束（与 --start_date 配合使用）"
    )

    # ── 路径参数 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--glorys_dir", type=str, default=r"D:\datasets\Glorys",
        help="GLORYS 数据目录（默认: D:\\datasets\\Glorys）"
    )
    parser.add_argument(
        "--af_thetao_dir", type=str, default=r"D:\datasets\AF_thetao",
        help="AF 温度数据目录（默认: D:\\datasets\\AF_thetao）"
    )
    parser.add_argument(
        "--af_so_dir", type=str, default=r"D:\datasets\AF_so",
        help="AF 盐度数据目录（默认: D:\\datasets\\AF_so）"
    )

    # ── 输出参数 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_dir", type=str, default="./rmse_analysis_results",
        help="结果保存目录（默认: ./rmse_analysis_results）"
    )
    parser.add_argument(
        "--save_csv", action="store_true",
        help="是否保存 CSV 结果文件"
    )
    parser.add_argument(
        "--save_plot", action="store_true",
        help="是否保存 RMSE 垂直廓线图"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="减少输出信息"
    )

    args = parser.parse_args()

    # ── 日期列表 ──────────────────────────────────────────────────────────
    if args.dates:
        dates = args.dates
    else:
        if args.end_date is None:
            parser.error("使用 --start_date 时必须同时指定 --end_date")
        dates = date_range(args.start_date, args.end_date)

    print(f"\n{'='*55}")
    print(f"  GLORYS vs AF  逐层 RMSE 分析")
    print(f"{'='*55}")
    print(f"  待分析日期: {len(dates)} 个")
    print(f"  GLORYS 路径   : {args.glorys_dir}")
    print(f"  AF thetao 路径: {args.af_thetao_dir}")
    print(f"  AF so 路径    : {args.af_so_dir}")
    print(f"{'='*55}")

    glorys_dir    = Path(args.glorys_dir)
    af_thetao_dir = Path(args.af_thetao_dir)
    af_so_dir     = Path(args.af_so_dir)

    # ── 目录检查 ──────────────────────────────────────────────────────────
    for name, p in [("GLORYS", glorys_dir),
                    ("AF_thetao", af_thetao_dir),
                    ("AF_so", af_so_dir)]:
        if not p.exists():
            print(f"  ✗ {name} 目录不存在: {p}")
            raise SystemExit(1)

    # ── 逐日期分析 ────────────────────────────────────────────────────────
    results = []
    for date_str in dates:
        r = analyze_single_date(
            date_str, glorys_dir, af_thetao_dir, af_so_dir,
            verbose=not args.quiet
        )
        if r is not None:
            results.append(r)

    if not results:
        print("\n✗ 没有成功分析任何日期，请检查数据路径和文件名。")
        raise SystemExit(1)

    print(f"\n✓ 成功分析 {len(results)}/{len(dates)} 个日期")

    # ── 打印结果表格 ──────────────────────────────────────────────────────
    mean_thetao, mean_so = print_rmse_table(results)

    # ── 可选：保存输出 ────────────────────────────────────────────────────
    if args.save_csv or args.save_plot:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.save_csv:
        save_csv(mean_thetao, mean_so, results, args.save_dir)

    if args.save_plot:
        plot_rmse_profile(mean_thetao, mean_so, results, args.save_dir)

    return results, mean_thetao, mean_so


if __name__ == "__main__":
    main()
