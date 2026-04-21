"""
Dataset Loader for AF Ocean Data Sources (AF_so + AF_thetao)

AF 数据结构说明:
- AF_thetao: D:\\datasets\\AF_thetao\\glorys_0.083deg_{YYYYMMDD}.nc  →  变量: thetao (1, 33, H, W)
- AF_so:     D:\\datasets\\AF_so\\glorys_0.083deg_{YYYYMMDD}.nc      →  变量: so     (1, 33, H, W)

load_single_date 返回与 OceanDatasetLoader 相同的嵌套字典格式:
{
    'SSS':        {'so':     ndarray(H, W)},          # AF_so  第 0 层（当天）
    'SST':        {'thetao': ndarray(H, W)},          # AF_thetao 第 0 层（当天）
    'Glorys':     {'thetao': ndarray(20, H, W),        # AF_thetao 选取深度层（当天，作为 label）
                   'so':     ndarray(20, H, W)},
    'Background': {'thetao': ndarray(20, H, W),        # AF_thetao 选取深度层（当天 -7 天，作为背景场）
                   'so':     ndarray(20, H, W)},
}
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple


class AFDatasetLoader:
    """Load ocean datasets from AF_so and AF_thetao folders."""

    def __init__(self, base_path: str = r"D:\datasets"):
        self.base_path = Path(base_path)
        self.af_so_path     = self.base_path / "AF_so"
        self.af_thetao_path = self.base_path / "AF_thetao"
        self.glorys_path    = self.base_path / "Glorys"

        # 从 33 个深度层中选取 20 层（与原始 OceanDatasetLoader 保持一致）
        self.depth_indices = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
            20, 21, 22, 23, 24, 25, 26, 28, 30, 32
        ]

    # ------------------------------------------------------------------
    # 私有工具
    # ------------------------------------------------------------------

    def _find_file(self, folder: Path, date_str: str) -> Path:
        """在 folder 中查找包含 date_str 的 .nc 文件。"""
        matches = sorted(folder.glob(f"*{date_str}*.nc"))
        if not matches:
            raise FileNotFoundError(
                f"在 {folder} 中找不到日期 {date_str} 的文件"
            )
        return matches[0]

    def _load_nc(self, file_path: Path, variable: str,
                 lon_range: Tuple[float, float] = (100, 160),
                 lat_range: Tuple[float, float] = (0, 50),
                 select_depth: bool = False,
                 unpack: bool = False) -> np.ndarray:
        """
        用 h5py 读取 NetCDF4 文件中的单个变量，裁剪经纬度，
        可选择性地按 depth_indices 取层。

        unpack=True：自动读取 scale_factor / add_offset 并解包（Glorys 打包数据）。

        返回 shape:
          select_depth=False : (H, W)
          select_depth=True  : (20, H, W)
        """
        with h5py.File(file_path, 'r') as ds:
            # 读取坐标
            lat = ds['latitude'][:]
            lon = ds['longitude'][:]

            # 找到经纬度索引范围
            lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])

            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]

            lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
            lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

            raw = ds[variable]  # (time, depth, lat, lon)

            # 读取打包参数（Glorys int16 打包）
            scale_factor = float(raw.attrs['scale_factor'][0]) if unpack and 'scale_factor' in raw.attrs else None
            add_offset   = float(raw.attrs['add_offset'][0])   if unpack and 'add_offset'   in raw.attrs else None
            fill_value   = float(raw.attrs['_FillValue'][0])   if unpack and '_FillValue'   in raw.attrs else None

            if select_depth:
                data = raw[0, self.depth_indices, lat_sl, lon_sl]   # (20, H, W)
            else:
                data = raw[0, 0, lat_sl, lon_sl]                    # (H, W)

        arr = np.array(data, dtype=np.float64)

        if unpack and scale_factor is not None:
            # 先把填充值标记为 NaN，再解包
            if fill_value is not None:
                arr[arr == fill_value] = np.nan
            arr = arr * scale_factor + add_offset
        else:
            # AF 数据：极大 / 极小值为陆地填充值
            arr[arr > 1e10]  = np.nan
            arr[arr < -1e10] = np.nan

        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load_single_date(self, date: str,
                         lon_slice: Tuple[float, float] = (100, 160),
                         lat_slice:  Tuple[float, float] = (0,   50),
                         isLog: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """
        加载指定日期的 AF 数据，返回与 OceanDatasetLoader.load_single_date 相同格式的字典。

        Args:
            date    : 目标日期，格式 'YYYYMMDD'（如 '20260202'）
            lon_slice: 经度范围 (min, max)
            lat_slice: 纬度范围 (min, max)
            isLog   : 是否打印加载进度

        Returns:
            {
                'SSS':        {'so':     ndarray(H, W)},           # AF_so   第 0 层（当天）
                'SST':        {'thetao': ndarray(H, W)},           # AF_thetao 第 0 层（当天）
                'Glorys':     {'thetao': ndarray(20, H, W),        # D:\datasets\Glorys（当天，target）
                               'so':     ndarray(20, H, W)},
                'Background': {'thetao': ndarray(20, H, W),        # AF_thetao 选取深度层（当天 -7 天）
                               'so':     ndarray(20, H, W)},
            }
        """
        bg_date = (datetime.strptime(date, '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')

        if isLog:
            print(f"\n{'='*60}")
            print(f"Loading AF data for date: {date}  (background: {bg_date})")
            print(f"{'='*60}")

        result: Dict[str, Dict[str, np.ndarray]] = {}

        # ── SSS：AF_so 第 0 层（当天）─────────────────────────────────────
        try:
            f = self._find_file(self.af_so_path, date)
            sss = self._load_nc(f, 'so', lon_slice, lat_slice, select_depth=False)
            result['SSS'] = {'so': sss}
            if isLog:
                print(f"\nSSS: {f.name}")
                print(f"  so: shape {sss.shape}")
        except Exception as e:
            if isLog:
                print(f"Error loading SSS: {e}")

        # ── SST：AF_thetao 第 0 层（当天）────────────────────────────────
        try:
            f = self._find_file(self.af_thetao_path, date)
            sst = self._load_nc(f, 'thetao', lon_slice, lat_slice, select_depth=False)
            result['SST'] = {'thetao': sst}
            if isLog:
                print(f"\nSST: {f.name}")
                print(f"  thetao: shape {sst.shape}")
        except Exception as e:
            if isLog:
                print(f"Error loading SST: {e}")

        # ── Glorys（label / target）：来自 D:\datasets\Glorys（当天）────
        try:
            f = self._find_file(self.glorys_path, date)
            t3d = self._load_nc(f, 'thetao', lon_slice, lat_slice, select_depth=True,  unpack=True)
            s3d = self._load_nc(f, 'so',     lon_slice, lat_slice, select_depth=True,  unpack=True)
            result['Glorys'] = {'thetao': t3d, 'so': s3d}
            if isLog:
                print(f"\nGlorys (label): {f.name}")
                print(f"  thetao: shape {t3d.shape}")
                print(f"  so:     shape {s3d.shape}")
        except Exception as e:
            if isLog:
                print(f"Error loading Glorys: {e}")

        # ── Background：AF_thetao + AF_so 选取深度层（date - 7 天）──────
        try:
            f_t = self._find_file(self.af_thetao_path, bg_date)
            f_s = self._find_file(self.af_so_path,     bg_date)
            t3d_bg = self._load_nc(f_t, 'thetao', lon_slice, lat_slice, select_depth=True)
            s3d_bg = self._load_nc(f_s, 'so',     lon_slice, lat_slice, select_depth=True)
            result['Background'] = {'thetao': t3d_bg, 'so': s3d_bg}
            if isLog:
                print(f"\nBackground: {f_t.name} / {f_s.name}")
                print(f"  thetao: shape {t3d_bg.shape}")
                print(f"  so:     shape {s3d_bg.shape}")
        except Exception as e:
            if isLog:
                print(f"Error loading Background: {e}")

        return result