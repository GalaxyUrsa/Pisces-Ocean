# -*- coding: utf-8 -*-
"""
SST NetCDF 文件重采样工具
将 0.25° 分辨率重采样到 0.125° 分辨率
并裁剪到指定区域：经度 100-160°E，纬度 0-50°N
"""

import xarray as xr
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# =========================
# 配置参数
# =========================
src_folder = r"F:\PythonWorkspace\predict_ts\download_utils\downloaded_data\SST"
dst_folder = r"F:\PythonWorkspace\predict_ts\download_utils\downloaded_data\SST_resample"

# 目标分辨率
target_resolution = 0.125

# 裁剪区域 (与 dataloader.py 保持一致)
lon_min, lon_max = 100, 159.875
lat_min, lat_max = 0, 49.875

# 计算网格点数
# 经度: 100 到 159.875, 步长 0.125 -> 480 个点
# 纬度: 0 到 49.875, 步长 0.125 -> 400 个点
n_lon = int((lon_max - lon_min) / target_resolution) + 1  # 480
n_lat = int((lat_max - lat_min) / target_resolution) + 1  # 400

# =========================
# 1. 创建输出文件夹
# =========================
os.makedirs(dst_folder, exist_ok=True)
print(f"输出文件夹: {dst_folder}")
print(f"裁剪区域: 经度 {lon_min}-{lon_max}°E, 纬度 {lat_min}-{lat_max}°N")

# =========================
# 2. 获取所有 nc 文件
# =========================
nc_files = list(Path(src_folder).glob("*.nc"))
print(f"找到 {len(nc_files)} 个 NetCDF 文件")

if len(nc_files) == 0:
    print("未找到任何 .nc 文件，退出程序")
    exit()

# =========================
# 3. 重采样处理
# =========================
for nc_file in tqdm(nc_files, desc="重采样进度"):
    try:
        # 读取原始文件
        ds = xr.open_dataset(nc_file)

        # 创建新的经纬度网格（0.125° 分辨率，覆盖裁剪区域）
        # 使用 linspace 确保精确的点数：480 个经度点，400 个纬度点
        new_lon = np.linspace(lon_min, lon_max, n_lon)
        new_lat = np.linspace(lat_min, lat_max, n_lat)

        # 使用线性插值进行重采样
        ds_resampled = ds.interp(
            lat=new_lat,
            lon=new_lon,
            method='linear'
        )

        # 保留原始属性并添加处理信息
        ds_resampled.attrs = ds.attrs
        ds_resampled.attrs['resampled_resolution'] = f"{target_resolution} degrees"
        ds_resampled.attrs['resampling_method'] = "linear interpolation"
        ds_resampled.attrs['spatial_extent'] = f"lon: {lon_min}-{lon_max}, lat: {lat_min}-{lat_max}"

        # 保存到新文件
        output_file = os.path.join(dst_folder, nc_file.name)
        ds_resampled.to_netcdf(output_file)

        # 关闭数据集
        ds.close()
        ds_resampled.close()

        print(f"[OK] 完成: {nc_file.name}")
        print(f"  原始维度: lat={len(ds.lat)}, lon={len(ds.lon)}")
        print(f"  新维度: lat={len(new_lat)}, lon={len(new_lon)}")

    except Exception as e:
        print(f"[ERROR] 处理 {nc_file.name} 时出错: {str(e)}")
        continue

print("\n=== 重采样完成 ===")
print(f"处理文件数: {len(nc_files)}")
print(f"输出目录: {dst_folder}")
print(f"分辨率: {target_resolution} degrees")
print(f"区域范围: 经度 {lon_min}-{lon_max}°E, 纬度 {lat_min}-{lat_max}°N")