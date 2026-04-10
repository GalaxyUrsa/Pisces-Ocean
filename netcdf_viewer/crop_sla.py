# -*- coding: utf-8 -*-
"""
SLA NetCDF 文件裁剪工具
裁剪到指定区域：经度 100-160°E，纬度 0-50°N
"""

import xarray as xr
import os
from pathlib import Path
from tqdm import tqdm

# =========================
# 配置参数
# =========================
src_folder = r"F:\PythonWorkspace\predict_ts\datasets\SLA_raw"
dst_folder = r"F:\PythonWorkspace\predict_ts\datasets\SLA"

# 裁剪区域
lon_min, lon_max = 100, 160
lat_min, lat_max = 0, 50

print(f"源文件夹: {src_folder}")
print(f"目标文件夹: {dst_folder}")
print(f"裁剪区域: 经度 {lon_min}-{lon_max}°E, 纬度 {lat_min}-{lat_max}°N\n")

# =========================
# 1. 创建输出文件夹
# =========================
os.makedirs(dst_folder, exist_ok=True)

# =========================
# 2. 获取所有 nc 文件
# =========================
nc_files = list(Path(src_folder).glob("*.nc"))
print(f"找到 {len(nc_files)} 个 NetCDF 文件\n")

if len(nc_files) == 0:
    print("未找到任何 .nc 文件，退出程序")
    exit()

# =========================
# 3. 裁剪处理
# =========================
success_count = 0
error_count = 0

for nc_file in tqdm(nc_files, desc="裁剪进度"):
    try:
        # 读取原始文件
        ds = xr.open_dataset(nc_file)

        # 检查坐标名称（可能是 lat/lon 或 latitude/longitude）
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

        print(f"\n处理: {nc_file.name}")
        print(f"  原始维度: {lat_name}={len(ds[lat_name])}, {lon_name}={len(ds[lon_name])}")

        # 裁剪到指定区域
        ds_cropped = ds.sel(
            {lat_name: slice(lat_min, lat_max),
             lon_name: slice(lon_min, lon_max)}
        )

        # 保留原始属性并添加处理信息
        ds_cropped.attrs = ds.attrs
        ds_cropped.attrs['spatial_extent'] = f"lon: {lon_min}-{lon_max}, lat: {lat_min}-{lat_max}"
        ds_cropped.attrs['processing'] = "cropped to specified region"

        # 保存到新文件
        output_file = os.path.join(dst_folder, nc_file.name)
        ds_cropped.to_netcdf(output_file)

        print(f"  裁剪后维度: {lat_name}={len(ds_cropped[lat_name])}, {lon_name}={len(ds_cropped[lon_name])}")
        print(f"  [OK] 已保存到: {output_file}")

        # 关闭数据集
        ds.close()
        ds_cropped.close()

        success_count += 1

    except Exception as e:
        print(f"\n[ERROR] 处理 {nc_file.name} 时出错: {str(e)}")
        error_count += 1
        continue

print("\n" + "="*60)
print("=== 裁剪完成 ===")
print(f"成功处理: {success_count} 个文件")
print(f"失败: {error_count} 个文件")
print(f"输出目录: {dst_folder}")
print(f"区域范围: 经度 {lon_min}-{lon_max}°E, 纬度 {lat_min}-{lat_max}°N")
