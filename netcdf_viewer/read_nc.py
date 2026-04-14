# -*- coding: utf-8 -*-
"""
NetCDF (.nc) 文件读取与元数据记录工具
"""

import xarray as xr
import pandas as pd
import os
from datetime import datetime


src_path = r"/Users/gary/Desktop/Pisces-Ocean/download_utils/downloaded_data/Glorys"

# =========================
# 配置参数
# =========================
nc_file = r'glorys_0.083deg_20260101.nc'

# 是否生成日志CSV文件（默认False）
save_log_to_csv = True
log_csv_file = "netcdf_metadata_log.csv"

# =========================
# 1. 读取 nc 文件
# =========================
full_nc_path = os.path.join(src_path, nc_file)
print(f"正在读取文件: {full_nc_path}")
ds = xr.open_dataset(full_nc_path)

# =========================
# 2. 查看文件结构
# =========================
print("=== 数据集信息 ===")
print(ds)

print("\n=== 变量列表 ===")
print(list(ds.data_vars))

# =========================
# 3. 提取元数据信息
# =========================
# 提取维度信息（更清晰的格式）
dim_info = ", ".join([f"{k}={v}" for k, v in ds.dims.items()])

metadata = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "file_path": full_nc_path,
    "file_name": os.path.basename(nc_file),
    "file_size_mb": round(os.path.getsize(full_nc_path) / (1024 * 1024), 2),
    "dimensions": dim_info,
    "dim_details": str(dict(ds.dims)),
    "variables": ", ".join(list(ds.data_vars)),
    "num_variables": len(list(ds.data_vars)),
    "coordinates": ", ".join(list(ds.coords)),
    "dataset_info": str(ds),
}

# 添加每个变量的详细信息
for var_name in list(ds.data_vars):
    var = ds[var_name]
    metadata[f"{var_name}_shape"] = str(var.shape)
    metadata[f"{var_name}_dtype"] = str(var.dtype)
    metadata[f"{var_name}_attrs"] = str(dict(var.attrs)) if var.attrs else "None"

print("\n=== 提取的元数据 ===")
for key, value in metadata.items():
    print(f"{key}: {value}")

# =========================
# 4. 保存元数据到CSV日志（可选）
# =========================
if save_log_to_csv:
    # 转换为DataFrame
    df_log = pd.DataFrame([metadata])

    # 如果文件存在，追加；否则创建新文件
    if os.path.exists(log_csv_file):
        df_log.to_csv(log_csv_file, mode='a', header=False, index=False, encoding="utf-8-sig")
        print(f"\n元数据已追加到 {log_csv_file}")
    else:
        df_log.to_csv(log_csv_file, mode='w', header=True, index=False, encoding="utf-8-sig")
        print(f"\n元数据已保存到新文件 {log_csv_file}")

# =========================
# 5. 关闭文件
# =========================
ds.close()
print("\nNetCDF 文件已关闭")