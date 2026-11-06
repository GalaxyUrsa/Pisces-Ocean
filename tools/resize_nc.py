import os
import glob
import numpy as np
import xarray as xr

# ========== 区域与分辨率配置 ==========
LAT_MIN, LAT_MAX = 0.0, 49.99
LON_MIN, LON_MAX = 100.0, 159.99

# 标准 0.25° 中心点网格（与 OISST/ERA5 对齐）
# 纬度：0.125, 0.375, ..., 49.875  → 共 200 个点
# 经度：100.125, 100.375, ..., 159.875 → 共 240 个点
NEW_LAT = np.arange(0.125, 49.85, 0.083)
NEW_LON = np.arange(100.125, 159.85, 0.083)


def resize_nc_to_025deg(input_path: str, output_path: str):
    """将单个 NC 文件插值到 0.25° 并保存。"""
    with xr.open_dataset(input_path) as ds:
        # 兼容坐标名
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"

        # 安全裁剪：处理纬度递减（如 GLORYS）或递增（如 OISST）
        lat_vals = ds[lat_name].values
        if lat_vals[0] > lat_vals[-1]:  # 递减：从北向南
            ds = ds.sel({
                lat_name: slice(LAT_MAX, LAT_MIN),
                lon_name: slice(LON_MIN, LON_MAX),
            })
        else:  # 递增：从南向北
            ds = ds.sel({
                lat_name: slice(LAT_MIN, LAT_MAX),
                lon_name: slice(LON_MIN, LON_MAX),
            })

        # ds["analysed_sst"] = ds["analysed_sst"] - 273.15
        # ds["analysed_sst"].attrs["units"] = "degC"

        # 双线性插值
        ds_out = ds.interp(
            {lat_name: NEW_LAT, lon_name: NEW_LON},
            method="linear",
            kwargs={"fill_value": "extrapolate"}  # 边缘外推，避免边界 NaN
        )

        # 统一坐标名
        rename_map = {}
        if lat_name != "latitude":
            rename_map[lat_name] = "latitude"
        if lon_name != "longitude":
            rename_map[lon_name] = "longitude"
        if rename_map:
            ds_out = ds_out.rename(rename_map)

        # 保存
        ds_out.to_netcdf(output_path)
        return ds_out


def batch_resize(input_folder: str, output_folder: str, pattern: str = "*.nc"):
    """
    批量处理文件夹内所有匹配 pattern 的 NC 文件。
    输出文件名保持原样（如需加后缀可自行修改）。
    """
    if not os.path.isdir(input_folder):
        print(f"错误：输入路径不存在：{input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 查找所有匹配文件
    search_path = os.path.join(input_folder, pattern)
    files = sorted(glob.glob(search_path))
    total = len(files)

    if total == 0:
        print(f"未找到匹配文件：{search_path}")
        return

    print(f"共找到 {total} 个文件，开始处理...\n")

    success = 0
    failed = 0

    for idx, in_path in enumerate(files, 1):
        fname = os.path.basename(in_path)
        out_path = os.path.join(output_folder, fname)

        # 如需要自动加后缀，取消下面这行注释：
        # out_path = out_path.replace(".nc", "_0.25deg.nc")

        print(f"[{idx}/{total}] 处理中：{fname}")

        try:
            resize_nc_to_025deg(in_path, out_path)
            print(f"       已保存 → {out_path}")
            success += 1
        except Exception as e:
            print(f"       失败：{e}")
            failed += 1

    print(f"\n完成：成功 {success} 个，失败 {failed} 个，输出目录：{output_folder}")


if __name__ == "__main__":
    # ========== 直接在这里改路径 ==========
    INPUT_FOLDER = r"D:\datasets\1"      # ← 原始数据文件夹
    OUTPUT_FOLDER = r"D:\datasets\2"   # ← 输出文件夹（自动创建）
    
    batch_resize(INPUT_FOLDER, OUTPUT_FOLDER)