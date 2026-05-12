"""
将输入文件夹中所有 .nc 文件的指定变量移除，保存到输出文件夹。
"""
import os
import shutil
import netCDF4 as nc

INPUT_DIR  = r"D:\datasets\Glorys_0.083deg"
OUTPUT_DIR = r"D:\datasets\Glorys_so_0.083deg"
REMOVE_VARS = ["vo", "uo", "thetao"]


def remove_variable(src_path, dst_path, var_names):
    skip = set(var_names)
    with nc.Dataset(src_path, "r") as src, nc.Dataset(dst_path, "w") as dst:
        # 复制全局属性
        dst.setncatts(src.__dict__)

        # 复制维度
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))

        # 复制变量（跳过要移除的）
        for name, var in src.variables.items():
            if name in skip:
                continue
            out_var = dst.createVariable(name, var.datatype, var.dimensions,
                                         chunksizes=var.chunking() if var.chunking() != "contiguous" else None)
            out_var.setncatts(var.__dict__)
            # 分块写入，避免大变量一次性读入内存导致 HDF error
            if var.ndim == 0:
                out_var.assignValue(var.getValue())
            elif var.shape[0] == 0:
                pass
            else:
                chunk = max(1, min(var.shape[0], 10))
                for i in range(0, var.shape[0], chunk):
                    out_var[i:i+chunk] = var[i:i+chunk]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nc_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".nc")]
    print(f"共找到 {len(nc_files)} 个 .nc 文件，移除变量: {REMOVE_VARS}")

    for fname in nc_files:
        src = os.path.join(INPUT_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(dst):
            print(f"  跳过（已存在）: {fname}")
            continue
        remove_variable(src, dst, REMOVE_VARS)
        print(f"  处理完成: {fname}")

    print("全部完成。")
