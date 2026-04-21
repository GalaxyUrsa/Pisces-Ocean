"""
快速查看 D:\\datasets\\AF_so 中所有文件第一层（表面层）的彩图
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

AF_SO_DIR = Path(r"D:\datasets\AF_so")
files = sorted(AF_SO_DIR.glob("*.nc"))

print(f"找到 {len(files)} 个文件：")
for f in files:
    print(f"  {f.name}")

fig, axes = plt.subplots(1, len(files), figsize=(8 * len(files), 6))
if len(files) == 1:
    axes = [axes]

for ax, fp in zip(axes, files):
    with h5py.File(fp, 'r') as ds:
        lat   = ds['latitude'][:]
        lon   = ds['longitude'][:]
        so    = ds['so'][0, 0, :, :]      # (H, W)  第 0 层（表面）
        depth = ds['depth'][0]

    # 将填充值（极大值）替换为 NaN
    so = np.array(so, dtype=np.float64)
    so[so > 1e10] = np.nan

    print(f"\n{fp.name}  —  depth[0] = {depth:.4f} m")
    print(f"  shape : {so.shape}")
    print(f"  valid : {np.sum(~np.isnan(so))} 个有效点")
    print(f"  range : {np.nanmin(so):.4f}  ~  {np.nanmax(so):.4f}  psu")

    im = ax.pcolormesh(lon, lat, so,
                       cmap='jet',
                       vmin=33, vmax=38,
                       shading='auto')
    plt.colorbar(im, ax=ax, label='Salinity (psu)', shrink=0.8)
    ax.set_title(f"{fp.stem}\n(depth[0] = {depth:.2f} m)", fontsize=12)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

plt.suptitle("AF_so  —  Surface Layer (Layer 0)", fontsize=14, fontweight='bold')
plt.tight_layout()

out = r"F:\PythonWorkspace\Pisces-Ocean\check_AF_so.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n✅ 图像已保存到: {out}")
plt.show()