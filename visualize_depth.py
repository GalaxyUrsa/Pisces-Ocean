import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset('glo12_rg_1d-m_20220601-20220601_3D-so_hcst_R20220615.nc')

# 所有33层深度值
all_depths = ds['depth'].values
np.set_printoptions(suppress=True)

# 选取的20层索引
DEPTH_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]
depth_values = all_depths[DEPTH_INDICES]

# 计算每个选中层的厚度：覆盖到与相邻选中层的中点
# 即从上一个选中层与当前层的中点，到当前层与下一个选中层的中点
n_sel = len(DEPTH_INDICES)
thickness = np.zeros(n_sel)
for k in range(n_sel):
    d_cur = depth_values[k]
    d_prev = depth_values[k - 1] if k > 0 else 0.0          # 上边界：上一选中层，或海面
    d_next = depth_values[k + 1] if k < n_sel - 1 else d_cur + (d_cur - depth_values[k - 1])  # 下边界外推
    thickness[k] = (d_cur - d_prev) / 2 + (d_next - d_cur) / 2

print(f"Selected {len(depth_values)} depth levels:")
for idx, d, t in zip(DEPTH_INDICES, depth_values, thickness):
    print(f"  index={idx:2d}  depth={d:8.2f} m  thickness={t:.2f} m")

# 画图
fig, ax = plt.subplots(figsize=(6, 10))
ax.barh(depth_values, thickness, height=thickness * 0.85, align='center', color='steelblue', edgecolor='white')

# 在每个 bar 右侧标注深度值
for d, t in zip(depth_values, thickness):
    ax.text(t + 0.5, d, f'{d:.1f} m', va='center', fontsize=7)

ax.set_xlabel('Thickness (m)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
ax.set_title(f'Selected {len(depth_values)} Depth Levels')
plt.tight_layout()
plt.savefig('depth_thickness.png', dpi=150)
plt.show()