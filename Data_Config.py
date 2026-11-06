import numpy as np

# 标签相对背景场的时间偏移天数（正数=未来，负数=过去），所有 OUTPUT_VARS 统一使用
BG_OFFSET_DAYS = 1

# 模型输入变量（背景场等）
# 每条格式：[folder, nc_var, output_name, opts]
# opts 中 select_depth=True 表示该变量有深度维（贡献 N_DEPTH 通道），否则贡献 1 通道
INPUT_VARS = [
    ['Glorys_thetao_0.083deg_50layer', 'thetao', 'bg_t_3d',     {'select_depth': True}],
    ['Glorys_so_0.083deg_50layer',     'so',     'bg_s_3d',     {'select_depth': True}],
    ['Glorys_uo_0.083deg_50layer',     'uo',     'bg_uo_3d',    {'select_depth': True}],
    ['Glorys_vo_0.083deg_50layer',     'vo',     'bg_vo_3d',    {'select_depth': True}],
    # ['era5_wind_u10_0.083deg',         'u10',    'bg_u10',      {}],
    # ['era5_wind_v10_0.083deg',         'v10',    'bg_v10',      {}],
    # ['era5_swh_0.083deg',              'swh',    'bg_swh',      {}],
    # ['era5_mwd_u_0.083deg',            'mwd_u',  'bg_mwd_u',    {}],
    # ['era5_mwd_v_0.083deg',            'mwd_v',  'bg_mwd_v',    {}],
]

# 监督标签变量（模型预测目标）
OUTPUT_VARS = [
    ['Glorys_thetao_0.083deg_50layer', 'thetao', 'label_t_3d',  {'select_depth': True, 'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_t_3d',  'weight': 1.0}],
    ['Glorys_so_0.083deg_50layer',     'so',     'label_s_3d',  {'select_depth': True, 'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_s_3d',  'weight': 0.1}],
    ['Glorys_uo_0.083deg_50layer',     'uo',     'label_uo_3d', {'select_depth': True, 'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_uo_3d', 'weight': 0.5}],
    ['Glorys_vo_0.083deg_50layer',     'vo',     'label_vo_3d', {'select_depth': True, 'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_vo_3d', 'weight': 0.5}],
    # ['era5_wind_u10_0.083deg',         'u10',    'label_u10',   {                      'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_u10',   'weight': 0.1}],
    # ['era5_wind_v10_0.083deg',         'v10',    'label_v10',   {                      'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_v10',   'weight': 0.1}],
    # ['era5_swh_0.083deg',              'swh',    'label_swh',   {                      'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_swh',   'weight': 0.1}],
    # ['era5_mwd_u_0.083deg',            'mwd_u',  'label_mwd_u', {                      'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_mwd_u', 'weight': 0.1}],
    # ['era5_mwd_v_0.083deg',            'mwd_v',  'label_mwd_v', {                      'bg_offset_days': BG_OFFSET_DAYS, 'bg_key': 'bg_mwd_v', 'weight': 0.1}],
]

data_index = INPUT_VARS + OUTPUT_VARS

# 兼容旧代码导入（inference/eval/fine_tune 等文件暂未迁移）
SURFACE_VARS = []
_SURFACE_INDEX = {
    'sss':  ['Glorys_so_surface_0.083deg', 'so',  'sss'],
    'sst':  ['Glorys_thetao_surface_0.083deg', 'thetao',  'sst'],
    'sla':  ['SLA', 'sla',  'sla'],
    'ugos': ['SLA', 'ugos', 'ugos'],
    'vgos': ['SLA', 'vgos', 'vgos'],
}

RAW_DATASET_PATH = r"G:\datasets"
NPZ_DIR          = r"F:\datasets_20layer_npz"

# 新训练的日志根目录，保存 checkpoint、曲线、归一化统计及源码快照。
LOG_ROOT = r"G:/Pisces-Ocean_v2.0_logs"

# train.py 和 train_AR.py 共用的起始日期范围（YYYYMMDD，包含首尾日期）
TRAIN_START_DATE = '20210108'
TRAIN_END_DATE   = '20241220'
VAL_START_DATE   = '20250101'
VAL_END_DATE     = '20251220'

# 预训练填 checkpoint 文件，续训填包含 last_checkpoint.pth 的目录；None 表示不启用。
# 同时设置时优先续训；预训练仅继承模型权重及 norm，不恢复优化器或 epoch。
TRAIN_PRETRAIN_PATH = None
TRAIN_RESUME_DIR    = None
AR_PRETRAIN_PATH    = "G:/Pisces-Ocean_v2.0_logs/pretrain_2021_2024_glo_glo_convnextv5_80layer/best_model.pth"  #"./logs/SwinUnet/best_model.pth"
AR_RESUME_DIR       = None

NAN_FILL_VALUE = 0  # 归一化后用于替换 NaN 的填充值（陆地/缺测区域）

# # 南中国海区域裁剪配置（105°E–125°E，5°N–21.7°N）
# # 原始网格：600(lat) × 720(lon)，分辨率 0.083°，范围 0–50°N / 100–160°E
CROP_ROW_START = 0   # 5°N
CROP_ROW_END   = 600  # 21.7°N
CROP_COL_START = 0   # 105°E
CROP_COL_END   = 720  # 125°E

# GLORYS12 标准50层深度（米），从 NetCDF 深度坐标直接取
GLORYS_ALL_DEPTHS = np.array([
    0.494025,    1.541375,    2.645675,    3.819488,    5.078224,
    6.440326,    7.929274,    9.572997,   11.405001,   13.467127,
   15.810017,   18.495564,   21.597761,   25.211399,   29.444731,
   34.434151,   40.343960,   47.373692,   55.764259,   65.807205,
   77.853851,   92.325691,  109.729004,  130.665993,  155.850769,
  186.125626,  222.475021,  266.040039,  318.126984,  380.213013,
  453.937988,  541.088989,  644.007019,  763.333008,  902.338989,
 1062.439941, 1245.291992, 1452.250977, 1684.283936, 1941.892944,
 2225.077881, 2533.335938, 2865.702881, 3220.819092, 3597.031982,
 3992.483887, 4405.224121, 4833.291016, 5274.784180, 5727.916504,
])

# 实际使用的深度层索引（从50层中选取，偏向上层和中层）
DEPTH_INDICES = [0, 7, 11, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
# DEPTH_INDICES = [0, 10, 15, 20, 25, 30]
# DEPTH_INDICES = [i for i in range(50)]

N_DEPTH = len(DEPTH_INDICES)  # 每个3D变量的深度层数，与 DEPTH_INDICES 保持一致

DEPTH_VALUES = GLORYS_ALL_DEPTHS[DEPTH_INDICES]  # 实际深度值，供 NetCDF 坐标使用

# 可视化配置：按 nc_var 名索引，供 viz 模块使用
VAR_VIZ_CONFIG = {
    'thetao': {'label': 'Temperature (°C)',  'vmin':  0,   'vmax': 40,  'err_min': -3.0,  'err_max': 3.0,  'cmap': 'RdYlBu_r'},
    'so':     {'label': 'Salinity (PSU)',    'vmin': 30,   'vmax': 40,  'err_min': -0.5,  'err_max': 0.5,  'cmap': 'RdYlBu_r'},
    'uo':     {'label': 'U-velocity (m/s)',  'vmin': -1,   'vmax':  1,  'err_min': -0.3,  'err_max': 0.3,  'cmap': 'RdBu_r'},
    'vo':     {'label': 'V-velocity (m/s)',  'vmin': -1,   'vmax':  1,  'err_min': -0.3,  'err_max': 0.3,  'cmap': 'RdBu_r'},
    'u10':    {'label': 'Wind U10 (m/s)',    'vmin': -20,  'vmax': 20,  'err_min': -5.0,  'err_max': 5.0,  'cmap': 'RdBu_r'},
    'v10':    {'label': 'Wind V10 (m/s)',    'vmin': -20,  'vmax': 20,  'err_min': -5.0,  'err_max': 5.0,  'cmap': 'RdBu_r'},
    'swh':    {'label': 'Wave Height (m)',   'vmin':  0,   'vmax': 10,  'err_min': -2.0,  'err_max': 2.0,  'cmap': 'YlOrRd'},
    'mwd_u':  {'label': 'Wave Dir U',        'vmin': -1,   'vmax':  1,  'err_min': -0.5,  'err_max': 0.5,  'cmap': 'RdBu_r'},
    'mwd_v':  {'label': 'Wave Dir V',        'vmin': -1,   'vmax':  1,  'err_min': -0.5,  'err_max': 0.5,  'cmap': 'RdBu_r'},
}
# 未在 VAR_VIZ_CONFIG 中定义的变量使用此 fallback
VAR_VIZ_DEFAULT = {'label': 'Value', 'vmin': -1, 'vmax': 1, 'err_min': -0.5, 'err_max': 0.5, 'cmap': 'RdYlBu_r'}

# ── 模型配置 ──────────────────────────────────────────────────────────────────
# from models.simple_convnext_net_v5 import ConvNeXtUNet as MODEL_CLASS
from models.simple_swin_unet_v1 import SwinUNet as MODEL_CLASS

IN_CHANNELS  = sum(N_DEPTH if e[3].get('select_depth') else 1 for e in INPUT_VARS)
OUT_CHANNELS = sum(N_DEPTH if e[3].get('select_depth') else 1 for e in OUTPUT_VARS)

MODEL_KWARGS = dict(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)


# ------------------------------------------------------------------------------------
# ── 自回归训练配置（参考 README_ocean_7day_autoregressive.md）───────────────────
AR_STEPS         = 5
AR_STEP_WEIGHTS  = [1] * AR_STEPS
AR_BLOCK_DAYS    = 15  # 每个 CPU 缓存块的连续起始日期数量，另补 AR_STEPS 天标签
