SURFACE_VARS = [
    # 'sss',  # Sea Surface Salinity
    # 'sst',  # Sea Surface Temperature
    # 'sla',  # Sea Level Anomaly
    # 'ugos',
    # 'vgos',
]

_SURFACE_INDEX = {
    'sss':  ['Glorys_so_surface_0.083deg', 'so',  'sss'],
    'sst':  ['Glorys_thetao_surface_0.083deg', 'thetao',  'sst'],
    'sla':  ['SLA', 'sla',  'sla'],
    'ugos': ['SLA', 'ugos', 'ugos'],
    'vgos': ['SLA', 'vgos', 'vgos'],
}

data_index = (
    [_SURFACE_INDEX[v] for v in SURFACE_VARS] +
    [
        ['Glorys_thetao_0.083deg',  'thetao', 'label_t_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['Glorys_so_0.083deg',      'so',     'label_s_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['AF_thetao_0.083deg',  'thetao', 'bg_t_3d',    {'select_depth': True}],
        ['AF_so_0.083deg',      'so',     'bg_s_3d',    {'select_depth': True}],
    ]
)

RAW_DATASET_PATH = r"D:\datasets"

NAN_FILL_VALUE = 0  # 归一化后用于替换 NaN 的填充值（陆地/缺测区域）

# # 南中国海区域裁剪配置（105°E–125°E，5°N–21.7°N）
# # 原始网格：600(lat) × 720(lon)，分辨率 0.083°，范围 0–50°N / 100–160°E
# CROP_ROW_START = 60   # 5°N
# CROP_ROW_END   = 260  # 21.7°N
# CROP_COL_START = 60   # 105°E
# CROP_COL_END   = 300  # 125°E

CROP_ROW_START = 0   # 5°N
CROP_ROW_END   = 600  # 21.7°N
CROP_COL_START = 0   # 105°E
CROP_COL_END   = 720  # 125°E