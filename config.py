# =============================================================================
# 消融实验配置 — 只需修改这里
# =============================================================================

# 参与输入的表面观测变量，注释掉对应行即可去掉该变量
SURFACE_VARS = [
    'sss',   # Sea Surface Salinity
    'sst',   # Sea Surface Temperature
    'sla',   # Sea Level Anomaly
]

# =============================================================================
# 以下内容自动推导，无需手动修改
# =============================================================================

# 数据索引：[folder, nc_variable, internal_name]
_SURFACE_INDEX = {
    'sss': ['SSS', 'sos', 'sss'],
    'sst': ['SST', 'sst', 'sst'],
    'sla': ['SLA', 'sla', 'sla'],
}

data_index = (
    [_SURFACE_INDEX[v] for v in SURFACE_VARS] +
    [
        ['Glorys',     'thetao', 'label_t_3d'],
        ['Glorys',     'so',     'label_s_3d'],
        ['Background', 'thetao', 'bg_t_3d'],
        ['Background', 'so',     'bg_s_3d'],
    ]
)

# 输入通道数：表面变量数 + bg_t_3d(20) + bg_s_3d(20)
IN_CHANNELS = len(SURFACE_VARS) + 40
OUT_CHANNELS = 40
