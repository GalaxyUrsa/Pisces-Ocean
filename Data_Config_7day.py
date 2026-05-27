SURFACE_VARS = []

_SURFACE_INDEX = {
    'sss':  ['Glorys_so_surface_0.083deg', 'so',  'sss'],
    'sst':  ['Glorys_thetao_surface_0.083deg', 'thetao',  'sst'],
    'sla':  ['SLA', 'sla',  'sla'],
    'ugos': ['SLA', 'ugos', 'ugos'],
    'vgos': ['SLA', 'vgos', 'vgos'],
}

# data_index_for_day[n] 对应第 n+1 天推理（n=0..6），bg_offset_days = -(n+1)
def make_data_index(day: int):
    """day: 1-based (1..7), bg_offset_days = -day"""
    return [
        ['Glorys_thetao_0.083deg', 'thetao', 'label_t_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['Glorys_so_0.083deg',     'so',     'label_s_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['AF_thetao_0.083deg',     'thetao', 'bg_t_3d',    {'select_depth': True, 'bg_offset_days': -day}],
        ['AF_so_0.083deg',         'so',     'bg_s_3d',    {'select_depth': True, 'bg_offset_days': -day}],
    ]

RAW_DATASET_PATH = r"D:\datasets"

NAN_FILL_VALUE = 0

CROP_ROW_START = 0
CROP_ROW_END   = 600
CROP_COL_START = 0
CROP_COL_END   = 720

# 7个权重文件路径，按天序排列
MODEL_PATHS = [
    r'./7_day/20260520_190103_finetune_1/best_model.pth',
    r'./7_day/20260525_151739_finetune_2/best_model.pth',
    r'./7_day/20260525_172424_finetune_3/best_model.pth',
    r'./7_day/20260525_203158_finetune_4/best_model.pth',
    r'./7_day/20260526_102858_finetune_5/best_model.pth',
    r'./7_day/20260526_124354_finetune_6/best_model.pth',
    r'./7_day/20260526_143943_finetune_7/best_model.pth',
]
