import copernicusmarine

USERNAME = "ghuang12"  # 替换为用户名
PASSWORD = "!Hjh123456789"  # 替换为密码

# 设置数据集和变量
dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m"
variables = ["so", "thetao"] 

# 设置空间范围（北极区域）
minimum_longitude = 100
maximum_longitude = 159.875
minimum_latitude = 0  # 北极区域起始纬度
maximum_latitude = 49.875  # 北极点
minimum_depth = 0.49402499198913574  # 表层数据
maximum_depth = 651

# 设置时间范围（2025年5月）
start_datetime = "2025-05-01T00:00:00"  # 5月起始时间
end_datetime = "2025-05-10T23:59:59"   # 5月结束时间

# 设置输出文件名
output_file = "sit_2025_05.nc"  # 包含年份和月份

# 下载数据
print("正在下载2025年5月的海冰数据...")
try:
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        output_filename=output_file,
        username=USERNAME,
        password=PASSWORD
    )
    print(f"数据已成功保存到 {output_file}")
except Exception as e:
    print(f"下载过程中出错: {e}")