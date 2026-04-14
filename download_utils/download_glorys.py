import copernicusmarine
import yaml
import os
from datetime import datetime, timedelta

# 加载账户配置
def load_credentials(yaml_path="config.yaml"):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['username'], config['password']

# 读取用户名和密码
USERNAME, PASSWORD = load_credentials()


# Cell thicknessEastward sea ice velocityEastward sea water velocityModel level number at sea floorNorthward sea ice velocityNorthward sea water velocityOcean mixed layer thickness defined by sigma thetaSea floor depth below geoidSea ice area fractionSea ice thicknessSea surface height above geoidSea
# water potential temperatureSea water potential temperature at sea floorSea water salinity

# Dataset configuration（保持原样）
dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
variables = ["so", "thetao", "uo", "vo"]

# Spatial domain（保持原样）
minimum_longitude = 100
maximum_longitude = 159.99
minimum_latitude = 0
maximum_latitude = 49.99

# Depth range（保持原样）
minimum_depth = 0.49402499198913574
maximum_depth = 651

# Time range（保持原样）
start_date = "2026-01-01"
end_date = "2026-02-28"

# Output directory（保持原样）
output_dir = "./downloaded_data/Glorys"
os.makedirs(output_dir, exist_ok=True)


def download_daily_data(start_date_str, end_date_str):
    """Download GLORYS data day by day."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    total_days = (end_dt - start_dt).days + 1
    success_count = 0
    fail_count = 0

    current_day = start_dt
    while current_day <= end_dt:
        day_str = current_day.strftime("%Y%m%d")
        day_start = f"{current_day.strftime('%Y-%m-%d')}T00:00:00"
        day_end = f"{current_day.strftime('%Y-%m-%d')}T23:59:59"
        
        output_file = os.path.join(output_dir, f"glorys_0.083deg_{day_str}.nc")

        if os.path.exists(output_file):
            print(f"[SKIP] {day_str}: File already exists")
            success_count += 1
            current_day += timedelta(days=1)
            continue

        print(f"\n[{success_count + fail_count + 1}/{total_days}] Downloading {day_str}...")

        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=minimum_longitude,
                maximum_longitude=maximum_longitude,
                minimum_latitude=minimum_latitude,
                maximum_latitude=maximum_latitude,
                start_datetime=day_start,
                end_datetime=day_end,
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
                output_filename=output_file,
                username=USERNAME,
                password=PASSWORD
                # 已移除：force_download=False（该参数已弃用）
            )
            print(f"[SUCCESS] Saved to {output_file}")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to download {day_str}: {e}")
            fail_count += 1

        current_day += timedelta(days=1)
    
    return success_count, fail_count


if __name__ == "__main__":
    print("=" * 60)
    print("GLORYS Daily Data Download Tool")
    print(f"User: {USERNAME}")
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 60)

    success, failed = download_daily_data(start_date, end_date)

    print("\n" + "=" * 60)
    print(f"Complete: {success} success, {failed} failed")
    print("=" * 60)