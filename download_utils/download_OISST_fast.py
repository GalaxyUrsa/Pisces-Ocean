import requests
import os
from datetime import datetime, timedelta

# Base URL
BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"

# Time range
start_date = "2026-01-01"
end_date = "2026-02-28"

# Output directory
output_dir = "./downloaded_data/SST"
os.makedirs(output_dir, exist_ok=True)


def download_daily(start_date_str, end_date_str):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    current_day = start_dt
    success, failed = 0, 0

    while current_day <= end_dt:
        day_str = current_day.strftime("%Y%m%d")
        year_month = current_day.strftime("%Y%m")

        filename = f"oisst-avhrr-v02r01.{day_str}.nc"
        url = f"{BASE_URL}/{year_month}/{filename}"
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print(f"⏭️  Already exists: {filename}")
            current_day += timedelta(days=1)
            continue

        print(f"Downloading: {filename}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r  {downloaded/total*100:.1f}%", end='', flush=True)
            print(f"\r✅ {filename}")
            success += 1

        except Exception as e:
            print(f"\r❌ {filename}: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            failed += 1

        current_day += timedelta(days=1)

    return success, failed


if __name__ == "__main__":
    print("=" * 80)
    print("NOAA OISST Download Tool")
    print("=" * 80)
    print(f"Time Range: {start_date} to {end_date}")
    print(f"Output Directory: {output_dir}")
    print()

    success, failed = download_daily(start_date, end_date)

    print("\n" + "=" * 80)
    print(f"✅ Success: {success}  ❌ Failed: {failed}")
    print("=" * 80)
