"""
GLORYS12 Data Download Guide and Script

This script helps you download GLORYS12 reanalysis data from Copernicus Marine Service
for training the SkyOcean reconstruction and forecast models.

Data Source: GLOBAL_ANALYSISFORECAST_PHY_001_024
Resolution: 0.083° (needs resampling to 0.125°)
Variables: thetao (temperature), so (salinity), uo (u velocity), vo (v velocity)
"""

import os
from datetime import datetime, timedelta

# ============================================================================
# STEP 1: Install Copernicus Marine Toolbox
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    GLORYS12 Data Download Guide                          ║
╔══════════════════════════════════════════════════════════════════════════╗

STEP 1: Install Copernicus Marine Toolbox
------------------------------------------

1. Install the toolbox:
   pip install copernicusmarine

2. Create a Copernicus Marine account (FREE):
   https://data.marine.copernicus.eu/register

3. Note your username and password

""")

# ============================================================================
# STEP 2: Download Commands
# ============================================================================

print("""
STEP 2: Download GLORYS12 Reanalysis Data
------------------------------------------

Product Information:
- Product ID: GLOBAL_ANALYSISFORECAST_PHY_001_024
- Dataset IDs:
  * Temperature: cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m
  * Salinity: cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m
  * Currents: cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m

Region:
- Longitude: 100°E to 160°E
- Latitude: 0°N to 50°N
- Depth: 0-650m (20 selected levels)

Depth Levels (meters):
0.49, 2.65, 5.08, 7.93, 11.41, 15.81, 21.60, 29.44, 40.34, 55.76,
77.85, 92, 130.67, 155.85, 186.13, 222.48, 318.13, 453.94, 643.57

""")

# ============================================================================
# STEP 3: Example Download Commands
# ============================================================================

def generate_download_commands(start_date, end_date, output_dir):
    """Generate download commands for a date range"""

    print(f"""
STEP 3: Download Commands for {start_date} to {end_date}
{'='*70}

Replace YOUR_USERNAME and YOUR_PASSWORD with your Copernicus credentials.

""")

    # Temperature
    print("# Download Temperature (thetao)")
    print(f"""copernicusmarine subset \\
    --dataset-id cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m \\
    --variable thetao \\
    --start-datetime {start_date}T00:00:00 \\
    --end-datetime {end_date}T23:59:59 \\
    --minimum-longitude 100 \\
    --maximum-longitude 160 \\
    --minimum-latitude 0 \\
    --maximum-latitude 50 \\
    --minimum-depth 0 \\
    --maximum-depth 650 \\
    --output-directory {output_dir}/temperature \\
    --username YOUR_USERNAME \\
    --password YOUR_PASSWORD
""")

    # Salinity
    print("\n# Download Salinity (so)")
    print(f"""copernicusmarine subset \\
    --dataset-id cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m \\
    --variable so \\
    --start-datetime {start_date}T00:00:00 \\
    --end-datetime {end_date}T23:59:59 \\
    --minimum-longitude 100 \\
    --maximum-longitude 160 \\
    --minimum-latitude 0 \\
    --maximum-latitude 50 \\
    --minimum-depth 0 \\
    --maximum-depth 650 \\
    --output-directory {output_dir}/salinity \\
    --username YOUR_USERNAME \\
    --password YOUR_PASSWORD
""")

    # Currents (u and v)
    print("\n# Download Currents (uo, vo)")
    print(f"""copernicusmarine subset \\
    --dataset-id cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m \\
    --variable uo \\
    --variable vo \\
    --start-datetime {start_date}T00:00:00 \\
    --end-datetime {end_date}T23:59:59 \\
    --minimum-longitude 100 \\
    --maximum-longitude 160 \\
    --minimum-latitude 0 \\
    --maximum-latitude 50 \\
    --minimum-depth 0 \\
    --maximum-depth 650 \\
    --output-directory {output_dir}/currents \\
    --username YOUR_USERNAME \\
    --password YOUR_PASSWORD
""")

# Example: Download 1 year of data
generate_download_commands("2023-01-01", "2023-12-31", "./glorys_data")

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Alternative: Python Script                            ║
╔══════════════════════════════════════════════════════════════════════════╗

You can also use Python to download data programmatically:
""")

print("""
import copernicusmarine

# Set your credentials
username = "YOUR_USERNAME"
password = "YOUR_PASSWORD"

# Download temperature data
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    variables=["thetao"],
    start_datetime="2023-01-01T00:00:00",
    end_datetime="2023-12-31T23:59:59",
    minimum_longitude=100,
    maximum_longitude=160,
    minimum_latitude=0,
    maximum_latitude=50,
    minimum_depth=0,
    maximum_depth=650,
    output_directory="./glorys_data/temperature",
    username=username,
    password=password
)

# Repeat for salinity and currents...
""")

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Data Processing Steps                                 ║
╔══════════════════════════════════════════════════════════════════════════╗

After downloading, you need to:

1. Merge Variables
   Combine temperature, salinity, and currents into single files per day

2. Resample to 0.125°
   GLORYS is at 0.083°, needs resampling to match other data

3. Select Depth Levels
   Extract only the 20 depth levels used by SkyOcean

4. Organize by Date
   Structure: ./input_data/GLORYS_REANALYSIS/YYYY/glorys_YYYYMMDD.nc

""")

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Recommended Download Strategy                         ║
╔══════════════════════════════════════════════════════════════════════════╗

For Training (Minimum):
- Training set: 2018-2022 (5 years, ~1825 days)
- Validation set: 2023 (1 year, ~365 days)
- Total: ~2190 days

For Best Results:
- Training set: 2015-2022 (8 years)
- Validation set: 2023 (1 year)
- Test set: 2024 (1 year)

Download Tips:
1. Download by year to avoid timeouts
2. Use multiple parallel downloads for different variables
3. Expect ~500MB-1GB per day for all variables
4. Total size: ~1-2TB for 5 years of data

Storage Requirements:
- Raw GLORYS data: ~1-2TB
- Processed data (0.125°): ~500GB-1TB
- Training checkpoints: ~10-50GB

""")

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Quick Start (Test with 1 month)                       ║
╔══════════════════════════════════════════════════════════════════════════╗

To test the pipeline, download just 1 month first:

copernicusmarine subset \\
    --dataset-id cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m \\
    --variable thetao \\
    --start-datetime 2023-01-01T00:00:00 \\
    --end-datetime 2023-01-31T23:59:59 \\
    --minimum-longitude 100 --maximum-longitude 160 \\
    --minimum-latitude 0 --maximum-latitude 50 \\
    --minimum-depth 0 --maximum-depth 650 \\
    --output-directory ./test_data \\
    --username YOUR_USERNAME --password YOUR_PASSWORD

This will download ~30 days (~15-30GB) to test your setup.

""")

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Need Help?                                            ║
╔══════════════════════════════════════════════════════════════════════════╗

Documentation:
- Copernicus Marine Toolbox: https://help.marine.copernicus.eu/
- Product Info: https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024

Common Issues:
1. Authentication failed → Check username/password
2. Timeout → Download smaller date ranges
3. Disk space → Ensure sufficient storage before downloading

For SkyOcean specific questions:
- GitHub: https://github.com/skyocean-kanhai/KanHai
- README: See data sources section

""")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Guide generated! Follow the steps above to download GLORYS data.")
    print("="*70)
