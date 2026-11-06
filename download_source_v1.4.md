# Ocean Data Sources — v1.4

This document describes all data sources used for ocean reconstruction model training and inference (v1.4 architecture).

## Table of Contents
- [Data Sources](#data-sources)
  - [GLORYS12 Reanalysis](#1-glorys12-reanalysis-copernicus-marine)
  - [ERA5 Wind Fields](#2-era5-10m-wind-fields-copernicus-cds)
  - [ERA5 Wave Fields](#3-era5-wave-fields-copernicus-cds)
- [Data Usage Summary](#data-usage-summary)
- [Preprocessing Notes](#preprocessing-notes)

---

## Data Sources

### 1. GLORYS12 Reanalysis (Copernicus Marine)

**Purpose**: Subsurface background fields (model input) and ground-truth labels (model output)

| Property | Value |
|----------|-------|
| **Full Name** | Global Ocean Physics Reanalysis |
| **Product ID** | GLOBAL_MULTIYEAR_PHY_001_030 |
| **Dataset ID** | cmems_mod_glo_phy_my_0.083deg_P1D-m |
| **Source** | Numerical model (GLORYS12V1) |
| **Spatial Resolution** | 1/12° × 1/12° (0.083°) |
| **Temporal Extent** | Jan 1993 – Dec 2025 |
| **Temporal Resolution** | Daily |
| **Processing Level** | L4 |
| **Vertical Layers** | 50 levels (depth 0.49 – 5728 m), subsampled to 20 |
| **Selected Variables** | thetao · so · uo · vo |
| **Domain** | 100°E – 160°E, 0°N – 50°N |

**Variable details**:

| Variable | Long Name | Units |
|----------|-----------|-------|
| `thetao` | Sea water potential temperature | °C |
| `so` | Sea water salinity | PSU |
| `uo` | Eastward sea water velocity | m/s |
| `vo` | Northward sea water velocity | m/s |

**Usage in model**:
- **Input (background)**: GLORYS at time T → 4 variables × 20 depth levels = **80 channels**
- **Output (label)**: GLORYS at time T+1 → 4 variables × 20 depth levels = **80 channels**
- The model predicts the residual (T+1 − T); final prediction = background + residual

**Data Access**:
```
https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
```

**Download tool**: `download_utils/download_glorys.py` (uses `copernicusmarine` Python client, downloads per-day NetCDF files)

---

### 2. ERA5 10m Wind Fields (Copernicus CDS)

**Purpose**: Surface atmospheric forcing input features

| Property | Value |
|----------|-------|
| **Full Name** | ERA5 hourly data on single levels |
| **Dataset ID** | reanalysis-era5-single-levels |
| **Source** | ECMWF reanalysis (IFS model) |
| **Native Spatial Resolution** | 0.25° × 0.25° |
| **Regridded Resolution** | 0.083° × 0.083° (to match GLORYS grid) |
| **Temporal Extent** | Jan 2015 – Dec 2025 |
| **Temporal Resolution** | Daily (12:00 UTC snapshot) |
| **Processing Level** | L4 |
| **Vertical Layers** | 1 (surface only) |
| **Selected Variables** | u10 · v10 |
| **Domain** | 100°E – 160°E, 0°N – 50°N |

**Variable details**:

| Variable | ERA5 Name | Long Name | Units |
|----------|-----------|-----------|-------|
| `u10` | `10m_u_component_of_wind` | 10m eastward wind | m/s |
| `v10` | `10m_v_component_of_wind` | 10m northward wind | m/s |

---

### 3. ERA5 Wave Fields (Copernicus CDS)

**Purpose**: Surface wave state input features

| Property | Value |
|----------|-------|
| **Full Name** | ERA5 hourly data on single levels |
| **Dataset ID** | reanalysis-era5-single-levels |
| **Source** | ECMWF reanalysis (WAM wave model) |
| **Native Spatial Resolution** | 0.5° × 0.5° |
| **Regridded Resolution** | 0.083° × 0.083° (to match GLORYS grid) |
| **Temporal Extent** | Jan 2015 – Dec 2025 |
| **Temporal Resolution** | Daily (12:00 UTC snapshot) |
| **Processing Level** | L4 |
| **Vertical Layers** | 1 (surface only) |
| **Selected Variables** | swh · mwd (decomposed to mwd_u / mwd_v) |
| **Domain** | 100°E – 160°E, 0°N – 50°N |

**Variable details**:

| Variable | ERA5 Name | Long Name | Units | Notes |
|----------|-----------|-----------|-------|-------|
| `swh` | `significant_height_of_combined_wind_waves_and_swell` | Significant wave height | m | Direct use |
| `mwd_u` | derived from `mean_wave_direction` | Wave direction east component | — | `−sin(mwd_rad)`, self-decomposed |
| `mwd_v` | derived from `mean_wave_direction` | Wave direction north component | — | `−cos(mwd_rad)`, self-decomposed |

> ERA5 provides `mean_wave_direction` (mwd) as a scalar angle in degrees (meteorological convention: direction waves come **from**, clockwise from north). mwd_u and mwd_v are not ERA5 native variables — they are derived locally to avoid angular discontinuity at 0°/360°.

**Usage in model**: 3 surface channels → **3 channels**

---

## Data Usage Summary

### Input Features (Model Inputs) — T

| Source | Variable | Channels |
|--------|----------|----------|
| GLORYS (T) | thetao (20 depth levels) | 20 |
| GLORYS (T) | so (20 depth levels) | 20 |
| GLORYS (T) | uo (20 depth levels) | 20 |
| GLORYS (T) | vo (20 depth levels) | 20 |
| ERA5 | u10 | 1 |
| ERA5 | v10 | 1 |
| ERA5 | swh | 1 |
| ERA5 | mwd_u | 1 |
| ERA5 | mwd_v | 1 |
| **Total** | | **85 channels** |

### Target Labels (Model Outputs) — T+1

| Source | Variable | Channels |
|--------|----------|----------|
| GLORYS (T+1) | thetao (20 depth levels) | 20 |
| GLORYS (T+1) | so (20 depth levels) | 20 |
| GLORYS (T+1) | uo (20 depth levels) | 20 |
| GLORYS (T+1) | vo (20 depth levels) | 20 |
| **Total** | | **80 channels** |

---

## Preprocessing Notes

**Depth subsampling**: 20 levels selected from GLORYS's 50 standard levels (indices `[0, 7, 11, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]`), covering 0.49 m to 644 m with finer resolution near the surface.

| Index | Depth (m) |
|-------|-----------|
| 0 | 0.49 |
| 7 | 9.57 |
| 11 | 18.50 |
| 14 | 29.44 |
| 16 | 40.34 |
| 18 | 55.76 |
| 19 | 65.81 |
| 20 | 77.85 |
| 21 | 92.33 |
| 22 | 109.73 |
| 23 | 130.67 |
| 24 | 155.85 |
| 25 | 186.13 |
| 26 | 222.48 |
| 27 | 266.04 |
| 28 | 318.13 |
| 29 | 380.21 |
| 30 | 453.94 |
| 31 | 541.09 |
| 32 | 644.01 |

**Spatial domain**: 100°E – 160°E, 0°N – 50°N (600 × 720 grid points at 0.083°)

**ERA5 regridding**: ERA5 wind (0.25°) and wave (0.5°) fields are bilinearly interpolated to the GLORYS 0.083° grid before use.

**NaN fill**: Land/missing grid points are filled with 0 after normalization (`NAN_FILL_VALUE = 0`).

**Normalization**: Z-score per variable, statistics computed on training set (2021-01-08 to 2024-12-30).
