# Ocean Data Sources

This document describes all data sources used for ocean reconstruction model training and inference.

## Table of Contents
- [Subsurface Data](#subsurface-data)
  - [Deep Layer Background (7-day lag)](#1-deep-layer-background-7-day-lag)
  - [GLORYS Reanalysis (Current)](#2-glorys-reanalysis-current)
- [Surface Observations](#surface-observations)
  - [Sea Level Anomaly (SLA)](#3-sea-level-anomaly-sla)
  - [Surface Geostrophic Currents (Ugos/Vgos)](#4-surface-geostrophic-currents-ugosvgos)
  - [Sea Surface Salinity (SSS)](#5-sea-surface-salinity-sss)
  - [Sea Surface Temperature (SST)](#6-sea-surface-temperature-sst)

---

## Subsurface Data

### 1. Deep Layer Background (7-day lag)

**Purpose**: Background subsurface fields from 7 days prior (used as input features)

| Property | Value |
|----------|-------|
| **Full Name** | Global Ocean Physics Reanalysis |
| **Product ID** | GLOBAL_MULTIYEAR_PHY_001_030 |
| **datasets ID** | cmems_mod_glo_phy_my_0.083deg_P1D-m |
| **Source** | Numerical models (GLORYS12V1) |
| **Spatial Resolution** | 1/12° × 1/12° (0.083°) |
| **Temporal Extent** | Jan 1993 - Dec 2025 |
| **Processing Level** | L4 |
| **Vertical Layers** | 80 depth levels |
| **Selected Variables** | • Sea water potential temperature (thetao)<br>• Sea water salinity (so)<br>• Eastward sea water velocity (uo)<br>• Northward sea water velocity (vo) |
| **Temporal Offset** | T - 7 days |

**Data Access**:
```
https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
```

---

### 2. GLORYS Reanalysis (Current)

**Purpose**: Ground truth subsurface fields at current time (used as labels)

| Property | Value |
|----------|-------|
| **Full Name** | Global Ocean Physics Reanalysis |
| **Product ID** | GLOBAL_MULTIYEAR_PHY_001_030 |
| **datasets ID** | cmems_mod_glo_phy_my_0.083deg_P1D-m |
| **Source** | Numerical models (GLORYS12V1) |
| **Spatial Resolution** | 1/12° × 1/12° (0.083°) |
| **Temporal Extent** | Jan 1993 - Dec 2025 |
| **Processing Level** | L4 |
| **Vertical Layers** | 80 depth levels |
| **Selected Variables** | • Sea water potential temperature (thetao)<br>• Sea water salinity (so)<br>• Eastward sea water velocity (uo)<br>• Northward sea water velocity (vo) |
| **Temporal Offset** | T (current time) |

**Data Access**:
```
https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
```

---

## Surface Observations

### 3. Sea Level Anomaly (SLA)

**Purpose**: Surface observation input feature

| Property | Value |
|----------|-------|
| **Full Name** | Global Ocean Gridded L4 Sea Surface Heights And Derived Variables NRT |
| **Product ID** | SEALEVEL_GLO_PHY_L4_NRT_008_046 |
| **datasets ID（daily）** | cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D |
| **Source** | Satellite observations (multi-mission altimetry) |
| **Spatial Resolution** | 1/8° × 1/8° (0.125°) |
| **Temporal Extent** | Oct 2022 - Jan 2026 |
| **Processing Level** | L4 |
| **Vertical Layers** | 1 (surface only) |
| **Selected Variables** | • Sea surface height above geoid (sla) |
| **Temporal Offset** | T (current time) |

**Data Access**:
```
https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description
```

---

### 4. Surface Geostrophic Currents (Ugos/Vgos)

**Purpose**: Surface observation input feature (derived from altimetry)

| Property | Value |
|----------|-------|
| **Full Name** | Global Ocean Gridded L4 Sea Surface Heights And Derived Variables NRT |
| **Product ID** | SEALEVEL_GLO_PHY_L4_NRT_008_046 |
| **datasets ID（daily）** | cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D |
| **Source** | Satellite observations (derived from altimetry) |
| **Spatial Resolution** | 1/8° × 1/8° (0.125°) |
| **Temporal Extent** | Oct 2022 - Jan 2026 |
| **Processing Level** | L4 |
| **Vertical Layers** | 1 (surface only) |
| **Selected Variables** | • Surface geostrophic eastward velocity (ugos)<br>• Surface geostrophic northward velocity (vgos) |
| **Temporal Offset** | T (current time) |

**Data Access**:
```
https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description
```

---

### 5. Sea Surface Salinity (SSS)

**Purpose**: Surface observation input feature

| Property | Value |
|----------|-------|
| **Full Name** | Multi Observation Global Ocean Sea Surface Salinity and Sea Surface Density |
| **Product ID** | MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013 |
| **datasets ID（daily）** | cmems_obs-mob_glo_phy-sss_nrt_multi_P1D |
| **Source** | In-situ observations & Satellite observations (SMOS, SMAP) |
| **Spatial Resolution** | 1/8° × 1/8° (0.125°) |
| **Temporal Extent** | Jan 1993 - Jan 2026 |
| **Processing Level** | L4 |
| **Vertical Layers** | 1 (surface only) |
| **Selected Variables** | • Sea surface salinity (sos) |
| **Temporal Offset** | T (current time) |

**Data Access**:
```
https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description
```

---

### 6. Sea Surface Temperature (SST)

**Purpose**: Surface observation input feature

| Property | Value |
|----------|-------|
| **Full Name** | NOAA Optimum Interpolation Sea Surface Temperature (OISST) V2.1 |
| **Product ID** | sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr |
| **datasets ID（daily）** | cmems_obs-mob_glo_phy-sss_nrt_multi_P1D |
| **Source** | Satellite observations (AVHRR) |
| **Spatial Resolution** | 1/4° × 1/4° (0.25°) |
| **Temporal Extent** | Sept 1981 - Jan 2026 |
| **Processing Level** | L4 |
| **Vertical Layers** | 1 (surface only) |
| **Selected Variables** | • Sea surface temperature (sst) |
| **Temporal Offset** | T (current time) |

**Data Access**:
```
https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/
```

---

## Data Usage Summary

### Input Features (Model Inputs)
1. **Surface observations** (T = current time):
   - SST (1 channel)
   - SSS (1 channel)
   - SLA (1 channel)
   - Ugos/Vgos (2 channels) - *optional*

2. **Background subsurface** (T - 7 days):
   - Temperature profile (20 depth levels)
   - Salinity profile (20 depth levels)

**Total input channels**: 3 + 20 + 20 = **43 channels** (or 45 with currents)

### Target Labels (Model Outputs)
- **GLORYS subsurface** (T = current time):
  - Temperature profile (20 depth levels)
  - Salinity profile (20 depth levels)

**Total output channels**: 20 + 20 = **40 channels**

---

## Notes

- All datasets are regridded to a common spatial grid (100°E-160°E, 0°N-50°N)
- Depth levels are subsampled from 80 to 20 levels for computational efficiency
- L4 processing level indicates gap-filled products
- Temporal offset "T - 7 days" means the background field is from 7 days before the target date