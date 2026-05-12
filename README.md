# Pisces-Ocean

English | **[简体中文](README_zh.md)**

A deep learning framework for 3D ocean subsurface reconstruction. Given satellite surface observations (SST, SSS) and a background forecast field, the model predicts the full 3D temperature and salinity profiles (20 depth levels) at the current time.

---

## Version
v1.1, Released on May 12th, 2026
- The data-loading and training pipeline was streamlined to facilitate data access, and the data were adapted to support multi-source input training.

---

## Overview

The model maps surface observations + a 1-day-lagged background forecast to the current 3D ocean state, trained against GLORYS reanalysis as ground truth.

**Input (42 channels)**:
- Sea Surface Salinity (SSS) — 1 channel (from AF surface field)
- Sea Surface Temperature (SST) — 1 channel (from AF surface field)
- Background temperature profile bg_t_3d (T − 1 day) — 20 channels
- Background salinity profile bg_s_3d (T − 1 day) — 20 channels

**Output (40 channels)**:
- Predicted temperature profile — 20 depth levels
- Predicted salinity profile — 20 depth levels

**Spatial domain**: 100°E–160°E, 0°N–50°N (0.083° resolution, 600×720 grid)

**Temporal resolution**: Daily

**Depth levels**: 20 selected from 80 available (0.49 m to ~644 m)

---

## Project Structure

```
Pisces-Ocean/
├── train.py                    # Training pipeline
├── inference.py                # Inference and evaluation
├── load_datasets.py            # NetCDF data loading
├── Data_Config.py              # Data source configuration
├── models/
│   ├── simple_convnext_net.py  # ConvNeXt U-Net (primary model)
│   ├── unet.py                 # Standard U-Net
│   ├── unet3d.py               # 3D U-Net
│   ├── HCANet.py               # Hybrid Conv-Attention Net
│   └── mymodel.py              # Simple baseline
├── download_utils/             # Data download scripts
│   ├── download_glorys.py
│   ├── download_analysis_forecast_thetao.py
│   ├── download_analysis_forecast_so.py
│   ├── download_analysis_forecast_thetao_surface.py
│   ├── download_analysis_forecast_so_surface.py
│   ├── download_OISST_SST.py
│   ├── download_OSTIA_SST.py
│   ├── download_multiobs_sss.py
│   ├── download_glorys_sst_surface.py
│   └── download_glorys_sss_surface.py
├── visualize_results_glory.py  # Result visualization
├── visualize_depth.py          # Depth profile visualization
├── analyze_glorys_af_rmse.py   # RMSE analysis
├── rmse_matrix.py              # RMSE matrix
├── read_nc_file.py             # NetCDF inspection utility
├── compare_nc.py               # NetCDF comparison utility
└── logs/                       # Training logs and checkpoints
```

---

## Model Architecture

The primary model ([models/simple_convnext_net.py](models/simple_convnext_net.py)) is a **ConvNeXt-based U-Net**:

```
Input (42ch)
    │
    ▼
proj_in: Conv2d → 64ch
    │
    ├── stage1 (ConvNeXt Block, 64ch) ──────────────────────────┐ skip
    │       ↓ down1 (stride=2)                                  │
    ├── stage2 (ConvNeXt Block, 128ch) ─────────────────┐ skip  │
    │       ↓ down2 (stride=2)                          │       │
    ├── stage3 (Bottleneck, 256ch × 2 blocks)           │       │
    │       ↓ up1 (bilinear ×2)                         │       │
    ├── fusion1 + stage4 (128ch) ←──────────────────────┘       │
    │       ↓ up2 (bilinear ×2)                                 │
    └── fusion2 + stage5 (64ch) ←───────────────────────────────┘
            │
            ├── head_temp → 20ch（温度）
            └── head_salt → 20ch（盐度）
                    │
                    ▼
             Output (40ch)
```

ConvNeXt Block: Depthwise Conv 7×7 → LayerNorm → Linear → GELU → Linear + Layer Scale + DropPath

---

## Data Sources

| Variable | Product | Resolution | Source |
|---|---|---|---|
| Ground truth thetao/so (labels) | GLORYS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | Copernicus Marine |
| Background forecast thetao/so | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024` | 1/12° | Copernicus Marine |
| Surface temperature SST | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024` (surface) | 1/12° | Copernicus Marine |
| Surface salinity SSS | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024` (surface) | 1/12° | Copernicus Marine |

> Data access requires a [Copernicus Marine](https://marine.copernicus.eu/) account and the `copernicusmarine` Python client.

---

## Setup

```bash
pip install torch numpy xarray netCDF4 copernicusmarine matplotlib timm tqdm
```

Configure the data path in [Data_Config.py](Data_Config.py):

```python
RAW_DATASET_PATH = r"/path/to/datasets"
```

Expected dataset directory structure:

```
datasets/
├── Glorys_thetao_0.083deg/
│   └── YYYYMMDD.nc
├── Glorys_so_0.083deg/
│   └── YYYYMMDD.nc
├── AF_thetao_0.083deg/
│   └── YYYYMMDD.nc
├── AF_so_0.083deg/
│   └── YYYYMMDD.nc
├── AF_thetao_surface_0.083deg/
│   └── YYYYMMDD.nc
└── AF_so_surface_0.083deg/
    └── YYYYMMDD.nc
```

---

## Usage

### 1. Download Data

```bash
python download_utils/download_glorys.py
python download_utils/download_analysis_forecast_thetao.py
python download_utils/download_analysis_forecast_so.py
python download_utils/download_analysis_forecast_thetao_surface.py
python download_utils/download_analysis_forecast_so_surface.py
```

### 2. Train

```bash
python train.py
```

Training configuration (edit in [train.py](train.py)):

| Parameter | Value |
|---|---|
| Training period | 2021-01-08 ~ 2024-12-31 |
| Validation period | 2025-01-01 ~ 2025-12-31 |
| Optimizer | Adam, lr=5e-5, weight_decay=1e-5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | 0.9 × MSE(temp) + 0.1 × MSE(salt) |
| Epochs | 50 |
| Batch size | 1 |
| Normalization | Z-score (stats computed from training set and cached) |

Checkpoints and loss curves are saved to `logs/<run_id>/`. Resume training by setting `resume_dir` to an existing log directory.

### 3. Inference

```bash
python inference.py --date 20260202 --model_path logs/<run_id>/best_model.pth
```

Outputs:
- Prediction NetCDF files (`inference_glory_results/`)
- RMSE / MAE / Pearson correlation per depth level
- Comparison metrics: model prediction vs background vs ground truth
- HTML visualization report

---

## Training Details

- **Normalization**: Z-score per variable; statistics are computed from the training set on first run and cached as `normalization_stats.json`
- **NaN handling**: Land and missing-data regions are excluded via mask and do not contribute to the loss
- **Loss function**: Weighted MSE — temperature weight 0.9, salinity weight 0.1
- **Baseline comparison**: Background field loss is computed alongside model loss during validation to measure improvement over the background
