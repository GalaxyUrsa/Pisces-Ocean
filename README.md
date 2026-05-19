# Pisces-Ocean

English | **[简体中文](README_zh.md)**

A deep learning framework for 3D ocean subsurface reconstruction. Given a background forecast field, the model predicts the full 3D temperature and salinity profiles (20 depth levels) at the next time step.

---

## Version

v1.2, Released on May 19th, 2026
- Adopted a pretrain + fine-tune two-stage paradigm: pretraining uses GLORYS reanalysis data; fine-tuning uses AF analysis-forecast as the background field.
- Loss function changed to residual prediction (pred = bg + residual), normalized by per-depth residual std to prevent surface layers from dominating gradients.
- Data loading optimized: folder index caching, DataLoader with persistent_workers and prefetch_factor.

v1.1, Released on May 12th, 2026
- The data-loading and training pipeline was streamlined to facilitate data access, and the data were adapted to support multi-source input training.

---

## Overview

### Task Definition

Given the 3D background field at day T, predict the 3D ocean temperature and salinity state at day T+1. The model outputs a **residual**, and the final prediction is: pred(T+1) = bg(T) + residual.

### Two-Stage Training Paradigm

**Stage 1: Pretraining (GLORYS → GLORYS)**

Pretrain using GLORYS reanalysis data: bg is GLORYS at day T, label is GLORYS at day T+1. The model learns the temporal evolution of 3D ocean structure.

**Stage 2: Fine-tuning (AF → GLORYS)**

Freeze the encoder and fine-tune using AF analysis-forecast as the background field (bg = AF(T)) against GLORYS ground truth (label = GLORYS(T+1)). This adapts the model to the AF→GLORYS systematic bias for operational forecasting.

### Input / Output

**Input (40 channels)**:
- Background temperature profile bg_t_3d (day T) — 20 channels
- Background salinity profile bg_s_3d (day T) — 20 channels

**Output (40 channels)**:
- Residual temperature profile — 20 depth levels
- Residual salinity profile — 20 depth levels

**Final prediction**: pred(T+1) = bg(T) + residual

**Spatial domain**: 100°E–160°E, 0°N–50°N (0.083° resolution, 600×720 grid)

**Temporal resolution**: Daily

**Depth levels**: 20 selected from 33 available (0.49 m to ~644 m)

---

## Project Structure

```
Pisces-Ocean/
├── train.py                    # Pretraining pipeline (GLORYS bg + GLORYS label)
├── fine_tune.py                # Fine-tuning pipeline (AF bg + GLORYS label, encoder frozen)
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
Input (40ch)
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
            ├── head_temp → 20ch (temperature residual)
            └── head_salt → 20ch (salinity residual)
                    │
                    ▼
             Output (40ch residual)
```

ConvNeXt Block: Depthwise Conv 7×7 → LayerNorm → Linear → GELU → Linear + Layer Scale + DropPath

**Frozen during fine-tuning**: `proj_in, stage1, down1, stage2, down2, stage3` (encoder)

**Trained during fine-tuning**: `up1, fusion1, stage4, up2, stage5, head_temp, head_salt` (decoder + output heads)

---

## Data Sources

| Variable | Product | Resolution | Usage |
|---|---|---|---|
| Ground truth thetao/so (labels) | GLORYS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | Pretrain + fine-tune label |
| Background forecast thetao/so | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024` | 1/12° | Fine-tune bg |
| Background reanalysis thetao/so | GLORYS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | Pretrain bg |

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
├── Glorys_thetao_0.083deg/       # Pretrain bg + label (day T and T+1)
│   └── glorys_0.083deg_YYYYMMDD.nc
├── Glorys_so_0.083deg/
│   └── glorys_0.083deg_YYYYMMDD.nc
├── AF_thetao_0.083deg/           # Fine-tune bg
│   └── YYYYMMDD.nc
└── AF_so_0.083deg/               # Fine-tune bg
    └── YYYYMMDD.nc
```

---

## Usage

### 1. Download Data

```bash
# GLORYS reanalysis (pretrain bg + label)
python download_utils/download_glorys.py

# AF analysis-forecast (fine-tune bg)
python download_utils/download_analysis_forecast_thetao.py
python download_utils/download_analysis_forecast_so.py
```

### 2. Pretrain (GLORYS → GLORYS)

Confirm bg uses GLORYS in [Data_Config.py](Data_Config.py):

```python
data_index = (
    [] +  # SURFACE_VARS is empty
    [
        ['Glorys_thetao_0.083deg', 'thetao', 'label_t_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['Glorys_so_0.083deg',     'so',     'label_s_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['Glorys_thetao_0.083deg', 'thetao', 'bg_t_3d',    {'select_depth': True}],
        ['Glorys_so_0.083deg',     'so',     'bg_s_3d',    {'select_depth': True}],
    ]
)
```

```bash
python train.py
```

Pretraining configuration (edit in [train.py](train.py)):

| Parameter | Value |
|---|---|
| Training period | 2023-01-08 ~ 2024-12-30 |
| Validation period | 2025-07-01 ~ 2025-12-19 |
| Optimizer | Adam, lr=1e-4, weight_decay=1e-5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss | 0.9 × MSE_norm(temp) + 0.1 × MSE_norm(salt), normalized by per-depth residual std |
| Epochs | 200 |
| Batch size | 1 (gradient accumulation × 4, effective batch=4) |
| Normalization | Z-score (stats computed from training set and cached as `normalization_stats.json`) |

### 3. Fine-tune (AF bg → GLORYS label)

Switch bg to AF in [Data_Config.py](Data_Config.py):

```python
data_index = (
    [] +  # SURFACE_VARS is empty
    [
        ['Glorys_thetao_0.083deg', 'thetao', 'label_t_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['Glorys_so_0.083deg',     'so',     'label_s_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['AF_thetao_0.083deg',     'thetao', 'bg_t_3d',    {'select_depth': True}],
        ['AF_so_0.083deg',         'so',     'bg_s_3d',    {'select_depth': True}],
    ]
)
```

Set the pretrained checkpoint path in [fine_tune.py](fine_tune.py):

```python
pretrain_path = './logs/<pretrain_run_id>/best_model.pth'
```

```bash
python fine_tune.py
```

Fine-tuning configuration:

| Parameter | Value |
|---|---|
| Frozen layers | Encoder (proj_in, stage1, down1, stage2, down2, stage3) |
| Trained layers | Decoder + output heads |
| Optimizer | Adam, lr=1e-5, weight_decay=1e-5 |
| Epochs | 50 |
| Normalization | AF stats recomputed and cached as `normalization_stats_af.json` |

### 4. Inference

```bash
python inference.py --date 20260202 --model_path logs/<run_id>/best_model.pth
```

Normalization statistics are loaded automatically from the checkpoint — no manual path needed.

Outputs:
- Prediction NetCDF files (`inference_glory_results/`)
- RMSE / MAE / Pearson correlation per depth level
- Comparison metrics: model prediction vs background vs ground truth
- HTML visualization report

---

## Training Details

- **Residual prediction**: The model outputs a residual; pred(T+1) = bg(T) + residual, avoiding direct absolute-value regression
- **Per-depth loss normalization**: Loss is normalized by per-depth residual std so each depth level contributes equally, preventing surface layers from dominating gradients
- **NaN handling**: Land and missing-data regions are excluded via mask; a pixel is valid only when both label and bg are non-NaN
- **Loss function**: Weighted MSE — temperature weight 0.9, salinity weight 0.1
- **Baseline comparison**: Background field loss (residual=0) is computed alongside model loss during validation; val/bg ratio < 1 indicates the model improves over the background
- **Resume training**: Set `resume_dir` to an existing log directory to continue from a checkpoint
