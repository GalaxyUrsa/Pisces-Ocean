# Pisces-Ocean

**[English](README.md)** | 简体中文

基于深度学习的三维海洋次表层重建框架。给定卫星表面观测（SST、SSS）和背景预报场，模型预测当前时刻完整的三维温盐剖面（20个深度层）。

---

## 版本
v1.1 发布于2026年5月12日
- 优化了数据加载与训练架构，使数据读取更加便捷，以支持不同输入源的训练。

## 项目概述

本项目将表面卫星观测与背景预报场（T-1天）作为输入，重建当前三维海洋状态，以 GLORYS 再分析数据作为真值标签进行训练。

**输入（42 channels）**：
- 海表盐度 SSS — 1 channel（来自 AF 表面场）
- 海表温度 SST — 1 channel（来自 AF 表面场）
- 背景温度剖面 bg_t_3d（T − 1 day）— 20 channels
- 背景盐度剖面 bg_s_3d（T − 1 day）— 20 channels

**输出（40 channels）**：
- 预测温度剖面 — 20 depth levels
- 预测盐度剖面 — 20 depth levels

**空间范围**：100°E–160°E，0°N–50°N（0.083°分辨率，600×720网格）

**时间分辨率**：逐日

**深度层**：从80层中选取20层（0.49 m 至 ~644 m）

---

## 项目结构

```
Pisces-Ocean/
├── train.py                    # 训练流程
├── inference.py                # 推理与评估
├── load_datasets.py            # NetCDF 数据加载
├── Data_Config.py              # 数据源配置
├── models/
│   ├── simple_convnext_net.py  # ConvNeXt U-Net（主模型）
│   ├── unet.py                 # Standard U-Net
│   ├── unet3d.py               # 3D U-Net
│   ├── HCANet.py               # Hybrid Conv-Attention Net
│   └── mymodel.py              # 简单基线模型
├── download_utils/             # 数据下载脚本
│   ├── download_glorys.py                            # GLORYS 再分析数据
│   ├── download_analysis_forecast_thetao.py          # AF 三维温度
│   ├── download_analysis_forecast_so.py              # AF 三维盐度
│   ├── download_analysis_forecast_thetao_surface.py  # AF 表面温度
│   ├── download_analysis_forecast_so_surface.py      # AF 表面盐度
│   ├── download_OISST_SST.py                         # NOAA OISST 海表温度
│   ├── download_OSTIA_SST.py                         # OSTIA 海表温度
│   ├── download_multiobs_sss.py                      # 多源融合海表盐度
│   ├── download_glorys_sst_surface.py
│   └── download_glorys_sss_surface.py
├── visualize_results_glory.py  # 结果可视化
├── visualize_depth.py          # 深度剖面可视化
├── analyze_glorys_af_rmse.py   # RMSE 分析
├── rmse_matrix.py              # RMSE 矩阵
├── read_nc_file.py             # NetCDF 查看工具
├── compare_nc.py               # NetCDF 对比工具
└── logs/                       # 训练日志与检查点
```

---

## 模型架构

主模型（[models/simple_convnext_net.py](models/simple_convnext_net.py)）为 **ConvNeXt U-Net**：

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

ConvNeXt Block 结构：Depthwise Conv 7×7 → LayerNorm → Linear → GELU → Linear + Layer Scale + DropPath

---

## 数据源

| 变量 | 产品 | 分辨率 | 来源 |
|---|---|---|---|
| 真值温盐 thetao/so（标签）| GLORYS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | Copernicus Marine |
| 背景预报温盐 thetao/so | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024` | 1/12° | Copernicus Marine |
| 表面温度 SST | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024`（表面层）| 1/12° | Copernicus Marine |
| 表面盐度 SSS | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024`（表面层）| 1/12° | Copernicus Marine |

> 数据访问需要 [Copernicus Marine](https://marine.copernicus.eu/) 账号及 `copernicusmarine` Python 客户端。

---

## 环境配置

```bash
pip install torch numpy xarray netCDF4 copernicusmarine matplotlib timm tqdm
```

在 [Data_Config.py](Data_Config.py) 中配置数据路径：

```python
RAW_DATASET_PATH = r"D:\datasets"   # 修改为本地数据集路径
```

数据目录结构示例：

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

## 使用方法

### 1. 下载数据

```bash
python download_utils/download_glorys.py
python download_utils/download_analysis_forecast_thetao.py
python download_utils/download_analysis_forecast_so.py
python download_utils/download_analysis_forecast_thetao_surface.py
python download_utils/download_analysis_forecast_so_surface.py
```

### 2. 训练

```bash
python train.py
```

训练配置（在 [train.py](train.py) 中修改）：

| 参数 | 值 |
|---|---|
| 训练时段 | 2021-01-08 ~ 2024-12-31 |
| 验证时段 | 2025-01-01 ~ 2025-12-31 |
| 优化器 | Adam, lr=5e-5, weight_decay=1e-5 |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=5) |
| 损失函数 | 0.9 × MSE(temp) + 0.1 × MSE(salt) |
| 训练轮数 | 50 |
| Batch size | 1 |
| 归一化 | Z-score（统计量从训练集计算并缓存）|

训练日志、检查点和损失曲线保存至 `logs/<run_id>/`。支持断点续训，将 `resume_dir` 设置为已有的日志目录路径即可。

### 3. 推理

```bash
python inference.py --date 20260202 --model_path logs/<run_id>/best_model.pth
```

输出内容：
- 预测结果 NetCDF 文件（`inference_glory_results/`）
- 逐深度层的 RMSE / MAE / Pearson 相关系数
- 模型预测 vs 背景场 vs 真值的对比指标
- HTML 可视化报告

---

## 训练细节

- **归一化**：对每个变量独立做 Z-score，统计量在首次训练时计算并缓存为 `normalization_stats.json`
- **NaN 处理**：陆地/缺测区域用 mask 排除，不参与损失计算
- **损失函数**：加权 MSE，温度权重 0.9，盐度权重 0.1
- **基线对比**：验证时同步计算背景场（bg）直接作为预测的损失，用于衡量模型相对背景场的提升幅度
