# Pisces-Ocean

**[English](README.md)** | 简体中文

基于深度学习的三维海洋次表层重建框架。给定背景预报场，模型预测下一时刻完整的三维温盐剖面（20个深度层）。

---

## 版本

v1.3 发布于2026年5月27日
- 新增7天独立预报（`inference_7day.py`）：7个专用模型，每个对应一个预报时效，分别用不同的 `bg_offset_days=-day` 训练，无误差累积。
- 新增自回归预报（`inference_autoregressive_forecast.py`）：单模型滚动推进 N 天，每步预测结果作为下一步背景场。
- 新增批量评估工具（`eval_batch.py`、`eval_batch_7day.py`）：在日期范围内批量评估，输出 CSV 指标和 HTML 报告。
- 将推理共享工具函数抽取至 `inference_utils.py`。
- 可视化脚本整理至 `viz/` 目录。

v1.2 发布于2026年5月19日
- 采用预训练 + 微调两阶段范式：预训练使用 GLORYS 再分析数据，微调使用 AF 分析预报场作为背景场。
- 损失函数改为残差预测（pred = bg + residual），按逐深度残差 std 归一化，解决深层梯度被表层淹没的问题。
- 数据加载优化：folder 索引缓存，DataLoader 增加 persistent_workers 和 prefetch_factor。

v1.1 发布于2026年5月12日
- 优化了数据加载与训练架构，使数据读取更加便捷，以支持不同输入源的训练。

---

## 项目概述

### 任务定义

给定 T 天的三维背景场，预测 T+1 天的三维海洋温盐状态。模型输出的是**残差**，最终预测 = bg(T) + residual。

### 两阶段训练范式

**阶段一：预训练（GLORYS → GLORYS）**

用 GLORYS 再分析数据自身做预训练：bg 取 T 天 GLORYS，label 取 T+1 天 GLORYS。让模型学习海洋三维结构的演化规律。

**阶段二：微调（AF → GLORYS）**

冻结编码器，用 AF 分析预报场作为背景场（bg = AF(T)），GLORYS 作为真值标签（label = GLORYS(T+1)）进行微调。让模型适应 AF→GLORYS 的系统偏差，用于实际预报场景。

### 输入输出

**输入（40 channels）**：
- 背景温度剖面 bg_t_3d（T 天）— 20 channels
- 背景盐度剖面 bg_s_3d（T 天）— 20 channels

**输出（40 channels）**：
- 残差温度剖面 — 20 depth levels
- 残差盐度剖面 — 20 depth levels

**最终预测**：pred(T+1) = bg(T) + residual

**空间范围**：100°E–160°E，0°N–50°N（0.083°分辨率，600×720网格）

**时间分辨率**：逐日

**深度层**：从33层中选取20层（0.49 m 至 ~644 m）

---

## 项目结构

```
Pisces-Ocean/
├── train.py                             # 预训练流程（GLORYS bg + GLORYS label）
├── fine_tune.py                         # 微调流程（AF bg + GLORYS label，冻结编码器）
├── inference.py                         # 单日推理与评估
├── inference_utils.py                   # 推理共享工具（load_model、prepare_input、指标计算）
├── inference_autoregressive_forecast.py # N天自回归滚动预报（单模型）
├── inference_7day.py                    # 7天独立预报（7个专用模型）
├── eval_batch.py                        # 日期范围批量评估 → CSV
├── eval_batch_7day.py                   # 7天预报批量评估 → CSV + HTML
├── load_datasets.py                     # NetCDF 数据加载
├── Data_Config.py                       # 数据源配置（预训练 / 单步微调）
├── Data_Config_7day.py                  # 7天预报数据源配置
├── models/
│   ├── simple_convnext_net.py           # ConvNeXt U-Net（主模型）
│   ├── unet.py                          # Standard U-Net
│   ├── unet3d.py                        # 3D U-Net
│   ├── HCANet.py                        # Hybrid Conv-Attention Net
│   └── mymodel.py                       # 简单基线模型
├── download_utils/                      # 数据下载脚本
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
├── viz/                                 # 可视化脚本
│   ├── visualize_depth.py
│   ├── visualize_autoregressive.py
│   └── visual_batch_eval.py
├── read_nc_file.py                      # NetCDF 查看工具
├── compare_nc.py                        # NetCDF 对比工具
└── logs/                                # 训练日志与检查点
```

---

## 模型架构

主模型（[models/simple_convnext_net.py](models/simple_convnext_net.py)）为 **ConvNeXt U-Net**：

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
            ├── head_temp → 20ch（温度残差）
            └── head_salt → 20ch（盐度残差）
                    │
                    ▼
             Output (40ch residual)
```

ConvNeXt Block 结构：Depthwise Conv 7×7 → LayerNorm → Linear → GELU → Linear + Layer Scale + DropPath

**微调时冻结的模块**：`proj_in, stage1, down1, stage2, down2, stage3`（编码器）

**微调时训练的模块**：`up1, fusion1, stage4, up2, stage5, head_temp, head_salt`（解码器+输出头）

---

## 数据源

| 变量 | 产品 | 分辨率 | 用途 |
|---|---|---|---|
| 真值温盐 thetao/so（标签）| GLORYS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | 预训练 + 微调 label |
| 背景预报温盐 thetao/so | AF `GLOBAL_ANALYSISFORECAST_PHY_001_024` | 1/12° | 微调 bg |
| 背景再分析温盐 thetao/so | GLORYS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | 预训练 bg |

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

数据目录结构：

```
datasets/
├── Glorys_thetao_0.083deg/       # 预训练 bg + label（T 天和 T+1 天）
│   └── glorys_0.083deg_YYYYMMDD.nc
├── Glorys_so_0.083deg/
│   └── glorys_0.083deg_YYYYMMDD.nc
├── AF_thetao_0.083deg/           # 微调 bg
│   └── YYYYMMDD.nc
└── AF_so_0.083deg/               # 微调 bg
    └── YYYYMMDD.nc
```

---

## 使用方法

### 1. 下载数据

```bash
# GLORYS 再分析（预训练 bg + label）
python download_utils/download_glorys.py

# AF 分析预报（微调 bg）
python download_utils/download_analysis_forecast_thetao.py
python download_utils/download_analysis_forecast_so.py
```

### 2. 预训练（GLORYS → GLORYS）

在 [Data_Config.py](Data_Config.py) 中确认 bg 使用 GLORYS：

```python
data_index = (
    [] +  # SURFACE_VARS 为空
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

预训练配置（在 [train.py](train.py) 中修改）：

| 参数 | 值 |
|---|---|
| 训练时段 | 2023-01-08 ~ 2024-12-30 |
| 验证时段 | 2025-07-01 ~ 2025-12-19 |
| 优化器 | Adam, lr=1e-4, weight_decay=1e-5 |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=3) |
| 损失函数 | 0.9 × MSE_norm(temp) + 0.1 × MSE_norm(salt)，按逐深度残差 std 归一 |
| 训练轮数 | 200 |
| Batch size | 1（梯度累积 × 4，等效 batch=4）|
| 归一化 | Z-score（统计量从训练集计算并缓存至 `normalization_stats.json`）|

### 3. 微调（AF bg → GLORYS label）

在 [Data_Config.py](Data_Config.py) 中切换 bg 为 AF：

```python
data_index = (
    [] +  # SURFACE_VARS 为空
    [
        ['Glorys_thetao_0.083deg', 'thetao', 'label_t_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['Glorys_so_0.083deg',     'so',     'label_s_3d', {'select_depth': True, 'bg_offset_days': -1}],
        ['AF_thetao_0.083deg',     'thetao', 'bg_t_3d',    {'select_depth': True}],
        ['AF_so_0.083deg',         'so',     'bg_s_3d',    {'select_depth': True}],
    ]
)
```

在 [fine_tune.py](fine_tune.py) 中设置预训练权重路径：

```python
pretrain_path = './logs/<pretrain_run_id>/best_model.pth'
```

```bash
python fine_tune.py
```

微调配置：

| 参数 | 值 |
|---|---|
| 冻结层 | 编码器（proj_in, stage1, down1, stage2, down2, stage3）|
| 训练层 | 解码器 + 输出头 |
| 优化器 | Adam, lr=1e-5, weight_decay=1e-5 |
| 训练轮数 | 50 |
| 归一化 | 重新计算 AF 数据的统计量，缓存至 `normalization_stats_af.json` |

### 4. 推理

```bash
python inference.py --date 20260202 --model_path logs/<run_id>/best_model.pth
```

推理时归一化统计量从 checkpoint 自动加载，无需手动指定。

输出内容：
- 预测结果 NetCDF 文件（`inference_glory_results/`）
- 逐深度层的 RMSE / MAE / Pearson 相关系数
- 模型预测 vs 背景场 vs 真值的对比指标
- HTML 可视化报告

### 5. 自回归预报

从起始日期滚动推进 N 天，每步预测结果作为下一步背景场，step 0 之后不再读取磁盘数据。

```bash
python inference_autoregressive_forecast.py \
    --start_date 20251220 \
    --n_days 10 \
    --model_path logs/<run_id>/best_model.pth \
    --save_dir ./autoregressive_results
```

逐步输出 NetCDF 文件，并生成包含 RMSE 趋势的 HTML 汇总报告。

### 6. 7天独立预报

使用7个独立微调的模型，每个对应一个预报时效（1-7天）。每个模型用 `bg_offset_days=-day` 训练，学习从 `day` 天前的背景场预测目标日，各步之间无误差累积。

在 [Data_Config_7day.py](Data_Config_7day.py) 中配置7个权重路径：

```python
MODEL_PATHS = [
    './7_day/<run_finetune_1>/best_model.pth',
    './7_day/<run_finetune_2>/best_model.pth',
    # ... 至第7天
]
```

```bash
python inference_7day.py --start_date 20251220 --save_dir ./results/7day_results
```

### 7. 批量评估

对单模型在日期范围内批量评估：

```bash
python eval_batch.py \
    --start 20250101 --end 20251231 \
    --model_path logs/<run_id>/best_model.pth \
    --out results.csv
```

输出每日 RMSE 的 CSV，同时包含背景场基线对比。

对7天预报系统批量评估：

```bash
python eval_batch_7day.py \
    --start 20250101 --end 20251231 \
    --step 7 \
    --save_dir ./results/batch_eval_results
```

输出逐预报时效的平均 RMSE 表、时间序列图和 HTML 报告。

---

## 训练细节

- **残差预测**：模型输出残差，pred(T+1) = bg(T) + residual，避免模型直接拟合绝对值
- **损失归一化**：按逐深度残差 std 归一化，使每个深度层在 loss 中贡献均等，防止表层主导梯度
- **NaN 处理**：陆地/缺测区域用 mask 排除，不参与损失计算；bg 和 label 同时为有效值才参与
- **损失函数**：加权 MSE，温度权重 0.9，盐度权重 0.1
- **基线对比**：验证时同步计算 bg baseline loss（residual=0），用于衡量模型相对背景场的提升幅度（val/bg ratio < 1 表示有效）
- **断点续训**：将 `resume_dir` 设置为已有的日志目录路径即可
