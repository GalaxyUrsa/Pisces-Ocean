# Pisces-Ocean

[English](README.md) | 简体中文

**v1.4** — 基于背景场预测三维海洋温度、盐度和流速的深度学习框架。项目负责人确认的当前最优模型为 **ConvNeXt UNet v6**，本指南以它为主线，FNO 和 Swin 系列作为备选模型介绍。

## 任务与数据

默认任务使用 T 天背景场预测 T+1 天状态。网络接收归一化后的背景通道，输出物理单位下的残差：

```text
input = normalize(background(T))
residual = model(input)
prediction(T+1) = background(T) + residual
```

深度和物理变量沿通道维堆叠，网络在水平二维网格上计算。“三维”指海洋场包含深度维，不表示采用三维卷积。

| 通道顺序 | NetCDF 变量 | 输入键名 | 标签键名 | 通道数 | 损失权重 |
|---|---|---|---|---:|---:|
| 1 | `thetao`（温度） | `bg_t_3d` | `label_t_3d` | 20 | 1.0 |
| 2 | `so`（盐度） | `bg_s_3d` | `label_s_3d` | 20 | 0.1 |
| 3 | `uo`（向东流速） | `bg_uo_3d` | `label_uo_3d` | 20 | 0.5 |
| 4 | `vo`（向北流速） | `bg_vo_3d` | `label_vo_3d` | 20 | 0.5 |

默认输入和输出均为 **80 通道**，按变量顺序排列，每个变量内部按所选深度排列。配置区域约为 100–160°E、0–50°N，分辨率为 1/12°，裁剪大小 **600 × 720**，时间分辨率为逐日。

原始深度坐标包含 50 层，当前选取以下 20 层，约覆盖 **0.49–644 m**：

```python
DEPTH_INDICES = [0, 7, 11, 14, 16, 18, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
```

实际深度值 `DEPTH_VALUES`、变量定义和裁剪设置见 [Data_Config.py](Data_Config.py)。风场和波浪输入目前只是注释中的配置示例，默认任务未启用。AR 流程要求输入输出变量及其顺序一致。

## 运行环境

先安装适合机器与 CUDA 环境的 PyTorch。训练脚本使用 `from torch.amp import GradScaler, autocast` 和 `GradScaler('cuda', ...)`，因此 **PyTorch 2.1 无法不经修改直接运行这些入口**。[PyTorch 2.3 源码](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/amp/__init__.py)已提供统一的 scaler，[2.1 源码](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/amp/__init__.py)未提供。请检查实际环境，不要仅依据环境目录名判断：

```bash
python -c "import torch; from torch.amp import GradScaler, autocast; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

安装 PyTorch 后，安装其他核心依赖：

```bash
pip install numpy xarray netCDF4 matplotlib timm tqdm
```

可选的数据下载依赖及测试工具：

```bash
pip install copernicusmarine pyyaml cdsapi
pip install pytest
```

文档核对环境为 Python 3.13.5、PyTorch 2.12.1、CPU；这不代表已经验证 GPU 显存需求、CUDA 兼容性或完整训练性能。ConvNeXt v6 需要 `timm`，独立的 Swin UNet 和 FNO 模型文件只导入 PyTorch。

## 快速开始

预处理、训练和推理命令均从仓库根目录运行。

1. 在 [Data_Config.py](Data_Config.py) 中，将生效的 `MODEL_CLASS` 导入替换为主模型：

   ```python
   from models.simple_convnext_net_v6 import ConvNeXtUNet as MODEL_CLASS
   ```

   当前工作区配置实际选中的是 `simple_convnext_net_v6_drop01`，其模块使用 DropPath 0.1，与本指南采用的原版 v6 不同。本次文档更新不修改实际配置。

2. 保留原有变量定义，将路径改成当前机器的目录：

   ```python
   RAW_DATASET_PATH = "/path/to/datasets"
   NPZ_DIR = "/path/to/datasets_20layer_npz"
   LOG_ROOT = "/path/to/training_logs"
   ```

3. 按下文准备四种变量的逐日 NetCDF，设置预处理输出目录和日期范围，然后运行：

   ```bash
   python preprocess_to_npz.py
   ```

4. 确认训练与验证日期，启动单步训练：

   ```bash
   python -u train.py
   ```

5. 使用保存的最佳 checkpoint 做推理，或作为 AR 训练的起始权重。相关配置及推理已知限制见后续章节。

## 数据准备

当前训练配置的背景和标签均来自 GLORYS 再分析数据。预期目录结构为：

```text
datasets/
├── Glorys_thetao_0.083deg_50layer/
├── Glorys_so_0.083deg_50layer/
├── Glorys_uo_0.083deg_50layer/
└── Glorys_vo_0.083deg_50layer/
    └── glorys_0.083deg_YYYYMMDD.nc
```

每个变量目录都需要包含对应的逐日文件。加载器根据 `.nc` 文件名中的八位日期建立索引，再读取配置的 NetCDF 变量，因此文件名前缀可以不同。

### 下载

现有 GLORYS 和可选 ERA5 数据说明见[数据源文档](download_source_v1.4.md)。下载前，在 `download_utils/config.yaml` 配置自己的账户信息，并修改各脚本的日期、空间与深度范围、输出目录。四个变量专用下载脚本当前请求深度至 5800 m。账户文件按工作目录查找，因此从 `download_utils` 内运行：

```bash
cd download_utils
python download_glorys_thetao.py
python download_glorys_so.py
python download_glorys_uo.py
python download_glorys_vo.py
cd ..
```

这些命令会访问外部服务并下载数据，已有文件时无需执行。AF 背景实验需要修改输入数据源，AF 不是当前默认背景。

### 逐日 NPZ 缓存

在 [preprocess_to_npz.py](preprocess_to_npz.py) 中设置 `OUTPUT_DIR`、`START_DATE`、`END_DATE`、`N_WORKERS`。**`OUTPUT_DIR` 独立于 `Data_Config.NPZ_DIR`，两处必须指向同一目录。** 脚本使用多进程，按配置选层和裁剪后保存压缩的 float32 数组，跳过已存在的日期文件。

```text
datasets_20layer_npz/
├── 20210108.npz
├── 20210109.npz
└── ...
```

默认每个 NPZ 包含 `bg_t_3d`、`bg_s_3d`、`bg_uo_3d`、`bg_vo_3d`，各数组形状为 `(20, H, W)`。训练通过读取下一天的背景键组装标签，不要求另存标签数组。训练读取器也支持完整深度缓存，前提是配置的深度索引可用。

单步训练优先读取有效的 NPZ 日期对；缺少文件或所需键时回退至 NetCDF。**AR 训练必须使用 NPZ**，会排除缺少 T 至 T+5 任意日期文件的起报日。预处理日期范围需要覆盖最后起报日之后的标签日。修改数据源、裁剪或选层后，需要对应的新缓存；仅重新运行预处理会跳过旧文件，不会刷新其内容。

## 配置说明

共享配置位于 [Data_Config.py](Data_Config.py)：

| 配置项 | 当前值／含义 |
|---|---|
| `INPUT_VARS`、`OUTPUT_VARS` | 有序的数据源、键名、选层设置及输出损失权重 |
| `BG_OFFSET_DAYS` | `1`：背景 T、标签 T+1；正数表示未来标签 |
| `IN_CHANNELS`、`OUT_CHANNELS` | 由变量定义计算，当前均为 80 |
| `CROP_ROW_START:CROP_ROW_END` | `0:600` |
| `CROP_COL_START:CROP_COL_END` | `0:720` |
| `TRAIN_START_DATE`、`TRAIN_END_DATE` | `20210108`–`20241220`，包含首尾日期 |
| `VAL_START_DATE`、`VAL_END_DATE` | `20250101`–`20251220`，包含首尾日期 |
| `TRAIN_PRETRAIN_PATH`、`TRAIN_RESUME_DIR` | 单步训练的起始 checkpoint 文件／续训目录 |
| `AR_PRETRAIN_PATH`、`AR_RESUME_DIR` | AR 的起始 checkpoint 文件／续训目录 |
| `AR_STEPS`、`AR_STEP_WEIGHTS` | `5`、`[1, 1, 1, 1, 1]` |
| `AR_BLOCK_DAYS` | 每个 CPU 缓存块最多包含 `15` 个连续起报日 |

`train.py` 和 `train_AR.py` 使用以上日期配置；`train_fine_tune.py` 在脚本内部单独定义日期和路径。

训练入口构造模型时**只传入** `in_channels` 和 `out_channels`。虽然配置中定义了 `MODEL_KWARGS`，但这些入口没有展开使用它。调整 FNO 宽度或 Swin 窗口需要修改模型默认值或模型构造调用，仅向 `MODEL_KWARGS` 添加参数不会生效。

## 主模型：ConvNeXt UNet v6

[ConvNeXt UNet v6](models/simple_convnext_net_v6.py) 在 80 输入／输出通道下共有 **52,327,600 个参数**。各阶段通道数为 160/320/640/320/160，模块数为 3/3/12/3/3。

```text
Input (B, 80, H, W)
  → stem: Conv3×3 + GELU + Conv2×2, stride 2
  → stage1: 160 channels, 3 blocks ───────────────────── skip s1
  → down1 + stage2: 320 channels, 3 blocks ─── skip s2
  → down2 + stage3: 640 channels, 12 blocks
  → up1: bilinear ×2 + Conv3×3
  → concatenate s2 + fusion1 + stage4: 320 channels, 3 blocks
  → up2: bilinear ×2 + Conv3×3
  → concatenate s1 + fusion2 + stage5: 160 channels, 3 blocks
  → up_final: bilinear ×2 + Conv3×3
  → four heads: Conv3×3 + GELU + Conv1×1, 20 channels each
Output (B, 80, H, W): temp / salt / uo / vo residuals
```

每个 ConvNeXt 模块包含 7×7 深度卷积、LayerNorm、4 倍通道扩展的 MLP、GELU、Layer Scale 和残差连接。原版 v6 的 DropPath 为 0.0。请使用能被 8 整除的 H、W 来保证跳跃连接尺寸对齐，600 × 720 满足此条件；原版 v6 不会自动填充任意尺寸。

### 备选模型

以下参数量统一按 80 输入／输出通道及各模型当前默认结构统计。参数量接近不代表精度、显存和速度相同。

| 模型 | 用作 `MODEL_CLASS` 的导入路径 | 参数量 | 主要配置 |
|---|---|---:|---|
| **ConvNeXt v6（主模型）** | `models.simple_convnext_net_v6.ConvNeXtUNet` | **52,327,600** | 通道 160/320/640/320/160 |
| FNO | `models.fno_net.FNONet` | 52,387,643 | 4 层，`width=113`、`modes=16` |
| Swin UNet | `models.simple_swin_unet_v1.SwinUNet` | 51,905,530 | `window_size=8`，模块数 3/3/12/3/3 |
| SwinIR | `models.simple_swin_ir_net_v3.SimpleSwinIR` | 4,926,336 | `embed_dim=224`，模块数 2/2/2 |

例如，将生效的模型导入替换为以下任意一种：

```python
from models.fno_net import FNONet as MODEL_CLASS
# or
from models.simple_swin_unet_v1 import SwinUNet as MODEL_CLASS
# or
from models.simple_swin_ir_net_v3 import SimpleSwinIR as MODEL_CLASS
```

加载权重时需要保持训练时的架构和通道布局。Swin UNet 文件已包含自身的注意力实现，可以单独复制到其他目录使用，仅依赖 PyTorch。

## 训练流程

### 单步训练

运行 `python -u train.py`。模型使用归一化输入学习物理单位残差 `target - background`。背景归一化使用训练集统计缓存，损失按逐深度残差标准差缩放。只有背景与标签均为有限值的位置参与损失，输入 NaN 在归一化后填为零。

总损失为 `1.0 × 温度 + 0.1 × 盐度 + 0.5 × uo + 0.5 × vo`，各分量为掩膜下按残差方差归一化的 MSE。该操作用于缩放不同深度的误差，不保证每层实际贡献相等。

`train.py` 内默认 batch size 为 1、训练 200 轮、学习率 `1e-4`、Adam weight decay 为 `1e-5`、DataLoader worker 数为 4、梯度累积为 1。ReduceLROnPlateau 的 factor 为 0.5、patience 为 2。训练循环使用 CUDA autocast 和梯度缩放。

没有起始 checkpoint 时，优先读取兼容的 `normalization_stats/normalization_stats.json`；缺失或不兼容则根据训练日期计算，并在本次日志目录保存副本。从 checkpoint 初始化或续训时，优先使用其中的 `norm_stats`。

加载起始权重时，将 `TRAIN_PRETRAIN_PATH` 设置为 checkpoint 文件，保持 `TRAIN_RESUME_DIR = None`。该方式继承权重和归一化统计，优化器和轮次重新开始。断点续训则将 `TRAIN_RESUME_DIR` 指向包含 `last_checkpoint.pth` 的目录，两者同时设置时优先续训。请确认断点文件存在；单步脚本在文件缺失时会在该目录重新开始训练状态。

### 自回归训练

将 `AR_PRETRAIN_PATH` 指向单步最佳 checkpoint，保持 `AR_RESUME_DIR = None`，然后运行：

```bash
python -u train_AR.py
```

只有初始背景来自真实数据，后续每一步都将预测状态重新归一化后作为输入，不使用 teacher forcing。默认优化目标为五天损失的加权**总和**，不是平均值。每天反向传播一次，并在天与天之间截断状态梯度，因此不是贯穿整段滚动序列的完整反向传播。

AR 要求 `BG_OFFSET_DAYS=1`、输入输出通道数及背景变量顺序一致。归一化优先继承 checkpoint，也可从其同目录的 `normalization_stats.json` 加载；未提供 checkpoint 时读取根目录统计缓存。AR 本身不会重新计算缺失的归一化统计。

默认学习率 `5e-5`、训练 200 轮、batch size 为 1、梯度累积为 4、不冻结编码器。最多 15 个连续起报日组成一个块，块间打乱，块内按日期训练。共享 CPU 块缓存默认使用 4 个 I/O 线程，DataLoader 自身的 `num_workers=0`；未来标签保留在 CPU，按天传输。缺失日期窗口及标签窗口跨入验证起始日的训练样本会被排除。

精确续训时设置 `AR_RESUME_DIR`。续训会核对 AR 配置、数据划分、冻结设置和加载策略，并恢复优化器、调度器、scaler 与随机数状态。配置不兼容的 checkpoint 在模型架构匹配时，可以改作起始权重使用。

### 微调

[train_fine_tune.py](train_fine_tune.py) 是独立入口，内部单独设置 `pretrain_path`、`resume_dir`、日期范围和统计缓存路径，修改后运行：

```bash
python -u train_fine_tune.py
```

默认训练 30 轮、学习率 `1e-4`、batch size 为 1、梯度累积为 1、DataLoader worker 数为 5。`_ENCODER_MODULES` 默认为空，因此默认**全量微调**。若要冻结 ConvNeXt v6 编码器，需要显式加入 `stem`、`stage1`、`down1`、`stage2`、`down2`、`stage3`。

与单步加载起始权重不同，此脚本在训练前读取或计算独立的微调统计缓存。它不会自动切换为 AF → GLORYS 实验，数据源仍取自 `Data_Config.py`。开展该实验时，应配置数据源，并准备相应缓存，或将脚本内的 `npz_dir = None` 以使用 NetCDF；仅修改源文件夹不会让旧 GLORYS NPZ 变成 AF 数据。

## Checkpoint、日志与指标

新训练保存到 `LOG_ROOT` 下，AR 和微调目录分别带 `_AR`、`_finetune` 后缀。

| 产物 | 内容／用途 |
|---|---|
| `train.py` 的 `best_model.pth` | **仅** `model_state_dict` 和 `norm_stats`，用于推理或起始权重 |
| `train.py` 的 `last_checkpoint.pth` | 权重、归一化、优化器、调度器、scaler、轮次、最优损失和指标历史，用于续训 |
| `train_AR.py` 的最佳和最后 checkpoint | 两者均包含完整训练状态、AR／数据／加载配置与随机数状态 |
| `train_fine_tune.py` 的最佳 checkpoint | 权重、归一化、优化器、轮次、验证损失和变量历史，不是两字段格式 |
| `train_fine_tune.py` 的最后 checkpoint | 完整续训状态，包含调度器、scaler 和损失历史 |
| `normalization_stats.json` | 本次训练的归一化统计 |
| 配置与模型 `.py` 文件 | `Data_Config.py` 和所选模型源文件的副本 |
| `source_snapshots/` | 续训或已有同名文件时生成的额外源码快照 |
| `loss_curve.png`、`loss_curve.csv` | 总体及逐变量训练／验证损失、背景基线和比值 |

源码快照只复制所选模型文件，不会复制其完整依赖树。Checkpoint 保存状态字典，不包含可执行的模型定义，仍需保留对应代码、配置和依赖。修改保存逻辑不会自动缩小已有 checkpoint。

单步验证以不变的输入背景作为基线；AR 各时效以初始背景保持不变作为持续性基线。模型／背景损失比小于 1，表示在该指标与数据划分上优于背景；逐变量损失未乘变量权重，总损失已加权。

## 推理

选择与训练 checkpoint 相同的模型及通道、深度配置，从仓库根目录运行：

```bash
python inference.py --date 20251220 --model_path /path/to/run/best_model.pth --save_dir ./results/inference_result
```

`--date` 表示**背景日 T**，默认时间偏移下评估还需要 T+1 标签。归一化从 checkpoint 加载，预测为 `background + residual`。推理脚本读取 NetCDF，不使用训练 NPZ 缓存。其他支持参数为 `--device`、`--profile_coords`，后者采用分号分隔的 `lat,lon` 坐标对。

入口计算总体及逐变量指标、逐深度 RMSE，在带时间戳的结果目录写入 `prediction_YYYYMMDD.nc`、`background_YYYYMMDD.nc`、`target_YYYYMMDD.nc`，并调用 HTML 可视化工具。文件名日期是传入的背景日期。

**已知限制：** NetCDF 保存工具按 `(prediction, background, target)` 返回路径，而 `inference.py` 按 `(prediction, target, background)` 解包，可能使报告中的标签和背景输入互换。此外，导出坐标按默认完整区域构造，并非读取源文件坐标；自定义裁剪需要核对坐标。在这些代码问题修复前，不应将报告正确性视为已经验证，本次 README 更新不修改这些问题。

## 项目结构

```text
Pisces-Ocean/
├── Data_Config.py             # 数据源、通道、日期、路径、模型、AR 设置
├── preprocess_to_npz.py       # 逐日 NetCDF → NPZ
├── train.py                   # 单步残差训练
├── train_AR.py                # 自回归训练
├── train_fine_tune.py         # 独立微调入口
├── inference.py              # 单日推理与评估
├── models/                   # ConvNeXt v6 及备选架构
├── utils/                    # 加载、推理、指标、快照、可视化
├── download_utils/           # 数据下载与预处理脚本
├── normalization_stats/      # 共享统计缓存
└── tests/                    # 模型和训练流程测试
```

## 验证

主流程与共享训练工具测试：

```bash
python -m pytest tests/test_training_config.py tests/test_training_metrics.py tests/test_training_snapshot.py tests/test_train_ar.py -q
```

备选模型测试：

```bash
python -m pytest tests/test_swin_unet_v1.py tests/test_fno_net.py -q -k "not configured_ocean_channels"
```

备选模型命令排除了 `test_configured_ocean_channels`，该测试要求 `Data_Config.py` 选中 FNO；其余测试可在保持 ConvNeXt v6 配置的情况下运行。部分测试需要 CUDA，无可用设备时会跳过。单元测试和尺寸、参数量检查不能替代实际数据集训练，也不能代替对上述推理报告限制的核对。

在文档核对环境中，`test_curves_csv_and_terminal_preserve_missing_history_and_zero_baselines` 在 Matplotlib 内异常退出，单独运行也能复现。训练配置、快照和 AR 测试集单独运行通过。本次文档更新不修复该绘图失败。

## 相对 v1.3 的变化

| 项目 | v1.3 文档 | v1.4 文档／当前实现 |
|---|---|---|
| 主模型 | 旧 `simple_convnext_net.py` | ConvNeXt UNet v6，52,327,600 参数 |
| 任务 | 温盐两变量、40 通道 | 温盐流四变量、80 通道 |
| 网络结构 | 64/128/256 通道示例 | 160/320/640/320/160，四输出头 |
| 深度来源 | 从 33 层选 20 层 | 从 50 层选 20 层，约 0.49–644 m |
| 配置 | 旧 `data_index` 示例与负时间偏移 | `INPUT_VARS`／`OUTPUT_VARS`，标签偏移 +1 |
| 数据流程 | 主要介绍 NetCDF | 逐日 NPZ 预处理、单步回退、AR 连续窗口 |
| 多步流程 | 旧预报脚本 | 当前五步 AR 训练与块缓存 |
| 微调 | 将 AF 背景及冻结编码器写为默认 | 显式配置数据源，默认全量微调 |
| 备选架构 | 旧模型列表 | 当前 FNO、Swin UNet、SwinIR 及参数量 |
| 训练产物 | 简略 checkpoint 说明 | 各入口保存格式、源码快照、变量指标及 CSV |
| 命令与结构 | 包含已删除文件 | 现存入口和路径，并说明已知限制 |

历史版本：**v1.3** 曾记录七天独立预报、自回归推理和批量评估，相关历史脚本在当前工作区已不存在；**v1.2** 引入残差预测及预训练／微调流程；**v1.1** 简化数据加载并支持多源输入。历史文档中出现过的能力不代表当前仍有可运行入口。

## 许可证

见 [LICENSE](LICENSE)。部分随仓库提供的模型实现另附许可证说明。
