# Pisces-Ocean

English | [简体中文](README_zh.md)

**v1.4** — A deep learning framework for predicting three-dimensional ocean temperature, salinity and currents from a background field. The current best model, as identified by the project owner, is **ConvNeXt UNet v6**. This guide uses it throughout; FNO and Swin models are alternatives.

## Task and data

The default task uses the background on day T to predict day T+1. The network takes normalized background channels and outputs a residual in physical units:

```text
input = normalize(background(T))
residual = model(input)
prediction(T+1) = background(T) + residual
```

Depth and physical variables are stacked as channels in a 2D network. “3D” describes the ocean fields, not a 3D convolutional architecture.

| Channel order | NetCDF variable | Input key | Target key | Channels | Loss weight |
|---|---|---|---|---:|---:|
| 1 | `thetao` (temperature) | `bg_t_3d` | `label_t_3d` | 20 | 1.0 |
| 2 | `so` (salinity) | `bg_s_3d` | `label_s_3d` | 20 | 0.1 |
| 3 | `uo` (eastward velocity) | `bg_uo_3d` | `label_uo_3d` | 20 | 0.5 |
| 4 | `vo` (northward velocity) | `bg_vo_3d` | `label_vo_3d` | 20 | 0.5 |

The default input and output each have **80 channels**, ordered by variable, then selected depth. The configured domain is approximately 100–160°E, 0–50°N, at 1/12° resolution, with a **600 × 720** crop and daily sampling.

The source depth coordinate has 50 levels. The current 20 selected indices cover approximately **0.49–644 m**:

```python
DEPTH_INDICES = [0, 7, 11, 14, 16, 18, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
```

See [Data_Config.py](Data_Config.py) for `DEPTH_VALUES`, variable definitions and crop settings. Wind and wave inputs are present as commented examples; they are not enabled in the default task. The AR pipeline requires matching input/output variables and order.

## Environment

Install a PyTorch build appropriate for the machine and its CUDA environment. The training scripts use `from torch.amp import GradScaler, autocast` and `GradScaler('cuda', ...)`; **PyTorch 2.1 cannot run these entrypoints unchanged**. The unified scaler is available in the [PyTorch 2.3 source](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/amp/__init__.py), but not in the [2.1 source](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/amp/__init__.py). Check the actual environment rather than its directory name:

```bash
python -c "import torch; from torch.amp import GradScaler, autocast; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

After installing PyTorch, install the other core dependencies:

```bash
pip install numpy xarray netCDF4 matplotlib timm tqdm
```

Optional data-download dependencies and test runner:

```bash
pip install copernicusmarine pyyaml cdsapi
pip install pytest
```

The documentation checks use Python 3.13.5 and PyTorch 2.12.1 on CPU. This does not establish GPU memory requirements, CUDA compatibility or end-to-end training performance. ConvNeXt v6 imports `timm`; the standalone Swin UNet and FNO model files only import PyTorch.

## Quick start

Run training, preprocessing and inference commands from the repository root.

1. In [Data_Config.py](Data_Config.py), replace the active `MODEL_CLASS` import with the primary model:

   ```python
   from models.simple_convnext_net_v6 import ConvNeXtUNet as MODEL_CLASS
   ```

   The current working configuration selects `simple_convnext_net_v6_drop01`, a variant whose blocks use DropPath 0.1. It is distinct from the original v6 used in this guide. This documentation update does not change the configuration.

2. Set paths for the current machine, keeping the existing variable definitions:

   ```python
   RAW_DATASET_PATH = "/path/to/datasets"
   NPZ_DIR = "/path/to/datasets_20layer_npz"
   LOG_ROOT = "/path/to/training_logs"
   ```

3. Prepare the four daily NetCDF variables described below. Set the preprocessing output directory and date range, then run:

   ```bash
   python preprocess_to_npz.py
   ```

4. Confirm training/validation dates and start single-step training:

   ```bash
   python -u train.py
   ```

5. Use the saved best checkpoint for inference or as the starting weights for AR training. See the corresponding sections for configuration and known inference limitations.

## Data preparation

The current training configuration uses GLORYS reanalysis for both background and targets. The expected directory layout is:

```text
datasets/
├── Glorys_thetao_0.083deg_50layer/
├── Glorys_so_0.083deg_50layer/
├── Glorys_uo_0.083deg_50layer/
└── Glorys_vo_0.083deg_50layer/
    └── glorys_0.083deg_YYYYMMDD.nc
```

Each variable directory contains its daily files. The loader indexes `.nc` filenames by an eight-digit date and loads the configured NetCDF variable, so a different filename prefix is acceptable.

### Download

See [data source notes](download_source_v1.4.md) for the existing GLORYS and optional ERA5 data descriptions. Before downloading, configure your own credentials in `download_utils/config.yaml` and edit each script's dates, spatial/depth range and output directory. The four variable-specific scripts currently request depths through 5800 m. Their credential lookup is relative to the working directory, so run them from `download_utils`:

```bash
cd download_utils
python download_glorys_thetao.py
python download_glorys_so.py
python download_glorys_uo.py
python download_glorys_vo.py
cd ..
```

These commands access an external service and download data; they are not required if the files already exist. AF background experiments require changing the input sources; AF is not the current default background.

### Daily NPZ cache

In [preprocess_to_npz.py](preprocess_to_npz.py), edit `OUTPUT_DIR`, `START_DATE`, `END_DATE` and `N_WORKERS`. **`OUTPUT_DIR` is defined separately from `Data_Config.NPZ_DIR`; set them to the same directory.** The script uses processes, writes compressed float32 arrays, selects the configured depths and crop, and skips existing daily files.

```text
datasets_20layer_npz/
├── 20210108.npz
├── 20210109.npz
└── ...
```

Each default NPZ contains `bg_t_3d`, `bg_s_3d`, `bg_uo_3d` and `bg_vo_3d`, each shaped `(20, H, W)`. Labels are assembled by reading the following day's background keys; label arrays do not need to be stored separately. The training readers also accept full-depth caches when the configured indices are available.

Single-step training prefers valid NPZ pairs and falls back to NetCDF when files or required keys are absent. **AR training requires NPZ files** and filters out origins without all T through T+5 files. Include the target days beyond the last requested origin when setting preprocessing dates. Changes to sources, crop or selected depths require a matching cache; rerunning preprocessing alone skips old files rather than refreshing them.

## Configuration

Shared settings are in [Data_Config.py](Data_Config.py):

| Setting | Current value / meaning |
|---|---|
| `INPUT_VARS`, `OUTPUT_VARS` | Ordered sources, keys, depth selection and output loss weights |
| `BG_OFFSET_DAYS` | `1`: background T, target T+1; positive means a future target |
| `IN_CHANNELS`, `OUT_CHANNELS` | Computed from variable definitions; currently 80 each |
| `CROP_ROW_START:CROP_ROW_END` | `0:600` |
| `CROP_COL_START:CROP_COL_END` | `0:720` |
| `TRAIN_START_DATE`, `TRAIN_END_DATE` | `20210108`–`20241220`, inclusive |
| `VAL_START_DATE`, `VAL_END_DATE` | `20250101`–`20251220`, inclusive |
| `TRAIN_PRETRAIN_PATH`, `TRAIN_RESUME_DIR` | Single-step initial checkpoint file / resume directory |
| `AR_PRETRAIN_PATH`, `AR_RESUME_DIR` | AR initial checkpoint file / resume directory |
| `AR_STEPS`, `AR_STEP_WEIGHTS` | `5`, `[1, 1, 1, 1, 1]` |
| `AR_BLOCK_DAYS` | `15` consecutive forecast origins per CPU cache block |

`train.py` and `train_AR.py` read these date settings. `train_fine_tune.py` defines its own dates and paths inside the script.

Training entrypoints construct the model with **only** `in_channels` and `out_channels`. Although `MODEL_KWARGS` exists, these entrypoints do not unpack it. Changing FNO width or Swin window size requires changing the model defaults or the construction call; adding those values to `MODEL_KWARGS` alone has no effect.

## Primary model: ConvNeXt UNet v6

[ConvNeXt UNet v6](models/simple_convnext_net_v6.py) has **52,327,600 parameters** with 80 input/output channels. Its stage widths are 160/320/640/320/160 and block counts are 3/3/12/3/3.

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

Each ConvNeXt block uses depthwise 7×7 convolution, LayerNorm, a 4× channel MLP, GELU, Layer Scale and a residual connection. Original v6 uses DropPath 0.0. Use H and W divisible by 8 to keep skip sizes aligned; 600 × 720 satisfies this requirement. Original v6 does not automatically pad arbitrary shapes.

### Alternative models

Counts below use 80 input/output channels and each model's current default architecture. Comparable parameter counts do not imply equal accuracy, memory use or speed.

| Model | Import for `MODEL_CLASS` | Parameters | Main configuration |
|---|---|---:|---|
| **ConvNeXt v6 (primary)** | `models.simple_convnext_net_v6.ConvNeXtUNet` | **52,327,600** | widths 160/320/640/320/160 |
| FNO | `models.fno_net.FNONet` | 52,387,643 | 4 layers, `width=113`, `modes=16` |
| Swin UNet | `models.simple_swin_unet_v1.SwinUNet` | 51,905,530 | `window_size=8`, depths 3/3/12/3/3 |
| SwinIR | `models.simple_swin_ir_net_v3.SimpleSwinIR` | 4,926,336 | `embed_dim=224`, depths 2/2/2 |

For example, replace the active model import with one of:

```python
from models.fno_net import FNONet as MODEL_CLASS
# or
from models.simple_swin_unet_v1 import SwinUNet as MODEL_CLASS
# or
from models.simple_swin_ir_net_v3 import SimpleSwinIR as MODEL_CLASS
```

Keep the training architecture and channel layout when loading its weights. The Swin UNet file contains its own attention implementation and can be copied elsewhere with PyTorch as its only dependency.

## Training workflows

### Single-step training

Run `python -u train.py`. It learns the physical residual `target - background` using normalized inputs. Background normalization uses cached training statistics; the loss divides residual error by the per-depth residual standard deviation. Only locations where both background and target are finite contribute to the loss. Input NaNs are filled with zero after normalization.

The weighted total is `1.0 × temperature + 0.1 × salinity + 0.5 × uo + 0.5 × vo`, where each component is a masked, variance-normalized MSE. This scales depth errors; it does not guarantee equal contributions from every depth.

Defaults inside `train.py` are batch size 1, 200 epochs, learning rate `1e-4`, Adam weight decay `1e-5`, 4 DataLoader workers and accumulation 1. ReduceLROnPlateau uses factor 0.5 and patience 2. CUDA autocast and gradient scaling are used by the training loop.

With no starting checkpoint, compatible statistics are loaded from `normalization_stats/normalization_stats.json`, or computed from training dates when absent/incompatible. A copy is saved in the run directory. A checkpoint's `norm_stats` takes precedence when initializing or resuming.

To initialize from trained weights, set `TRAIN_PRETRAIN_PATH` to a checkpoint file and leave `TRAIN_RESUME_DIR = None`. This loads weights and normalization, with a fresh optimizer and epoch count. To resume, set `TRAIN_RESUME_DIR` to a directory containing `last_checkpoint.pth`. Resume takes precedence if both options are set. Verify the file exists: the single-step script otherwise starts a new training state in that directory.

### Autoregressive training

Set `AR_PRETRAIN_PATH` to the single-step best checkpoint and leave `AR_RESUME_DIR = None`, then run:

```bash
python -u train_AR.py
```

Only the initial background comes from data. Each predicted state becomes the next input, with normalization applied again; there is no teacher forcing. The default objective is the weighted **sum of five daily losses**, not their mean. Gradients are backpropagated daily and the state is detached between days, so this is not full backpropagation through the rollout.

AR requires `BG_OFFSET_DAYS=1`, matching input/output channels and background variable order. Normalization is inherited from the checkpoint, or loaded from its adjacent `normalization_stats.json`; without a checkpoint it uses the root statistics cache. AR does not compute missing normalization statistics itself.

Defaults are learning rate `5e-5`, 200 epochs, batch size 1, accumulation 4 and encoder freezing disabled. Origins are shuffled by blocks of up to 15 consecutive days. A shared CPU block cache uses 4 I/O threads by default; the DataLoader itself uses `num_workers=0`. Future targets stay on CPU until needed. Incomplete windows and training windows overlapping the validation start are excluded.

For exact continuation, set `AR_RESUME_DIR` to its run directory. Resume validates AR settings, data split, freeze setting and loading strategy, and restores optimizer/scheduler/scaler and RNG state. An incompatible checkpoint can instead be used as starting weights when its model architecture matches.

### Fine-tuning

[train_fine_tune.py](train_fine_tune.py) is a separate entrypoint with its own `pretrain_path`, `resume_dir`, date ranges and statistics-cache path. Edit these before running:

```bash
python -u train_fine_tune.py
```

Defaults are 30 epochs, learning rate `1e-4`, batch size 1, accumulation 1 and 5 DataLoader workers. `_ENCODER_MODULES` is empty, so **all parameters are trainable by default**. For ConvNeXt v6 encoder freezing, explicitly include `stem`, `stage1`, `down1`, `stage2`, `down2`, `stage3`.

Unlike the single-step initialization path, this script loads or computes its separate fine-tuning statistics cache before training. It is not automatically an AF → GLORYS experiment: sources still come from `Data_Config.py`. For such an experiment, configure the sources and provide matching caches or set the script's `npz_dir = None` to use NetCDF; a stale GLORYS NPZ cache does not become AF data when a source folder is changed.

## Checkpoints, logs and metrics

New runs are created under `LOG_ROOT`; AR and fine-tuning runs have `_AR` and `_finetune` suffixes.

| Artifact | Contents / use |
|---|---|
| `best_model.pth` from `train.py` | **Only** `model_state_dict` and `norm_stats`; inference or starting weights |
| `last_checkpoint.pth` from `train.py` | Weights, normalization, optimizer, scheduler, scaler, epoch, best loss and metric histories; resume |
| Best and last checkpoints from `train_AR.py` | Both contain full training state, AR/data/loading configuration and RNG state |
| Best checkpoint from `train_fine_tune.py` | Weights, normalization, optimizer, epoch, validation loss and variable history; not the two-field format |
| Last checkpoint from `train_fine_tune.py` | Full continuation state, including scheduler/scaler and loss histories |
| `normalization_stats.json` | Run-specific normalization statistics |
| Configuration and model `.py` files | Copies of `Data_Config.py` and the selected model's source file |
| `source_snapshots/` | Additional source snapshots on resume or when files already exist |
| `loss_curve.png`, `loss_curve.csv` | Total and per-variable train/validation losses, background baselines and ratios |

The source snapshot copies the selected model file, not its entire dependency tree. A checkpoint stores state dictionaries, not executable model definitions; keep the corresponding code, configuration and dependencies. Old checkpoints do not shrink automatically when the saving code changes.

Single-step validation compares against the unchanged input background. AR compares each lead day against persistence of the initial background. A model/background loss ratio below 1 means improvement for that metric and split; the per-variable losses are shown without variable weights, while the total is weighted.

## Inference

Select the same model and channel/depth configuration used to train the checkpoint. From the repository root:

```bash
python inference.py --date 20251220 --model_path /path/to/run/best_model.pth --save_dir ./results/inference_result
```

`--date` is the **background date T**; with the default offset, evaluation also reads the T+1 target. Normalization is loaded from the checkpoint and prediction is `background + residual`. The script loads NetCDF data, not the training NPZ cache. Other supported options are `--device` and `--profile_coords` (semicolon-separated `lat,lon` pairs).

The entrypoint computes aggregate and per-variable metrics and per-depth RMSE, writes `prediction_YYYYMMDD.nc`, `background_YYYYMMDD.nc`, `target_YYYYMMDD.nc` under a timestamped result directory, and invokes the HTML visualization helper. The filename date is the supplied background date.

**Known limitations:** the NetCDF helper returns paths as `(prediction, background, target)`, while `inference.py` unpacks them as `(prediction, target, background)`. This can swap target/background inputs to the report. Exported coordinates are also constructed for the default full domain rather than read from source files, so custom crops need a coordinate review. Treat report correctness as unverified until these code issues are addressed; this README update does not fix them.

## Project layout

```text
Pisces-Ocean/
├── Data_Config.py             # sources, channels, dates, paths, model, AR settings
├── preprocess_to_npz.py       # daily NetCDF → NPZ
├── train.py                   # single-step residual training
├── train_AR.py                # autoregressive training
├── train_fine_tune.py         # separate fine-tuning entrypoint
├── inference.py              # single-date inference/evaluation
├── models/                   # ConvNeXt v6 and alternative architectures
├── utils/                    # loading, inference, metrics, snapshots, visualization
├── download_utils/           # data acquisition/preparation scripts
├── normalization_stats/      # shared statistics cache
└── tests/                    # model and training-pipeline tests
```

## Validation

For the primary workflow and shared training utilities:

```bash
python -m pytest tests/test_training_config.py tests/test_training_metrics.py tests/test_training_snapshot.py tests/test_train_ar.py -q
```

For the alternative models:

```bash
python -m pytest tests/test_swin_unet_v1.py tests/test_fno_net.py -q -k "not configured_ocean_channels"
```

The alternative-model command excludes `test_configured_ocean_channels`, which specifically requires FNO to be selected in `Data_Config.py`; the remaining tests can run while ConvNeXt v6 stays selected. Some tests require CUDA and skip when unavailable. Unit tests and shape/parameter checks do not replace training on the actual dataset or checking the inference report limitations above.

In the documentation-check environment, `test_curves_csv_and_terminal_preserve_missing_history_and_zero_baselines` aborts inside Matplotlib, including when run alone. The training configuration, snapshot and AR suites pass separately. This documentation update does not resolve the plotting failure.

## Changes from v1.3

| Topic | v1.3 documentation | v1.4 documentation / current implementation |
|---|---|---|
| Primary model | Old `simple_convnext_net.py` | ConvNeXt UNet v6, 52,327,600 parameters |
| Task | Temperature/salinity, 40 channels | Temperature/salinity/uo/vo, 80 channels |
| Architecture | 64/128/256 channel example | 160/320/640/320/160, four output heads |
| Depth source | 20 selected from 33 | 20 selected from 50, about 0.49–644 m |
| Configuration | Old `data_index` examples and negative offsets | `INPUT_VARS`/`OUTPUT_VARS`, target offset +1 |
| Data pipeline | NetCDF-focused instructions | Daily NPZ preprocessing, single-step fallback, AR windows |
| Multi-step workflow | Old forecast scripts | Current five-step AR training and block cache |
| Fine-tuning | AF background and frozen encoder presented as defaults | Sources configured explicitly; full fine-tuning by default |
| Alternatives | Old model list | Current FNO, Swin UNet and SwinIR with parameter counts |
| Training outputs | Brief checkpoint description | Entrypoint-specific checkpoints, snapshots, variable metrics and CSV |
| Commands/layout | Includes removed files | Existing entrypoints and paths, with known limitations |

Earlier releases: **v1.3** documented independent seven-day forecasts, autoregressive inference and batch evaluation; those historical scripts are absent from the current working tree. **v1.2** introduced residual prediction and pretraining/fine-tuning. **v1.1** streamlined data loading and multi-source inputs. These historical capabilities are not current runnable entrypoints merely because they appeared in earlier documentation.

## License

See [LICENSE](LICENSE). Some bundled model implementations also carry their own license notices.
