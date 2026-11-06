from utils.load_datasets import OceanDatasetLoader
from utils.training_snapshot import save_training_sources
from utils.training_metrics import VariableLossMeter, record_variable_metrics, plot_loss_curves, loss_ratio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from Data_Config import INPUT_VARS, OUTPUT_VARS, data_index, RAW_DATASET_PATH, NPZ_DIR, CROP_ROW_START, \
CROP_ROW_END, CROP_COL_START, CROP_COL_END, NAN_FILL_VALUE, DEPTH_INDICES, N_DEPTH, BG_OFFSET_DAYS, MODEL_CLASS as mymodel, IN_CHANNELS, OUT_CHANNELS
from Data_Config import (
    TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
    TRAIN_PRETRAIN_PATH, TRAIN_RESUME_DIR,
    LOG_ROOT,
)

def _select_npz_depths(data, entries):
    """按 Data_Config.DEPTH_INDICES 选择 NPZ 中三维变量的深度层。"""
    for entry in entries:
        if not entry[3].get('select_depth'):
            continue

        key = entry[2]
        if key not in data:
            continue

        arr = data[key]
        if arr.ndim != 3:
            raise ValueError(
                f"NPZ variable {key!r} should have shape (depth, H, W), "
                f"got {arr.shape}"
            )

        # 同时兼容完整深度缓存和已经按当前配置切过层的缓存。
        if arr.shape[0] == N_DEPTH:
            continue
        if not DEPTH_INDICES or max(DEPTH_INDICES) >= arr.shape[0]:
            raise ValueError(
                f"NPZ variable {key!r} has {arr.shape[0]} depth levels, "
                f"but DEPTH_INDICES={DEPTH_INDICES}"
            )
        data[key] = arr[DEPTH_INDICES]

    return data


def _weighted_residual_loss(residual, target_residual, masks, norm_stats, return_components=False):
    """Shared weighted loss after the target has been expressed as a residual."""
    diff = residual - target_residual

    total_loss = 0.0
    components = {}
    ch = 0
    for entry in OUTPUT_VARS:
        name = entry[2]
        opts = entry[3]
        n_ch = N_DEPTH if opts.get('select_depth') else 1
        weight = opts.get('weight', 1.0)

        residual_key = name.replace('label_', 'residual_').replace('_3d', '')
        std_val = norm_stats.get(residual_key, {}).get('std', 1.0)

        # Keep variance normalization in FP32. Some residual standard
        # deviations are very small and can overflow FP16 gradients.
        d = diff[:, ch:ch + n_ch].float()
        m = masks[:, ch:ch + n_ch]
        std = torch.tensor(
            std_val, dtype=torch.float32, device=d.device
        ).view(1, -1, 1, 1) if isinstance(std_val, list) else std_val
        var_loss = ((d / std) ** 2 * m).sum() / m.sum().clamp(min=1)
        total_loss = total_loss + weight * var_loss
        if return_components:
            components[name] = var_loss
        ch += n_ch

    return (total_loss, components) if return_components else total_loss


def Hybrid_loss(
    bg, residual, targets, masks, norm_stats, targets_are_residual=False, return_components=False
):
    """
    残差预测损失，自动适配 OUTPUT_VARS 中的任意变量组合。
    bg / residual / targets: 物理单位 (B, C, H, W)。
    targets_are_residual=True 时，targets 已经是 target - background，
    此时 bg 应为 None；供 single-step 减少数据传输使用。
    通道顺序与 OUTPUT_VARS 一致，每个 3D 变量占 N_DEPTH 通道，2D 变量占 1 通道。
    """
    if targets_are_residual:
        if bg is not None:
            raise ValueError(
                "bg must be None when targets_are_residual=True"
            )
        target_residual = targets
    else:
        if bg is None:
            raise ValueError(
                "bg is required when targets_are_residual=False"
            )
        target_residual = targets - bg

    return _weighted_residual_loss(
        residual, target_residual, masks, norm_stats, return_components=return_components
    )


def _stats_worker(args):
    """多进程 worker：计算单个日期的统计量，返回可序列化的累加结果。"""
    date_str, npz_dir_str, base_path_str = args
    from pathlib import Path
    import numpy as np
    from Data_Config import INPUT_VARS, OUTPUT_VARS, data_index, N_DEPTH, BG_OFFSET_DAYS
    from Data_Config import CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END

    var_names      = [e[2] for e in INPUT_VARS + OUTPUT_VARS]
    residual_names = [
        e[2].replace('label_', 'residual_').replace('_3d', '')
        for e in OUTPUT_VARS if e[3].get('select_depth')
    ]

    # 加载数据
    data = None
    if npz_dir_str is not None:
        from datetime import datetime, timedelta
        bg_path    = Path(npz_dir_str) / f"{date_str}.npz"
        target_str = (datetime.strptime(date_str, '%Y%m%d') + timedelta(days=BG_OFFSET_DAYS)).strftime('%Y%m%d')
        label_path = Path(npz_dir_str) / f"{target_str}.npz"
        if bg_path.exists() and label_path.exists():
            bg_npz    = dict(np.load(bg_path))
            label_npz = dict(np.load(label_path))
            input_keys = {entry[2] for entry in INPUT_VARS}
            label_keys = {entry[3]['bg_key'] for entry in OUTPUT_VARS}
            if input_keys.issubset(bg_npz) and label_keys.issubset(label_npz):
                data = dict(bg_npz)
                for entry in OUTPUT_VARS:
                    data[entry[2]] = label_npz[entry[3]['bg_key']]
                data = _select_npz_depths(data, INPUT_VARS + OUTPUT_VARS)
    if data is None:
        from utils.load_datasets import OceanDatasetLoader
        loader = OceanDatasetLoader(base_path_str)
        raw = loader.load_single_date(date_str, data_index, isLog=False)
        data = {}
        for key, arr in raw.items():
            if arr.ndim == 2:
                data[key] = arr[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
            else:
                data[key] = arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

    # 计算标量统计
    s, s2, cnt = {}, {}, {}
    for var in var_names:
        arr = data.get(var)
        if arr is None:
            s[var] = 0.0; s2[var] = 0.0; cnt[var] = 0
            continue
        m = ~np.isnan(arr)
        v = arr[m].astype(np.float64)
        s[var]   = float(v.sum())
        s2[var]  = float((v ** 2).sum())
        cnt[var] = int(m.sum())

    # 计算残差统计（per-depth）
    rs, rs2, rcnt = {}, {}, {}
    for entry in OUTPUT_VARS:
        if not entry[3].get('select_depth'):
            continue
        label_name = entry[2]
        bg_name    = entry[3]['bg_key']
        res_name   = label_name.replace('label_', 'residual_').replace('_3d', '')
        if label_name not in data or bg_name not in data:
            rs[res_name]   = np.zeros(N_DEPTH, np.float64)
            rs2[res_name]  = np.zeros(N_DEPTH, np.float64)
            rcnt[res_name] = np.zeros(N_DEPTH, np.int64)
            continue
        res_arr    = data[label_name].astype(np.float64) - data[bg_name].astype(np.float64)
        valid_mask = ~np.isnan(res_arr)
        res_safe   = np.where(valid_mask, res_arr, 0.0)
        rs[res_name]   = res_safe.sum(axis=(1, 2))
        rs2[res_name]  = (res_safe ** 2).sum(axis=(1, 2))
        rcnt[res_name] = valid_mask.sum(axis=(1, 2))

    return s, s2, cnt, rs, rs2, rcnt


def compute_normalization_stats(date_list, dataloader, save_path='normalization_stats.json',
                                npz_dir=None, n_workers=8):
    """
    计算训练集的归一化统计量（均值和标准差），多进程并行加速。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("\n" + "="*60)
    print("Computing normalization statistics...")
    print("="*60)

    var_names = [e[2] for e in INPUT_VARS + OUTPUT_VARS]
    residual_names = [
        e[2].replace('label_', 'residual_').replace('_3d', '')
        for e in OUTPUT_VARS if e[3].get('select_depth')
    ]

    sums   = {v: 0.0                          for v in var_names}
    sums_sq= {v: 0.0                          for v in var_names}
    counts = {v: 0                            for v in var_names}
    rsums  = {v: np.zeros(N_DEPTH, np.float64) for v in residual_names}
    rsums_sq={v: np.zeros(N_DEPTH, np.float64) for v in residual_names}
    rcounts= {v: np.zeros(N_DEPTH, np.int64)   for v in residual_names}

    args_list = [(d, str(npz_dir) if npz_dir else None, str(dataloader.base_path))
                 for d in date_list]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_stats_worker, a): a[0] for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            s, s2, cnt, rs, rs2, rcnt = fut.result()
            for v in var_names:
                sums[v]    += s[v]
                sums_sq[v] += s2[v]
                counts[v]  += cnt[v]
            for v in residual_names:
                rsums[v]    += rs[v]
                rsums_sq[v] += rs2[v]
                rcounts[v]  += rcnt[v]

    # 计算均值和标准差
    stats = {}
    # 标量变量
    for var in var_names:
        if counts[var] > 0:
            mean = sums[var] / counts[var]
            var_val = (sums_sq[var] / counts[var]) - (mean ** 2)
            std = np.sqrt(max(var_val, 1e-8))  # 避免除零

            stats[var] = {
                'mean': float(mean),
                'std': float(std),
                'count': int(counts[var])
            }

            print(f"{var:12s}: mean={mean:8.4f}, std={std:8.4f}, count={counts[var]:,}")
        else:
            print(f"Warning: No valid data for {var}")
            stats[var] = {'mean': 0.0, 'std': 1.0, 'count': 0}

    # 残差变量（per-depth）
    for var in residual_names:
        cnt = rcounts[var]
        cnt_safe = np.maximum(cnt, 1)
        mean = rsums[var] / cnt_safe
        var_val = rsums_sq[var] / cnt_safe - mean ** 2
        std = np.sqrt(np.maximum(var_val, 1e-8))
        std = np.where(cnt == 0, 1.0, std)
        mean = np.where(cnt == 0, 0.0, mean)

        stats[var] = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'count': cnt.tolist(),
        }
        print(f"{var:12s} per-depth std: {[f'{s:.4f}' for s in std]}")

    # 保存统计量
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nNormalization statistics saved to: {save_path}")

    return stats


def load_normalization_stats(stats_path='normalization_stats.json'):
    """
    加载归一化统计量

    Args:
        stats_path: 统计量文件路径

    Returns:
        stats: 统计量字典
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    print(f"Loaded normalization statistics from: {stats_path}")
    return stats


def normalization_stats_are_compatible(stats):
    """检查缓存统计是否与当前变量和深度配置一致。"""
    for entry in INPUT_VARS + OUTPUT_VARS:
        if entry[2] not in stats:
            return False

    for entry in OUTPUT_VARS:
        if not entry[3].get('select_depth'):
            continue
        residual_key = entry[2].replace(
            'label_', 'residual_'
        ).replace('_3d', '')
        residual_stats = stats.get(residual_key)
        if residual_stats is None:
            return False
        if len(residual_stats.get('std', [])) != N_DEPTH:
            return False

    return True


def denormalize(data, var_name, stats):
    """
    反归一化函数

    Args:
        data: 归一化后的数据 (numpy array or torch tensor)
        var_name: 变量名
        stats: 统计量字典

    Returns:
        denormalized_data: 反归一化后的数据
    """
    mean = stats[var_name]['mean']
    std = stats[var_name]['std']

    if isinstance(data, torch.Tensor):
        return data * std + mean
    else:
        return data * std + mean


class OceanDataset(Dataset):
    """自定义数据集类（支持Z-score归一化，优先从 .npz 缓存加载）"""
    def __init__(self, date_list, dataloader, norm_stats=None, npz_dir=None):
        """
        Args:
            date_list: 日期列表，格式为 ['20250501', '20250502', ...]
            dataloader: OceanDatasetLoader 实例
            norm_stats: 归一化统计量字典，如果为None则不进行归一化
            npz_dir:    预处理 .npz 文件目录，None 表示只走 NetCDF 路径
        """
        self.date_list = date_list
        self.dataloader = dataloader
        self.norm_stats = norm_stats
        self.npz_dir = Path(npz_dir) if npz_dir is not None else None

    def __len__(self):
        return len(self.date_list)

    def _normalize(self, data, var_name):
        if self.norm_stats is None:
            return data
        mean = self.norm_stats[var_name]['mean']
        std = self.norm_stats[var_name]['std']
        return (data - mean) / std

    def _load_data(self, date_str):
        """优先读 .npz（bg + target date），没有则回退到 NetCDF。"""
        if self.npz_dir is not None:
            bg_path    = self.npz_dir / f"{date_str}.npz"
            target_str = (datetime.strptime(date_str, '%Y%m%d') + timedelta(days=BG_OFFSET_DAYS)).strftime('%Y%m%d')
            label_path = self.npz_dir / f"{target_str}.npz"
            if bg_path.exists() and label_path.exists():
                with np.load(bg_path) as bg_npz, np.load(label_path) as label_npz:
                    input_keys = {entry[2] for entry in INPUT_VARS}
                    label_keys = {entry[3]['bg_key'] for entry in OUTPUT_VARS}
                    if (input_keys.issubset(bg_npz.files)
                            and label_keys.issubset(label_npz.files)):
                        data = {key: bg_npz[key] for key in bg_npz.files}
                        for entry in OUTPUT_VARS:
                            data[entry[2]] = label_npz[entry[3]['bg_key']]
                        return _select_npz_depths(
                            data, INPUT_VARS + OUTPUT_VARS
                        )

        raw_data = self.dataloader.load_single_date(date_str, data_index, isLog=False)
        # NetCDF 路径需要裁剪，npz 已预先裁剪好
        data = {}
        for key, arr in raw_data.items():
            if arr.ndim == 2:
                data[key] = arr[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
            else:
                data[key] = arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
        return data

    def __getitem__(self, idx):
        date_str = self.date_list[idx]
        data = self._load_data(date_str)

        # 模型输入：按 INPUT_VARS 顺序归一化并拼接
        input_arrays = []
        for entry in INPUT_VARS:
            key = entry[2]
            arr = data[key]
            arr_norm = self._normalize(arr, key) if self.norm_stats is not None else arr
            if arr_norm.ndim == 2:
                arr_norm = np.expand_dims(arr_norm, axis=0)
            input_arrays.append(arr_norm)
        inputs = np.concatenate(input_arrays, axis=0).astype(
            np.float32, copy=False
        )

        residual_arrays = []
        mask_arrays = []
        for e in OUTPUT_VARS:
            target = data[e[2]]
            bg = data[e[3]['bg_key']]
            if target.ndim == 2:
                target = np.expand_dims(target, axis=0)
                bg = np.expand_dims(bg, axis=0)
            mask_arrays.append(np.isfinite(target) & np.isfinite(bg))
            residual_arrays.append(target - bg)
        target_residual = np.concatenate(
            residual_arrays, axis=0
        ).astype(np.float32, copy=False)
        mask = np.concatenate(mask_arrays, axis=0)

        # NaN → 0
        np.nan_to_num(inputs, copy=False, nan=NAN_FILL_VALUE)
        np.nan_to_num(target_residual, copy=False, nan=NAN_FILL_VALUE)

        return (torch.from_numpy(inputs),
                torch.from_numpy(target_residual),
                torch.from_numpy(mask))


def train_epoch(model, dataloader, optimizer, scaler, device, norm_stats, accum_steps=1,
                return_metrics=False):
    """训练一个 epoch；return_metrics=True 时附带未加权的逐变量损失及 ratio。"""
    model.train()
    total_loss = 0.0
    meter = VariableLossMeter()
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc='Training')
    for step, (inputs, target_residual, masks) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        target_residual = target_residual.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 前向：模型输出物理量纲残差，直接监督 target - background
        with autocast('cuda'):
            residual = model(inputs)
            loss = Hybrid_loss(
                None,
                residual,
                target_residual,
                masks,
                norm_stats,
                targets_are_residual=True,
                return_components=return_metrics,
            )
            if return_metrics:
                loss, components = loss
                with torch.no_grad():
                    _, baseline_components = Hybrid_loss(
                        None, torch.zeros_like(residual), target_residual, masks, norm_stats,
                        targets_are_residual=True, return_components=True,
                    )
                meter.update(components, baseline_components)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        total_loss += loss.item() * accum_steps  # 还原为真实损失值记录

        # 每 accum_steps 步或最后一个 step 才更新参数
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            max_grad_norm = getattr(model, 'max_grad_norm', None)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 更新进度条
        pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.6f}"})

    result = total_loss / len(dataloader)
    return (result, meter.compute()) if return_metrics else result


def validate_epoch(model, dataloader, device, norm_stats, return_metrics=False):
    """验证一个 epoch；可附带逐变量指标，默认保持总损失和基线二元返回值。"""
    model.eval()
    total_loss = 0.0
    total_bg_loss = 0.0
    meter = VariableLossMeter()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, target_residual, masks in pbar:
            inputs = inputs.to(device, non_blocking=True)
            target_residual = target_residual.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda'):
                residual = model(inputs)
                loss = Hybrid_loss(
                    None,
                    residual,
                    target_residual,
                    masks,
                    norm_stats,
                    targets_are_residual=True,
                    return_components=return_metrics,
                )

                zero_residual = torch.zeros_like(residual)
                bg_loss = Hybrid_loss(
                    None,
                    zero_residual,
                    target_residual,
                    masks,
                    norm_stats,
                    targets_are_residual=True,
                    return_components=return_metrics,
                )
                if return_metrics:
                    loss, components = loss
                    bg_loss, baseline_components = bg_loss
                    meter.update(components, baseline_components)

            total_loss += loss.item()
            total_bg_loss += bg_loss.item()

            pbar.set_postfix({'pred': f"{loss.item():.6f}", 'bg': f"{bg_loss.item():.6f}"})

    result = total_loss / len(dataloader), total_bg_loss / len(dataloader)
    return (*result, meter.compute()) if return_metrics else result


def main(pretrain_path=TRAIN_PRETRAIN_PATH, resume_dir=TRAIN_RESUME_DIR):
    # -------------------------------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 训练参数
    batch_size = 1
    num_epochs = 200
    learning_rate = 1e-4
    num_workers = 4
    accum_steps = 1

    # 创建或恢复 log 目录
    if resume_dir is not None:
        log_dir = resume_dir
        print(f"Resuming from: {log_dir}")
    else:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(LOG_ROOT, run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")

    save_training_sources(log_dir, mymodel, resume=resume_dir is not None)
    last_ckpt_path = os.path.join(log_dir, 'last_checkpoint.pth')
    checkpoint_path = last_ckpt_path if resume_dir is not None else pretrain_path
    ckpt = None
    if checkpoint_path is not None and (resume_dir is None or os.path.exists(checkpoint_path)):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 归一化参数
    use_normalization = True  # 是否使用归一化
    norm_stats_path = os.path.join(log_dir, 'normalization_stats.json')
    root_norm_stats_path = 'normalization_stats/normalization_stats.json'  # 根目录缓存路径

    # -------------------------------------------------------------------------------------------------
    # 准备数据
    # -------------------------------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Preparing datasets...")
    print("="*60)

    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)

    start_train = datetime.strptime(TRAIN_START_DATE, '%Y%m%d').date()
    end_train = datetime.strptime(TRAIN_END_DATE, '%Y%m%d').date()
    start_val = datetime.strptime(VAL_START_DATE, '%Y%m%d').date()
    end_val = datetime.strptime(VAL_END_DATE, '%Y%m%d').date()

    def generate_date_list(start, end):
        delta = end - start
        return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]

    train_dates = generate_date_list(start_train, end_train)
    val_dates = generate_date_list(start_val, end_val)

    print(f"Training samples: {len(train_dates)}")
    print(f"Validation samples: {len(val_dates)}")

    # -------------------------------------------------------------------------------------------------
    # 计算或加载归一化统计量
    # -------------------------------------------------------------------------------------------------
    norm_stats = ckpt.get('norm_stats') if ckpt is not None else None
    if use_normalization:
        if norm_stats is not None:
            if not normalization_stats_are_compatible(norm_stats):
                raise ValueError("Checkpoint normalization stats are incompatible with current variables or depths")
            print("\nInherited normalization statistics from checkpoint.")
        elif os.path.exists(root_norm_stats_path):
            norm_stats = load_normalization_stats(root_norm_stats_path)
            if normalization_stats_are_compatible(norm_stats):
                print(
                    f"\nFound compatible normalization stats at root: "
                    f"{root_norm_stats_path}, skipping computation."
                )
            else:
                print(
                    f"\nCached normalization stats are incompatible with "
                    f"N_DEPTH={N_DEPTH}; recomputing."
                )
                norm_stats = None

        if norm_stats is None:
            norm_stats = compute_normalization_stats(train_dates, dataloader, root_norm_stats_path, npz_dir=NPZ_DIR)

        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Saved to log dir: {norm_stats_path}")
    else:
        print("\nNormalization is disabled.")

    # 残差损失 std 已包含在 norm_stats 里，直接传给 Hybrid_loss 使用

    # -------------------------------------------------------------------------------------------------
    # 创建数据集
    # -------------------------------------------------------------------------------------------------
    train_dataset = OceanDataset(train_dates, dataloader, norm_stats, npz_dir=NPZ_DIR)
    val_dataset = OceanDataset(val_dates, dataloader, norm_stats, npz_dir=NPZ_DIR)
    # train_dataset = OceanDataset(train_dates, dataloader, norm_stats)
    # val_dataset = OceanDataset(val_dates, dataloader, norm_stats)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )

    # -------------------------------------------------------------------------------------------------
    # 创建模型
    # -------------------------------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)

    # 这里需要导入您的模型
    # from models import ReconstructionModel
    # model = ReconstructionModel(
    #     in_channels=43,   # 输入通道数
    #     out_channels=40,  # 输出通道数
    #     base_channels=64
    # ).to(device)

    model = mymodel(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # -------------------------------------------------------------------------------------------------
    # 创建优化器和损失函数
    # -------------------------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler('cuda', init_scale=1.0)

    # -------------------------------------------------------------------------------------------------
    # 断点续训：加载 checkpoint
    # -------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    bg_loss_history = []
    variable_loss_history = []

    if resume_dir is not None and ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        train_loss_history = ckpt['train_loss_history']
        val_loss_history = ckpt['val_loss_history']
        bg_loss_history = ckpt.get('bg_loss_history', [])
        variable_loss_history = ckpt.get('variable_loss_history', [None] * len(train_loss_history))
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_val_loss={best_val_loss:.6f}")
    elif resume_dir is not None:
        print(f"Warning: resume_dir set but no checkpoint found at {last_ckpt_path}, starting from scratch.")
    elif ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded pretrained weights: {pretrain_path}")
    del ckpt


    # -------------------------------------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, norm_stats, accum_steps, return_metrics=True)
        print(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        # 验证
        val_loss, bg_loss, val_metrics = validate_epoch(
            model, val_loader, device, norm_stats, return_metrics=True)
        print(f"Val Loss: {val_loss:.6f}  |  BG Baseline: {bg_loss:.6f}  |  Ratio: {loss_ratio(val_loss, bg_loss):.4f}")
        val_loss_history.append(val_loss)
        bg_loss_history.append(bg_loss)
        record_variable_metrics(variable_loss_history, train_metrics, val_metrics)

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'norm_stats': norm_stats,
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f"Best model saved! Val Loss: {val_loss:.6f}")

        # 每个epoch保存断点（用于续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'bg_loss_history': bg_loss_history,
            'norm_stats': norm_stats,
            'variable_loss_history': variable_loss_history,
        }, last_ckpt_path)

        plot_loss_curves(train_loss_history, val_loss_history, os.path.join(log_dir, 'loss_curve.png'),
                         bg_loss_history, variable_loss_history)

        # 先保存本轮指标与断点，再检查 early stop。
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            print(f"Early stopping: lr={current_lr:.2e} < 1e-6")
            break

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)
    return log_dir


if __name__ == "__main__":
    main()

