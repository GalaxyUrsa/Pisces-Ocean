"""独立 AR 训练：各预测日等权求和，每天反传并截断状态梯度。

仅初始状态来自真实背景，后续输入全部来自模型预测，不做 teacher forcing。
时间块之间打乱、块内按日期训练，共享一份 CPU 缓存；归一化继承 checkpoint。
运行：python train_AR.py
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import perf_counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from utils.training_snapshot import save_training_sources
from utils.training_metrics import VariableLossMeter, record_variable_metrics, plot_loss_curves

from Data_Config import (
    INPUT_VARS, OUTPUT_VARS, NPZ_DIR, N_DEPTH, DEPTH_INDICES, BG_OFFSET_DAYS,
    MODEL_CLASS as mymodel, IN_CHANNELS, OUT_CHANNELS,
    AR_STEPS, AR_STEP_WEIGHTS, AR_BLOCK_DAYS,
    TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
    AR_PRETRAIN_PATH, AR_RESUME_DIR,
    LOG_ROOT,
)


def ar_config():
    """记录影响滚动预测和损失尺度的配置，防止误续训旧版断点。"""
    return {
        'steps': AR_STEPS, 'weights': list(AR_STEP_WEIGHTS),
        'truncate_every': 1, 'reduction': 'sum',
        'depth_indices': list(DEPTH_INDICES),
        'inputs': INPUT_VARS, 'outputs': OUTPUT_VARS,
    }


def validate_ar_config():
    if BG_OFFSET_DAYS != 1 or IN_CHANNELS != OUT_CHANNELS:
        raise ValueError("AR requires BG_OFFSET_DAYS=1 and matching input/output channels")
    if [e[2] for e in INPUT_VARS] != [e[3]['bg_key'] for e in OUTPUT_VARS]:
        raise ValueError("AR input and output background variables must have identical order")
    if AR_STEPS < 1 or len(AR_STEP_WEIGHTS) != AR_STEPS:
        raise ValueError("AR_STEP_WEIGHTS must contain one weight per rollout day")
    weights = np.asarray(AR_STEP_WEIGHTS, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("AR step weights must be finite and positive")


def _validate_norm_stats(stats):
    for entry in OUTPUT_VARS:
        key = entry[3]['bg_key']
        width = N_DEPTH if entry[3].get('select_depth') else 1
        for field in ('mean', 'std'):
            value = np.asarray(stats[key][field])
            if value.size not in (1, width) or not np.isfinite(value).all():
                raise ValueError(f"Incompatible normalization {key}.{field}")
        if (np.asarray(stats[key]['std']) <= 0).any():
            raise ValueError(f"Normalization std must be positive: {key}")
        residual_key = entry[2].replace('label_', 'residual_').replace('_3d', '')
        std = np.asarray(stats[residual_key]['std'])
        if std.size != width or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Incompatible per-depth residual std: {residual_key}")


def load_normalization_stats(checkpoint=None, checkpoint_path=None):
    stats = checkpoint.get('norm_stats') if checkpoint is not None else None
    if stats is None:
        path = (Path(checkpoint_path).parent / 'normalization_stats.json'
                if checkpoint_path else Path('normalization_stats/normalization_stats.json'))
        if not path.is_file():
            raise FileNotFoundError(f"Normalization statistics not found: {path}; use a checkpoint with norm_stats")
        with path.open(encoding='utf-8') as stream:
            stats = json.load(stream)
        print(f"Loaded normalization: {path}")
    else:
        print("Inherited normalization statistics from checkpoint")
    _validate_norm_stats(stats)
    return stats


def _normalize_bg(bg_phys, norm_stats, mask):
    channels = []
    start = 0
    for entry in OUTPUT_VARS:
        width = N_DEPTH if entry[3].get('select_depth') else 1
        stats = norm_stats[entry[3]['bg_key']]
        mean = torch.as_tensor(stats['mean'], dtype=torch.float32, device=bg_phys.device).view(1, -1, 1, 1)
        std = torch.as_tensor(stats['std'], dtype=torch.float32, device=bg_phys.device).view(1, -1, 1, 1)
        normalized = (bg_phys[:, start:start + width] - mean) / std
        channels.append(torch.where(mask[:, start:start + width], normalized, 0.0))
        start += width
    return torch.cat(channels, dim=1)


def Hybrid_loss(bg, residual, targets, masks, norm_stats, return_components=False):
    """物理残差误差，按变量权重和逐深度残差 std 缩放，始终使用 FP32。"""
    diff = residual.float() - (targets.float() - bg.float())
    total_loss = 0.0
    components = {}
    start = 0
    for entry in OUTPUT_VARS:
        width = N_DEPTH if entry[3].get('select_depth') else 1
        key = entry[2].replace('label_', 'residual_').replace('_3d', '')
        std = torch.as_tensor(norm_stats[key]['std'], dtype=torch.float32,
                              device=diff.device).view(1, -1, 1, 1)
        mask = masks[:, start:start + width]
        error = torch.where(mask, diff[:, start:start + width], 0.0)
        loss = ((error / std) ** 2).sum() / mask.sum().clamp(min=1)
        total_loss = total_loss + entry[3].get('weight', 1.0) * loss
        if return_components:
            components[entry[2]] = loss
        start += width
    return (total_loss, components) if return_components else total_loss


def _load_state(npz_dir, date_str):
    """缓存可为完整深度或已选择的深度；返回物理单位 (C, H, W)。"""
    channels = []
    with np.load(Path(npz_dir) / f'{date_str}.npz') as data:
        for entry in OUTPUT_VARS:
            key, opts = entry[3]['bg_key'], entry[3]
            arr = np.asarray(data[key], dtype=np.float32)
            if opts.get('select_depth'):
                if arr.ndim != 3:
                    raise ValueError(f"{date_str}: {key} must have shape (depth, H, W)")
                if arr.shape[0] != N_DEPTH:
                    if not DEPTH_INDICES or max(DEPTH_INDICES) >= arr.shape[0]:
                        raise ValueError(f"{date_str}: incompatible depth count for {key}: {arr.shape}")
                    arr = arr[DEPTH_INDICES]
            elif arr.ndim == 2:
                arr = arr[None]
            if arr.ndim != 3 or arr.shape[0] != (N_DEPTH if opts.get('select_depth') else 1):
                raise ValueError(f"{date_str}: invalid shape for {key}: {arr.shape}")
            channels.append(arr)
    return np.concatenate(channels, axis=0)


class OceanARDataset(Dataset):
    """缓存当前时间块；返回物理初始场、未来目标序列及初始有效掩膜。"""

    def __init__(self, date_list, npz_dir, block_days=None, io_workers=0):
        self.block_days = AR_BLOCK_DAYS if block_days is None else block_days
        if not isinstance(self.block_days, int) or self.block_days < 1:
            raise ValueError("AR block_days must be a positive integer")
        if not isinstance(io_workers, int) or io_workers < 0:
            raise ValueError("AR io_workers must be a non-negative integer")
        self.io_workers = io_workers
        self.npz_dir = Path(npz_dir)
        if not self.npz_dir.is_dir():
            raise FileNotFoundError(f"AR NPZ directory not found: {self.npz_dir}")
        self.date_list = []
        for date_str in sorted(date_list):
            origin = datetime.strptime(date_str, '%Y%m%d')
            if all((self.npz_dir / f"{(origin + timedelta(days=k)):%Y%m%d}.npz").is_file()
                   for k in range(AR_STEPS + 1)):
                self.date_list.append(date_str)
        if not self.date_list:
            raise ValueError("No complete AR windows: each sample needs the initial day and all target days")
        skipped = len(date_list) - len(self.date_list)
        if skipped:
            print(f"Excluded {skipped} incomplete AR windows")
        self.blocks = []
        self._block_ids = []
        previous = None
        for index, date_str in enumerate(self.date_list):
            current = datetime.strptime(date_str, '%Y%m%d')
            if (previous is None or current != previous + timedelta(days=1)
                    or index - self.blocks[-1][0] == self.block_days):
                self.blocks.append((index, index + 1))
            else:
                self.blocks[-1] = (self.blocks[-1][0], index + 1)
            self._block_ids.append(len(self.blocks) - 1)
            previous = current
        self.reset_io_stats()

    def __len__(self):
        return len(self.date_list)

    def clear_cache(self):
        self._cache = None
        self._cached_block = None

    def reset_io_stats(self):
        self.clear_cache()
        self.file_reads = 0
        self.block_loads = 0
        self.load_seconds = 0.0

    def _load_block(self, block_id):
        if self._cached_block == block_id:
            return
        self.clear_cache()
        start, stop = self.blocks[block_id]
        origin = datetime.strptime(self.date_list[start], '%Y%m%d')
        started = perf_counter()
        try:
            days = stop - start + AR_STEPS
            first = _load_state(self.npz_dir, origin.strftime('%Y%m%d'))
            self.file_reads += 1
            cache = np.empty((days, *first.shape), dtype=np.float32)
            cache[0] = first
            del first
            read_lock = Lock()

            def load_day(day):
                date_str = (origin + timedelta(days=day)).strftime('%Y%m%d')
                state = _load_state(self.npz_dir, date_str)
                with read_lock:
                    self.file_reads += 1
                if state.shape != cache.shape[1:]:
                    raise ValueError(f"{date_str}: AR grid shape changed within block: {state.shape}")
                cache[day] = state

            if self.io_workers:
                # 任务直接写各自的日期切片，只返回 None，避免 Future 保留整天数组。
                with ThreadPoolExecutor(max_workers=min(self.io_workers, days - 1)) as pool:
                    for _ in pool.map(load_day, range(1, days)):
                        pass
            else:
                for day in range(1, days):
                    load_day(day)
            self._cache = cache
            self._cached_block = block_id
            self.block_loads += 1
        finally:
            self.load_seconds += perf_counter() - started

    def __getitem__(self, idx):
        block_id = self._block_ids[idx]
        self._load_block(block_id)
        offset = idx - self.blocks[block_id][0]
        initial = torch.from_numpy(self._cache[offset])
        targets = torch.from_numpy(self._cache[offset + 1:offset + AR_STEPS + 1])
        return initial, targets, torch.isfinite(initial)


class ARBlockBatchSampler(Sampler):
    """打乱块的顺序，块内按日期组 batch，保留每块末尾不足 batch_size 的样本。"""

    def __init__(self, dataset, batch_size, shuffle=False, generator=None):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return sum((stop - start + self.batch_size - 1) // self.batch_size
                   for start, stop in self.dataset.blocks)

    def __iter__(self):
        order = (torch.randperm(len(self.dataset.blocks), generator=self.generator).tolist()
                 if self.shuffle else range(len(self.dataset.blocks)))
        for block_id in order:
            start, stop = self.dataset.blocks[block_id]
            for index in range(start, stop, self.batch_size):
                yield list(range(index, min(index + self.batch_size, stop)))


def _optimizer_step(model, optimizer, scaler, pending, accum_steps):
    # 每批按样本数加权；包括不足一个累积窗口的最后一批。
    parameters = [p for p in model.parameters() if p.grad is not None]
    for parameter in parameters:
        parameter.grad.mul_(accum_steps / pending)
    scaler.unscale_(optimizer)
    finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item()
    if not finite and not scaler.is_enabled():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite AR gradients")
    scaler.step(optimizer)  # AMP 遇到 Inf/NaN 时跳过更新并降低 scale。
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return int(finite)


def train_epoch(model, dataloader, optimizer, scaler, device, norm_stats, accum_steps=1,
                return_metrics=False):
    """训练一个 epoch：逐日反传并截断梯度，按样本数累积更新。"""
    if accum_steps < 1:
        raise ValueError("accum_steps must be positive")
    if dataloader.num_workers != 0:
        raise ValueError("AR DataLoader requires num_workers=0; use main(num_workers=N) for shared-cache I/O threads")
    nominal_batch_size = dataloader.batch_size or dataloader.batch_sampler.batch_size
    dataset = dataloader.dataset
    dataset.reset_io_stats()
    model.train()
    totals = np.zeros(AR_STEPS)
    meter = VariableLossMeter()
    samples = 0
    pending = 0
    updates = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            progress = tqdm(dataloader, desc='Training')
            for initial, targets, initial_mask in progress:
                state = initial.to(device, dtype=torch.float32, non_blocking=True)
                mask = initial_mask.to(device, non_blocking=True)
                state = torch.where(mask, state, 0.0)
                persistence = state.clone() if return_metrics else None
                batch_size = state.shape[0]
                day_losses = []
                for day in range(AR_STEPS):
                    # 未来目标留在 CPU，每次只传入当天标签。
                    target = targets[:, day].to(device, non_blocking=True)
                    valid = mask & torch.isfinite(target)
                    if not valid.any():
                        raise ValueError(f"No valid supervision for rollout day {day + 1}")
                    target = torch.where(valid, target, 0.0)
                    with autocast(device.type, enabled=device.type == 'cuda'):
                        residual = model(_normalize_bg(state, norm_stats, mask))
                    loss = Hybrid_loss(state, residual, target, valid, norm_stats,
                                       return_components=return_metrics)
                    if return_metrics:
                        loss, components = loss
                        with torch.no_grad():
                            _, baseline_components = Hybrid_loss(
                                persistence, torch.zeros_like(persistence), target, valid, norm_stats,
                                return_components=True,
                            )
                        meter.update(components, baseline_components,
                                     weight=batch_size * AR_STEP_WEIGHTS[day])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite AR loss on day {day + 1}")
                    value = loss.detach().item()
                    day_losses.append(value)
                    totals[day] += value * batch_size
                    scaler.scale(loss * AR_STEP_WEIGHTS[day] * batch_size / accum_steps).backward()
                    # 预测数值继续滚动，每天断开梯度；初始海洋掩膜保持不变。
                    state = torch.where(mask, state + residual.detach(), 0.0).detach()
                    del residual, target, valid, loss

                samples += batch_size
                pending += batch_size
                if pending >= accum_steps * nominal_batch_size:
                    updates += _optimizer_step(model, optimizer, scaler, pending, accum_steps)
                    pending = 0
                progress.set_postfix(loss=f"{np.dot(day_losses, AR_STEP_WEIGHTS):.6f}")
            if pending:
                updates += _optimizer_step(model, optimizer, scaler, pending, accum_steps)
    finally:
        dataset.clear_cache()
        tqdm.write(f"Training I/O: blocks={dataset.block_loads}, NPZ reads={dataset.file_reads}, "
                   f"block loading={dataset.load_seconds:.2f}s")
    if not samples:
        raise ValueError("No AR samples were processed")
    if not updates:
        raise FloatingPointError("No optimizer updates completed; inspect AMP overflow and gradients")
    result = float(np.dot(totals / samples, AR_STEP_WEIGHTS))
    return (result, meter.compute(scale=sum(AR_STEP_WEIGHTS))) if return_metrics else result


def validate_epoch(model, dataloader, device, norm_stats, return_metrics=False):
    """验证一个 epoch，返回逐日损失、逐日持久性基线及各自加权总损失。"""
    if dataloader.num_workers != 0:
        raise ValueError("AR DataLoader requires num_workers=0; use main(num_workers=N) for shared-cache I/O threads")
    dataset = dataloader.dataset
    dataset.reset_io_stats()
    model.eval()
    totals = np.zeros(AR_STEPS)
    baseline_totals = np.zeros(AR_STEPS)
    meter = VariableLossMeter()
    samples = 0

    try:
        with torch.no_grad():
            progress = tqdm(dataloader, desc='Validation')
            for initial, targets, initial_mask in progress:
                state = initial.to(device, dtype=torch.float32, non_blocking=True)
                mask = initial_mask.to(device, non_blocking=True)
                state = torch.where(mask, state, 0.0)
                # 基线始终使用初始场，并与预测使用相同的有效掩膜。
                persistence = state.clone()
                batch_size = state.shape[0]
                day_losses = []
                for day in range(AR_STEPS):
                    # 未来目标留在 CPU，每次只传入当天标签。
                    target = targets[:, day].to(device, non_blocking=True)
                    valid = mask & torch.isfinite(target)
                    if not valid.any():
                        raise ValueError(f"No valid supervision for rollout day {day + 1}")
                    target = torch.where(valid, target, 0.0)
                    with autocast(device.type, enabled=device.type == 'cuda'):
                        residual = model(_normalize_bg(state, norm_stats, mask))
                    loss = Hybrid_loss(state, residual, target, valid, norm_stats,
                                       return_components=return_metrics)
                    if return_metrics:
                        loss, components = loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite AR loss on day {day + 1}")
                    value = loss.detach().item()
                    day_losses.append(value)
                    totals[day] += value * batch_size
                    baseline = Hybrid_loss(persistence, torch.zeros_like(persistence), target, valid, norm_stats,
                                           return_components=return_metrics)
                    if return_metrics:
                        baseline, baseline_components = baseline
                        meter.update(components, baseline_components,
                                     weight=batch_size * AR_STEP_WEIGHTS[day])
                    baseline_totals[day] += baseline.item() * batch_size
                    del baseline
                    # 后续输入使用预测场，初始海洋掩膜保持不变。
                    state = torch.where(mask, state + residual.detach(), 0.0).detach()
                    del residual, target, valid, loss

                samples += batch_size
                progress.set_postfix(loss=f"{np.dot(day_losses, AR_STEP_WEIGHTS):.6f}")
    finally:
        dataset.clear_cache()
        tqdm.write(f"Validation I/O: blocks={dataset.block_loads}, NPZ reads={dataset.file_reads}, "
                   f"block loading={dataset.load_seconds:.2f}s")
    if not samples:
        raise ValueError("No AR samples were processed")
    day_losses = totals / samples
    day_baseline = baseline_totals / samples
    result = (day_losses, day_baseline,
              float(np.dot(day_losses, AR_STEP_WEIGHTS)),
              float(np.dot(day_baseline, AR_STEP_WEIGHTS)))
    return (*result, meter.compute(scale=sum(AR_STEP_WEIGHTS))) if return_metrics else result


# 保留旧入口名称，兼容已有调用。
train_epoch_ar = train_epoch
validate_epoch_ar = validate_epoch


def freeze_encoder(model):
    modules = {'stem', 'proj_in', 'stage1', 'down1', 'stage2', 'down2', 'stage3'}
    for name, parameter in model.named_parameters():
        if name.split('.')[0] in modules:
            parameter.requires_grad = False


def _rng_state(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'sampler': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    generator.set_state(state['sampler'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def data_loading_config(batch_size):
    return {'strategy': 'block_shuffle_v1', 'block_days': AR_BLOCK_DAYS, 'batch_size': batch_size}


def _validate_resume(checkpoint, data_split, freeze_encoder_enabled, batch_size=1):
    if checkpoint.get('ar_config') != ar_config():
        raise ValueError("AR checkpoint uses different training semantics; load it as pretrain_path instead")
    if checkpoint.get('data_split') != data_split:
        raise ValueError("AR resume dates differ from checkpoint")
    if checkpoint.get('freeze_encoder_enabled') != freeze_encoder_enabled:
        raise ValueError("AR resume freeze setting differs from checkpoint")
    if checkpoint.get('data_loading') != data_loading_config(batch_size):
        raise ValueError("AR checkpoint uses a different data loading strategy; load it as pretrain_path instead")


def main(pretrain_path=AR_PRETRAIN_PATH, resume_dir=AR_RESUME_DIR,
         num_epochs=200, num_workers=4):
    """num_workers 控制当前块的读取线程数；0 为串行，DataLoader 始终在主进程。"""
    # --------------------------------------------------------------------------------
    # 配置参数
    # --------------------------------------------------------------------------------
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("AR num_workers must be a non-negative integer")
    validate_ar_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    freeze_encoder_enabled = False
    batch_size = 1
    learning_rate = 5e-5
    accum_steps = 4
    seed = 42

    # --------------------------------------------------------------------------------
    # 准备日期与有效数据窗口
    # --------------------------------------------------------------------------------
    start_train = datetime.strptime(TRAIN_START_DATE, '%Y%m%d').date()
    end_train = datetime.strptime(TRAIN_END_DATE, '%Y%m%d').date()
    start_val = datetime.strptime(VAL_START_DATE, '%Y%m%d').date()
    end_val = datetime.strptime(VAL_END_DATE, '%Y%m%d').date()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    def dates_between(start, end):
        return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range((end - start).days + 1)]

    train_dates = dates_between(start_train, end_train)
    # 标签窗口也必须早于验证起始日，而不只是起报日早于验证期。
    isolated_dates = [d for d in train_dates
                      if datetime.strptime(d, '%Y%m%d').date() + timedelta(days=AR_STEPS) < start_val]
    print(f"Excluded {len(train_dates) - len(isolated_dates)} training windows overlapping validation")
    train_dates = isolated_dates

    # 创建数据集时过滤缺失窗口，续训校验使用实际有效日期。
    train_dataset = OceanARDataset(train_dates, NPZ_DIR, io_workers=num_workers)
    val_dataset = OceanARDataset(dates_between(start_val, end_val), NPZ_DIR, io_workers=num_workers)
    data_split = {'train': train_dataset.date_list, 'validation': val_dataset.date_list}
    print(f"Complete windows: train={len(train_dataset)}, validation={len(val_dataset)}")

    # --------------------------------------------------------------------------------
    # 加载断点与归一化统计量
    # --------------------------------------------------------------------------------
    checkpoint_path = Path(resume_dir) / 'last_checkpoint.pth' if resume_dir else (
        Path(pretrain_path) if pretrain_path else None)
    # 完整训练 checkpoint 在 CPU 加载，避免把旧 Adam 状态复制到 GPU。
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if resume_dir:
        _validate_resume(checkpoint, data_split, freeze_encoder_enabled, batch_size)
    norm_stats = load_normalization_stats(checkpoint, checkpoint_path)
    log_dir = Path(resume_dir) if resume_dir else Path(LOG_ROOT) / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_AR')
    log_dir.mkdir(parents=True, exist_ok=True)
    save_training_sources(log_dir, mymodel, resume=bool(resume_dir))
    with (log_dir / 'normalization_stats.json').open('w', encoding='utf-8') as stream:
        json.dump(norm_stats, stream, indent=2)

    # --------------------------------------------------------------------------------
    # 创建数据加载器
    # --------------------------------------------------------------------------------
    loader_options = dict(num_workers=0, pin_memory=False)
    print(f"AR block loading: I/O threads={num_workers} (0=serial), DataLoader workers=0")
    train_sampler = ARBlockBatchSampler(train_dataset, batch_size, shuffle=True, generator=generator)
    val_sampler = ARBlockBatchSampler(val_dataset, batch_size)
    print(f"AR blocks: train={len(train_dataset.blocks)}, validation={len(val_dataset.blocks)}, "
          f"up to {AR_BLOCK_DAYS} origins + {AR_STEPS} target days per block")
    # DataLoader 初始化的 RNG 与块采样器分离，续训时保持相同块顺序。
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              generator=torch.Generator().manual_seed(seed + 1), **loader_options)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            generator=torch.Generator().manual_seed(seed + 2), **loader_options)

    # --------------------------------------------------------------------------------
    # 创建模型并加载权重
    # --------------------------------------------------------------------------------
    model = mymodel(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights: {checkpoint_path}")
    if freeze_encoder_enabled:
        freeze_encoder(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # --------------------------------------------------------------------------------
    # 创建优化器与学习率调度器
    # --------------------------------------------------------------------------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    scaler = GradScaler(enabled=device.type == 'cuda')
    # --------------------------------------------------------------------------------
    # 恢复续训状态
    # --------------------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    bg_loss_history = []
    variable_loss_history = []
    if resume_dir:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        _restore_rng(checkpoint['rng_state'], generator)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_select_metric']
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']
        bg_loss_history = checkpoint['bg_loss_history']
        variable_loss_history = checkpoint.get('variable_loss_history', [None] * len(train_loss_history))
    del checkpoint

    # --------------------------------------------------------------------------------
    # 训练循环
    # --------------------------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}, lr={optimizer.param_groups[0]['lr']:.2e}")
        # 训练与验证
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, norm_stats, accum_steps, return_metrics=True)
        day_losses, _, val_loss, bg_loss, val_metrics = validate_epoch(
            model, val_loader, device, norm_stats, return_metrics=True)
        print("Validation per day: " + " ".join(f"D{d + 1}={v:.6f}" for d, v in enumerate(day_losses)))
        ratio = val_loss / bg_loss if bg_loss else float('nan')
        print(f"Train sum={train_loss:.6f}, validation sum={val_loss:.6f}, persistence sum={bg_loss:.6f}, ratio={ratio:.4f}")
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        bg_loss_history.append(bg_loss)
        record_variable_metrics(variable_loss_history, train_metrics, val_metrics)
        # 更新学习率
        scheduler.step(val_loss)

        # 保存完整断点；最佳模型与最后断点使用相同字段。
        improved = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        payload = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'select_metric': val_loss, 'best_select_metric': best_val_loss,
            'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history,
            'bg_loss_history': bg_loss_history, 'norm_stats': norm_stats,
            'variable_loss_history': variable_loss_history,
            'ar_config': ar_config(), 'data_split': data_split,
            'data_loading': data_loading_config(batch_size),
            'freeze_encoder_enabled': freeze_encoder_enabled, 'rng_state': _rng_state(generator),
        }
        names = ['last_checkpoint.pth', 'best_model.pth'] if improved else ['last_checkpoint.pth']
        for name in names:
            path = log_dir / name
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)

        # 保存曲线后再检查停止条件，保留当前 epoch 的训练状态。
        plot_loss_curves(train_loss_history, val_loss_history, log_dir / 'loss_curve.png',
                         bg_loss_history, variable_loss_history)
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            break
    print(f"AR training completed. Best sum of daily validation losses: {best_val_loss:.6f}")
    return log_dir


if __name__ == "__main__":
    main()
