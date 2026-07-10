from load_datasets import OceanDatasetLoader
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from pathlib import Path
from datetime import date, datetime, timedelta
matplotlib.use('Agg')  # 使用非交互式后端

from Data_Config_AR import INPUT_VARS, OUTPUT_VARS, data_index, RAW_DATASET_PATH, NPZ_DIR, CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END, NAN_FILL_VALUE, N_DEPTH, BG_OFFSET_DAYS, MODEL_CLASS as mymodel, IN_CHANNELS, OUT_CHANNELS
from Data_Config_AR import AR_STEPS, AR_STEP_WEIGHTS, AR_WARMUP_EPOCHS

# from models.mymodel import SimpleModel as mymodel
# from models.unet import UNet as mymodel
# from models.HCANet import HCANet as mymodel
# from models.unet3d import UNet3D as mymodel
# from models.simple_convnext_net_v3 import ConvNeXtUNet as mymodel  # 原版 2.3M
# from models.simple_convnext_net_v2 import ConvNeXtUNet as mymodel  # V2: 15M + 梯度检查点


def Hybrid_loss(bg, residual, targets, masks, norm_stats):
    """
    残差预测损失，自动适配 OUTPUT_VARS 中的任意变量组合。
    bg / residual / targets: 物理单位 (B, C, H, W)
    通道顺序与 OUTPUT_VARS 一致，每个 3D 变量占 N_DEPTH 通道，2D 变量占 1 通道。
    """
    pred = bg + residual
    diff = pred - targets

    total_loss = 0.0
    ch = 0
    for entry in OUTPUT_VARS:
        name   = entry[2]           # output_name，如 label_t_3d
        opts   = entry[3]
        n_ch   = N_DEPTH if opts.get('select_depth') else 1
        weight = opts.get('weight', 1.0)

        # 残差 key 约定：label_t_3d → residual_t，label_s_3d → residual_s
        residual_key = name.replace('label_', 'residual_').replace('_3d', '')
        std_val = norm_stats.get(residual_key, {}).get('std', 1.0)

        d = diff[:, ch:ch + n_ch]
        m = masks[:, ch:ch + n_ch]
        std = torch.tensor(std_val, dtype=torch.float32, device=d.device).view(1, -1, 1, 1) if isinstance(std_val, list) else std_val
        var_loss = ((d / std) ** 2 * m).sum() / m.sum().clamp(min=1)
        total_loss = total_loss + weight * var_loss
        ch += n_ch

    return total_loss


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
            data      = dict(bg_npz)
            for entry in OUTPUT_VARS:
                data[entry[2]] = label_npz[entry[3]['bg_key']]
    if data is None:
        from load_datasets import OceanDatasetLoader
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


def _ar_collate_fn(batch):
    """过滤掉 __getitem__ 返回 None 的样本（连续 8 天不完整）。"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class OceanARDataset(Dataset):
    """AR 数据集：每样本返回连续 8 天序列（第 0 天初始 bg + 第 1~7 天 label）。"""

    def __init__(self, date_list, dataloader, norm_stats=None, npz_dir=None):
        self.date_list  = date_list
        self.dataloader = dataloader
        self.norm_stats = norm_stats
        self.npz_dir    = Path(npz_dir) if npz_dir is not None else None

    def __len__(self):
        return len(self.date_list)

    def _load_one_day(self, date_str):
        """加载单天 .npz，返回 bg_arrays (C,H,W) 和 label_arrays (C,H,W)。缺失返回 None。"""
        if self.npz_dir is not None:
            bg_path    = self.npz_dir / f"{date_str}.npz"
            target_str = (datetime.strptime(date_str, '%Y%m%d') + timedelta(days=BG_OFFSET_DAYS)).strftime('%Y%m%d')
            label_path = self.npz_dir / f"{target_str}.npz"
            if not (bg_path.exists() and label_path.exists()):
                return None, None
            bg_npz    = dict(np.load(bg_path))
            label_npz = dict(np.load(label_path))
            data      = dict(bg_npz)
            for entry in OUTPUT_VARS:
                data[entry[2]] = label_npz[entry[3]['bg_key']]
        else:
            return None, None  # 暂不支持 NetCDF 路径的序列加载

        bg_arrays = []
        for e in OUTPUT_VARS:
            arr = data[e[3]['bg_key']]
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 0)
            bg_arrays.append(arr)
        bg = np.concatenate(bg_arrays, axis=0).astype(np.float32)

        label_arrays = []
        for e in OUTPUT_VARS:
            arr = data[e[2]]
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 0)
            label_arrays.append(arr)
        label = np.concatenate(label_arrays, axis=0).astype(np.float32)

        return bg, label

    def __getitem__(self, idx):
        base_date = datetime.strptime(self.date_list[idx], '%Y%m%d')
        date_strs  = [(base_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(AR_STEPS + 1)]

        bg_list, label_list = [], []
        for date_str in date_strs:
            bg, label = self._load_one_day(date_str)
            if bg is None:
                return None  # 任意一天缺失，跳过该样本
            bg_list.append(bg)
            label_list.append(label)

        # bg_phys_seq: (8, C, H, W)  第 0~7 天背景场真值
        bg_phys_seq = np.stack(bg_list, axis=0)
        # labels_seq: (7, C, H, W)   第 1~7 天 label（对应步 0~6 的预测目标）
        labels_seq  = np.stack(label_list[1:], axis=0)

        masks_seq = (~np.isnan(labels_seq) & ~np.isnan(bg_phys_seq[1:])).astype(np.float32)

        bg_phys_seq = np.nan_to_num(bg_phys_seq, nan=NAN_FILL_VALUE)
        labels_seq  = np.nan_to_num(labels_seq,  nan=NAN_FILL_VALUE)

        return (torch.from_numpy(bg_phys_seq),
                torch.from_numpy(labels_seq),
                torch.from_numpy(masks_seq))


def train_epoch(model, dataloader, optimizer, scaler, device, norm_stats, accum_steps=1):
    """训练一个epoch，支持梯度累积（accum_steps>1 模拟更大 batch size）"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc='Training')
    for step, (inputs, bg_phys, targets, masks) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        bg_phys = bg_phys.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 前向：混合精度，模型输出物理量纲的残差，pred = bg + residual
        with autocast('cuda'):
            residual = model(inputs)
            loss = Hybrid_loss(bg_phys, residual, targets, masks, norm_stats) / accum_steps

        scaler.scale(loss).backward()

        total_loss += loss.item() * accum_steps  # 还原为真实损失值记录

        # 每 accum_steps 步或最后一个 step 才更新参数
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 更新进度条
        pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.6f}"})

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device, norm_stats):
    """验证一个epoch，同时计算 pred loss 和 bg baseline loss"""
    model.eval()
    total_loss = 0.0
    total_bg_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, bg_phys, targets, masks in pbar:
            inputs = inputs.to(device, non_blocking=True)
            bg_phys = bg_phys.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda'):
                residual = model(inputs)
                loss = Hybrid_loss(bg_phys, residual, targets, masks, norm_stats)

                zero_residual = torch.zeros_like(residual)
                bg_loss = Hybrid_loss(bg_phys, zero_residual, targets, masks, norm_stats)

            total_loss += loss.item()
            total_bg_loss += bg_loss.item()

            pbar.set_postfix({'pred': f"{loss.item():.6f}", 'bg': f"{bg_loss.item():.6f}"})

    return total_loss / len(dataloader), total_bg_loss / len(dataloader)


def plot_loss_curves(train_losses, val_losses, save_path='loss_curve.png', bg_losses=None):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    if bg_losses:
        plt.plot(epochs, bg_losses, 'g--', label='BG Baseline Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 标记最佳验证损失
    min_val_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_idx]
    plt.plot(min_val_idx + 1, min_val_loss, 'r*', markersize=15,
             label=f'Best Val Loss: {min_val_loss:.6f} (Epoch {min_val_idx + 1})')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")

    # 同时保存为CSV
    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        header = 'epoch,train_loss,val_loss'
        if bg_losses:
            header += ',bg_loss'
        f.write(header + '\n')
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            row = f'{i},{train_loss:.8f},{val_loss:.8f}'
            if bg_losses:
                row += f',{bg_losses[i-1]:.8f}'
            f.write(row + '\n')
    print(f"Loss data saved to: {csv_path}")


if __name__=="__main__":
    # -------------------------------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 训练参数
    batch_size = 1
    num_epochs = 200
    learning_rate = 1e-4
    num_workers = 5
    accum_steps = 4  # 梯度累积步数，等效 batch size = batch_size * accum_steps

    # 断点续训：设置为已有的 log 目录路径（如 'logs/20250507_120000'），None 表示从头训练
    # resume_dir = "./logs/20260627_094743"
    resume_dir = None

    # 创建或恢复 log 目录
    if resume_dir is not None:
        log_dir = resume_dir
        print(f"Resuming from: {log_dir}")
    else:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('logs', run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")

    # 归一化参数
    use_normalization = True  # 是否使用归一化
    norm_stats_path = os.path.join(log_dir, 'normalization_stats.json')
    root_norm_stats_path = 'normalization_stats.json'  # 根目录缓存路径

    # -------------------------------------------------------------------------------------------------
    # 准备数据
    # -------------------------------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Preparing datasets...")
    print("="*60)

    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)

    # 生成日期列表（示例：使用已有的10个日期）
    # 实际使用时，您需要根据实际情况生成完整的日期列表
    # train_dates = ['20250501', '20250502', '20250503', '20250504', '20250505',
    #                '20250506', '20250507', '']
    # val_dates = ['20250508', '20250509', '20250510']
    start_train = date(2021, 1,  8)
    end_train   = date(2024, 12, 30)
    start_val   = date(2025, 1,  1)
    end_val     = date(2025, 12, 20)
    # start_train = date(2021, 1,  8)
    # end_train   = date(2021, 1, 30)
    # start_val   = date(2021, 2,  1)
    # end_val     = date(2021, 2, 20)

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
    norm_stats = None
    if use_normalization:
        if os.path.exists(root_norm_stats_path):
            # 根目录已有缓存，直接跳过计算
            print(f"\nFound cached normalization stats at root: {root_norm_stats_path}, skipping computation.")
            norm_stats = load_normalization_stats(root_norm_stats_path)
            # 同时复制一份到本次 log 目录
            import shutil
            shutil.copy(root_norm_stats_path, norm_stats_path)
            print(f"Copied to log dir: {norm_stats_path}")
        else:
            # 根目录没有缓存，计算并同时保存到根目录和 log 目录
            norm_stats = compute_normalization_stats(train_dates, dataloader, root_norm_stats_path, npz_dir=NPZ_DIR)
            import shutil
            shutil.copy(root_norm_stats_path, norm_stats_path)
            print(f"Copied to log dir: {norm_stats_path}")
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
    scaler = GradScaler('cuda')

    # -------------------------------------------------------------------------------------------------
    # 断点续训：加载 checkpoint
    # -------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    bg_loss_history = []

    last_ckpt_path = os.path.join(log_dir, 'last_checkpoint.pth')
    if resume_dir is not None and os.path.exists(last_ckpt_path):
        print(f"\nLoading checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
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
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_val_loss={best_val_loss:.6f}")
    elif resume_dir is not None:
        print(f"Warning: resume_dir set but no checkpoint found at {last_ckpt_path}, starting from scratch.")


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
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, norm_stats, accum_steps)
        print(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        # 验证
        val_loss, bg_loss = validate_epoch(model, val_loader, device, norm_stats)
        print(f"Val Loss: {val_loss:.6f}  |  BG Baseline: {bg_loss:.6f}  |  Ratio: {val_loss/bg_loss:.4f}")
        val_loss_history.append(val_loss)
        bg_loss_history.append(bg_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # LR 过小时 early stop
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            print(f"Early stopping: lr={current_lr:.2e} < 1e-6")
            break

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_stats': norm_stats,
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f"✓ Best model saved! Val Loss: {val_loss:.6f}")

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
        }, last_ckpt_path)

        # 每个epoch后绘制损失曲线
        plot_loss_curves(train_loss_history, val_loss_history, os.path.join(log_dir, 'loss_curve.png'), bg_loss_history)

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)

