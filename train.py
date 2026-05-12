from load_datasets import OceanDatasetLoader
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from datetime import date, datetime, timedelta
matplotlib.use('Agg')  # 使用非交互式后端

from Data_Config import SURFACE_VARS, _SURFACE_INDEX, data_index, CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END

# from models.mymodel import SimpleModel as mymodel
# from models.unet import UNet as mymodel
# from models.HCANet import HCANet as mymodel
# from models.unet3d import UNet3D as mymodel
from models.simple_convnext_net import ConvNeXtUNet as mymodel
# from models.simple_convnext_net_0429 import ConvNeXtUNet as mymodel

IN_CHANNELS = len(SURFACE_VARS) + 40  # surface vars + bg_t_3d(20) + bg_s_3d(20)


def Hybrid_loss(targets, outputs, masks, temp_weight=0.9, salt_weight=0.1):
    # temperature loss
    temp_loss = ((outputs[:, :20] - targets[:, :20]) ** 2 * masks[:, :20]).sum() / masks[:, :20].sum().clamp(min=1)
    # salinity loss
    salinity_loss = ((outputs[:, 20:40] - targets[:, 20:40]) ** 2 * masks[:, 20:40]).sum() / masks[:, 20:40].sum().clamp(min=1)
    return temp_weight * temp_loss + salt_weight * salinity_loss


def compute_normalization_stats(date_list, dataloader, save_path='normalization_stats.json'):
    """
    计算训练集的归一化统计量（均值和标准差）

    Args:
        date_list: 训练集日期列表
        dataloader: OceanDatasetLoader 实例
        save_path: 统计量保存路径

    Returns:
        stats: 包含每个变量均值和标准差的字典
    """
    print("\n" + "="*60)
    print("Computing normalization statistics...")
    print("="*60)

    # 初始化累加器
    sums = {}
    sums_sq = {}
    counts = {}

    # 定义需要归一化的变量（输入和输出）
    var_names = SURFACE_VARS + ['bg_t_3d', 'bg_s_3d', 'label_t_3d', 'label_s_3d']

    for var in var_names:
        sums[var] = 0.0
        sums_sq[var] = 0.0
        counts[var] = 0

    # 遍历所有训练样本
    for date in tqdm(date_list, desc="Processing samples"):
        raw_data = dataloader.load_single_date(date, data_index, isLog=False)

        # raw_data is already a flat {output_name: array} dict
        data = raw_data

        # 对每个变量计算统计量
        for var in var_names:
            arr = data[var]
            # 只统计非NaN值
            valid_mask = ~np.isnan(arr)
            valid_data = arr[valid_mask]

            if len(valid_data) > 0:
                sums[var] += np.sum(valid_data)
                sums_sq[var] += np.sum(valid_data ** 2)
                counts[var] += len(valid_data)

    # 计算均值和标准差
    stats = {}
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


class OceanDataset(Dataset):
    """自定义数据集类（支持Z-score归一化）"""
    def __init__(self, date_list, dataloader, norm_stats=None):
        """
        Args:
            date_list: 日期列表，格式为 ['20250501', '20250502', ...]
            dataloader: OceanDatasetLoader 实例
            norm_stats: 归一化统计量字典，如果为None则不进行归一化
        """
        self.date_list = date_list
        self.dataloader = dataloader
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.date_list)

    def _normalize(self, data, var_name):
        """对数据进行Z-score归一化"""
        if self.norm_stats is None:
            return data

        mean = self.norm_stats[var_name]['mean']
        std = self.norm_stats[var_name]['std']

        return (data - mean) / std

    def __getitem__(self, idx):
        date = self.date_list[idx]

        # 加载单个日期的数据
        raw_data = self.dataloader.load_single_date(date, data_index, isLog=False)

        # raw_data is already a flat {output_name: array} dict
        data = raw_data

        # 对每个变量进行归一化（在处理NaN之前）
        if self.norm_stats is not None:
            for v in SURFACE_VARS:
                data[v] = self._normalize(data[v], v)
            data['bg_t_3d'] = self._normalize(data['bg_t_3d'], 'bg_t_3d')
            data['bg_s_3d'] = self._normalize(data['bg_s_3d'], 'bg_s_3d')
            data['label_t_3d'] = self._normalize(data['label_t_3d'], 'label_t_3d')
            data['label_s_3d'] = self._normalize(data['label_s_3d'], 'label_s_3d')

        # 构建输入：surface vars + bg_t_3d(20) + bg_s_3d(20)
        inputs = np.concatenate(
            [np.expand_dims(data[v], axis=0) for v in SURFACE_VARS] +
            [data['bg_t_3d'], data['bg_s_3d']],
            axis=0
        ).astype(np.float32)  # (IN_CHANNELS, H, W)

        # 构建标签：label_t_3d(20,400,480) + label_s_3d(20,400,480)
        # 总共: 20 + 20 = 40 个通道
        labels = np.concatenate([
            data['label_t_3d'],                       # (20, 400, 480)
            data['label_s_3d']                        # (20, 400, 480)
        ], axis=0).astype(np.float32)                 # (40, 400, 480)

        # # 创建mask（用于处理NaN值）：在通道维度取 all，得到 (H, W) 再扩展回 (40, H, W)
        # mask_1 = ~np.isnan(labels)   # (40, H, W)
        # mask_2 = (~np.isnan(inputs)).all(axis=0, keepdims=True)  # (1, H, W)
        # mask = mask_1 & mask_2       # broadcast → (40, H, W)
        mask = ~np.isnan(labels)

        # 裁剪到南中国海区域 (C, H, W) → (C, 200, 240)
        inputs = inputs[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
        labels = labels[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
        mask   = mask[:,   CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

        # 将NaN替换为0
        inputs = np.nan_to_num(inputs, nan=0.0)
        labels = np.nan_to_num(labels, nan=0.0)

        return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(mask.astype(np.float32))


def train_epoch(model, dataloader, optimizer, device, accum_steps=1):
    """训练一个epoch，支持梯度累积（accum_steps>1 模拟更大 batch size）"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc='Training')
    for step, (inputs, targets, masks) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # 前向传播，损失除以累积步数以保持梯度量级一致
        outputs = model(inputs)
        loss = Hybrid_loss(outputs, targets, masks) / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps  # 还原为真实损失值记录

        # 每 accum_steps 步或最后一个 step 才更新参数
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        # 更新进度条
        pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.6f}"})

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    """验证一个epoch，同时计算 pred loss 和 bg baseline loss"""
    model.eval()
    total_loss = 0.0
    total_bg_loss = 0.0

    # bg 在输入中的通道索引：surface_vars 之后的前20通道是 bg_t_3d，接着20通道是 bg_s_3d
    n_surface = len(SURFACE_VARS)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, targets, masks in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算 pred loss
            loss = Hybrid_loss(outputs, targets, masks)
            total_loss += loss.item()

            # 计算 bg baseline loss（bg 直接作为预测）
            bg = inputs[:, n_surface:n_surface + 40]  # (B, 40, H, W)
            bg_loss = Hybrid_loss(bg, targets, masks)
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
    num_epochs = 50
    learning_rate = 5e-5
    num_workers = 8
    accum_steps = 1  # 梯度累积步数，等效 batch size = batch_size * accum_steps

    # 断点续训：设置为已有的 log 目录路径（如 'logs/20250507_120000'），None 表示从头训练
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

    dataloader = OceanDatasetLoader()

    # 生成日期列表（示例：使用已有的10个日期）
    # 实际使用时，您需要根据实际情况生成完整的日期列表
    # train_dates = ['20250501', '20250502', '20250503', '20250504', '20250505',
    #                '20250506', '20250507', '']
    # val_dates = ['20250508', '20250509', '20250510']
    start_train = date(2021,  1,  8)
    end_train   = date(2024, 12, 31)
    start_val   = date(2025,  1,  1)
    end_val     = date(2025, 12, 31)

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
            norm_stats = compute_normalization_stats(train_dates, dataloader, root_norm_stats_path)
            import shutil
            shutil.copy(root_norm_stats_path, norm_stats_path)
            print(f"Copied to log dir: {norm_stats_path}")
    else:
        print("\nNormalization is disabled.")

    # -------------------------------------------------------------------------------------------------
    # 创建数据集
    # -------------------------------------------------------------------------------------------------
    train_dataset = OceanDataset(train_dates, dataloader, norm_stats)
    val_dataset = OceanDataset(val_dates, dataloader, norm_stats)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
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

    model = mymodel(in_channels=IN_CHANNELS, out_channels=40).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # -------------------------------------------------------------------------------------------------
    # 创建优化器和损失函数
    # -------------------------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
        train_loss = train_epoch(model, train_loader, optimizer, device, accum_steps)
        print(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        # 验证
        val_loss, bg_loss = validate_epoch(model, val_loader, device)
        print(f"Val Loss: {val_loss:.6f}  |  BG Baseline: {bg_loss:.6f}  |  Ratio: {val_loss/bg_loss:.4f}")
        val_loss_history.append(val_loss)
        bg_loss_history.append(bg_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_stats': norm_stats,  # 保存归一化统计量
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f"✓ Best model saved! Val Loss: {val_loss:.6f}")

        # 每个epoch保存断点（用于续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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

