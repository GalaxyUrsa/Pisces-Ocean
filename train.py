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

# from models.mymodel import SimpleModel as mymodel
# from models.unet import UNet as mymodel
# from models.HCANet import HCANet as mymodel
# from models.unet3d import UNet3D as mymodel
from models.simple_convnext_net import ConvNeXtUNet as mymodel

# =============================================================================
# 消融实验配置 — 只需修改这里，注释掉对应行即可去掉该变量
# =============================================================================
SURFACE_VARS = [
    'sss',  # Sea Surface Salinity
    'sst',  # Sea Surface Temperature
    'sla',  # Sea Level Anomaly
    'ugos',
    'vgos',
]

# 数据索引（自动推导，无需手动修改）
_SURFACE_INDEX = {
    'sss':  ['SSS', 'sos',  'sss'],
    'sst':  ['SST', 'sst',  'sst'],
    'sla':  ['SLA', 'sla',  'sla'],
    'ugos': ['SLA', 'ugos', 'ugos'],
    'vgos': ['SLA', 'vgos', 'vgos'],
}
data_index = (
    [_SURFACE_INDEX[v] for v in SURFACE_VARS] +
    [
        ['Glorys',     'thetao', 'label_t_3d'],
        ['Glorys',     'so',     'label_s_3d'],
        ['Background', 'thetao', 'bg_t_3d'],
        ['Background', 'so',     'bg_s_3d'],
    ]
)
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
        raw_data = dataloader.load_single_date(date, isLog=False)

        # 扁平化数据
        data = {}
        for folder, var, name in data_index:
            data[name] = raw_data[folder][var]

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
        raw_data = self.dataloader.load_single_date(date, isLog=False)

        # 扁平化数据
        data = {}
        for folder, var, name in data_index:
            data[name] = raw_data[folder][var]

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
        ).astype(np.float32)  # (IN_CHANNELS, 400, 480)

        # 构建标签：label_t_3d(20,400,480) + label_s_3d(20,400,480)
        # 总共: 20 + 20 = 40 个通道
        labels = np.concatenate([
            data['label_t_3d'],                       # (20, 400, 480)
            data['label_s_3d']                        # (20, 400, 480)
        ], axis=0).astype(np.float32)                 # (40, 400, 480)

        # 创建mask（用于处理NaN值）
        mask = ~np.isnan(labels)

        # 将NaN替换为0
        inputs = np.nan_to_num(inputs, nan=0.0)
        labels = np.nan_to_num(labels, nan=0.0)

        return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(mask.astype(np.float32))


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets, masks in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(inputs)

        # 计算损失（只在有效区域，masked mean）
        # loss = ((outputs - targets) ** 2 * masks).sum() / masks.sum().clamp(min=1)
        loss = Hybrid_loss(outputs, targets, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, targets, masks in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失（只在有效区域，masked mean）
            # loss = ((outputs - targets) ** 2 * masks).sum() / masks.sum().clamp(min=1)
            loss = Hybrid_loss(outputs, targets, masks)

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

    return total_loss / len(dataloader)


def plot_loss_curves(train_losses, val_losses, save_path='loss_curve.png'):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

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
        f.write('epoch,train_loss,val_loss\n')
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f'{i},{train_loss:.8f},{val_loss:.8f}\n')
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
    learning_rate = 1e-4
    num_workers = 4

    # 创建本次训练的 log 目录
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # 归一化参数
    use_normalization = True  # 是否使用归一化
    norm_stats_path = os.path.join(log_dir, 'normalization_stats.json')

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
    start_train = date(2024, 7, 8)
    end_train = date(2025, 9, 30)
    start_val = date(2025, 10, 1)
    end_val = date(2026, 1, 2)

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
        if os.path.exists(norm_stats_path):
            print(f"\nFound existing normalization stats: {norm_stats_path}")
            user_input = input("Use existing stats? (y/n): ").strip().lower()
            if user_input == 'y':
                norm_stats = load_normalization_stats(norm_stats_path)
            else:
                norm_stats = compute_normalization_stats(train_dates, dataloader, norm_stats_path)
        else:
            norm_stats = compute_normalization_stats(train_dates, dataloader, norm_stats_path)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # -------------------------------------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        # 验证
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Val Loss: {val_loss:.6f}")
        val_loss_history.append(val_loss)

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

        # 每个epoch后绘制损失曲线
        plot_loss_curves(train_loss_history, val_loss_history, os.path.join(log_dir, 'loss_curve.png'))

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)

