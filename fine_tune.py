"""
Fine-tune a GLORYS-pretrained ConvNeXtUNet on AF background data.

Strategy:
  - Encoder (proj_in, stage1, down1, stage2, down2, stage3) is frozen.
  - Decoder + output heads (up1, fusion1, stage4, up2, stage5, head_temp, head_salt) are trained.
  - Normalization stats are recomputed from AF training data.
  - Loss and dataset logic are identical to train.py.

Usage:
    python fine_tune.py
"""

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
import shutil
from datetime import date, datetime, timedelta
matplotlib.use('Agg')

from Data_Config import SURFACE_VARS, data_index, RAW_DATASET_PATH, \
    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END, NAN_FILL_VALUE
from models.simple_convnext_net import ConvNeXtUNet as mymodel
from train import (
    compute_normalization_stats,
    load_normalization_stats,
    OceanDataset,
    Hybrid_loss,
    train_epoch,
    validate_epoch,
    plot_loss_curves,
)

IN_CHANNELS = len(SURFACE_VARS) + 40


# 编码器模块名：这些参数冻结，不参与梯度更新
_ENCODER_MODULES = {'proj_in', 'stage1', 'down1', 'stage2', 'down2', 'stage3'}


def freeze_encoder(model: mymodel):
    for name, param in model.named_parameters():
        top = name.split('.')[0]
        if top in _ENCODER_MODULES:
            param.requires_grad = False

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters:   {frozen:,}")
    print(f"Trainable parameters:{trainable:,}")


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # 配置
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 预训练权重路径（GLORYS 训练的 best_model.pth）
    pretrain_path = './logs/fine_tune/best_model.pth'

    # 微调超参
    batch_size   = 1
    num_epochs   = 50
    learning_rate = 1e-5      # 比预训练小一个量级
    num_workers  = 8
    accum_steps  = 4

    # 断点续训：None 表示从头微调
    resume_dir = None

    # log 目录
    if resume_dir is not None:
        log_dir = resume_dir
        print(f"Resuming from: {log_dir}")
    else:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_finetune'
        log_dir = os.path.join('logs', run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")

    # AF stats 缓存路径（与 GLORYS stats 分开存，不覆盖）
    af_stats_root = 'normalization_stats_af.json'
    af_stats_log  = os.path.join(log_dir, 'normalization_stats.json')

    # -------------------------------------------------------------------------
    # 数据（与 train.py 保持相同格式，日期范围按需调整）
    # -------------------------------------------------------------------------
    start_train = date(2023,  1,  8)
    end_train   = date(2024, 12, 30)
    start_val   = date(2025,  7,  1)
    end_val     = date(2025, 12, 19)

    def generate_date_list(start, end):
        delta = end - start
        return [(start + timedelta(days=i)).strftime('%Y%m%d')
                for i in range(delta.days + 1)]

    train_dates = generate_date_list(start_train, end_train)
    val_dates   = generate_date_list(start_val,   end_val)
    print(f"Training samples:   {len(train_dates)}")
    print(f"Validation samples: {len(val_dates)}")

    dataloader = OceanDatasetLoader(RAW_DATASET_PATH)

    # -------------------------------------------------------------------------
    # 归一化统计量（AF 数据重新算）
    # -------------------------------------------------------------------------
    if os.path.exists(af_stats_root):
        print(f"\nFound cached AF stats: {af_stats_root}")
        norm_stats = load_normalization_stats(af_stats_root)
        shutil.copy(af_stats_root, af_stats_log)
        print(f"Copied to log dir: {af_stats_log}")
    else:
        print("\nComputing AF normalization stats...")
        norm_stats = compute_normalization_stats(train_dates, dataloader, af_stats_root)
        shutil.copy(af_stats_root, af_stats_log)
        print(f"Copied to log dir: {af_stats_log}")

    residual_t_std = torch.tensor(norm_stats['residual_t']['std'],
                                  dtype=torch.float32, device=device).view(1, 20, 1, 1)
    residual_s_std = torch.tensor(norm_stats['residual_s']['std'],
                                  dtype=torch.float32, device=device).view(1, 20, 1, 1)
    print(f"residual_t std (per depth): {residual_t_std.flatten().cpu().tolist()}")
    print(f"residual_s std (per depth): {residual_s_std.flatten().cpu().tolist()}")

    # -------------------------------------------------------------------------
    # 数据集 & DataLoader
    # -------------------------------------------------------------------------
    train_dataset = OceanDataset(train_dates, dataloader, norm_stats)
    val_dataset   = OceanDataset(val_dates,   dataloader, norm_stats)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    # -------------------------------------------------------------------------
    # 模型：加载预训练权重，冻结编码器
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading pretrained model...")
    print("=" * 60)

    model = mymodel(in_channels=IN_CHANNELS, out_channels=40).to(device)
    ckpt = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded from: {pretrain_path}  (epoch {ckpt['epoch']})")

    freeze_encoder(model)

    # -------------------------------------------------------------------------
    # 优化器：只传可训练参数
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # -------------------------------------------------------------------------
    # 断点续训
    # -------------------------------------------------------------------------
    start_epoch      = 0
    best_val_loss    = float('inf')
    train_loss_history = []
    val_loss_history   = []
    bg_loss_history    = []

    last_ckpt_path = os.path.join(log_dir, 'last_checkpoint.pth')
    if resume_dir is not None and os.path.exists(last_ckpt_path):
        print(f"\nLoading checkpoint: {last_ckpt_path}")
        saved = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start_epoch        = saved['epoch'] + 1
        best_val_loss      = saved['best_val_loss']
        train_loss_history = saved['train_loss_history']
        val_loss_history   = saved['val_loss_history']
        bg_loss_history    = saved.get('bg_loss_history', [])
        print(f"Resumed from epoch {saved['epoch'] + 1}, best_val_loss={best_val_loss:.6f}")
    elif resume_dir is not None:
        print(f"Warning: resume_dir set but no checkpoint at {last_ckpt_path}, starting from scratch.")

    # -------------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting fine-tuning (encoder frozen)...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            residual_t_std, residual_s_std, accum_steps
        )
        print(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        val_loss, bg_loss = validate_epoch(
            model, val_loader, device,
            residual_t_std, residual_s_std
        )
        print(f"Val Loss: {val_loss:.6f}  |  BG Baseline: {bg_loss:.6f}  |  Ratio: {val_loss/bg_loss:.4f}")
        val_loss_history.append(val_loss)
        bg_loss_history.append(bg_loss)

        scheduler.step(val_loss)

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

        plot_loss_curves(
            train_loss_history, val_loss_history,
            os.path.join(log_dir, 'loss_curve.png'),
            bg_loss_history
        )

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)