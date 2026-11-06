"""
Fine-tune a pretrained ConvNeXtUNet on a different date range or data split.

Strategy:
  - Encoder modules listed in _ENCODER_MODULES are frozen (empty = full fine-tune).
  - All other training logic is identical to train.py.
  - Normalization stats are recomputed from fine-tune training data.

Usage:
    python fine_tune.py
"""

from utils.load_datasets import OceanDatasetLoader
from utils.training_snapshot import save_training_sources
from utils.training_metrics import record_variable_metrics, loss_ratio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import os
import shutil
from datetime import date, datetime, timedelta

from Data_Config import INPUT_VARS, data_index, RAW_DATASET_PATH, NPZ_DIR, N_DEPTH, MODEL_CLASS as mymodel, IN_CHANNELS, OUT_CHANNELS
from Data_Config import LOG_ROOT
from train import (
    compute_normalization_stats,
    load_normalization_stats,
    OceanDataset,
    train_epoch,
    validate_epoch,
    plot_loss_curves,
)

IN_CHANNELS = sum(N_DEPTH if e[3].get('select_depth') else 1 for e in INPUT_VARS)

# 要冻结的顶层模块名，空集 = 全量微调
# _ENCODER_MODULES: set = {'proj_in', 'stage1', 'down1', 'stage2', 'down2', 'stage3'}
_ENCODER_MODULES: set = {}


def freeze_encoder(model):
    for name, param in model.named_parameters():
        if name.split('.')[0] in _ENCODER_MODULES:
            param.requires_grad = False
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── 超参 ──────────────────────────────────────────────────────────────────
    pretrain_path = './logs/pretrain_2021_2024_glo_glo_convnextv3_50_layer_100epoch/best_model.pth'
    batch_size    = 1
    num_epochs    = 30
    learning_rate = 1e-4
    num_workers   = 5
    accum_steps   = 1
    npz_dir       = NPZ_DIR
    root_stats_path = 'normalization_stats/normalization_stats_finetune_2_day.json'

    # 断点续训：None 表示从预训练权重开始
    resume_dir = None
    # resume_dir = "./logs/20260704_203255_finetune_6"

    # ── 日期范围 ───────────────────────────────────────────────────────────────
    start_train = date(2021,  1,  8)
    end_train   = date(2024, 12, 30)
    start_val   = date(2025, 1,  1)
    end_val     = date(2025, 12, 20)

    def generate_date_list(start, end):
        return [(start + timedelta(days=i)).strftime('%Y%m%d')
                for i in range((end - start).days + 1)]

    train_dates = generate_date_list(start_train, end_train)
    val_dates   = generate_date_list(start_val,   end_val)
    print(f"Training samples:   {len(train_dates)}")
    print(f"Validation samples: {len(val_dates)}")

    # ── log 目录 ───────────────────────────────────────────────────────────────
    if resume_dir is not None:
        log_dir = resume_dir
        print(f"Resuming from: {log_dir}")
    else:
        log_dir = os.path.join(LOG_ROOT, datetime.now().strftime('%Y%m%d_%H%M%S') + '_finetune')
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")

    save_training_sources(log_dir, mymodel, resume=resume_dir is not None)

    # ── 归一化统计量 ───────────────────────────────────────────────────────────
    
    log_stats_path  = os.path.join(log_dir, 'normalization_stats.json')
    dataloader      = OceanDatasetLoader(RAW_DATASET_PATH)

    if os.path.exists(root_stats_path):
        print(f"\nFound cached normalization stats: {root_stats_path}")
        norm_stats = load_normalization_stats(root_stats_path)
        shutil.copy(root_stats_path, log_stats_path)
    else:
        norm_stats = compute_normalization_stats(
            train_dates, dataloader, root_stats_path, npz_dir=npz_dir, n_workers=8)
        shutil.copy(root_stats_path, log_stats_path)

    # ── 数据集 ─────────────────────────────────────────────────────────────────
    train_dataset = OceanDataset(train_dates, dataloader, norm_stats, npz_dir=npz_dir)
    val_dataset   = OceanDataset(val_dates,   dataloader, norm_stats, npz_dir=npz_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=1)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=1)

    # ── 模型 ───────────────────────────────────────────────────────────────────
    model = mymodel(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 优化器 ─────────────────────────────────────────────────────────────────
    start_epoch       = 0
    best_val_loss     = float('inf')
    train_loss_history = []
    val_loss_history   = []
    bg_loss_history    = []
    variable_loss_history = []

    last_ckpt_path = os.path.join(log_dir, 'last_checkpoint.pth')

    if resume_dir is not None and os.path.exists(last_ckpt_path):
        print(f"\nLoading checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=1e-5)
        freeze_encoder(model)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler = GradScaler('cuda')
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch        = ckpt['epoch'] + 1
        best_val_loss      = ckpt['best_val_loss']
        train_loss_history = ckpt['train_loss_history']
        val_loss_history   = ckpt['val_loss_history']
        bg_loss_history    = ckpt.get('bg_loss_history', [])
        variable_loss_history = ckpt.get('variable_loss_history', [None] * len(train_loss_history))
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_val_loss={best_val_loss:.6f}")
    else:
        # 从预训练权重加载
        print(f"\nLoading pretrained weights: {pretrain_path}")
        ckpt = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        freeze_encoder(model)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler = GradScaler('cuda')
        if resume_dir is not None:
            print(f"Warning: resume_dir set but no checkpoint found at {last_ckpt_path}, starting from pretrain.")

    # ── 训练循环 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, norm_stats, accum_steps, return_metrics=True)
        print(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        val_loss, bg_loss, val_metrics = validate_epoch(
            model, val_loader, device, norm_stats, return_metrics=True)
        print(f"Val Loss: {val_loss:.6f}  |  BG Baseline: {bg_loss:.6f}  |  Ratio: {loss_ratio(val_loss, bg_loss):.4f}")
        val_loss_history.append(val_loss)
        bg_loss_history.append(bg_loss)
        record_variable_metrics(variable_loss_history, train_metrics, val_metrics)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'norm_stats': norm_stats,
                'variable_loss_history': variable_loss_history,
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f"✓ Best model saved! Val Loss: {val_loss:.6f}")

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

        plot_loss_curves(train_loss_history, val_loss_history,
                         os.path.join(log_dir, 'loss_curve.png'), bg_loss_history, variable_loss_history)
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print(f"Early stopping: lr < 1e-7")
            break

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)
