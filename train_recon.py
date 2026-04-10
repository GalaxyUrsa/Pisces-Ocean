"""
Ocean Reconstruction Model Training Script

This script trains the ocean reconstruction model that reconstructs 3D ocean fields
from surface observations and background subsurface data.

Usage:
    python train_recon.py --config config_recon.json
    python train_recon.py --epochs 100 --batch_size 4 --lr 1e-4
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ReconstructionModel, count_parameters
from dataset import OceanReconstructionDataset
from utils import (
    generate_date_list,
    load_normalization_stats,
    MetricTracker,
    compute_metrics,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    export_to_onnx,
    Logger,
    get_lr,
    set_seed
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Ocean Reconstruction Model')

    # Data paths
    parser.add_argument('--surface_dir', type=str, default='./input_data/SLA',
                        help='Directory containing SLA data')
    parser.add_argument('--sst_dir', type=str, default='./input_data/SST',
                        help='Directory containing SST data')
    parser.add_argument('--sss_dir', type=str, default='./input_data/SSS',
                        help='Directory containing SSS data')
    parser.add_argument('--background_dir', type=str, default='./input_data/DEEP_LAYER_BACKGROUND',
                        help='Directory containing background subsurface data')
    parser.add_argument('--target_dir', type=str, default='./input_data/GLORYS_REANALYSIS',
                        help='Directory containing target GLORYS data')
    parser.add_argument('--src_dir', type=str, default='./src',
                        help='Directory containing normalization statistics')

    # Training dates
    parser.add_argument('--train_start', type=str, default='2018-01-01',
                        help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2022-12-31',
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--val_start', type=str, default='2023-01-01',
                        help='Validation start date (YYYY-MM-DD)')
    parser.add_argument('--val_end', type=str, default='2023-12-31',
                        help='Validation end date (YYYY-MM-DD)')

    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels in the model')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/recon',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/recon',
                        help='Directory to save logs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Early stopping
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides other arguments)')

    args = parser.parse_args()

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    return args


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    tracker = MetricTracker(['loss', 'rmse', 'mae', 'corr'])

    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets, masks in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss (only on valid regions)
        loss = criterion(outputs * masks, targets * masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, targets, masks)
            metrics['loss'] = loss.item()
            tracker.update(metrics, inputs.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.6f}",
            'rmse': f"{metrics['rmse']:.6f}"
        })

    return tracker.get_averages()


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    tracker = MetricTracker(['loss', 'rmse', 'mae', 'corr'])

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, targets, masks in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs * masks, targets * masks)

            # Compute metrics
            metrics = compute_metrics(outputs, targets, masks)
            metrics['loss'] = loss.item()
            tracker.update(metrics, inputs.size(0))

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.6f}",
                'rmse': f"{metrics['rmse']:.6f}"
            })

    return tracker.get_averages()


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add min validation loss marker
    min_val_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_idx]
    plt.plot(min_val_idx + 1, min_val_loss, 'r*', markersize=15,
             label=f'Best Val Loss: {min_val_loss:.6f} (Epoch {min_val_idx + 1})')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Also save loss history to CSV
    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f'{i},{train_loss:.8f},{val_loss:.8f}\n')


def main():
    """Main training function"""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = Logger(args.log_dir, 'train_recon.log')
    logger.log("=" * 80)
    logger.log("Ocean Reconstruction Model Training")
    logger.log("=" * 80)
    logger.log(f"Arguments: {vars(args)}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load normalization statistics
    logger.log("Loading normalization statistics...")
    mean, std = load_normalization_stats(args.src_dir)
    logger.log(f"Normalization stats loaded: mean shape {mean.shape}, std shape {std.shape}")

    # Generate date lists
    logger.log("Generating date lists...")
    train_dates = generate_date_list(args.train_start, args.train_end)
    val_dates = generate_date_list(args.val_start, args.val_end)
    logger.log(f"Training samples: {len(train_dates)}")
    logger.log(f"Validation samples: {len(val_dates)}")

    # Create datasets
    logger.log("Creating datasets...")

    # Get background offset from args if available, otherwise use default
    background_offset = getattr(args, 'background_offset', -7)

    train_dataset = OceanReconstructionDataset(
        date_list=train_dates,
        surface_dir=args.surface_dir,
        sst_dir=args.sst_dir,
        sss_dir=args.sss_dir,
        background_dir=args.background_dir,
        target_dir=args.target_dir,
        mean=mean,
        std=std,
        background_offset=background_offset
    )

    val_dataset = OceanReconstructionDataset(
        date_list=val_dates,
        surface_dir=args.surface_dir,
        sst_dir=args.sst_dir,
        sss_dir=args.sss_dir,
        background_dir=args.background_dir,
        target_dir=args.target_dir,
        mean=mean,
        std=std,
        background_offset=background_offset
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    logger.log("Creating model...")
    model = ReconstructionModel(
        in_channels=85,
        out_channels=80,
        base_channels=args.base_channels
    ).to(device)

    logger.log(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        logger.log(f"Resuming from checkpoint: {args.resume}")
        model, start_epoch, _ = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch += 1

    # Initialize loss history tracking
    train_loss_history = []
    val_loss_history = []

    # Training loop
    logger.log("Starting training...")
    logger.log("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        logger.log(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.log(f"Learning rate: {get_lr(optimizer):.2e}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.log_metrics(epoch + 1, 'Train', train_metrics)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        logger.log_metrics(epoch + 1, 'Val', val_metrics)

        # Record loss history
        train_loss_history.append(train_metrics['loss'])
        val_loss_history.append(val_metrics['loss'])

        # Update learning rate
        scheduler.step(val_metrics['loss'])

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            logger.log(f"New best validation loss: {best_val_loss:.6f}")

        # Always save the latest checkpoint
        latest_checkpoint_path = os.path.join(args.checkpoint_dir, 'recon_last.pth')
        save_checkpoint(
            model, optimizer, epoch, val_metrics['loss'],
            latest_checkpoint_path, is_best=False
        )

        # Save best checkpoint if this is the best model
        if is_best:
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'recon_best.pth')
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                best_checkpoint_path, is_best=False
            )
            logger.log(f"Best checkpoint saved: {best_checkpoint_path}")

        # Plot loss curves after each epoch
        loss_plot_path = os.path.join(args.log_dir, 'loss_curve.png')
        plot_loss_curves(train_loss_history, val_loss_history, loss_plot_path)

        # Early stopping
        if early_stopping(val_metrics['loss']):
            logger.log(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Export to ONNX
    logger.log("\nExporting model to ONNX...")
    onnx_path = os.path.join(args.checkpoint_dir, 'recon_model.onnx')
    export_to_onnx(model, (1, 85, 400, 480), onnx_path, device)
    logger.log(f"Model exported to: {onnx_path}")

    logger.log("=" * 80)
    logger.log("Training completed!")
    logger.log(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
