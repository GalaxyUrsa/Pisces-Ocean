"""
Ocean Forecast Model Training Script

This script trains the ocean forecast models that predict future ocean states
from current observations. Separate models are trained for each lead time (1-10 days).

Usage:
    python train_forecast.py --lead_days 1 --config config_forecast.json
    python train_forecast.py --lead_days 5 --epochs 100 --batch_size 4 --lr 1e-4
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ForecastModel, count_parameters
from dataset import OceanForecastDataset
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
    parser = argparse.ArgumentParser(description='Train Ocean Forecast Model')

    # Data paths
    parser.add_argument('--surface_dir', type=str, default='./input_data/SLA',
                        help='Directory containing SLA data')
    parser.add_argument('--sst_dir', type=str, default='./input_data/SST',
                        help='Directory containing SST data')
    parser.add_argument('--sss_dir', type=str, default='./input_data/SSS',
                        help='Directory containing SSS data')
    parser.add_argument('--current_dir', type=str, default='./output_data/recon',
                        help='Directory containing current subsurface data (reconstruction results)')
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

    # Forecast parameters
    parser.add_argument('--lead_days', type=int, default=1,
                        help='Forecast lead time in days (1-10)')
    parser.add_argument('--train_all_leads', action='store_true',
                        help='Train models for all lead times (1-10)')

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
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/forecast',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/forecast',
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


def train_single_lead(args, lead_days):
    """Train model for a single lead time"""

    # Set random seed
    set_seed(args.seed + lead_days)  # Different seed for each lead time

    # Create directories for this lead time
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'lead{lead_days}')
    log_dir = os.path.join(args.log_dir, f'lead{lead_days}')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = Logger(log_dir, f'train_lead{lead_days}.log')
    logger.log("=" * 80)
    logger.log(f"Ocean Forecast Model Training - Lead {lead_days} Days")
    logger.log("=" * 80)
    logger.log(f"Arguments: {vars(args)}")

    # Set device
    device = torch.device(args.device if torch.c_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load normalization statistics
    logger.log("Loading normalization statistics...")
    mean, std = load_normalization_stats(args.src_dir)
    logger.log(f"Normalization stats loaded: mean shape {mean.shape}, std shape {std.shape}")

    # Generate date lists (exclude dates too close to boundaries)
    logger.log("Generating date lists...")
    train_dates = generate_date_list(args.train_start, args.train_end)
    val_dates = generate_date_list(args.val_start, args.val_end)

    # Remove dates that don't have future datavailable
    train_dates = train_dates[:-lead_days]
    val_dates = val_dates[:-lead_days]

    logger.log(f"Training samples: {len(train_dates)}")
    logger.log(f"Validation samples: {len(val_dates)}")

    # Create datasets
    logger.log("Creating datasets...")
    train_dataset = OceanForecastDataset(
        date_list=train_dates,
        surface_dir=args.surface_dir,
        sst_dir=args.sst_dir,
        sss_dir=args.sss_dir,
        current_dir=args.current_dir,
        target_dir=args.target_dir,
        mean=mean,
        std=std,
        lead_days=lead_days
    )

    val_dataset = OceanForecastDataset(
        date_list=val_dates,
        surface_dir=args.surface_dir,
        sst_dir=args.sst_dir,
        sss_dir=args.sss_dir,
        current_dir=args.current_dir,
        target_dir=args.target_dir,
        mean=mean,
        std=std,
        lead_days=lead_days
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
    model = ForecastModel(
        in_channels=85,
        out_channels=81,
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

        # Update learning rate
        scheduler.step(val_metrics['loss'])

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            logger.log(f"New best validation loss: {best_val_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0 or is_best:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'lead{lead_days}_epoch{epoch + 1}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                checkpoint_path, is_best
            )
            logger.log(f"Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if early_stopping(val_metrics['loss']):
            logger.log(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Export to ONNX
    logger.log("\nExporting model to ONNX...")
    onnx_path = os.path.join(checkpoint_dir, f'lead{lead_days}_model.onnx')
    export_to_onnx(model, (1, 85, 400, 480), onnx_path, device)
    logger.log(f"Model exported to: {onnx_path}")

    logger.log("=" * 80)
    logger.log(f"Training completed for lead {lead_days} days!")
    logger.log(f"Best validation loss: {best_val_loss:.6f}")

    return best_val_loss


def main():
    """Main training function"""
    args = parse_args()

    if args.train_all_leads:
        # Train models for all lead times (1-10 days)
        print("=" * 80)
        print("Training forecast models for all lead times (1-10 days)")
        print("=" * 80)

        results = {}
        for lead in range(1, 11):
            print(f"\n{'=' * 80}")
            print(f"Training model for lead time: {lead} days")
            print(f"{'=' * 80}\n")

            best_loss = train_single_lead(args, lead)
            results[f'lead{lead}'] = best_loss

            print(f"\nCompleted training for lead {lead} days")
            print(f"Best validation loss: {best_loss:.6f}\n")

        # Print summary
        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)
        for lead, loss in results.items():
            print(f"{lead}: Best Val Loss = {loss:.6f}")
        print("=" * 80)

    else:
        # Train model for single lead time
        if args.lead_days < 1 or args.lead_days > 10:
            raise ValueError("lead_days must be between 1 and 10")

        train_single_lead(args, args.lead_days)


if __name__ == '__main__':
    main()
