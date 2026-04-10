"""
Quick Training Test Script

This script uses the existing sample data to test the training pipeline.
It adapts the dataset to use reconstruction results as training targets.
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

from models import ReconstructionModel, count_parameters
from utils import (
    generate_date_list,
    load_normalization_stats,
    MetricTracker,
    compute_metrics,
    save_checkpoint,
    export_to_onnx,
    Logger,
    get_lr,
    set_seed
)

# Import the dataset loading functions
from dataset import date_series, find_files_by_keyword_and_dates
import xarray as xr
from datetime import datetime, timedelta, date
from typing import Tuple, List
from torch.utils.data import Dataset


class QuickTestDataset(Dataset):
    """
    Quick test dataset using available sample data.
    Uses reconstruction results as both input background and training targets.
    """

    def __init__(
        self,
        date_list: List[str],
        surface_dir: str,
        sst_dir: str,
        sss_dir: str,
        recon_dir: str,  # Use reconstruction results
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.date_list = date_list
        self.surface_dir = surface_dir
        self.sst_dir = sst_dir
        self.sss_dir = sss_dir
        self.recon_dir = recon_dir
        self.mean = mean
        self.std = std

        # Depth indices
        self.depth_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]
        self.deep_vars = ['thetao', 'so', 'uo', 'vo']

        # Filter dates to only include those with all required files
        self.valid_dates = self._filter_valid_dates()
        print(f"Found {len(self.valid_dates)} valid dates out of {len(date_list)}")

    def _filter_valid_dates(self) -> List[str]:
        """Filter dates that have all required data files"""
        valid = []
        for date_str in self.date_list:
            try:
                # Check SLA
                sla_files = find_files_by_keyword_and_dates(self.surface_dir, ["allsat"], [date_str])
                if date_str not in sla_files:
                    continue

                # Check SST
                sst_files = find_files_by_keyword_and_dates(self.sst_dir, ["oisst"], [date_str])
                if date_str not in sst_files:
                    continue

                # Check SSS
                sss_file = f"{self.sss_dir}/model_output_sss_{date_str}.nc"
                if not os.path.exists(sss_file):
                    continue

                # Check reconstruction file
                year = date_str[:4]
                recon_file = f"{self.recon_dir}/{year}/recon_{date_str}.nc"
                if not os.path.exists(recon_file):
                    continue

                valid.append(date_str)
            except Exception as e:
                continue

        return valid

    def __len__(self) -> int:
        return len(self.valid_dates)

    def _load_surface_data(self, date_str: str) -> np.ndarray:
        """Load surface observation data"""
        # Load SLA and velocity
        sla_files = find_files_by_keyword_and_dates(self.surface_dir, ["allsat"], [date_str])
        sla_file = sla_files[date_str][-1]
        ds_sla = xr.open_dataset(sla_file)

        sur_sla = ds_sla[['sla']].sel(
            longitude=slice(100, 160), latitude=slice(0, 50)
        ).to_array().values.reshape(-1, 400, 480)

        sur_var = ds_sla[['ugos', 'vgos']].sel(
            longitude=slice(100, 160), latitude=slice(0, 50)
        ).to_array().values.reshape(-1, 400, 480)

        # Load SST
        sst_files = find_files_by_keyword_and_dates(self.sst_dir, ["oisst"], [date_str])
        sst_file = sst_files[date_str][0]
        sur_sst = xr.open_dataset(sst_file)[["sst"]].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values.reshape(-1, 400, 480)

        # Load SSS
        sss_file = f"{self.sss_dir}/model_output_sss_{date_str}.nc"
        sur_so = xr.open_dataset(sss_file)[['sss_output']].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values.reshape(-1, 400, 480)

        # Concatenate
        surface_data = np.concatenate((sur_sla, sur_sst, sur_so, sur_var), axis=0)
        return surface_data.astype(np.float32)

    def _load_recon_data(self, date_str: str) -> np.ndarray:
        """Load reconstruction result data"""
        year = date_str[:4]
        recon_file = f"{self.recon_dir}/{year}/recon_{date_str}.nc"

        ds = xr.open_dataset(recon_file)
        depth_var = ds[self.deep_vars].sel(
            longitude=slice(100, 159.875), latitude=slice(0, 49.875)
        ).to_array().values

        # Reshape to (80, 400, 480)
        depth_var = depth_var.reshape(-1, 400, 480)
        return depth_var.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training sample"""
        date_str = self.valid_dates[idx]

        try:
            # Load surface data (5 channels)
            surface_data = self._load_surface_data(date_str)

            # Load reconstruction data as both background and target (80 channels)
            # In real training, background would be from 7 days ago
            # Here we use the same data for simplicity
            background_data = self._load_recon_data(date_str)
            target_data = self._load_recon_data(date_str)

            # Concatenate input: surface + background = 85 channels
            input_data = np.concatenate([surface_data, background_data], axis=0)

            # Create mask for valid data
            mask = ~np.isnan(target_data)

            # Normalize input
            input_data = (input_data - self.mean.reshape(85, 1, 1)) / self.std.reshape(85, 1, 1)

            # Handle NaN values
            input_data = np.nan_to_num(input_data, nan=0.0)
            target_data = np.nan_to_num(target_data, nan=0.0)

            return (
                torch.from_numpy(input_data).float(),
                torch.from_numpy(target_data).float(),
                torch.from_numpy(mask.astype(np.float32))
            )

        except Exception as e:
            print(f"Error loading data for date {date_str}: {e}")
            raise


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


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Quick Test Training')

    # Data paths
    parser.add_argument('--surface_dir', type=str, default='./input_data/SLA')
    parser.add_argument('--sst_dir', type=str, default='./input_data/SST')
    parser.add_argument('--sss_dir', type=str, default='./input_data/SSS')
    parser.add_argument('--recon_dir', type=str, default='./output_data/recon')
    parser.add_argument('--src_dir', type=str, default='./src')

    # Training dates (use available data)
    parser.add_argument('--train_start', type=str, default='2025-07-01')
    parser.add_argument('--train_end', type=str, default='2025-07-05')
    parser.add_argument('--val_start', type=str, default='2025-07-01')
    parser.add_argument('--val_end', type=str, default='2025-07-02')

    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Reduced for quick testing')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Set to 0 for debugging')

    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/test')
    parser.add_argument('--log_dir', type=str, default='./logs/test')
    parser.add_argument('--save_interval', type=int, default=2)

    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = Logger(args.log_dir, 'train_test.log')
    logger.log("=" * 80)
    logger.log("Quick Test Training")
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

    # Create datasets
    logger.log("Creating datasets...")
    train_dataset = QuickTestDataset(
        date_list=train_dates,
        surface_dir=args.surface_dir,
        sst_dir=args.sst_dir,
        sss_dir=args.sss_dir,
        recon_dir=args.recon_dir,
        mean=mean,
        std=std
    )

    val_dataset = QuickTestDataset(
        date_list=val_dates,
        surface_dir=args.surface_dir,
        sst_dir=args.sst_dir,
        sss_dir=args.sss_dir,
        recon_dir=args.recon_dir,
        mean=mean,
        std=std
    )

    logger.log(f"Training samples: {len(train_dataset)}")
    logger.log(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.log("ERROR: No valid training samples found!")
        logger.log("Please check that you have:")
        logger.log("  - SLA files in input_data/SLA/")
        logger.log("  - SST files in input_data/SST/")
        logger.log("  - SSS files in input_data/SSS/")
        logger.log("  - Reconstruction files in output_data/recon/YYYY/")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
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

    # Training loop
    logger.log("Starting training...")
    logger.log("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        logger.log(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.log(f"Learning rate: {get_lr(optimizer):.2e}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.log_metrics(epoch + 1, 'Train', train_metrics)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        logger.log_metrics(epoch + 1, 'Val', val_metrics)

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            logger.log(f"New best validation loss: {best_val_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0 or is_best:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'test_epoch{epoch + 1}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                checkpoint_path, is_best
            )
            logger.log(f"Checkpoint saved: {checkpoint_path}")

    # Export to ONNX
    logger.log("\nExporting model to ONNX...")
    onnx_path = os.path.join(args.checkpoint_dir, 'test_model.onnx')
    export_to_onnx(model, (1, 85, 400, 480), onnx_path, device)
    logger.log(f"Model exported to: {onnx_path}")

    logger.log("=" * 80)
    logger.log("Training completed!")
    logger.log(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
