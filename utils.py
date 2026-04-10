"""
Utility Functions for Training

This module provides utility functions for training, validation, logging,
and model conversion.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple
from pathlib import Path


def generate_date_list(start_date: str, end_date: str) -> List[str]:
    """
    Generate a list of dates between start_date and end_date.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        List[str]: List of dates in YYYYMMDD format
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return date_list


def load_normalization_stats(src_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load normalization statistics for input data.

    Args:
        src_dir (str): Directory containing normalization files

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and std arrays (85 channels each)
    """
    # Load surface data statistics (5 channels)
    surface_mean = np.load(os.path.join(src_dir, 'Satellite_cmems_mean.npy'))
    surface_std = np.load(os.path.join(src_dir, 'Satellite_cmems_std.npy'))

    # Load subsurface data statistics (80 channels, excluding SLA at index 0)
    glorys_mean = np.load(os.path.join(src_dir, 'glorys_all_channel_mean.npy'))
    glorys_std = np.load(os.path.join(src_dir, 'glorys_all_channel_std.npy'))

    # Combine: surface (5) + subsurface (80) = 85 channels
    input_mean = np.concatenate([surface_mean, glorys_mean[1:81]], axis=0)
    input_std = np.concatenate([surface_std, glorys_std[1:81]], axis=0)

    return input_mean, input_std


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """Track multiple metrics during training"""
    def __init__(self, metrics: List[str]):
        self.metrics = {metric: AverageMeter() for metric in metrics}

    def update(self, metric_dict: Dict[str, float], n: int = 1):
        for key, value in metric_dict.items():
            if key in self.metrics:
                self.metrics[key].update(value, n)

    def get_averages(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self.metrics.items()}

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        pred (torch.Tensor): Predictn        target (torch.Tensor): Target values
        mask (torch.Tensor): Valid data mask

    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Apply mask
    pred_masked = pred * mask
    target_masked = target * mask

    # MSE
    mse = torch.mean((pred_masked - target_masked) ** 2)

    # RMSE
    rmse = torch.sqrt(mse)

    # MAE
    mae = torch.mean(torch.abs(pred_masked - target_masked))

    # Correlation coefficient
    pred_flat = pred_masked.view(-1)
    target_flat = target_masked.view(-1)

    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)

    numerator = torch.sum((pred_flat - pred_mean) * (target_flat - target_mean))
    denominator = torch.sqrt(torch.sum((pred_flat - pred_mean) ** 2) * torch.sum((target_flat - target_mean) ** 2))

    corr = numerator / (denominator + 1e-8)

    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'corr': corr.item()
    }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score (float): Current validation score

        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint.

    Args:
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        save_path (str): Path to save checkpoint
        is_best (bool): Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cuda'
) -> Tuple[nn.Module, int, float]:
    """
    Load model checkpoint.

    Args:
        model (nn.Module): Model to load weights into
        checkpoint_path (str): Path to checkpoint file
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        device (str): Device to load model to

    Returns:
        Tuple[nn.Module, int, float]: Model, epoch, loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)

    return model, epoch, loss


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    save_path: str,
    device: str = 'cuda'
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model (nn.Module): Model to export
        input_shape (Tuple[int, ...]): Input tensor shape (e.g., (1, 85, 400, 480))
        save_path (str): Path to save ONNX model
        device (str): Device to use for export
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to ONNX: {save_path}")


class Logger:
    """Simple logger for training progress"""
    def __init__(self, log_dir: str, log_file: str = 'training.log'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("=" * 80 + "\n")

    def log(self, message: str, print_console: bool = True):
        """Log a message to file and optionally print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

        if print_console:
            print(log_message)

    def log_metrics(self, epoch: int, phase: str, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        message = f"Epoch {epoch} - {phase}: "
        message += ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.log(message)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Test utility functions
    print("Testing utility functions...")

    # Test date generation
    dates = generate_date_list("2025-01-01", "2025-01-10")
    print(f"Generated dates: {dates}")

    # Test metric tracker
    tracker = MetricTracker(['loss', 'rmse', 'mae'])
    tracker.update({'loss': 0.5, 'rmse': 0.3, 'mae': 0.2})
    tracker.update({'loss': 0.4, 'rmse': 0.25, 'mae': 0.18})
    print(f"Average metrics: {tracker.get_averages()}")
