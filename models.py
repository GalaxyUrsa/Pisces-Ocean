"""
Ocean Reconstruction and Forecast Model Architectures

This module implements the neural network architectures for ocean state reconstruction
and forecasting, inspired by ConvIR (Convolutional Image Restoration) framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block with normalization and activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolutional kernel
        stride (int): Stride of convolution
        padding (int): Padding size
        use_bn (bool): Whether to use batch normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Args:
        channels (int): Number of channels
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling block for encoder.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block for decoder.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.up(x)))
        x = self.res(x)
        return x


class ReconstructionModel(nn.Module):
    """
    Ocean Reconstruction Model: 85 channels -> 80 channels

    Input: Surface observations (5 channels) + Background subsurface (80 channels)
    Output: Reconstructed 3D ocean state (80 channels: 4 variables × 20 depth levels)

    Args:
        in_channels (int): Number of input channels (default: 85)
        out_channels (int): Number of output channels (default: 80)
        base_channels (int): Base number of channels for the network (default: 64)
    """
    def __init__(self, in_channels=85, out_channels=80, base_channels=64):
        super(ReconstructionModel, self).__init__()

        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels)

        # Encoder
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)      # 400x480 -> 200x240
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)  # 200x240 -> 100x120
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8)  # 100x120 -> 50x60

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )

        # Decoder
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)      # 50x60 -> 100x120
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)      # 100x120 -> 200x240
        self.up3 = UpsampleBlock(base_channels * 2, base_channels)          # 200x240 -> 400x480

        # Output convolution
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (B, 85, 400, 480)

        Returns:
            torch.Tensor: Output tensor of shape (B, 80, 400, 480)
        """
        # Initial feature extraction
        x = self.init_conv(x)

        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Bottleneck
        x = self.bottleneck(x3)

        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Output
        out = self.out_conv(x)

        return out


class ForecastModel(nn.Module):
    """
    Ocean Forecast Model: 85 channels -> 81 channels

    Input: Surface observations (5 channels) + Current subsurface state (80 channels)
    Output: Forecasted ocean state (81 channels: SLA + 4 variables × 20 depth levels)

    Args:
        in_channels (int): Number of input channels (default: 85)
        out_channels (int): Number of output channels (default: 81)
        base_channels (int): Base number of channels for the network (default: 64)
    """
    def __init__(self, in_channels=85, out_channels=81, base_channels=64):
        super(ForecastModel, self).__init__()

        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels)

        # Encoder
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )

        # Decoder
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpsampleBlock(base_channels * 2, base_channels)

        # Output convolution
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (B, 85, 400, 480)

        Returns:
            torch.Tensor: Output tensor of shape (B, 81, 400, 480)
        """
        # Initial feature extraction
        x = self.init_conv(x)

        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Bottleneck
        x = self.bottleneck(x3)

        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Output
        out = self.out_conv(x)

        return out


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing Reconstruction Model...")
    recon_model = ReconstructionModel()
    dummy_input = torch.randn(2, 85, 400, 480)
    output = recon_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(recon_model):,}")

    print("\nTesting Forecast Model...")
    forecast_model = ForecastModel()
    output = forecast_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(forecast_model):,}")