import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D UNet，兼容现有 2D 数据接口。
    输入: (B, in_channels, H, W)  — 如 (B, 45, 400, 480)
    输出: (B, out_channels, H, W) — 如 (B, 40, 400, 480)

    内部将 20 层温盐背景场重组为 3D 体数据，表面变量广播到 20 层深度，
    经过 3D UNet 处理后再 reshape 回 2D 格式。

    Args:
        in_channels: 输入通道数（如 45 = 5 surface + 20 bg_t + 20 bg_s）
        out_channels: 输出通道数（如 40 = 20 label_t + 20 label_s）
        n_surface: 表面变量数量（默认 5）
        depth: 深度层数（默认 20）
        base_channels: 基础通道数（默认 32）
    """
    def __init__(self, in_channels, out_channels, n_surface=5, depth=20, base_channels=32):
        super().__init__()
        self.n_surface = n_surface
        self.depth = depth

        # 3D UNet 的输入通道: surface(广播) + bg_t + bg_s = n_surface + 2
        unet_in = n_surface + 2
        # 3D UNet 的输出通道: label_t + label_s = 2
        unet_out = 2

        # Encoder (2 levels)
        self.enc1 = DoubleConv3D(unet_in, base_channels)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)

        # 深度维度只有 20，池化两次后变成 5，用 (2,2,2) 没问题
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(base_channels * 2, base_channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)

        self.out_conv = nn.Conv3d(base_channels, unet_out, 1)

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        return: (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        D = self.depth
        ns = self.n_surface

        # 分离表面变量和 3D 背景场
        surface = x[:, :ns, :, :]                    # (B, ns, H, W)
        bg_t = x[:, ns:ns+D, :, :]                   # (B, 20, H, W)
        bg_s = x[:, ns+D:ns+2*D, :, :]               # (B, 20, H, W)

        # 表面变量广播到 D 层: (B, ns, H, W) -> (B, ns, D, H, W)
        surface_3d = surface.unsqueeze(2).expand(B, ns, D, H, W)

        # 背景场重组为 3D: (B, 20, H, W) -> (B, 1, D, H, W)
        bg_t_3d = bg_t.unsqueeze(1)                   # (B, 1, D, H, W)
        bg_s_3d = bg_s.unsqueeze(1)                   # (B, 1, D, H, W)

        # 拼接: (B, ns+2, D, H, W)
        x3d = torch.cat([surface_3d, bg_t_3d, bg_s_3d], dim=1)

        # 3D UNet 编码-解码
        e1 = self.enc1(x3d)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out_3d = self.out_conv(d1)  # (B, 2, D, H, W)

        # 拆分并 reshape 回 2D 格式
        out_t = out_3d[:, 0, :, :, :]  # (B, D, H, W)
        out_s = out_3d[:, 1, :, :, :]  # (B, D, H, W)

        # 拼接: (B, 40, H, W)
        return torch.cat([out_t, out_s], dim=1)
