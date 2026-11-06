"""二维 FNO：深度和物理变量作为通道，输出归一化后的预测残差。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """在二维频域中混合通道，仅保留低频模态。"""

    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        # 用实数存储复权重的实部/虚部，兼容 GradScaler 和梯度裁剪。
        shape = (in_channels, out_channels, modes, modes, 2)
        scale = 1.0 / in_channels
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x):
        batch, _, height, width = x.shape
        # CUDA 的半精度 FFT 对尺寸有限制；频域分支固定使用 FP32。
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm='ortho')
            output = spectrum.new_zeros(
                batch, self.out_channels, height, width // 2 + 1
            )
            modes_w = min(self.modes, width // 2 + 1)
            # 正负频率在小网格上也不重叠；height=1 时只有正频率。
            modes_pos = min(self.modes, (height + 1) // 2)
            modes_neg = min(self.modes, height // 2)
            weight = torch.view_as_complex(self.weight_pos.float())
            output[:, :, :modes_pos, :modes_w] = torch.einsum(
                'bihw,iohw->bohw',
                spectrum[:, :, :modes_pos, :modes_w],
                weight[:, :, :modes_pos, :modes_w],
            )
            if modes_neg:
                weight = torch.view_as_complex(self.weight_neg.float())
                output[:, :, -modes_neg:, :modes_w] = torch.einsum(
                    'bihw,iohw->bohw',
                    spectrum[:, :, -modes_neg:, :modes_w],
                    weight[:, :, :modes_neg, :modes_w],
                )
            output = torch.fft.irfft2(output, s=(height, width), norm='ortho')
        return output.to(dtype=x.dtype)


class FNONet(nn.Module):
    """(B, in_channels, H, W) -> (B, out_channels, H, W)。

    四层频域卷积与逐点线性分支；不在模型内部叠加背景场。
    默认隐藏通道为 113，每个空间方向保留 16 个低频模态。
    80 输入/输出通道时有 52,387,643 个参数。
    坐标通道描述相对位置，边界补零缓解 FFT 的周期拼接影响。
    """

    def __init__(self, in_channels=24, out_channels=24, width=113, modes=16):
        super().__init__()
        self.padding = 8
        self.stem = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes) for _ in range(4)
        ])
        self.local_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x):
        batch, _, height, width = x.shape
        rows = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        cols = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        rows = rows.view(1, 1, height, 1).expand(batch, 1, height, width)
        cols = cols.view(1, 1, 1, width).expand(batch, 1, height, width)
        x = self.stem(torch.cat([x, rows, cols], dim=1))
        x = F.pad(x, (0, self.padding, 0, self.padding))
        for index, (spectral, local) in enumerate(zip(
                self.spectral_layers, self.local_layers)):
            x = spectral(x) + local(x)
            if index < len(self.spectral_layers) - 1:
                x = F.gelu(x)
        return self.head(x[:, :, :height, :width])
