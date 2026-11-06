import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from timm.models import register_model
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.Module):
    """
        LayerNorm that supports two data formats: channels_last (default) or channels_first.
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs
        with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt_Block(nn.Module):
    """
        ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtUNet(nn.Module):
    """AR 训练用四变量（temp/salt/uo/vo）版本，编码器结构与 simple_convnext_net_v3 一致，去掉风场/波浪输出头。

    out_channels 必须能被 4 整除（对应 temp/salt/uo/vo 四个变量各占 out_channels // 4 个深度通道）。
    """

    def __init__(self, in_channels=24, out_channels=24):
        super().__init__()
        assert out_channels % 4 == 0, f"out_channels 必须是 4 的倍数（temp/salt/uo/vo），当前 {out_channels}"
        n_depth = out_channels // 4

        # 通道链: in -> 160 -> 320 -> 640(bottleneck) -> 320 -> 160 -> out
        # stem 用 2x2 stride=2 卷积在最开始就把特征图下采样一半（要求 H, W 为偶数），减少后续所有 stage 的计算量
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 160, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(160, 160, kernel_size=2, stride=2),
        )

        # 编码器
        self.stage1 = nn.Sequential(*[ConvNeXt_Block(160) for _ in range(3)])
        self.down1 = nn.Conv2d(160, 320, kernel_size=3, stride=2, padding=1)

        self.stage2 = nn.Sequential(*[ConvNeXt_Block(320) for _ in range(3)])
        self.down2 = nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.stage3 = nn.Sequential(*[ConvNeXt_Block(640) for _ in range(12)])

        # 解码器
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(640, 320, kernel_size=3, padding=1)
        )
        self.fusion1 = nn.Conv2d(640, 320, kernel_size=1)   # cat(320, 320) -> 320
        self.stage4 = nn.Sequential(*[ConvNeXt_Block(320) for _ in range(3)])

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(320, 160, kernel_size=3, padding=1)
        )
        self.fusion2 = nn.Conv2d(320, 160, kernel_size=1)    # cat(160, 160) -> 160
        self.stage5 = nn.Sequential(*[ConvNeXt_Block(160) for _ in range(3)])

        # stem 下采样了 2 倍，这里上采样回原分辨率，保证输出能和输入 bg 做残差相加
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(160, 160, kernel_size=3, padding=1)
        )

        # 分层输出头（温/盐/流速 uo/vo 分离，两层增加任务特定非线性）
        self.head_temp = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(160, n_depth, kernel_size=1)
        )
        self.head_salt = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(160, n_depth, kernel_size=1)
        )
        self.head_uo = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(160, n_depth, kernel_size=1)
        )
        self.head_vo = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(160, n_depth, kernel_size=1)
        )

    def forward(self, x):
        x = self.stem(x)                                  # (B, 160, H/2, W/2)

        s1 = self.stage1(x)                               # (B, 160, H/2, W/2)
        x  = self.down1(s1)                                # (B, 320, H/4, W/4)

        s2 = self.stage2(x)                                # (B, 320, H/4, W/4)
        x  = self.down2(s2)                                # (B, 640, H/8, W/8)

        x = self.stage3(x)                                 # (B, 640, H/8, W/8)

        x = self.up1(x)                                    # (B, 320, H/4, W/4)
        x = self.fusion1(torch.cat([x, s2], dim=1))        # (B, 320, H/4, W/4)
        x = self.stage4(x)

        x = self.up2(x)                                    # (B, 160, H/2, W/2)
        x = self.fusion2(torch.cat([x, s1], dim=1))        # (B, 160, H/2, W/2)
        x = self.stage5(x)

        x = self.up_final(x)                               # (B, 160, H, W)

        temp = self.head_temp(x)
        salt = self.head_salt(x)
        uo   = self.head_uo(x)
        vo   = self.head_vo(x)
        return torch.cat([temp, salt, uo, vo], dim=1)  # (B, out_channels, H, W)
