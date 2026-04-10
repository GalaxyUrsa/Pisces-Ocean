import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from timm.models import register_model

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
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
    r""" ConvNeXt Block. There are two equivalent implementations:
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
    def __init__(self, in_channels=45, out_channels=40):
        super().__init__()
        # 输入投影
        self.proj_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # 编码器（下采样路径）
        self.stage1 = nn.Sequential(*[ConvNeXt_Block(64) for _ in range(1)])
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 480->240
        
        self.stage2 = nn.Sequential(*[ConvNeXt_Block(128) for _ in range(1)])
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 240->120
        
        # Bottleneck（大尺度背景场）
        self.stage3 = nn.Sequential(*[ConvNeXt_Block(256) for _ in range(3)])  # 建议改为6而非9
        
        # 解码器（上采样路径）
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1)
        )
        # 关键：跳跃连接融合通道数 = 上采样后(128) + stage2输出(128) = 256
        self.fusion1 = nn.Conv2d(256, 128, kernel_size=1)  # 压缩通道
        self.stage4 = nn.Sequential(*[ConvNeXt_Block(128) for _ in range(1)])
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )
        self.fusion2 = nn.Conv2d(128, 64, kernel_size=1)  # 64+64=128 -> 64
        self.stage5 = nn.Sequential(*[ConvNeXt_Block(64) for _ in range(1)])
        
        # 分层输出头（温盐分离）
        self.head_temp = nn.Conv2d(64, 20, kernel_size=3, padding=1)
        self.head_salt = nn.Conv2d(64, 20, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (B, 45, 480, 400)
        x = self.proj_in(x)  # (B, 64, 480, 400)
        
        # 编码
        s1 = self.stage1(x)  # 保存用于跳跃连接
        x = self.down1(s1)   # (B, 128, 240, 200)
        
        s2 = self.stage2(x)  # 保存
        x = self.down2(s2)   # (B, 256, 120, 100)
        
        x = self.stage3(x)   # (B, 256, 120, 100)
        
        # 解码 + 跳跃连接
        x = self.up1(x)      # (B, 128, 240, 200)
        x = torch.cat([x, s2], dim=1)  # 跳跃连接：(B, 256, 240, 200)
        x = self.fusion1(x)  # (B, 128, 240, 200)
        x = self.stage4(x)
        
        x = self.up2(x)      # (B, 64, 480, 400)
        x = torch.cat([x, s1], dim=1)  # 跳跃连接：(B, 128, 480, 400)
        x = self.fusion2(x)  # (B, 64, 480, 400)
        x = self.stage5(x)
        
        # 输出
        temp = self.head_temp(x)
        salt = self.head_salt(x)
        return torch.cat([temp, salt], dim=1)  # (B, 40, 480, 400)