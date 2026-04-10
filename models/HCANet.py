import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import os
sys.path.append(os.getcwd())

# m_seed = 1
# # 设置seed
# torch.manual_seed(m_seed)
# torch.cuda.manual_seed_all(m_seed)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Multi-Scale Feed-Forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*3, kernel_size=(1,1,1), bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
        # self.dwconv2 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        # self.dwconv3 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)


        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1,1,1), bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        # x1 = self.dwconv1(x1)
        # x2 = self.dwconv2(x2)
        # x3 = self.dwconv3(x3)
        x = F.gelu(x1)*x2*x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2)      
        return x



##########################################################################
## Convolution and Attention Fusion Module  (CAFM)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)

        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)


        # global SA
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output =  out + out_conv

        return output



##########################################################################
## CAMixing Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=31, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=(3,3,3), stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.proj(x)
        x = x.squeeze(2)
        return x



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        # x = x.unsqueeze(2)
        x = self.body(x)
        # x = x.squeeze(2)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        # x = x.unsqueeze(2)
        x = self.body(x)
        # x = x.squeeze(2)
        return x

##########################################################################
##---------- HCANet -----------------------
class HCANet(nn.Module):
    def __init__(self, in_channels=31, out_channels=31, base_channels=48,
                 num_blocks=[2,3,3], num_refinement_blocks=1, heads=[1,2,4],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        c = base_channels

        self.patch_embed = OverlapPatchEmbed(in_channels, c)

        # Encoder
        self.enc1  = nn.Sequential(*[TransformerBlock(c,   heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])
        self.pool1 = Downsample(c)
        self.enc2  = nn.Sequential(*[TransformerBlock(c*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])])
        self.pool2 = Downsample(c*2)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[TransformerBlock(c*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[2])])

        # Decoder
        self.up2     = Upsample(c*4)
        self.reduce2 = nn.Conv3d(c*4, c*2, kernel_size=(1,1,1), bias=bias)
        self.dec2    = nn.Sequential(*[TransformerBlock(c*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])])

        self.up1  = Upsample(c*2)
        self.dec1 = nn.Sequential(*[TransformerBlock(c*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(c*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_refinement_blocks)])

        self.out_conv = nn.Conv3d(c*2, out_channels, kernel_size=(3,3,3), stride=1, padding=1, bias=bias)
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(self.patch_embed(x))
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder with skip connections
        d2 = self.reduce2(torch.cat([self.up2(b), e2], dim=1).unsqueeze(2)).squeeze(2)
        d2 = self.dec2(d2)

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.refinement(d1)
        return self.out_conv(out.unsqueeze(2)).squeeze(2) + self.skip_proj(x)

if __name__ == "__main__":
    model = HCANet()
    inputs = torch.ones([2, 31, 128, 128])
    outputs = model(inputs)
    print(outputs.size())