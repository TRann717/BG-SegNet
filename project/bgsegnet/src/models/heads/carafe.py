"""
CARAFE: Content-Aware ReAssembly of FEatures.

Lightweight variant used only for the final upsampling stage in the decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..neck.ppm import make_gn


class CARAFEKernelPred(nn.Module):
    """Predict per-location upsampling kernels for CARAFE."""
    
    def __init__(self, in_channels, kernel_up=5, k_encoder=3, compress_rate=16):
        """
        Args:
            in_channels: number of input feature channels.
            kernel_up: spatial size of upsampling kernel (kernel_up x kernel_up).
            k_encoder: kernel size for content encoder convolution.
            compress_rate: channel compression ratio for encoder.
        """
        super().__init__()
        self.kernel_up = kernel_up
        self.compress_rate = compress_rate
        
        mid_channels = max(in_channels // compress_rate, 8)
        
        # Content encoder: extract local context to predict upsampling kernels.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, k_encoder, padding=k_encoder//2, bias=False),
            make_gn(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, kernel_up * kernel_up, 1)  # 预测每个位置的kernel
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize conv weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: input feature map (B, C, H, W).
        Returns:
            Upsampling kernels (B, kernel_up^2, H, W) normalized with softmax.
        """
        kernel_pred = self.encoder(x)
        kernel_pred = F.softmax(kernel_pred, dim=1)
        return kernel_pred


class CARAFEModule(nn.Module):
    """CARAFE upsampling module."""
    
    def __init__(self, in_channels, scale=2, kernel_up=5, k_encoder=3, compress_rate=16):
        """
        Args:
            in_channels: number of input feature channels.
            scale: upsampling factor.
            kernel_up: spatial kernel size for content-aware reassembly.
            k_encoder: encoder kernel size.
            compress_rate: channel compression ratio.
        """
        super().__init__()
        self.scale = scale
        self.kernel_up = kernel_up
        self.pad = kernel_up // 2
        
        # 核预测器
        self.kernel_pred = CARAFEKernelPred(
            in_channels, kernel_up, k_encoder, compress_rate
        )
    
    def forward(self, x):
        """
        Args:
            x: input feature map (B, C, H, W).
        Returns:
            Upsampled feature map (B, C, H*scale, W*scale).
        """
        B, C, H, W = x.shape
        H_up = H * self.scale
        W_up = W * self.scale
        
        kernel_pred = self.kernel_pred(x)  # (B, k^2, H, W)
        
        kernel_pred = F.interpolate(
            kernel_pred, size=(H_up, W_up), 
            mode='bilinear', align_corners=False
        )  # (B, k^2, H_up, W_up)
        
        x_pad = F.pad(x, [self.pad, self.pad, self.pad, self.pad], mode='reflect')
        
        x_unfold = F.unfold(
            x_pad, 
            kernel_size=self.kernel_up,
            dilation=1,
            stride=1,
            padding=0
        )  # (B, C*k^2, H*W)
        
        x_unfold = x_unfold.view(B, C, self.kernel_up*self.kernel_up, H, W)
        
        x_unfold_up = F.interpolate(
            x_unfold.view(B, -1, H, W),
            size=(H_up, W_up),
            mode='nearest'
        ).view(B, C, self.kernel_up*self.kernel_up, H_up, W_up)
        
        kernel_pred = kernel_pred.unsqueeze(1)
        output = (x_unfold_up * kernel_pred).sum(dim=2)  # (B, C, H_up, W_up)
        
        return output


class LightCARAFE(nn.Module):
    """Lightweight CARAFE wrapper used only for the last upsampling step."""
    
    def __init__(self, in_channels, scale=2):
        """
        Args:
            in_channels: number of input feature channels.
            scale: nominal upsampling factor.
        """
        super().__init__()
        
        self.carafe = CARAFEModule(
            in_channels=in_channels,
            scale=scale,
            kernel_up=5,      # 5x5上采样核
            k_encoder=3,      # 3x3编码器
            compress_rate=16  # 高压缩率
        )
    
    def forward(self, x, target_size=None):
        """
        Args:
            x: input feature map.
            target_size: target spatial size (H, W); if provided, overrides `scale`.
        Returns:
            Upsampled feature map.
        """
        if target_size is not None:
            H, W = x.shape[-2:]
            target_H, target_W = target_size
            
            scale_h = target_H / H
            scale_w = target_W / W
            
            if scale_h > 1.5 and scale_w > 1.5:
                x = self.carafe(x)  # 2x上采样
                
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:
            x = self.carafe(x)
        
        return x