"""
Decoder heads for semantic segmentation (main decoder and auxiliary deep-supervision head).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..neck.ppm import make_gn


class DecoderHead(nn.Module):
    """Main segmentation decoder head."""
    
    class GuidedUp(nn.Module):
        def __init__(self, k: int = 3):
            super().__init__()
            self.k = k
            self.pad = k // 2
            # Predict per-pixel k×k spatial weights from a single-channel guide map.
            self.kernel_pred = nn.Conv2d(1, k * k, 3, padding=1, bias=True)
        
        def forward(self, logits: torch.Tensor, guide: torch.Tensor):
            # logits: (B,C,H,W), guide: (B,1,H,W) ∈ [0,1]
            B, C, H, W = logits.shape
            K2 = self.k * self.k
            w = self.kernel_pred(guide)                    # (B,K2,H,W)
            w = w.view(B, K2, -1)
            w = torch.softmax(w, dim=1).view(B, K2, H, W)  

            x_pad = F.pad(logits, [self.pad, self.pad, self.pad, self.pad], mode='reflect')
            x_unf = F.unfold(x_pad, kernel_size=self.k, stride=1)  # (B, C*K2, H*W)
            x_unf = x_unf.view(B, C, K2, H, W)
            y = (x_unf * w.unsqueeze(1)).sum(dim=2)                # (B, C, H, W)
            return y
    
    def __init__(self, in_channels, num_classes, dropout=0.1, up_type="bilinear"):
        """
        Args:
            in_channels: number of input feature channels.
            num_classes: number of semantic classes.
            dropout: dropout rate in decoder features.
            up_type: upsampling type ("bilinear" | "carafe").
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.up_type = up_type
        
        mid = in_channels // 2
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            make_gn(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        if up_type == "carafe":
            from .carafe import LightCARAFE
            self.up2a = LightCARAFE(mid, scale=2)  # 1/4 → 1/2
            self.up2b = LightCARAFE(mid, scale=2)  # 1/2 → 1×
        else:
            self.up2a = self.up2b = None
        
        self.classifier = nn.Conv2d(mid, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def init_bias_with_priors(self, priors):
        """
        Initialize final classifier bias using class priors.
        Args:
            priors: list of per-class pixel frequencies (length = num_classes, sum ≈ 1).
        """
        if priors is None:
            return
        assert len(priors) == self.num_classes, f"Priors length {len(priors)} != num_classes {self.num_classes}"
        
        pi = np.clip(np.array(priors, dtype=np.float32), 1e-6, 1.0)
        logpi = np.log(pi)
        logpi = logpi - logpi.mean()
        
        with torch.no_grad():
            last_conv = self.classifier
            assert isinstance(last_conv, nn.Conv2d) and last_conv.out_channels == self.num_classes
            last_conv.bias.copy_(torch.from_numpy(logpi).to(last_conv.bias.device))
    def forward(self, x, out_size=None):
        """
        Args:
            x: input feature map (B, C, H, W).
            out_size: target output size (H, W).
        Returns:
            Segmentation logits (B, num_classes, H_out, W_out).
        """
        x = self.feat(x)                              # (B, mid, H/4, W/4)
        
        if self.up_type == "carafe":
            x = self.up2a(x)                          # (B, mid, H/2, W/2)
            x = self.up2b(x)                          # (B, mid, H, W)
        else:
            if out_size is not None:
                x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        
        if out_size is not None and x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        
        logits = self.classifier(x)                   # (B, C, H, W)
        return logits


class AuxHead(nn.Module):
    """Auxiliary segmentation head for deep supervision."""
    
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels: number of input feature channels.
            num_classes: number of semantic classes.
        """
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            make_gn(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, out_size=None):
        """
        Args:
            x: input feature map (B, C, H, W).
            out_size: output size (H, W).
        Returns:
            Segmentation logits (B, num_classes, H_out, W_out).
        """
        x = self.conv(x)
        
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        
        return x
