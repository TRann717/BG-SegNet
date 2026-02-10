"""
Boundary detection head and soft boundary generators for medical image segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from ..neck.ppm import make_gn


class BoundaryHead(nn.Module):
    """Boundary detection head with optional class-aware and direction branches."""
    
    def __init__(self, in_channels, num_classes=1, with_dir=False):
        """
        Args:
            in_channels: number of input feature channels.
            num_classes: number of output classes (1 for binary boundary, >1 for class-aware boundaries).
            with_dir: whether to predict an auxiliary boundary direction field.
        """
        super().__init__()
        self.num_classes = num_classes
        self.with_dir = with_dir
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            make_gn(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            make_gn(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
        if with_dir:
            # Direction field head: 2D unit vector field used with cosine similarity losses.
            self.dir_head = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                make_gn(32), 
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1)
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
            out_size: optional output spatial size (H, W).
        Returns:
            dict with:
              'b_logits': (B, num_classes|1, H_out, W_out) boundary logits
              'dir': (B, 2, H_out, W_out) optional direction field
        """
        b = self.conv(x)
        d = self.dir_head(x) if self.with_dir else None
        
        if out_size is not None:
            b = F.interpolate(b, size=out_size, mode='bilinear', align_corners=False)
            if d is not None:
                d = F.interpolate(d, size=out_size, mode='bilinear', align_corners=False)
        
        return {'b_logits': b, 'dir': d}


def normalize_dir_field(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize a 2D direction field to unit vectors.

    Args:
        d: (B, 2, H, W) direction field or None.
    """
    if d is None:
        return None
    dx, dy = d[:, 0:1], d[:, 1:2]
    norm = torch.sqrt(dx * dx + dy * dy + eps)
    return torch.cat([dx / norm, dy / norm], dim=1)


def generate_boundary_from_mask(mask: torch.Tensor, kernel_size:int=3, num_classes:int=None):
    """
    Generate a binary boundary map from a label mask using a morphological gradient.

    Supports multi-class masks by computing per-class boundaries and taking the union.
    Implemented on GPU via max-pooling approximations of dilation/erosion.

    Args:
        mask: (B,H,W) long tensor of class indices.
        kernel_size: size of the structuring element.
        num_classes: number of classes (if None, inferred from mask).
    Returns:
        (B,1,H,W) binary boundary map.
    """
    if mask.dim()==2:
        mask = mask.unsqueeze(0)
    B,H,W = mask.shape
    if num_classes is None:
        C = int(mask.max().item()+1)
    else:
        C = num_classes

    k = kernel_size
    pad = (k-1)//2
    boundary = torch.zeros((B,1,H,W), device=mask.device, dtype=torch.float32)

    for c in range(1, C):  # 跳过背景
        fg = (mask==c).float().unsqueeze(1)  # (B,1,H,W)
        dil = F.max_pool2d(fg, kernel_size=k, stride=1, padding=pad)
        ero = -F.max_pool2d(-fg, kernel_size=k, stride=1, padding=pad)
        bnd = (dil - ero).clamp_min(0.0)     # (B,1,H,W)
        boundary = torch.maximum(boundary, bnd)
    return boundary


def _single_boundary(mask, kernel_size):
    """Generate a binary boundary map for a single 2D label mask."""
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 腐蚀操作
    eroded = cv2.erode(mask, kernel, iterations=1)
    # 边界 = 原图 - 腐蚀图
    boundary = mask - eroded
    
    return torch.from_numpy(boundary.astype(np.float32))


def generate_soft_boundary_from_mask(
    mask: torch.Tensor,
    kernel_size: int = 3,
    num_classes: int = None,
    band_pixels: int = 6,
    sigma: float = 3.0
):
    """
    Soft boundary band via distance transform on the union of class-wise edges.

      1) compute a stable hard boundary using a morphological gradient
      2) run distance transform from every non-boundary pixel to nearest boundary
      3) soft(x) = exp(- d(x)^2 / (2 σ^2)), truncated to 0 beyond `band_pixels`

    Returns:
        (B,1,H,W) float32 ∈ [0,1] soft boundary weights.
    """
    dev = mask.device
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    B, H, W = mask.shape
    if num_classes is None:
        C = int(mask.max().item() + 1)
    else:
        C = num_classes
    
    out = []
    k = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    
    for b in range(B):
        m = mask[b].detach().cpu().numpy().astype(np.uint8)  # (H,W)
        # 多类边界并集
        boundary = np.zeros((H, W), np.uint8)
        for c in range(1, C):
            fg = (m == c).astype(np.uint8)
            grad = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, kernel)
            boundary = np.maximum(boundary, grad)
        
        # 若整幅图没有边界，直接给出零权重，避免 distanceTransform 异常尺度
        if boundary.sum() == 0:
            soft = np.zeros((H, W), dtype=np.float32)
            out.append(torch.from_numpy(soft).view(1, H, W))
            continue
        
        # 距离变换：到最近边界（边界像素为0）的欧氏距离
        inv = (boundary == 0).astype(np.uint8)
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float64)
        
        # 先记录"带宽外"掩码，再对距离做裁剪，避免 (大数)**2 溢出
        if band_pixels is not None and band_pixels > 0:
            far_mask = dist > float(band_pixels)
            # 裁剪距离后再平方，数值稳定
            np.minimum(dist, float(band_pixels), out=dist)
        else:
            far_mask = None
        
        denom = 2.0 * (float(sigma) ** 2) + 1e-12
        # 用 float64 做平方/指数，最后转回 float32
        soft = np.exp(-np.square(dist, dtype=np.float64) / denom).astype(np.float32, copy=False)
        if far_mask is not None:
            soft[far_mask] = 0.0
        
        out.append(torch.from_numpy(soft).view(1, H, W))
    
    out = torch.stack(out, dim=0).to(dev).float()  # (B,1,H,W)
    return out


def generate_soft_boundary_per_class(
    mask: torch.Tensor,
    num_classes: int,
    kernel_size: int = 3,
    band_pixels: int = 6,
    sigma: float = 3.0,
):
    """
    Class-aware soft boundary bands S_c(x) with numerically stable distance transforms.

    Args:
        mask: (B, H, W) segmentation mask.
        num_classes: number of semantic classes.
        kernel_size: size of morphological kernel for gradients.
        band_pixels: maximum band width in pixels.
        sigma: Gaussian decay parameter for distance weighting.
    Returns:
        (B, C, H, W) per-class soft boundary maps.
    """
    dev = mask.device
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    B, H, W = mask.shape
    C = num_classes
    out = []
    k = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    
    for b in range(B):
        m = mask[b].detach().cpu().numpy().astype(np.uint8)  # (H,W)
        sc = []
        for c in range(C):
            if c == 0:
                # 背景边界可选：通常不监督；给0面罩
                sc.append(np.zeros((H, W), dtype=np.float32))
                continue
                
            fg = (m == c).astype(np.uint8)
            grad = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, kernel)
            
            if grad.sum() == 0:
                sc.append(np.zeros((H, W), dtype=np.float32))
                continue
                
            inv = (grad == 0).astype(np.uint8)
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float64)
            
            if band_pixels is not None and band_pixels > 0:
                np.minimum(dist, float(band_pixels), out=dist)
                
            denom = 2.0 * (float(sigma) ** 2) + 1e-12
            soft = np.exp(-np.square(dist, dtype=np.float64) / denom).astype(np.float32, copy=False)
            
            if band_pixels and band_pixels > 0:
                soft[dist >= float(band_pixels) - 1e-6] = 0.0
                
            sc.append(soft)
            
        out.append(np.stack(sc, axis=0))  # (C,H,W)
        
    out = torch.from_numpy(np.stack(out, axis=0)).to(dev).float()  # (B,C,H,W)
    return out