"""
Lightweight boundary gating module for medical image segmentation backbones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..neck.ppm import make_gn
from .boundary_head import normalize_dir_field


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BoundaryGate(nn.Module):
    """
    Lightweight boundary gate.

    1x1 conv followed by sigmoid yields a spatial gate g, which is used to refine
    features via shallow convs. Supports depthwise separable convs, SE channel
    attention, orientation-aware basis kernels, and uncertainty-aware residual
    strength (alpha).
    """
    def __init__(self, in_channels, refine_channels=None, use_depthwise=True, use_se=True):
        """
        Args:
            in_channels: number of input feature channels.
            refine_channels: number of channels in refinement branch (defaults to in_channels).
            use_depthwise: whether to use depthwise separable convolutions.
            use_se: whether to enable SE channel attention.
        """
        super().__init__()
        
        # Gate projection: produce a single-channel gating map g(x,y) in [0,1].
        self.g_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        # Mild initial suppression: bias≈-0.5 → g≈sigmoid(-0.5)≈0.38; avoids over-suppressing early on.
        nn.init.constant_(self.g_proj.bias, -0.5)
        
        # Refinement network
        mid = in_channels if refine_channels is None else refine_channels
        
        if use_depthwise:
            # Depthwise-separable refinement: parameter-efficient.
            self.refine = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                make_gn(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, mid, 1, bias=False),
                make_gn(mid),
                nn.ReLU(inplace=True),
            )
        else:
            # Standard conv refinement.
            self.refine = nn.Sequential(
                nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
                make_gn(mid),
                nn.ReLU(inplace=True),
            )
        
        # Optional SE channel attention.
        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock(mid, reduction=16)
        
        # Optional output projection (if channel dimension changes).
        self.out_proj = nn.Conv2d(mid, in_channels, 1) if mid != in_channels else nn.Identity()
        
        # === OABG: orientation-aware depthwise kernels (4 basis directions) ===
        # Four depthwise 3x3 kernels as steerable bases: horizontal, vertical, and two diagonals.
        self.use_oabg = True
        self.dw_h = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.dw_v = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.dw_d1= nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.dw_d2= nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        for m in [self.dw_h, self.dw_v, self.dw_d1, self.dw_d2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.mix1x1 = nn.Sequential(nn.Conv2d(mid, mid, 1, bias=False), make_gn(mid), nn.ReLU(inplace=True))

        # === Uncertainty head for U-aware alpha (U-Aware α) ===
        self.u_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            make_gn(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1)
        )
        self.u_sigma = 1.0

        # === Class mixing branch (binary: background / foreground 1x1 mapping) ===
        self.class_mixing = True
        self.delta_bg = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.delta_fg = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        nn.init.zeros_(self.delta_bg.weight); nn.init.zeros_(self.delta_fg.weight)
        
    def forward(self, feats, alpha=0.2, g_ext=None, adaptive_alpha=False, 
                alpha_min=0.15, alpha_max=0.4, gamma_alpha=1.0, 
                u_map=None, dir_map=None, p_fg=None, use_oabg=True, 
                use_uaware=True, use_classmix=True):
        """
        Args:
            feats: input feature map (B, C, H, W).
            alpha: base residual refinement strength.
            g_ext: optional external gate map (B, 1, H, W) already in [0,1].
            adaptive_alpha: enable spatially adaptive alpha driven by gate strength.
            alpha_min: minimum spatial alpha.
            alpha_max: maximum spatial alpha.
            gamma_alpha: exponent controlling how strongly alpha concentrates on edges.
            u_map: optional uncertainty map (B, 1, H, W) to down-weight noisy regions.
            dir_map: orientation field (B, 2, H, W) for orientation-aware gating (OABG).
            p_fg: foreground probability (B, 1, H, W) used in class mixing.
            use_oabg: whether to apply orientation-aware depthwise kernels.
            use_uaware: whether to enable uncertainty-aware modulation.
            use_classmix: whether to enable class mixing refinement.
        Returns:
            Refined feature map (B, C, H, W).
        """
        if g_ext is None:
            g = torch.sigmoid(self.g_proj(feats))  # (B, 1, H, W)
        else:
            g = g_ext
        
        if adaptive_alpha:
            g_pow = torch.pow(g, gamma_alpha)  # (B, 1, H, W)
            alpha_spatial = alpha_min + (alpha_max - alpha_min) * g_pow
            # alpha_spatial形状: (B, 1, H, W)
        else:
            alpha_spatial = alpha
        
        gated = feats * g
        
        # Local refinement on gated features.
        delta = self.refine(gated)
        
        # Orientation-aware steerable combination of depthwise kernels.
        if use_oabg and dir_map is not None:
            d = normalize_dir_field(dir_map)
            tx, ty = d[:, 0:1], d[:, 1:2]  # approximate tangent direction
            w_h  = torch.abs(tx)
            w_v  = torch.abs(ty)
            w_d1 = torch.abs((tx + ty) * 0.7071)
            w_d2 = torch.abs((tx - ty) * 0.7071)
            w_sum = (w_h + w_v + w_d1 + w_d2).clamp_min(1e-6)
            w_h, w_v, w_d1, w_d2 = w_h / w_sum, w_v / w_sum, w_d1 / w_sum, w_d2 / w_sum
            y = ( self.dw_h(delta) * w_h + self.dw_v(delta) * w_v 
                + self.dw_d1(delta)* w_d1 + self.dw_d2(delta)* w_d2 )
            delta = self.mix1x1(y)
        
        if self.use_se:
            delta = self.se_block(delta)
        
        delta = self.out_proj(delta)
        
        if use_uaware:
            if u_map is None:
                u_map = torch.sigmoid(self.u_head(feats))
            damp = (1.0 - u_map).clamp(0, 1)
        else:
            damp = 1.0
        
        out = feats + (alpha_spatial * damp) * delta
        
        if use_classmix and (p_fg is not None):
            d_bg = self.delta_bg(out)
            d_fg = self.delta_fg(out)
            out = out + (1 - p_fg) * d_bg + p_fg * d_fg
        
        return out