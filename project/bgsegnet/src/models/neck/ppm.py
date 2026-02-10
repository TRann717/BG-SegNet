"""
Neck components: Pyramid Pooling Module (PPM) and UPerNet-style FPN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(num_channels: int, num_groups: int = 32):
    """Create GroupNorm with a group count that divides num_channels."""
    g = min(num_groups, num_channels)
    for gg in reversed(range(1, g+1)):
        if num_channels % gg == 0:
            return nn.GroupNorm(gg, num_channels)
    # Fallback: instance norm when no divisor is found.
    return nn.GroupNorm(1, num_channels)


class PPM(nn.Module):
    """Pyramid Pooling Module for multi-scale context aggregation."""
    
    def __init__(self, in_channels, bins=(1, 2, 3, 6), out_channels=256):
        """
        Args:
            in_channels: number of input feature channels.
            bins: list of pooling bin sizes.
            out_channels: number of output channels.
        """
        super().__init__()
        self.bins = bins
        self.stages = nn.ModuleList()
        
        for bin_size in bins:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, out_channels // len(bins), 1, bias=False),
                make_gn(out_channels // len(bins)),
                nn.ReLU(inplace=True)
            ))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            make_gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
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
    
    def forward(self, x):
        """
        Args:
            x: input feature map (B, C, H, W).
        Returns:
            fused feature map (B, out_channels, H, W).
        """
        h, w = x.size()[2:]
        
        priors = [x]
        for stage in self.stages:
            prior = stage(x)
            prior = F.interpolate(prior, size=(h, w), mode='bilinear', align_corners=False)
            priors.append(prior)
        
        out = torch.cat(priors, dim=1)
        out = self.bottleneck(out)
        
        return out


class UPerNet(nn.Module):
    """
    UPerNet neck: PPM on top of backbone + FPN-style top-down pathway.

    Final feature is obtained by concatenating p4, upsampled p8, upsampled p16,
    and upsampled PPM output, followed by a fusion conv.
    """
    def __init__(self, in_channels_dict, out_channels=256, ppm_bins=(1,2,3,6)):
        super().__init__()
        C4 = in_channels_dict["F4"]; C8 = in_channels_dict["F8"]; C16 = in_channels_dict["F16"]
        self.ppm = PPM(in_channels=C16, bins=ppm_bins, out_channels=out_channels)

        # lateral 1x1
        self.lat4 = nn.Conv2d(C4, out_channels, 1, bias=False)
        self.lat8 = nn.Conv2d(C8, out_channels, 1, bias=False)
        self.lat16= nn.Conv2d(C16, out_channels, 1, bias=False)

        # smooth 3x3
        self.smooth4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                     make_gn(out_channels), nn.ReLU(inplace=True))
        self.smooth8 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                     make_gn(out_channels), nn.ReLU(inplace=True))
        self.smooth16= nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                     make_gn(out_channels), nn.ReLU(inplace=True))

        # final fusion after concat [p2, p3↑, p4↑, ppm↑]
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 3, padding=1, bias=False),
            make_gn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, return_pyramid=False):
        f4, f8, f16 = feats["F4"], feats["F8"], feats["F16"]
        p4 = self.lat4(f4)                 # 1/4
        p8 = self.lat8(f8)                 # 1/8
        p16= self.lat16(f16)               # 1/16
        ppm= self.ppm(f16)                 # 1/16

        # top-down
        p8  = p8  + F.interpolate(p16, size=p8.shape[-2:],  mode='bilinear', align_corners=False)
        p4  = p4  + F.interpolate(p8,  size=p4.shape[-2:],  mode='bilinear', align_corners=False)
        p16 = self.smooth16(p16); p8 = self.smooth8(p8); p4 = self.smooth4(p4)

        # upsample to 1/4
        p8_up   = F.interpolate(p8,  size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p16_up  = F.interpolate(p16, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        ppm_up  = F.interpolate(ppm, size=p4.shape[-2:], mode='bilinear', align_corners=False)

        out = torch.cat([p4, p8_up, p16_up, ppm_up], dim=1)
        out = self.fuse(out)               # (B, out_channels, H/4, W/4)
        
        if not return_pyramid:
            return out
        
        # 返回平滑后的 p4/p8/p16 与 PPM，用于多尺度边界
        return {
            "out": out,    # 1/4 fused
            "p4": p4,      # 1/4  
            "p8": p8,      # 1/8
            "p16": p16,    # 1/16
            "ppm": ppm     # 1/16
        }