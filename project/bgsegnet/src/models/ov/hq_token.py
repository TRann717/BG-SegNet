# src/models/ov/hq_token.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HQTokenAdapter(nn.Module):
    """
    Lightweight HQ-token branch: takes selected decoder features and outputs
    a single high-quality lesion mask.

    Can be attached after the main decoder as an additional 1×1 conv + upsample head.
    """
    def __init__(self, in_channels:int, mid:int=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1)
        )

    def forward(self, feats: torch.Tensor, out_size_hw: tuple[int,int]):
        """
        Args:
            feats: decoder feature map (B,C,Hs,Ws).
            out_size_hw: target spatial size (H,W).
        Returns:
            mask_logits: high-quality mask logits (B,1,H,W).
        """
        logits = self.conv(feats)  # (B,1,Hs,Ws)
        mask_logits = F.interpolate(logits, size=out_size_hw, mode="bilinear", align_corners=False)
        return mask_logits
