# src/models/pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnMaskPooling(nn.Module):
    """
    Attention-based mask pooling:
      r_i = sum_{(x,y) in mask} softmax(phi(F_xy)) * F_xy
    """
    def __init__(self, in_channels: int, temperature: float = 1.0):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.temperature = temperature

    def forward(self, feats: torch.Tensor, mask_down: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, C, Hs, Ws)
        mask_down: (B, 1, Hs, Ws) in {0,1}
        returns:
          region_vecs: (B, C)
        """
        B, C, Hs, Ws = feats.shape
        mask = (mask_down > 0.5).float()
        # avoid empty mask
        eps = 1e-6
        masked = feats * mask
        logits = self.score(feats) / max(self.temperature, 1e-6)  # (B,1,Hs,Ws)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(logits.view(B, 1, -1), dim=-1).view(B, 1, Hs, Ws)  # (B,1,Hs,Ws)
        region = (attn * feats).sum(dim=[2, 3])  # (B, C)
        # Normalize by sum of attn on mask to be safe
        denom = (attn * mask).sum(dim=[2,3]).clamp_min(eps)  # (B,1)
        region = region / denom
        return region  # (B,C)

def downsample_mask(mask: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """
    mask: (B,1,H,W) -> bilinear to (B,1,Hs,Ws), keep binary by thresholding
    """
    mask_down = F.interpolate(mask.float(), size=size_hw, mode="bilinear", align_corners=False)
    return (mask_down > 0.5).float()
