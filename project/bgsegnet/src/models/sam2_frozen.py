# src/models/sam2_frozen.py
import torch
import torch.nn as nn

class SAM2Frozen(nn.Module):
    """
    Thin wrapper that exposes a stable feature-extraction interface for SAM2.

    The real SAM2 model is injected from outside via a callable `feature_fn`,
    so this module does not depend on any private APIs from specific repos.
    """
    def __init__(self, feature_fn, feat_keys=("F8","F16")):
        """
        Args:
            feature_fn: callable(images: Tensor) -> dict[str, Tensor]
                Expected to return e.g. {"F8": (B,C8,H/8,W/8), "F16": (B,C16,H/16,W/16), ...}.
        """
        super().__init__()
        self.feature_fn = feature_fn
        self.feat_keys = feat_keys
        self.eval()  # 默认冻结
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.feature_fn(images)
        return {k: feats[k] for k in self.feat_keys if k in feats}
