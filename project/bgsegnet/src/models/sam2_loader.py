"""
Utilities for loading SAM2.1 and wrapping it as a feature extractor.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path


def load_sam2_model(sam2_path, device='cuda'):
    """
    Load a SAM2.1 model from the given path.
    """
    sam2_dir = Path(sam2_path)
    if sam2_dir.exists():
        sys.path.insert(0, str(sam2_dir))
    
    try:
        sys.path.insert(0, "/root/autodl-tmp/BG-SegNet/project/third_party/sam2")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        import hydra
        
        # Build the model using the official Hydra config.
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            
        config_dir = "/root/autodl-tmp/BG-SegNet/project/third_party/sam2/sam2/configs"
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="sam2.1/sam2.1_hiera_b+.yaml")
            OmegaConf.resolve(cfg)
            
        # Instantiate model from Hydra config.
        from hydra.utils import instantiate
        sam2_model = instantiate(cfg.model, _recursive_=True)
        
        # Load checkpoint weights (without `weights_only` for broader PyTorch compatibility).
        ckpt_path = str(sam2_dir / "sam2.1_hiera_base_plus.pt")
        if Path(ckpt_path).exists():
            sd = torch.load(ckpt_path, map_location="cpu") 
            if "model" in sd:
                sam2_model.load_state_dict(sd["model"], strict=True)
            elif "models" in sd:
                sam2_model.load_state_dict(sd["models"], strict=True)
            else:
                sam2_model.load_state_dict(sd, strict=True)
                
        sam2_model = sam2_model.to(device)
        sam2_model.eval()
        
        print(f"Successfully loaded SAM2.1 from {sam2_path}")
        return sam2_model
        
    except Exception as e:
        print(f"Failed to load SAM2: {e}")
        raise RuntimeError(f"Cannot load SAM2 model from {sam2_path}. Error: {e}")





class DummyImageEncoder(nn.Module):
    """
    Dummy image encoder used as a lightweight stand-in for SAM2 in debugging.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 768, 4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(768, 768, 2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(768, 768, 2, stride=2, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: input images (B, 3, H, W).
        Returns:
            dict of multi-scale features.
        """
        f4 = self.conv1(x)
        f8 = self.conv2(f4)
        f16 = self.conv3(f8)
        
        return {
            "res2": f4,   # 1/4
            "res3": f8,   # 1/8
            "res4": f16,  # 1/16
        }


def _pick_first_present(d, keys):
    """Return the first value in `d` whose key is present and non-None."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class SAM2FeatureExtractor(nn.Module):
    """
    SAM2 feature extractor wrapper (freezable, with auto channel probing).
    """
    def __init__(self, sam2_path, freeze=True, img_mean=(123.675,116.28,103.53), img_std=(58.395,57.12,57.375)):
        super().__init__()
        self.sam2 = load_sam2_model(sam2_path) 
        self._frozen = freeze
        if freeze:
            for p in self.sam2.parameters():
                p.requires_grad_(False)
            self.sam2.eval()
        # Preprocessing parameters (ImageNet-style BGR mean/std; replace with official values if available).
        self.register_buffer("img_mean", torch.tensor(img_mean).view(1,3,1,1), persistent=False)
        self.register_buffer("img_std",  torch.tensor(img_std).view(1,3,1,1), persistent=False)

    def _preprocess(self, images):
        img_mean = self.img_mean.to(images.device)
        img_std = self.img_std.to(images.device)
        
        images = images.float()
        
        if images.max() <= 1.0:
            images = images * 255.0
            
        return (images - img_mean) / img_std

    def _forward_impl(self, images):
        x = self._preprocess(images)
        # Prefer SAM2's `forward_image` API when available.
        if hasattr(self.sam2, 'forward_image'):
            backbone_out = self.sam2.forward_image(x)
            feats = _pick_first_present(backbone_out, 
                                       ["backbone_fpn", "fpn_features", "fpn", "features"])
            if feats is None:
                feats = backbone_out
        elif hasattr(self.sam2, 'image_encoder'):
            feats = self.sam2.image_encoder(x)
        else:
            feats = self.sam2(x)
        
        # Normalize naming to a common multi-scale interface.
        out = {}
        if isinstance(feats, tuple):
            feats = list(feats)
        
        if isinstance(feats, list):
            # Common SAM2 layout: [low_res, mid_res, high_res] = [1/16, 1/8, 1/4].
            if len(feats) >= 3:
                out["F16"] = feats[0]  #  1/16
                out["F8"] = feats[1]   #  1/8
                out["F4"] = feats[2]   #  1/4
            elif len(feats) == 2:
                out["F16"] = feats[0]
                out["F8"] = feats[1]
                out["F4"] = nn.functional.interpolate(feats[1], scale_factor=2, mode='bilinear', align_corners=False)
            else:
                out["F16"] = feats[0]
                out["F8"]  = nn.functional.interpolate(feats[0], scale_factor=2, mode='bilinear', align_corners=False)
                out["F4"]  = nn.functional.interpolate(feats[0], scale_factor=4, mode='bilinear', align_corners=False)
        elif isinstance(feats, dict):
            out["F4"]  = _pick_first_present(feats, ["res2", "feat_s4", "low_res", "s4"])
            out["F8"]  = _pick_first_present(feats, ["res3", "feat_s8", "mid_res", "s8"])
            out["F16"] = _pick_first_present(feats, ["res4", "feat_s16", "high_res", "s16"])
            
            if out["F16"] is None and out["F8"] is not None:
                out["F16"] = nn.functional.interpolate(out["F8"], scale_factor=0.5, mode='bilinear', align_corners=False)
            if out["F8"] is None and out["F16"] is not None:
                out["F8"] = nn.functional.interpolate(out["F16"], scale_factor=2, mode='bilinear', align_corners=False)
            if out["F4"] is None and out["F8"] is not None:
                out["F4"] = nn.functional.interpolate(out["F8"], scale_factor=2, mode='bilinear', align_corners=False)
        else:
            out["F16"] = feats
            out["F8"]  = nn.functional.interpolate(feats, scale_factor=2, mode='bilinear', align_corners=False)
            out["F4"]  = nn.functional.interpolate(feats, scale_factor=4, mode='bilinear', align_corners=False)
        return out

    def forward(self, images):
        if self._frozen:
            with torch.no_grad():
                return self._forward_impl(images)
        else:
            return self._forward_impl(images)
