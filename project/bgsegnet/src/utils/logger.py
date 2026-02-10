# src/utils/logger.py
from __future__ import annotations
import csv
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from PIL import Image


def increment_path(path: Path, sep: str = '') -> Path:
    """
    returns path or path{sep}2, path{sep}3, ...
    Similar to Ultralytics-style exp/exp2/exp3 auto-incremented experiment dirs.
    """
    if not path.exists():
        return path
    for i in range(2, 10_000):
        p = Path(f"{path}{sep}{i}")
        if not p.exists():
            return p
    return path

def colorize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a label mask to a simple RGB visualization.

    Args:
        mask: (H,W) int64 [0..C-1]
    Returns:
        (3,H,W) float in [0,1], with a small fixed color palette repeating over classes.
    """
    if mask.ndim == 3:
        mask = mask[0]
    H, W = mask.shape
    out = torch.zeros(3, H, W, dtype=torch.float32)
    palette = torch.tensor([
        [0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0],
        [255,0,255], [0,255,255], [255,127,0]
    ], dtype=torch.float32)
    idx = mask.clamp_min(0).to(torch.long) % palette.shape[0]
    rgb = palette[idx] / 255.0  # (H,W,3)
    out[0] = rgb[...,0]; out[1] = rgb[...,1]; out[2] = rgb[...,2]
    return out

def overlay_mask(img: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Overlay a colorized segmentation mask on top of an image.

    Args:
        img: (3,H,W) float in [0,255] or [0,1].
        mask: (H,W) long label map.
    Returns:
        (3,H,W) float in [0,1].
    """
    if img.max() > 1.5:
        img = img / 255.0
    m = colorize_mask(mask)  # 0..1
    return (1 - alpha) * img + alpha * m

class TBLogger:
    def __init__(self, save_dir: Path, cfg: dict, config_path: Optional[str] = None):
        """
        Args:
            save_dir: experiment directory (e.g., runs/exp or runs/exp2).
        """
        self.save_dir = save_dir
        self.weights_dir = save_dir / "weights"
        self.labels_dir = save_dir / "labels"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        if config_path and Path(config_path).exists():
            shutil.copy(config_path, self.save_dir / "config.yaml")

        tb_enable = bool(cfg.get("log", {}).get("tensorboard", {}).get("enable", True))
        self.tb = SummaryWriter(str(save_dir)) if tb_enable else None

        self.log_every = int(cfg.get("log", {}).get("tensorboard", {}).get("log_every", 50))
        self.img_log = bool(cfg.get("log", {}).get("tensorboard", {}).get("images", True))
        self.max_samples = int(cfg.get("log", {}).get("tensorboard", {}).get("max_samples", 4))

        self.csv_path = self.save_dir / "results.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch",
                    "train/loss","val/loss",
                    "train/miou","val/miou",
                    "train/macc","val/macc",
                    "train/overall_acc","val/overall_acc",
                    "val/mbiou",
                    "lr_group0","lr_group1"
                ])

    def add_scalar_dict(self, d: Dict[str, float], step: int):
        if self.tb is None:
            return
        for k, v in d.items():
            self.tb.add_scalar(k, float(v), step)

    def add_lr(self, optimizer: torch.optim.Optimizer, step: int):
        if self.tb is None:
            return
        for i, g in enumerate(optimizer.param_groups):
            self.tb.add_scalar(f"lr/group{i}", float(g.get("lr", 0.0)), step)

    def add_images_batch(self, images: torch.Tensor, gts: torch.Tensor,
                         preds: torch.Tensor, tag: str, step: int):
        if self.tb is None or not self.img_log:
            return
        # 只取前 N 张
        n = min(images.shape[0], self.max_samples)
        grids = []
        for i in range(n):
            im = images[i].detach().cpu().float()
            gt = gts[i].detach().cpu().long()
            pr = preds[i].detach().cpu().long()
            over_gt = overlay_mask(im, gt)
            over_pr = overlay_mask(im, pr)
            grid = torch.stack([im/255.0 if im.max()>1.5 else im, over_gt, over_pr], dim=0)  # (3,3,H,W)
            grid = vutils.make_grid(grid, nrow=3)  # (3,H,3W)
            grids.append(grid)
        if grids:
            grid_all = vutils.make_grid(grids, nrow=1)
            self.tb.add_image(tag, grid_all, step)
    
    def save_image_samples(self, images: torch.Tensor, gts: torch.Tensor, 
                          preds: torch.Tensor, epoch: int, tag: str = 'train'):
        """Save image/mask/prediction samples to disk for qualitative inspection."""
        if not self.img_log:
            return
            
        images_dir = self.save_dir / 'images' / tag
        images_dir.mkdir(parents=True, exist_ok=True)
        
        n = min(images.shape[0], self.max_samples)
        for i in range(n):
            img = images[i].detach().cpu().float()
            gt = gts[i].detach().cpu().long()
            pred = preds[i].detach().cpu().long()
            
            if img.max() <= 1.5:
                img = img * 255.0
            
            img_norm = img / 255.0  # 归一化到[0,1]
            gt_colored = colorize_mask(gt)
            pred_colored = colorize_mask(pred)
            
            gt_overlay = overlay_mask(img, gt, alpha=0.5)
            pred_overlay = overlay_mask(img, pred, alpha=0.5)
            
            row1 = torch.cat([img_norm, gt_overlay, pred_overlay], dim=2)
            row2 = torch.cat([gt_colored, pred_colored, torch.zeros_like(gt_colored)], dim=2)
            combined = torch.cat([row1, row2], dim=1)
            
            save_path = images_dir / f'epoch_{epoch:03d}_sample_{i:02d}.png'
            vutils.save_image(combined, save_path, normalize=False)

    def write_csv_row(self, epoch: int, payload: Dict[str, Any]):
        vals = [
            epoch,
            payload.get("train/loss", 0), payload.get("val/loss", 0),
            payload.get("train/miou", 0), payload.get("val/miou", 0),
            payload.get("train/macc", 0), payload.get("val/macc", 0),
            payload.get("train/overall_acc", 0), payload.get("val/overall_acc", 0),
            payload.get("val/mbiou", 0),
            payload.get("lr/group0", 0), payload.get("lr/group1", 0)
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(vals)

    def close(self):
        if self.tb:
            self.tb.flush()
            self.tb.close()