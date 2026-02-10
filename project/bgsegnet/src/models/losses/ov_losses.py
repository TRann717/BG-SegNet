# src/losses/ov_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)

def info_nce(z: torch.Tensor, t_pos: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    z: (B,Dt) normalized
    t_pos: (B,Dt) normalized (positive text prototype)
    """
    B, Dt = z.shape
    sims = torch.mm(z, t_pos.t()) / temperature  # (B,B)
    labels = torch.arange(B, device=z.device)
    return F.cross_entropy(sims, labels)

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps=1e-6):
    """
    logits: (B,1,H,W), targets: (B,1,H,W) in {0,1}
    """
    prob = torch.sigmoid(logits)
    num = 2 * (prob * targets).sum(dim=[1,2,3])
    den = (prob.pow(2) + targets.pow(2)).sum(dim=[1,2,3]) + eps
    loss = 1 - (num + eps) / (den + eps)
    return loss.mean()

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha=0.25, gamma=2.0):
    """
    Binary focal for masks
    """
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = - w * (1 - pt).pow(gamma) * pt.clamp_min(1e-6).log()
    return loss.mean()
