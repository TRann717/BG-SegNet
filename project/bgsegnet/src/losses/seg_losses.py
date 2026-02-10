"""
Semantic segmentation loss.

Integrates region losses (CE + Dice/Tversky) with unified soft boundary supervision
and optional geometric band regularizers, aligned with current model outputs
and configuration.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Soft boundary construction (shared implementation from boundary_head.py).
from ..models.heads.boundary_head import (
    generate_soft_boundary_from_mask,
    generate_soft_boundary_per_class,
)

# ====================== Base region losses ======================

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            logits, targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp_min(0), num_classes=C).permute(0,3,1,2).float()
        mask = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * mask
        targets_oh = targets_oh * mask
        start_c = 1 if self.ignore_background and C > 1 else 0
        loss = 0.0
        valid = 0
        for c in range(start_c, C):
            p = probs[:, c]; t = targets_oh[:, c]
            inter = (p * t).sum(dim=(1,2))
            denom = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
            d = (2*inter + self.smooth) / (denom + self.smooth)
            loss += (1 - d).mean()
            valid += 1
        return loss / max(valid, 1)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, ignore_index=255, ignore_background=True):
        super().__init__()
        self.alpha = alpha; self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp_min(0), num_classes=C).permute(0,3,1,2).float()
        mask = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * mask; targets_oh = targets_oh * mask
        start_c = 1 if self.ignore_background and C > 1 else 0
        loss = 0.0; valid = 0
        for c in range(start_c, C):
            p = probs[:, c]; t = targets_oh[:, c]
            tp = (p*t).sum(dim=(1,2))
            fp = (p*(1-t)).sum(dim=(1,2))
            fn = ((1-p)*t).sum(dim=(1,2))
            denom = tp + self.alpha*fp + self.beta*fn + self.smooth
            loss += (1 - (tp + self.smooth)/denom).mean()
            valid += 1
        return loss / max(valid, 1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ignore_index = ignore_index
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        p = torch.exp(-ce)
        loss = (1 - p)**self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


# ---------------------- 统一软边界（类无关/多尺度） ----------------------

class UnifiedBoundaryLoss(nn.Module):
    def __init__(self,
                 focal_weight=0.6,
                 bce_weight=0.3,
                 l1_weight=0.3,
                 grad_weight=0.0,
                 ms_weight=0.1,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 band_pixels=6,
                 eps=1e-6):
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.bce_weight   = float(bce_weight)
        self.l1_weight    = float(l1_weight)
        self.grad_weight  = float(grad_weight)
        self.ms_weight    = float(ms_weight)
        self.focal_alpha  = float(focal_alpha)
        self.focal_gamma  = float(focal_gamma)
        self.band_pixels  = int(band_pixels)
        self.eps          = float(eps)
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', kx); self.register_buffer('sobel_y', ky)

    def _grad(self, x):
        # Ensure Sobel kernels follow input tensor dtype/device (AMP-friendly).
        sobel_x = self.sobel_x.to(x.device).to(x.dtype)
        sobel_y = self.sobel_y.to(x.device).to(x.dtype)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return gx, gy

    @staticmethod
    def _band_mask(soft):
        return (soft > 0.01).float()

    def forward(self, pred_logits, soft_target, b4_logits=None, b8_logits=None):
        if soft_target.dim() == 3:
            soft_target = soft_target.unsqueeze(1)
        band = self._band_mask(soft_target)
        area = band.sum().clamp_min(1.0)
        pred_prob = torch.sigmoid(pred_logits)

        # 1) focal on band
        lf = pred_prob.new_tensor(0.0)
        if self.focal_weight > 0:
            pos = - soft_target * (1 - pred_prob).pow(self.focal_gamma) * torch.log(pred_prob.clamp_min(1e-6))
            neg = - (1 - soft_target) * pred_prob.pow(self.focal_gamma) * torch.log((1 - pred_prob).clamp_min(1e-6))
            pos = self.focal_alpha * pos; neg = (1 - self.focal_alpha) * neg
            lf = ((pos + neg) * band).sum() / area

        # 2) bce on band
        lb = pred_prob.new_tensor(0.0)
        if self.bce_weight > 0:
            bce = F.binary_cross_entropy_with_logits(pred_logits, soft_target, reduction='none')
            lb = (bce * band).sum() / area

        # 3) l1 on band
        ll1 = pred_prob.new_tensor(0.0)
        if self.l1_weight > 0:
            ll1 = (torch.abs(pred_prob - soft_target) * band).sum() / area

        # 4) gradient alignment (optional)
        lg = pred_prob.new_tensor(0.0)
        if self.grad_weight > 0:
            pgx, pgy = self._grad(pred_prob); ggx, ggy = self._grad(soft_target)
            pn = torch.sqrt(pgx**2 + pgy**2 + 1e-6); gn = torch.sqrt(ggx**2 + ggy**2 + 1e-6)
            cos = (pgx/pn)*(ggx/gn) + (pgy/pn)*(ggy/gn)
            lg = ((1 - cos) * band).sum() / area

        # 5) multi-scale consistency at 1/4 with 1/8 (optional)
        lms = pred_prob.new_tensor(0.0)
        if self.ms_weight > 0 and (b4_logits is not None) and (b8_logits is not None):
            b8_up = F.interpolate(torch.sigmoid(b8_logits), size=b4_logits.shape[-2:], mode='bilinear', align_corners=False)
            b4_p  = torch.sigmoid(b4_logits)
            soft4 = F.interpolate(soft_target, size=b4_logits.shape[-2:], mode='bilinear', align_corners=False)
            band4 = self._band_mask(soft4)
            lms = (torch.abs(b4_p - b8_up) * band4).sum() / band4.sum().clamp_min(1.0)

        total = self.focal_weight*lf + self.bce_weight*lb + self.l1_weight*ll1 + self.grad_weight*lg + self.ms_weight*lms
        return total, {'focal':lf, 'bce':lb, 'l1':ll1, 'grad':lg, 'ms':lms}


# ====================== Geometric band regularizers (optional) ======================

class BandSupConTripletLoss(nn.Module):
    """
    Cosine triplet-hinge loss on band geometry at 1/4 resolution.

    Extracts mean feature vectors from three regions around the lesion boundary
    (boundary band / inner band / outer band) and enforces semantic separation.
    """
    def __init__(self, margin=0.2, band_pixels=6):
        super().__init__()
        self.m = float(margin); self.band = int(band_pixels)

    @staticmethod
    def _mean(feats: torch.Tensor, mask: torch.Tensor, eps=1e-6):
        # feats: (B,C,h,w) L2-normalized; mask: (B,1,h,w) in {0,1}.
        num = (feats * mask).sum(dim=(2,3))
        den = mask.sum(dim=(2,3)).clamp_min(eps)
        v = num / den
        return F.normalize(v, dim=1)

    def forward(self, feats_1_4: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, h, w = feats_1_4.shape
        t = F.interpolate(targets.float().unsqueeze(1), size=(h,w), mode='nearest').squeeze(1).long()
        fg = (t == 1).float().unsqueeze(1)                      # (B,1,h,w)
        k = max(3, self.band*2-1); pad = (k-1)//2
        dil = F.max_pool2d(fg, k, 1, pad)
        ero = -F.max_pool2d(-fg, k, 1, pad)
        boundary = (dil - ero).clamp_min(0.0)
        band = (F.max_pool2d(boundary, 3, 1, 1) > 0).float()
        inside = (fg * (1 - boundary)).float()
        outside= ((1 - fg) * (1 - boundary)).float()
        z = F.normalize(feats_1_4, dim=1)
        vb = self._mean(z, band + 1e-3)
        vi = self._mean(z, inside + 1e-3)
        vo = self._mean(z, outside + 1e-3)
        s_bi = (vb*vi).sum(dim=1)
        s_bo = (vb*vo).sum(dim=1)
        s_io = (vi*vo).sum(dim=1)
        loss = F.relu(self.m + s_bi).mean() + F.relu(self.m + s_bo).mean() + 0.5*F.relu(self.m + s_io).mean()
        return loss


class CurvAwareLoss(nn.Module):
    """
    Curvature-aware boundary regularization within the lesion band.

      L = E[ |∂P/∂t| ] + λ * E[ ReLU(m_n - |∂P/∂n|) ]

    where ∂P/∂t is the tangential gradient (encouraging smoothness along the contour)
    and ∂P/∂n is the normal gradient (encouraging sharp transitions across the contour).
    """
    def __init__(self, min_normal=0.3, lam=1.0, band_pixels=6):
        super().__init__()
        self.mn = float(min_normal); self.lam = float(lam); self.band = int(band_pixels)
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx', kx); self.register_buffer('sy', ky)
    def _grad(self, x):
        gx = F.conv2d(x, self.sx.to(x.device).to(x.dtype), padding=1)
        gy = F.conv2d(x, self.sy.to(x.device).to(x.dtype), padding=1)
        return gx, gy
    def forward(self, prob_fg: torch.Tensor, targets: torch.Tensor, dir_1_4: torch.Tensor | None = None):
        B,_,H,W = prob_fg.shape
        fg = (targets==1).float().unsqueeze(1)
        k = max(3, self.band*2-1); pad=(k-1)//2
        dil = F.max_pool2d(fg, k,1,pad); ero = -F.max_pool2d(-fg, k,1,pad)
        boundary = (dil - ero).clamp_min(0.0)
        band = (F.max_pool2d(boundary,3,1,1)>0).float()
        area = band.sum().clamp_min(1.0)
        gx, gy = self._grad(prob_fg)
        # approximate tangent (dx,dy) from direction if provided (upsampled), otherwise orthogonal to grad
        if dir_1_4 is not None:
            d = F.interpolate(dir_1_4, size=(H,W), mode='bilinear', align_corners=False)
            dx, dy = d[:,0:1], d[:,1:2]
            nrm = torch.sqrt(dx*dx + dy*dy + 1e-6); dx, dy = dx/nrm, dy/nrm
        else:
            bx, by = gx, gy
            nrm = torch.sqrt(bx*bx + by*by + 1e-6)
            dx, dy = -by/nrm, bx/nrm
        grad_t = torch.abs(dx*gx + dy*gy)
        grad_n = torch.abs(-dy*gx + dx*gy)
        loss = (grad_t*band).sum()/area + self.lam*(F.relu(self.mn - grad_n)*band).sum()/area
        return loss


# ---------------------- 组装：CompoundLoss ----------------------

class CompoundLoss(nn.Module):
    """
    Composite loss for medical image segmentation.

    Combines:
      - main region loss (CE + Dice/Tversky)
      - optional unified soft boundary supervision
      - optional geometric band regularizers

    Depends on config keys:
      - model.boundary.*
      - loss.*

    Forward usage:
      loss, stats = criterion(outputs, targets)
      where `outputs` is produced by `BG-SegNet.forward(...)`.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        data_cfg  = config['data']
        loss_cfg  = config['loss']
        bd_cfg    = config['model']['boundary']

        cls_w = data_cfg.get('class_weights', None)
        self.ce_w = float(loss_cfg.get('ce_weight', 1.0))
        self.dice_w = float(loss_cfg.get('dice_weight', 0.0))
        self.ignore_index = int(data_cfg.get('ignore_index', 255))

        if cls_w is not None:
            w = torch.tensor(cls_w, dtype=torch.float32)
        else:
            w = None
        self.ce = CrossEntropyLoss2d(weight=w, ignore_index=self.ignore_index, reduction='mean')

        if bool(loss_cfg.get('use_tversky', False)):
            self.overlap = TverskyLoss(
                alpha=float(loss_cfg.get('tversky_alpha',0.3)),
                beta=float(loss_cfg.get('tversky_beta',0.7)),
                ignore_index=self.ignore_index,
                ignore_background=bool(loss_cfg.get('dice_ignore_background', True))
            )
        else:
            self.overlap = DiceLoss(
                ignore_index=self.ignore_index,
                ignore_background=bool(loss_cfg.get('dice_ignore_background', True))
            )

        # Unified soft boundary supervision
        self.bd_enable       = bool(bd_cfg.get('enable', False))
        self.bd_class_aware  = bool(bd_cfg.get('class_aware', False))
        self.bd_multiscale   = bool(bd_cfg.get('multiscale', False))
        self.bd_weight_max   = float(bd_cfg.get('weight', 0.0))
        self._bd_scalar      = 1.0  # curriculum scalar adjusted by Trainer.set_boundary_scalar()
        self.bd_loss = UnifiedBoundaryLoss(
            focal_weight=float(bd_cfg.get('focal_weight', 0.6)),
            bce_weight=float(bd_cfg.get('bce_weight', 0.3)),
            l1_weight=float(bd_cfg.get('l1_weight', 0.3)),
            grad_weight=float(bd_cfg.get('grad_weight', 0.0)),
            ms_weight=float(bd_cfg.get('ms_weight', 0.1)),
            focal_alpha=float(bd_cfg.get('focal_alpha', 0.25)),
            focal_gamma=float(bd_cfg.get('focal_gamma', 2.0)),
            band_pixels=int(bd_cfg.get('band_pixels', 6)),
        )
        self.bd_kernel = int(bd_cfg.get('k_size', 3))
        self.bd_sigma  = float(bd_cfg.get('sigma', 3.0))
        self.bd_band   = int(bd_cfg.get('band_pixels', 6))
        self.num_classes = int(data_cfg.get('num_classes', 2))

        # Geometric band regularizers
        self.band_supcon_enable = bool(loss_cfg.get('band_supcon_enable', False))
        self.band_supcon = BandSupConTripletLoss(
            margin=float(loss_cfg.get('band_supcon_margin', 0.2)),
            band_pixels=int(bd_cfg.get('band_pixels', 6)),
        )
        self.band_supcon_w = float(loss_cfg.get('band_supcon_weight', 0.3))

        self.curv_aware_enable = bool(loss_cfg.get('curv_aware_enable', False))
        self.curv_aware = CurvAwareLoss(
            min_normal=float(loss_cfg.get('curv_min_normal', 0.3)),
            lam=float(loss_cfg.get('curv_aware_weight', 0.2)),
            band_pixels=int(bd_cfg.get('band_pixels', 6)),
        )

        # Auxiliary segmentation head (weight from model.head.aux_weight)
        self.aux_w = float(config['model']['head'].get('aux_weight', 0.4))

        self.num_classes = int(data_cfg['num_classes'])

    def set_boundary_scalar(self, s: float):
        self._bd_scalar = float(s)

    def forward(self, outputs: dict, targets: torch.Tensor):
        """
        Args:
            outputs: model outputs dictionary
            targets: ground-truth label map, shape (B,H,W), dtype long
        """
        loss_dict = {}
        total = 0.0

        logits = outputs['logits']  # (B,C,H,W)
        l_ce = self.ce(logits, targets) * self.ce_w
        total += l_ce
        loss_dict['ce'] = l_ce.detach()

        if self.dice_w > 0:
            l_overlap = self.overlap(logits, targets) * self.dice_w
            total += l_overlap
            loss_dict['dice'] = l_overlap.detach()

        if ('aux' in outputs) and (self.aux_w > 0):
            aux = outputs['aux']
            l_aux = self.ce(aux, targets) * self.aux_w
            total += l_aux
            loss_dict['aux_ce'] = l_aux.detach()

        if self.bd_enable and ('boundary' in outputs):
            if self.bd_class_aware and 'boundary_b4' in outputs:
                b4 = outputs['boundary_b4']  # (B, C, H/4, W/4)
                b8 = outputs.get('boundary_b8', None)  # (B, C, H/8, W/8)
                
                soft_per_class = generate_soft_boundary_per_class(
                    mask=targets,
                    num_classes=self.num_classes,
                    kernel_size=self.bd_kernel,
                    band_pixels=self.bd_band,
                    sigma=self.bd_sigma
                )  # (B, C, H, W)
                
                bd_loss_total = 0.0
                bd_parts_accum = {}
                valid_classes = 0
                
                for c in range(1, self.num_classes):  # skip background class 0
                    b4_c = b4[:, c:c+1]  # (B, 1, H/4, W/4)
                    soft_c = soft_per_class[:, c:c+1]  # (B, 1, H, W)
                    
                    pred_1x_c = F.interpolate(b4_c, size=soft_c.shape[-2:], mode='bilinear', align_corners=False)
                    
                    b8_c = b8[:, c:c+1] if b8 is not None else None
                    
                    l_bd_c, bd_parts_c = self.bd_loss(pred_1x_c, soft_c, b4_logits=b4_c, b8_logits=b8_c)
                    bd_loss_total += l_bd_c
                    valid_classes += 1
                    
                    for k, v in bd_parts_c.items():
                        if k not in bd_parts_accum:
                            bd_parts_accum[k] = 0.0
                        bd_parts_accum[k] += v
                
                if valid_classes > 0:
                    bd_loss_total = bd_loss_total / valid_classes
                    for k in bd_parts_accum:
                        bd_parts_accum[k] = bd_parts_accum[k] / valid_classes
                
                w_bd = self.bd_weight_max * self._bd_scalar
                total = total + w_bd * bd_loss_total
                loss_dict['boundary_total'] = bd_loss_total.detach()
                for k, v in bd_parts_accum.items():
                    loss_dict[f'bd_{k}'] = v.detach()
                    
            else:
                soft = generate_soft_boundary_from_mask(
                    mask=targets,
                    kernel_size=self.bd_kernel,
                    num_classes=self.num_classes,
                    band_pixels=self.bd_band,
                    sigma=self.bd_sigma
                )  # (B,1,H,W)
                b4 = outputs.get('boundary_b4', None)
                b8 = outputs.get('boundary_b8', None)
                pred_1x = outputs.get('boundary', None)
                
                if pred_1x is None and b4 is not None:
                    if b4.shape[1] > 1:
                        b4 = b4.max(dim=1, keepdim=True)[0]
                    pred_1x = F.interpolate(b4, size=soft.shape[-2:], mode='bilinear', align_corners=False)
                
                if pred_1x is not None:
                    if pred_1x.shape[1] > 1:
                        pred_1x = pred_1x.max(dim=1, keepdim=True)[0]
                    
                    w_bd = self.bd_weight_max * self._bd_scalar
                    l_bd, bd_parts = self.bd_loss(pred_1x, soft, b4_logits=b4, b8_logits=b8)
                    total = total + w_bd * l_bd
                    loss_dict['boundary_total'] = l_bd.detach()
                    for k, v in bd_parts.items():
                        loss_dict[f'bd_{k}'] = v.detach()

        if 'dir_b4' in outputs and self.config['model']['boundary'].get('dir_weight', 0) > 0:
            dir_weight = float(self.config['model']['boundary']['dir_weight'])
            dir_b4 = outputs['dir_b4']  # (B, 2, H/4, W/4)
            
            # Use gradient of soft boundary map as ground-truth direction field.
            if self.bd_class_aware:
                soft_per_class = generate_soft_boundary_per_class(
                    mask=targets, num_classes=self.num_classes,
                    kernel_size=self.bd_kernel, band_pixels=self.bd_band, sigma=self.bd_sigma
                )
                soft_union = soft_per_class[:, 1:].sum(dim=1, keepdim=True).clamp_max(1.0)
            else:
                soft_union = generate_soft_boundary_from_mask(
                    mask=targets,
                    kernel_size=self.bd_kernel,
                    num_classes=self.num_classes,
                    band_pixels=self.bd_band,
                    sigma=self.bd_sigma
                )
            
            soft_4 = F.interpolate(soft_union, size=dir_b4.shape[-2:], mode='bilinear', align_corners=False)
            
            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=soft_4.dtype, device=soft_4.device).view(1,1,3,3)
            sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=soft_4.dtype, device=soft_4.device).view(1,1,3,3)
            gx = F.conv2d(soft_4, sobel_x, padding=1)
            gy = F.conv2d(soft_4, sobel_y, padding=1)
            norm = torch.sqrt(gx**2 + gy**2 + 1e-6)
            gt_dir = torch.cat([gx/norm, gy/norm], dim=1)
            
            pred_norm = torch.sqrt(dir_b4[:,0:1]**2 + dir_b4[:,1:2]**2 + 1e-6)
            pred_dir = dir_b4 / pred_norm
            
            # Cosine similarity loss between predicted and GT direction fields within boundary band.
            cos_sim = (pred_dir * gt_dir).sum(dim=1, keepdim=True)
            band_4 = (soft_4 > 0.01).float()
            dir_loss = ((1 - cos_sim) * band_4).sum() / band_4.sum().clamp_min(1.0)
            
            total = total + dir_weight * dir_loss
            loss_dict['bd_dir'] = dir_loss.detach()

        if self.config['model']['boundary'].get('consis_weight', 0) > 0 and 'boundary_b4' in outputs and 'boundary_b8' in outputs:
            consis_weight = float(self.config['model']['boundary']['consis_weight'])
            b4 = outputs['boundary_b4']
            b8 = outputs['boundary_b8']
            
            if b4.shape[1] > 1:
                b4 = b4.max(dim=1, keepdim=True)[0]
            if b8.shape[1] > 1:
                b8 = b8.max(dim=1, keepdim=True)[0]
            
            b8_up = F.interpolate(torch.sigmoid(b8), size=b4.shape[-2:], mode='bilinear', align_corners=False)
            b4_prob = torch.sigmoid(b4)
            
            soft = generate_soft_boundary_from_mask(
                mask=targets,
                kernel_size=self.bd_kernel,
                num_classes=self.num_classes,
                band_pixels=self.bd_band,
                sigma=self.bd_sigma
            )
            soft_4 = F.interpolate(soft, size=b4.shape[-2:], mode='bilinear', align_corners=False)
            band_4 = (soft_4 > 0.01).float()
            
            consis_loss = (torch.abs(b4_prob - b8_up) * band_4).sum() / band_4.sum().clamp_min(1.0)
            total = total + consis_weight * consis_loss
            loss_dict['bd_cons'] = consis_loss.detach()

        if self.config['model']['boundary'].get('margin_weight', 0) > 0 and 'feat_1_4' in outputs:
            margin_weight = float(self.config['model']['boundary']['margin_weight'])
            feat = outputs['feat_1_4']  # (B, C, H/4, W/4)
            
            soft = generate_soft_boundary_from_mask(
                mask=targets,
                kernel_size=self.bd_kernel,
                num_classes=self.num_classes,
                band_pixels=self.bd_band,
                sigma=self.bd_sigma
            )
            soft_4 = F.interpolate(soft, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            band_4 = (soft_4 > 0.01).float().squeeze(1)  # (B, H/4, W/4)
            
            targets_4 = F.interpolate(targets.unsqueeze(1).float(), size=feat.shape[-2:], mode='nearest').squeeze(1).long()
            
            margin_loss = 0.0
            for b in range(feat.shape[0]):
                band_idx = (band_4[b] > 0).nonzero(as_tuple=False)
                if band_idx.shape[0] < 10:
                    continue
                
                K = min(band_idx.shape[0], 100)
                perm = torch.randperm(band_idx.shape[0], device=band_idx.device)[:K]
                sampled = band_idx[perm]
                
                labels = targets_4[b, sampled[:,0], sampled[:,1]]
                feats = feat[b, :, sampled[:,0], sampled[:,1]].T
                feats = F.normalize(feats, dim=1)
                
                fg_feats = feats[labels == 1]
                bg_feats = feats[labels == 0]
                
                if fg_feats.shape[0] > 0 and bg_feats.shape[0] > 0:
                    fg_mean = fg_feats.mean(dim=0, keepdim=True)
                    bg_mean = bg_feats.mean(dim=0, keepdim=True)
                    sim = (fg_mean * bg_mean).sum()
                    margin_loss += F.relu(0.2 + sim)
            
            if feat.shape[0] > 0:
                margin_loss = margin_loss / feat.shape[0]
            
            total = total + margin_weight * margin_loss
            loss_dict['bd_margin'] = margin_loss.detach() if isinstance(margin_loss, torch.Tensor) else torch.tensor(margin_loss, device=total.device)

        if self.band_supcon is not None and 'feat_1_4' in outputs:
            try:
                l_bctr = self.band_supcon(outputs['feat_1_4'], targets)
                total = total + self.band_supcon_w * l_bctr
                loss_dict['band_supcon'] = l_bctr.detach()
            except Exception:
                loss_dict['band_supcon'] = torch.tensor(0.0, device=total.device)
                
        if self.curv_aware_enable and self.curv_aware is not None:
            try:
                with torch.no_grad():
                    if outputs['logits'].shape[1] > 1:
                        prob_fg = torch.softmax(outputs['logits'], dim=1)[:,1:2]
                    else:
                        prob_fg = torch.sigmoid(outputs['logits'])
                dir_1_4 = outputs.get('dir_b4', None)
                l_curv = self.curv_aware(prob_fg, targets, dir_1_4)
                total = total + l_curv  # Weight already embedded in CurvAwareLoss.lam
                loss_dict['curv_aware'] = l_curv.detach()
            except Exception:
                loss_dict['curv_aware'] = torch.tensor(0.0, device=total.device)

        return total, loss_dict