"""
BG-SegNet: dermoscopic lesion segmentation network (PH2 variant).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .neck.ppm import UPerNet
from .heads.decoder_head import DecoderHead, AuxHead
from .heads.boundary_head import BoundaryHead, normalize_dir_field
from .heads.boundary_gate import BoundaryGate
from .ov.hq_token import HQTokenAdapter
from .sam2_loader import SAM2FeatureExtractor


class CSegNet(nn.Module):
    """BG-SegNet segmentation network with boundary-aware decoder."""
    
    def __init__(self, config):
        super().__init__()
        
        model_cfg = config['model']
        data_cfg = config['data']
        
        # SAM2 backbone (optionally frozen).
        self.backbone = SAM2FeatureExtractor(model_cfg['sam2_path'], freeze=model_cfg['freeze_backbone'])

        # Probe backbone feature channels with a dummy input at configured img_size.
        H, W = data_cfg['img_size']
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W, device="cuda" if torch.cuda.is_available() else "cpu")
            feats_probe = self.backbone(dummy)
        in_channels_dict = {
            "F4": feats_probe["F4"].shape[1],
            "F8": feats_probe["F8"].shape[1],
            "F16": feats_probe["F16"].shape[1],
        }

        # Neck: UPerNet (PPM + FPN) using probed channel dimensions.
        self.neck = UPerNet(
            in_channels_dict=in_channels_dict,  # 使用自检的通道，不用config里的
            out_channels=model_cfg['neck']['out_channels'],
            ppm_bins=model_cfg['neck']['ppm_bins']
        )
        
        # Main segmentation head.
        self.head = DecoderHead(
            in_channels=model_cfg['neck']['out_channels'],
            num_classes=data_cfg['num_classes'],
            dropout=model_cfg['head']['dropout'],
            up_type=model_cfg['head'].get('up_type', 'bilinear')
        )
        
        # Bias initialization using per-class pixel frequency priors (if provided).
        class_freq = data_cfg.get('class_freq', None)
        if class_freq is not None:
            self.head.init_bias_with_priors(class_freq)
        
        # Auxiliary segmentation head for deep supervision.
        self.use_aux = model_cfg['head']['aux_loss']
        if self.use_aux:
            self.aux_head = AuxHead(
                in_channels=in_channels_dict['F8'],  # 使用自检的通道
                num_classes=data_cfg['num_classes']
            )
        
        # Boundary supervision heads and boundary gate.
        self.use_boundary = model_cfg['boundary']['enable']
        self.use_gum = model_cfg['head'].get('gum', False)  # guided upsampling
        self.oabg_enable = model_cfg['boundary'].get('oabg_enable', False)
        self.uaware_enable = model_cfg['boundary'].get('u_aware_alpha', False)
        self.classmix_enable = model_cfg['boundary'].get('class_mixing', False)
        
        if self.use_boundary:
            # Lightweight boundary gate for feature refinement (SE + adaptive alpha).
            self.boundary_gate = BoundaryGate(
                in_channels=model_cfg['neck']['out_channels'],
                use_depthwise=True,
                use_se=model_cfg['boundary'].get('use_se', True)
            )
            self.boundary_refine_alpha = model_cfg['boundary'].get('refine_alpha', 0.2)
            self.adaptive_alpha = model_cfg['boundary'].get('adaptive_alpha', True)
            self.alpha_min = model_cfg['boundary'].get('alpha_min', 0.15)
            self.alpha_max = model_cfg['boundary'].get('alpha_max', 0.35)  # 收紧上限
            self.gamma_alpha = model_cfg['boundary'].get('gamma_alpha', 1.0)
            
            # Boundary supervision heads (used only in loss), with optional 1/4 and 1/8 scales.
            self.boundary_multiscale = model_cfg['boundary'].get('multiscale', False)
            
            # Class-aware boundary configuration.
            self.class_aware_boundary = model_cfg['boundary'].get('class_aware', False)
            num_cls = data_cfg['num_classes'] if self.class_aware_boundary else 1
            with_dir = model_cfg['boundary'].get('with_dir', False) or (model_cfg['boundary'].get('dir_weight', 0.0) > 0)
            self.softmax_beta = model_cfg['boundary'].get('softmax_beta', 6.0)
            
            self.boundary_head_4 = BoundaryHead(
                in_channels=model_cfg['neck']['out_channels'],
                num_classes=num_cls, 
                with_dir=with_dir
            )
            if self.boundary_multiscale:
                self.boundary_head_8 = BoundaryHead(
                    in_channels=model_cfg['neck']['out_channels'],
                    num_classes=num_cls, 
                    with_dir=with_dir
                )
            
            # Gating fusion and gradient flow strategy.
            gate_cfg = model_cfg['boundary'].get('gate', {})
            self.g_lambda_f4 = float(gate_cfg.get('lambda_f4', 0.6))
            self.g_lambda_f8 = float(gate_cfg.get('lambda_f8', 0.4))
            self._gate_detach = not bool(gate_cfg.get('allow_gate_grad', False))
        
        # Optional HQ boundary enhancement (disabled by default, can be enabled for low bIoU).
        self.use_hq = False
        if self.use_hq:
            self.hq_adapter = HQTokenAdapter(
                in_channels=model_cfg['neck']['out_channels'],
                mid=256
            )
    
    def set_gate_detach(self, detach: bool = True):
        """Control whether the gate signal is detached (used for progressive unfreezing)."""
        self._gate_detach = bool(detach)
    
    def forward(self, images):
        """
        Args:
            images: input images (B, 3, H, W).
        Returns:
            dict with:
                "logits": main segmentation logits (B, num_classes, H, W)
                "aux": auxiliary logits (B, num_classes, H, W) if enabled
                "boundary": boundary logits (B, 1 or C, H, W) if enabled
                "hq_mask": HQ mask logits (B, 1, H, W) if enabled
        """
        B, _, H, W = images.shape
        
        # Basic NaN/Inf check on inputs.
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"[ERROR] Input images contain NaN/Inf! Replacing with zeros.")
            images = torch.where(torch.isnan(images) | torch.isinf(images), torch.zeros_like(images), images)
        
        # 1. Extract SAM2 multi-scale features.
        features = self.backbone(images)
        
        # Sanity-check backbone features for numerical issues.
        for k, v in features.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"[ERROR] Backbone feature {k} contains NaN/Inf! Stats: min={v.min()}, max={v.max()}, nan_count={torch.isnan(v).sum()}")
                # 尝试修复：替换为0
                features[k] = torch.where(torch.isnan(v) | torch.isinf(v), torch.zeros_like(v), v)
        
        # 2. Neck fusion.
        if self.use_boundary and self.boundary_multiscale:
            neck_out_dict = self.neck(features, return_pyramid=True)
            neck_out = neck_out_dict["out"]          # 1/4 fused
            p4 = neck_out_dict["p4"]                 # 1/4
            p8 = neck_out_dict["p8"]                 # 1/8
            
            # Numerical checks for neck outputs.
            if torch.isnan(neck_out).any() or torch.isinf(neck_out).any():
                print(f"[ERROR] Neck output contains NaN/Inf! Stats: min={neck_out.min()}, max={neck_out.max()}, nan_count={torch.isnan(neck_out).sum()}")
                neck_out = torch.where(torch.isnan(neck_out) | torch.isinf(neck_out), torch.zeros_like(neck_out), neck_out)
            if torch.isnan(p4).any() or torch.isinf(p4).any():
                print(f"[ERROR] Neck p4 contains NaN/Inf!")
                p4 = torch.where(torch.isnan(p4) | torch.isinf(p4), torch.zeros_like(p4), p4)
            if torch.isnan(p8).any() or torch.isinf(p8).any():
                print(f"[ERROR] Neck p8 contains NaN/Inf!")
                p8 = torch.where(torch.isnan(p8) | torch.isinf(p8), torch.zeros_like(p8), p8)
        else:
            neck_out = self.neck(features)
            if torch.isnan(neck_out).any() or torch.isinf(neck_out).any():
                print(f"[ERROR] Neck output contains NaN/Inf! Stats: min={neck_out.min()}, max={neck_out.max()}, nan_count={torch.isnan(neck_out).sum()}")
                neck_out = torch.where(torch.isnan(neck_out) | torch.isinf(neck_out), torch.zeros_like(neck_out), neck_out)
        
        # 2.5. Boundary-guided refinement (if enabled).
        if self.use_boundary:
            # 1/4 boundary logits (possibly multi-channel if class-aware).
            b4_pack = self.boundary_head_4(neck_out, out_size=None)
            b4 = b4_pack['b_logits']                     # (B,C|1,H/4,W/4)
            dir4 = b4_pack.get('dir', None)
            g4_all = torch.sigmoid(b4)                   # (B,C|1,H/4,W/4)
            
            # Compute foreground probability p_fg for class mixing in boundary gate.
            if self.classmix_enable and hasattr(self, 'head'):
                with torch.no_grad():
                    coarse_logits = self.head.classifier(self.head.feat(neck_out))
                    p_fg = torch.softmax(coarse_logits, dim=1)[:, 1:2]  # 前景概率
            else:
                p_fg = None
            
            # Aggregate multi-class boundaries into a class-agnostic edge strength.
            if g4_all.shape[1] > 1:
                w = torch.softmax(self.softmax_beta * b4, dim=1)     # (B,C,H/4,W/4)
                g4 = (w * torch.sigmoid(b4)).sum(dim=1, keepdim=True)
            else:
                g4 = g4_all
            
            # Optional 1/8-scale boundary.
            if self.boundary_multiscale:
                b8_pack = self.boundary_head_8(p8, out_size=None)
                b8 = b8_pack['b_logits']
                dir8 = b8_pack.get('dir', None)
                g8_all = torch.sigmoid(b8)
                
                if g8_all.shape[1] > 1:
                    w8 = torch.softmax(self.softmax_beta * b8, dim=1)
                    g8 = (w8 * torch.sigmoid(b8)).sum(dim=1, keepdim=True)
                else:
                    g8 = g8_all
                    
                g8_up = F.interpolate(g8, size=g4.shape[-2:], mode='bilinear', align_corners=False)
                g = self.g_lambda_f4 * g4 + self.g_lambda_f8 * g8_up
                
                # Boundary tensors used by loss functions.
                b_for_loss = {
                  'b4': b4, 'b8': b8,
                  'dir4': dir4, 'dir8': dir8
                }
            else:
                g = g4
                b_for_loss = {'b4': b4, 'dir4': dir4}
            
            # Progressive decoupling: detach early, then unfreeze gate gradients later.
            if self._gate_detach:
                g = g.detach()
            
            # Normalize direction field if orientation-aware gate is enabled.
            dir_map = normalize_dir_field(dir4) if self.oabg_enable and dir4 is not None else None
            
            neck_out = self.boundary_gate(
                neck_out, 
                alpha=self.boundary_refine_alpha, 
                g_ext=g,
                adaptive_alpha=self.adaptive_alpha,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                gamma_alpha=self.gamma_alpha,
                dir_map=dir_map,
                p_fg=p_fg,
                use_oabg=self.oabg_enable,
                use_uaware=self.uaware_enable,
                use_classmix=self.classmix_enable
            )
        
        # 3. Main segmentation head.
        if self.use_gum and self.use_boundary:
            # Guided upsampling: use boundary strength as spatial guide.
            feat_mid = self.head.feat(neck_out)  # (B, mid, H/4, W/4)
            
            if self.head.up_type == "carafe":
                feat_mid = self.head.up2a(feat_mid)  # (B, mid, H/2, W/2)
                feat_mid = self.head.up2b(feat_mid)  # (B, mid, H, W)
            else:
                feat_mid = F.interpolate(feat_mid, size=(H, W), mode='bilinear', align_corners=False)
            
            guide = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
            
            logits_raw = self.head.classifier(feat_mid)  # (B, C, H, W)
            
            if not hasattr(self, 'gum'):
                self.gum = self.head.GuidedUp(k=3).to(logits_raw.device)
            logits = self.gum(logits_raw, guide)
        else:
            logits = self.head(neck_out, out_size=(H, W))
        
        # Final numerical checks on logits; fall back to safe defaults if corrupted.
        if torch.isnan(logits).all() or torch.isinf(logits).all():
            print(f"[ERROR] All logits are NaN/Inf! This indicates model parameters may be corrupted.")
            print(f"       Logits stats: min={logits.min()}, max={logits.max()}, nan_count={torch.isnan(logits).sum()}, inf_count={torch.isinf(logits).sum()}")
            logits = torch.zeros_like(logits)
            logits[:, 0] = -0.1  # 稍微偏向背景
            logits[:, 1] = 0.1   # 稍微偏向前景
        elif torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[WARN] Some logits are NaN/Inf, replacing with safe values")
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
        
        outputs = {"logits": logits}
        
        # Provide 1/4-scale features for geometric boundary losses.
        if self.use_boundary:
            outputs["feat_1_4"] = neck_out  # (B, 256, H/4, W/4)
        
        if self.use_aux and self.training:
            aux_out = self.aux_head(features["F8"], out_size=(H, W))
            outputs["aux"] = aux_out
        
        # 5. Boundary head outputs for loss/monitoring (also during validation).
        if self.use_boundary:
            if self.boundary_multiscale:
                b1x = F.interpolate(b_for_loss['b4'], size=(H, W), mode='bilinear', align_corners=False)
                outputs["boundary"] = b1x
                outputs["boundary_b4"] = b_for_loss['b4']   # (B,C|1,H/4,W/4)
                outputs["boundary_b8"] = b_for_loss['b8']   # (B,C|1,H/8,W/8)
                if b_for_loss.get('dir4') is not None:
                    outputs["dir_b4"] = b_for_loss['dir4']
                    outputs["dir_b8"] = b_for_loss['dir8']
            else:
                b1x = F.interpolate(b_for_loss['b4'], size=(H, W), mode='bilinear', align_corners=False)
                outputs["boundary"] = b1x
                outputs["boundary_b4"] = b_for_loss['b4']
                if b_for_loss.get('dir4') is not None:
                    outputs["dir_b4"] = b_for_loss['dir4']
        
        if self.use_hq:
            hq_mask = self.hq_adapter(neck_out, out_size_hw=(H, W))
            outputs["hq_mask"] = hq_mask
        
        return outputs


def build_model(config):
    """Build PH2 variant of BG-SegNet from a configuration dictionary."""
    model = CSegNet(config)
    return model