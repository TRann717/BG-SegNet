"""
BG-SegNet: medical image segmentation network (generic variant).
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

        # Probe backbone feature channels using a dummy input at configured img_size.
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
            in_channels_dict=in_channels_dict,  
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
        
        # Initialize classifier bias with class-frequency priors if available.
        class_freq = data_cfg.get('class_freq', None)
        if class_freq is not None:
            self.head.init_bias_with_priors(class_freq)
        
        # Auxiliary head for deep supervision.
        self.use_aux = model_cfg['head']['aux_loss']
        if self.use_aux:
            self.aux_head = AuxHead(
                in_channels=in_channels_dict['F8'],  
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
            self.alpha_max = model_cfg['boundary'].get('alpha_max', 0.35)  
            self.gamma_alpha = model_cfg['boundary'].get('gamma_alpha', 1.0)
            
            # Boundary supervision heads (only used in loss), with optional 1/4 and 1/8 scales.
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
            
            # Gating fusion and gradient-flow strategy.
            gate_cfg = model_cfg['boundary'].get('gate', {})
            self.g_lambda_f4 = float(gate_cfg.get('lambda_f4', 0.6))
            self.g_lambda_f8 = float(gate_cfg.get('lambda_f8', 0.4))
            self._gate_detach = not bool(gate_cfg.get('allow_gate_grad', False))
        
        # Optional HQ boundary enhancement (disabled by default, may be enabled if bIoU is low).
        self.use_hq = False
        if self.use_hq:
            self.hq_adapter = HQTokenAdapter(
                in_channels=model_cfg['neck']['out_channels'],
                mid=256
            )
    
    def set_gate_detach(self, detach: bool = True):
        """Switch whether gate activations are detached (for progressive unfreezing)."""
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
        
        # 1. Extract SAM2 multi-scale features.
        features = self.backbone(images)
        
        # 2. Neck fusion.
        if self.use_boundary and self.boundary_multiscale:
            neck_out_dict = self.neck(features, return_pyramid=True)
            neck_out = neck_out_dict["out"]          # 1/4 fused
            p4 = neck_out_dict["p4"]                 # 1/4
            p8 = neck_out_dict["p8"]                 # 1/8
        else:
            neck_out = self.neck(features)
        
        # 2.5. Boundary-guided refinement (if enabled).
        if self.use_boundary:
            # 1/4 boundary logits (may be multi-channel if class-aware).
            b4_pack = self.boundary_head_4(neck_out, out_size=None)
            b4 = b4_pack['b_logits']                     # (B,C|1,H/4,W/4)
            dir4 = b4_pack.get('dir', None)
            g4_all = torch.sigmoid(b4)                   # (B,C|1,H/4,W/4)
            
            # Estimate coarse foreground probability p_fg for class-mixing in boundary gate.
            if self.classmix_enable and hasattr(self, 'head'):
                with torch.no_grad():
                    coarse_logits = self.head.classifier(self.head.feat(neck_out))
                    p_fg = torch.softmax(coarse_logits, dim=1)[:, 1:2]  
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
                
                # Boundary tensors passed to loss functions.
                b_for_loss = {
                  'b4': b4, 'b8': b8,
                  'dir4': dir4, 'dir8': dir8
                }
            else:
                g = g4
                b_for_loss = {'b4': b4, 'dir4': dir4}
            
            # Progressive decoupling: detach gate early in training, unfreeze later.
            if self._gate_detach:
                g = g.detach()
            
            # Normalize boundary direction field if OABG is enabled.
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
            # Guided upsampling with boundary strength as guide.
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
        
        outputs = {"logits": logits}
        
        # 1/4-scale features used by geometric boundary losses.
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
    """Build generic BG-SegNet model from configuration."""
    model = CSegNet(config)
    return model
