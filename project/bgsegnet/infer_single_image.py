import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'bgsegnet' / 'src'))

from src.config.loader import load_config
from src.models.cseg_net import CSegNet


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def preprocess(img_bgr: np.ndarray, size_hw: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize to (H,W), convert to float tensor [0,1] in RGB order."""
    orig_h, orig_w = img_bgr.shape[:2]
    H, W = size_hw
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    img_f32 = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return tensor, (orig_h, orig_w)


def save_mask(mask01: np.ndarray, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, (mask01 * 255).astype(np.uint8))


def load_model_from_ckpt(config_path: str, checkpoint_path: str, device: torch.device) -> CSegNet:
    config = load_config(config_path)
    model = CSegNet(config).to(device)
    model.eval()

    # Run a dummy forward pass so lazily created submodules (e.g., GUM) are built.
    try:
        H, W = tuple(config['data']['img_size'])
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W, device=device)
            _ = model(dummy)
    except Exception as e:
        print(f"[WARN] Dummy forward failed (will still try to load): {e}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # Strip DataParallel-style 'module.' prefixes if present.
    new_state = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        new_state[nk] = v

    # Strict load to ensure all weights (including GUM) are correctly restored.
    missing, unexpected = model.load_state_dict(new_state, strict=True)
    if missing:
        print(f"[INFO] Missing keys: {missing}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {unexpected}")
    return model


def infer_single(config_path: str, checkpoint_path: str, image_path: str, out_path: str,
                 threshold: float = 0.5, device_str: str | None = None):
    device = torch.device(device_str or ('cuda' if torch.cuda.is_available() else 'cpu'))
    config = load_config(config_path)

    
    model = load_model_from_ckpt(config_path, checkpoint_path, device)

    
    img_bgr = load_image_bgr(image_path)
    H, W = tuple(config['data']['img_size'])
    inp, (orig_h, orig_w) = preprocess(img_bgr, (H, W))
    inp = inp.to(device)

    with torch.no_grad():
        outputs = model(inp)
       
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                preds = outputs['logits']
            elif 'main' in outputs:
                preds = outputs['main']
            elif 'pred' in outputs:
                preds = outputs['pred']
            elif 'output' in outputs:
                preds = outputs['output']
            else:
                first_key = list(outputs.keys())[0]
                print(f"[WARN] Using first output key '{first_key}' instead of 'logits'")
                preds = outputs[first_key]
        else:
            preds = outputs

        if preds.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {preds.shape}")

       
        if preds.shape[1] == 2:
            probs = torch.softmax(preds, dim=1)
            fg_prob = probs[:, 1:2]
        elif preds.shape[1] == 1:
            fg_prob = torch.sigmoid(preds)
        else:
            probs = torch.softmax(preds[:, :2], dim=1)
            fg_prob = probs[:, 1:2]

        bin_mask = (fg_prob >= threshold).to(torch.uint8)  # 1x1xHxW
       
        bin_mask = F.interpolate(bin_mask.float(), size=(orig_h, orig_w), mode='nearest').to(torch.uint8)

    mask_np = bin_mask.squeeze(0).squeeze(0).cpu().numpy()
    save_mask(mask_np, out_path)
    print(f"Saved binary mask to: {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description='Single-image inference for BG-SegNet (binary mask output).')
    p.add_argument('--config', type=str, default="/root/autodl-tmp/BG-SegNet/project/bgsegnet/config.yaml", help='Path to YAML config used for training')
    p.add_argument('--checkpoint', type=str,  default="/root/autodl-tmp/BG-SegNet/best.pth", help='Path to model checkpoint (.pth/.pt)')
    p.add_argument('--image', type=str,  default="/root/autodl-tmp/BG-SegNet/对比实验/fusc_0014.png", help='Path to input image')
    p.add_argument('--out', type=str,  default="/root/autodl-tmp/BG-SegNet/对比实验/my_model_results/fusc_0005_mask.png", help='Path to save binary mask (png)')
    p.add_argument('--threshold', type=float, default=0.5, help='Foreground threshold (default=0.5)')
    p.add_argument('--device', type=str,  default="cuda", help='cuda|cpu (default auto)')
    return p


def main():
    args = build_argparser().parse_args()
    infer_single(args.config, args.checkpoint, args.image, args.out, args.threshold, args.device)


if __name__ == '__main__':
    main() 
