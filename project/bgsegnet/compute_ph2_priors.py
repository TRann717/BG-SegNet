"""
Compute PH2 pixel-class priors (class_freq) and suggest class_weights.

PH2 raw structure expected:
  <ph2_root>/PH2 Dataset images/IMDxxx/IMDxxx_lesion*/<mask>.(bmp|png|jpg|jpeg)

This follows the same pairing logic as `src/data/dataset_mask.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def collect_ph2_lesion_masks(ph2_root: Path) -> List[Path]:
    images_root = ph2_root / "PH2 Dataset images"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing folder: {images_root}")

    mask_paths: List[Path] = []
    imd_dirs = [d for d in images_root.iterdir() if d.is_dir() and d.name.startswith("IMD")]
    imd_dirs = sorted(imd_dirs, key=lambda p: p.name)

    for imd_dir in imd_dirs:
        lesion_id = imd_dir.name
        lesion_dirs = list(imd_dir.glob(f"{lesion_id}_lesion*"))
        if not lesion_dirs:
            continue
        lesion_dir = lesion_dirs[0]

        mask_candidates = sorted(
            list(lesion_dir.glob("*.bmp"))
            + list(lesion_dir.glob("*.png"))
            + list(lesion_dir.glob("*.jpg"))
            + list(lesion_dir.glob("*.jpeg"))
        )
        if not mask_candidates:
            continue
        mask_paths.append(mask_candidates[0])

    if not mask_paths:
        raise RuntimeError(f"No lesion masks found under: {images_root}")
    return mask_paths


def compute_binary_mask_fg_ratio(mask_paths: List[Path]) -> Tuple[float, int, int]:
    fg = 0
    total = 0
    bad = 0

    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            bad += 1
            continue
        # PH2 lesion masks are typically 0/255; be robust to any non-zero foreground.
        m_fg = (m > 0).astype(np.uint8)
        fg += int(m_fg.sum())
        total += int(m_fg.size)

    if total <= 0:
        raise RuntimeError("No valid masks read; total pixels is 0.")

    fg_ratio = float(fg) / float(total)
    return fg_ratio, total, bad


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ph2_root",
        type=str,
        default=r"/root/autodl-tmp/BG-SegNet/PH2 Dataset/PH2Dataset",
        help="PH2Dataset root (contains 'PH2 Dataset images')",
    )
    args = parser.parse_args()

    root = Path(args.ph2_root)
    mask_paths = collect_ph2_lesion_masks(root)
    fg_ratio, total_px, bad = compute_binary_mask_fg_ratio(mask_paths)
    bg_ratio = 1.0 - fg_ratio

    # Suggested weights:
    # - linear inverse frequency can be too aggressive; sqrt is a stable default for CE.
    inv = bg_ratio / max(fg_ratio, 1e-12)
    w_fg_sqrt = float(np.sqrt(inv))
    w_fg_linear = float(inv)

    print(f"[PH2] masks: {len(mask_paths)} (bad_reads={bad})")
    print(f"[PH2] total_pixels: {total_px}")
    print(f"[PH2] class_freq: [bg={bg_ratio:.6f}, fg={fg_ratio:.6f}]  (sum={bg_ratio+fg_ratio:.6f})")
    print(f"[PH2] suggested class_weights (CE): [1.0, sqrt(bg/fg)={w_fg_sqrt:.3f}]")
    print(f"[PH2] (aggressive) class_weights (CE): [1.0, (bg/fg)={w_fg_linear:.3f}]")


if __name__ == "__main__":
    main()

