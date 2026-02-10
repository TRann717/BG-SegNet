#!/usr/bin/env python
"""
BG-SegNet: 5-fold cross-validation training on the PH2 dataset.

Goals:
- Train the BG-SegNet model (build_model + Trainer) on PH2 using the raw directory layout.
- Leverage MaskSegDataset's automatic parsing of original PH2 structure (no manual reorganization).
- Report key metrics: Dice, mIoU, precision, recall, FPS, parameter count, GPU memory, and HD95.

Notes:
- Dice/mIoU/precision/recall come from Trainer.validate -> RunningScore.
- HD95 is derived from BoundaryIoU (requires model.use_boundary=True).
- FPS is measured here with a dedicated forward-only timing loop.
- Parameter count and approximate model memory are computed analytically (params * 4 bytes).
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import cv2

# 添加 src 到路径
import sys
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_mask import get_dataloader
from src.models.cseg_net import build_model
from src.engine.trainer import Trainer
from src.utils.logger import TBLogger, increment_path


def set_seed(seed: int, cudnn_benchmark: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def check_ph2_masks(mask_paths: List[Path]) -> Tuple[int, int]:
    """
    Quick check of PH2 lesion masks for emptiness.

    Returns:
        (total_masks, zero_mask_count)
    """
    zero_count = 0
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            zero_count += 1
            continue
        if m.max() == 0:
            zero_count += 1
    return len(mask_paths), zero_count


def metrics_from_confusion_matrix(cm: np.ndarray) -> Dict[str, float]:
    """
    Compute Dice/precision/recall for the foreground class (class 1)
    from a 2x2 confusion matrix (rows = GT, columns = predictions).
    """
    eps = 1e-6
    cm = cm.astype(np.float64)
    if cm.shape != (2, 2):
        return {"dice": float("nan"), "precision": float("nan"), "recall": float("nan")}

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {"dice": float(dice), "precision": float(precision), "recall": float(recall)}


@torch.no_grad()
def measure_fps(model: torch.nn.Module, dataloader, device: torch.device, warmup: int = 5) -> float:
    """
    Measure inference throughput (FPS) by timing forward passes only.
    """
    model.eval()
    times: List[float] = []
    num_images = 0

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)

        # warmup：避免首批 cuDNN/Kernel 初始化影响
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            times.append(t1 - t0)
            num_images += images.size(0)

    total = float(np.sum(times)) if len(times) > 0 else 0.0
    if total <= 0 or num_images <= 0:
        return 0.0
    return num_images / total


def compute_model_stats(model: torch.nn.Module, img_size: Tuple[int, int], device: torch.device) -> Dict[str, Optional[float]]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = total_params * 4 / (1024 ** 2)  # float32 估算

    # FLOPs are intentionally omitted because external SAM2 + GuidedUp are not
    # compatible with standard FLOPs profilers; only param/memory stats are reported.
    flops_g = None

    gpu_mem_mb = None
    if device.type == "cuda":
        try:
            print("[DEBUG] current allocated(MB) before dummy:",
            torch.cuda.memory_allocated() / (1024 ** 2))
            torch.cuda.reset_peak_memory_stats()
            dummy = torch.randn(1, 3, img_size[0], img_size[1], device=device)
            _ = model(dummy)
            torch.cuda.synchronize()
            gpu_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            torch.cuda.empty_cache()
        except Exception:
            gpu_mem_mb = None

    return {
        "Params(M)": float(total_params / 1e6),
        "TrainableParams(M)": float(trainable_params / 1e6),
        "FLOPs(G)": flops_g,
        "ModelMem(MB)": float(param_memory_mb),
        "GPUMemInfer(MB)": gpu_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="BG-SegNet PH2 5-fold training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="bgsegnet config.yaml path",
    )
    parser.add_argument(
        "--ph2_root",
        type=str,
        default="/root/autodl-tmp/BG-SegNet/PH2 Dataset/PH2Dataset",
        help="PH2Dataset root directory (should contain 'PH2 Dataset images' subdirectory).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds K (default 5).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["data"]["dataset_root"] = args.ph2_root
    config["data"]["cv_num_folds"] = int(args.folds)
    config["data"]["cv_seed"] = config["data"].get("cv_seed", 42)

    seed = int(config["train"]["seed"])
    cudnn_benchmark = bool(config["train"].get("cudnn_benchmark", False))
    set_seed(seed, cudnn_benchmark=cudnn_benchmark)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PH2 root: {config['data']['dataset_root']}")
    print(f"Using {config['data']['cv_num_folds']}-fold cross-validation, seed={config['data']['cv_seed']}")

    img_size = tuple(config["data"]["img_size"])

    dice_list: List[float] = []
    miou_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []
    hd95_list: List[float] = []
    fps_list: List[float] = []

    model_stats_global: Optional[Dict[str, Optional[float]]] = None

    for fold in range(int(config["data"]["cv_num_folds"])):
        print(f"\n------ Fold {fold + 1}/{config['data']['cv_num_folds']} ------")
        config["data"]["cv_fold_index"] = fold

        # Dataloaders: here 'val' is treated as the test split of this fold.
        train_loader = get_dataloader(config, split="train")
        val_loader = get_dataloader(config, split="val")
        print(f"Train dataset (fold {fold}): {len(train_loader.dataset)} samples")
        print(f"Test  dataset (fold {fold}): {len(val_loader.dataset)} samples")
        assert len(train_loader.dataset) > 0, f"Training dataset is empty in fold {fold}!"
        assert len(val_loader.dataset) > 0, f"Test dataset is empty in fold {fold}!"

        # Sanity check: PH2 masks should contain foreground for most lesions.
        if getattr(val_loader.dataset, "mask_files", None):
            total_masks, zero_masks = check_ph2_masks(val_loader.dataset.mask_files)
            if zero_masks > 0:
                print(f"[WARN] Fold {fold}: {zero_masks}/{total_masks} val masks are all background;")
                print(f"       this may indicate path/filename issues or failed reads, and will degrade Dice/Precision/Recall.")
        
        # === Debug: inspect a few train/val masks for correctness ===
        print("\n[DEBUG] 检查数据加载格式...")
        sample_train = train_loader.dataset[0]
        sample_val = val_loader.dataset[0]
        print(f"  - Train sample mask shape: {sample_train['mask'].shape}, dtype: {sample_train['mask'].dtype}, unique: {torch.unique(sample_train['mask'])}")
        print(f"  - Val sample mask shape: {sample_val['mask'].shape}, dtype: {sample_val['mask'].dtype}, unique: {torch.unique(sample_val['mask'])}")
        print(f"  - Train sample mask foreground pixels: {(sample_train['mask'] == 1).sum().item()}")
        print(f"  - Val sample mask foreground pixels: {(sample_val['mask'] == 1).sum().item()}")
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        print(f"  - Train batch mask shape: {train_batch['mask'].shape}, dtype: {train_batch['mask'].dtype}")
        print(f"  - Val batch mask shape: {val_batch['mask'].shape}, dtype: {val_batch['mask'].dtype}")
        print(f"  - Train batch mask foreground pixels: {(train_batch['mask'] == 1).sum().item()}")
        print(f"  - Val batch mask foreground pixels: {(val_batch['mask'] == 1).sum().item()}")

        # Build a fresh model per fold to avoid parameter leakage across folds.
        model = build_model(config).to(device)
        
        # Check for NaN/Inf in initialized parameters.
        has_nan_params = False
        nan_param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"[ERROR] Fold {fold}: Model parameter {name} contains NaN/Inf! nan_count={nan_count}, inf_count={inf_count}")
                    nan_param_names.append(name)
                    has_nan_params = True
        if has_nan_params:
            print(f"[ERROR] Fold {fold}: Model has corrupted parameters after initialization!")
            print(f"        Corrupted parameters: {nan_param_names}")
            print(f"        Reinitializing model...")
            torch.cuda.empty_cache()
            model = build_model(config).to(device)
            # 再次检查
            for name, param in model.named_parameters():
                if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                    print(f"[ERROR] Fold {fold}: Model parameter {name} STILL contains NaN/Inf after reinitialization!")
                    raise RuntimeError(f"Model initialization failed for fold {fold}")
        
        print(f"[INFO] Fold {fold}: Model initialized successfully, all parameters are clean.")

        # Compute model statistics only once (on the first fold).
        if fold == 0 and model_stats_global is None:
            print("Computing model statistics...")
            model_stats_global = compute_model_stats(model, img_size, device)
            print(
                f"[MODEL] Params(M)={model_stats_global['Params(M)']:.3f}, "
                f"FLOPs(G)={model_stats_global['FLOPs(G)'] if model_stats_global['FLOPs(G)'] is not None else 'NA'}, "
                f"ModelMem(MB)={model_stats_global['ModelMem(MB)']:.3f}, "
                f"GPU-Mem(Infer,MB)={model_stats_global['GPUMemInfer(MB)'] if model_stats_global['GPUMemInfer(MB)'] is not None else 'NA'}"
            )

        # Per-fold logging directory.
        base_dir = Path(config["log"]["outdir"]) / f"{config['experiment']}_ph2_fold{fold}"
        save_dir = increment_path(base_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = TBLogger(save_dir, config, config_path=args.config)

        trainer = Trainer(model, config, device, save_dir=save_dir, logger=logger)
        trainer.fit(train_loader, val_loader, start_epoch=1)

        # Evaluate this fold: metrics include mIoU/Dice/precision/recall/HD95.
        val_loss, val_scores = trainer.validate(val_loader)

        # === Debug: inspect logits, predictions, GT masks, and confusion matrix ===
        print("\n[DEBUG] 检查验证集数据流...")
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            images = sample_batch["image"].to(device)
            masks_gt = sample_batch["mask"].to(device)
            
            # Squeeze channel dimension if masks are (B,1,H,W).
            if masks_gt.dim() == 4 and masks_gt.size(1) == 1:
                print(f"  [WARN] GT mask 是 4D (B,1,H,W)，压缩为 (B,H,W)")
                masks_gt = masks_gt.squeeze(1)
            
            outputs = model(images)
            preds = outputs["logits"].argmax(dim=1)
            
            print(f"  - GT mask shape: {masks_gt.shape}, dtype: {masks_gt.dtype}, unique: {torch.unique(masks_gt)}")
            print(f"  - Pred mask shape: {preds.shape}, dtype: {preds.dtype}, unique: {torch.unique(preds)}")
            print(f"  - GT foreground pixels: {(masks_gt == 1).sum().item()}")
            print(f"  - Pred foreground pixels: {(preds == 1).sum().item()}")
            print(f"  - Logits shape: {outputs['logits'].shape}, logits range: [{outputs['logits'].min().item():.2f}, {outputs['logits'].max().item():.2f}]")
            print(f"  - Logits class 0 mean: {outputs['logits'][:, 0].mean().item():.2f}, class 1 mean: {outputs['logits'][:, 1].mean().item():.2f}")
        
        # Manually compute a small confusion matrix for sanity-check.
        from src.engine.metrics import RunningScore
        debug_metrics = RunningScore(num_classes=2, ignore_index=255)
        debug_metrics.update(preds.cpu(), masks_gt.cpu())
        debug_scores = debug_metrics.get_scores()
        print(f"  - 手动计算的混淆矩阵:\n{debug_metrics.confusion_matrix}")
        print(f"  - debug_scores 的所有键: {list(debug_scores.keys())}")
        print(f"  - 手动计算的 Dice: {debug_scores.get('dice', float('nan')):.4f}, Precision: {debug_scores.get('precision', float('nan')):.4f}, Recall: {debug_scores.get('recall', float('nan')):.4f}")
        if 'iou_per_class' in debug_scores:
            print(f"  - IoU per class: {debug_scores['iou_per_class']}")
        if 'dice_per_class' in debug_scores:
            print(f"  - Dice per class: {debug_scores['dice_per_class']}")
        if 'precision_per_class' in debug_scores:
            print(f"  - Precision per class: {debug_scores['precision_per_class']}")
        if 'recall_per_class' in debug_scores:
            print(f"  - Recall per class: {debug_scores['recall_per_class']}")
        
        print(f"\n[DEBUG] Trainer.validate 返回的 scores:")
        print(f"  - dice: {val_scores.get('dice', 'NOT FOUND')}")
        print(f"  - precision: {val_scores.get('precision', 'NOT FOUND')}")
        print(f"  - recall: {val_scores.get('recall', 'NOT FOUND')}")
        print(f"  - miou: {val_scores.get('miou', 'NOT FOUND')}")
        if 'iou_per_class' in val_scores:
            print(f"  - iou_per_class: {val_scores['iou_per_class']}")
        if 'dice_per_class' in val_scores:
            print(f"  - dice_per_class: {val_scores['dice_per_class']}")

        # FPS measured with separate timing on the validation loader.
        fps = measure_fps(model, val_loader, device)

        miou = float(val_scores.get("miou", 0.0))
        # Backward-compatible: derive Dice/precision/recall from confusion matrix
        # if validate() did not explicitly return them.
        if "dice" in val_scores and "precision" in val_scores and "recall" in val_scores:
            dice = float(val_scores.get("dice", 0.0))
            precision = float(val_scores.get("precision", 0.0))
            recall = float(val_scores.get("recall", 0.0))
        else:
            cm = getattr(trainer.val_metrics, "confusion_matrix", None)
            if cm is not None:
                m = metrics_from_confusion_matrix(cm)
                dice, precision, recall = m["dice"], m["precision"], m["recall"]
                print(f"[INFO] Fallback metrics from confusion matrix: Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            else:
                dice, precision, recall = 0.0, 0.0, 0.0
        hd95 = float(val_scores.get("hd95", 0.0)) if "hd95" in val_scores else 0.0

        print(
            f"\n[FOLD {fold + 1}] "
            f"Dice={dice:.4f}, "
            f"mIoU={miou:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"FPS={fps:.2f}, "
            f"HD95={hd95:.2f} (像素), "
            f"Loss={val_loss:.4f}"
        )

        dice_list.append(dice)
        miou_list.append(miou)
        precision_list.append(precision)
        recall_list.append(recall)
        hd95_list.append(hd95)
        fps_list.append(fps)

        logger.close()

    def mean_std(x: List[float]) -> Tuple[float, float]:
        return float(np.mean(x)), float(np.std(x))

    dice_mean, dice_std = mean_std(dice_list)
    miou_mean, miou_std = mean_std(miou_list)
    prec_mean, prec_std = mean_std(precision_list)
    rec_mean, rec_std = mean_std(recall_list)
    hd95_mean, hd95_std = mean_std(hd95_list)
    fps_mean, fps_std = mean_std(fps_list)

    print("\n" + "=" * 60)
    print("====== PH2 5-fold cross-validation summary ======")
    print(f"Dice:      {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"mIoU:      {miou_mean:.4f} ± {miou_std:.4f}")
    print(f"Precision: {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"Recall:    {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"FPS:       {fps_mean:.2f} ± {fps_std:.2f}")
    print(f"HD95:      {hd95_mean:.2f} ± {hd95_std:.2f} (pixels)")

    if model_stats_global is not None:
        print("\nModel Statistics:")
        print(f"  - Params(M):           {model_stats_global['Params(M)']:.3f}")
        if model_stats_global.get("TrainableParams(M)") is not None:
            print(f"  - TrainableParams(M):  {model_stats_global['TrainableParams(M)']:.3f}")
        if model_stats_global.get("FLOPs(G)") is not None:
            print(f"  - FLOPs(G):            {model_stats_global['FLOPs(G)']:.3f}")
        else:
            print("  - FLOPs(G):            NA (GuidedUp FLOPs ignored)")
        print(f"  - ModelMem(MB):        {model_stats_global['ModelMem(MB)']:.3f}")
        if model_stats_global.get("GPUMemInfer(MB)") is not None:
            print(f"  - GPU Mem(MB):         {model_stats_global['GPUMemInfer(MB)']:.3f} (inference peak)")

    print("=" * 60)


if __name__ == "__main__":
    main()

