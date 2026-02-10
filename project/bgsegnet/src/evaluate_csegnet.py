#!/usr/bin/env python3
"""
Evaluation script for BG-SegNet on YOLO-style segmentation test sets.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import sys
sys.path.append('bgsegnet/src')

from models.cseg_net import CSegNet
from config.loader import load_config
from engine.metrics import RunningScore, BoundaryIoU


class YOLOSegDataset:
    """YOLO-format segmentation dataset loader for evaluation."""
    
    def __init__(self, data_root: str, split: str = "test", img_size: Tuple[int, int] = (640, 640)):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        
        self.img_dir = self.data_root / split / "images"
        self.label_dir = self.data_root / split / "labels"
        
        self.image_paths = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        self.label_paths = sorted(list(self.label_dir.glob("*.txt")))
        
        print(f"Found {len(self.image_paths)} images and {len(self.label_paths)} labels in {split} split")
        
        if len(self.image_paths) != len(self.label_paths):
            print(f"Warning: Image and label counts don't match!")
            print(f"Images: {len(self.image_paths)}, Labels: {len(self.label_paths)}")
            
            # 只保留有对应标签的图像
            self._filter_matching_pairs()
            
            print(f"After filtering: {len(self.image_paths)} valid image-label pairs")
    
    def _filter_matching_pairs(self):
        """Filter image/label pairs to keep only those with a matching annotation."""
        valid_pairs = []
        
        for img_path in self.image_paths:
            # 获取图像文件名（不含扩展名）
            img_name = img_path.stem
            
            # 查找对应的标签文件
            label_path = self.label_dir / f"{img_name}.txt"
            
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
            else:
                print(f"Warning: No label found for image {img_name}")
        
        self.image_paths = [pair[0] for pair in valid_pairs]
        self.label_paths = [pair[1] for pair in valid_pairs]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_paths) or idx >= len(self.label_paths):
            raise IndexError(f"Index {idx} out of range. Images: {len(self.image_paths)}, Labels: {len(self.label_paths)}")
        
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        
        label_path = self.label_paths[idx]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 3:  # requires at least class id and a few vertices
                            class_id = int(parts[0])
                            if class_id == 0:  # class 0 = foreground lesion
                                coords = []
                                for i in range(1, len(parts), 2):
                                    if i + 1 < len(parts):
                                        x = float(parts[i])
                                        y = float(parts[i + 1])
                                        coords.append([x, y])
                                
                                if len(coords) >= 3:  # at least 3 points to form a polygon
                                    h, w = img.shape[:2]
                                    coords = np.array(coords)
                                    coords[:, 0] *= w  # x坐标
                                    coords[:, 1] *= h  # y坐标
                                    coords = coords.astype(np.int32)
                                    
                                    cv2.fillPoly(mask, [coords], 1)
            except Exception as e:
                print(f"Warning: Error reading label file {label_path}: {e}")
                # fall back to all-zero mask
        
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return img, mask, img_path.name


class CSegNetEvaluator:
    """Evaluator for BG-SegNet models on binary lesion segmentation."""
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str, 
                 pred_threshold: float = 0.5, num_samples: int = 50):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.pred_threshold = pred_threshold
        self.num_samples = num_samples
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "comparison_images").mkdir(exist_ok=True)
        
        self.config = load_config(config_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        self.running_score = RunningScore(num_classes=2, ignore_index=255)
        
        # Boundary IoU with a small dilation ratio suitable for fine lesion contours.
        self.boundary_iou = BoundaryIoU(num_classes=2, dilation_ratio=0.01)
        
        self.band_width = 6  # 边界带宽度（像素）
        self.band_ious = []  # 存储每个样本的边界带IoU
        
        self.total_inference_time = 0.0
        self.total_frames = 0
        self.comparison_images = []
        
    def _load_model(self) -> CSegNet:
        """Load BG-SegNet model and checkpoint weights."""
        print(f"Loading model from config: {self.config_path}")
        model = CSegNet(self.config)
        
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device,weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model loaded successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _make_overlay(self, background_img: Image.Image, bin_mask: np.ndarray, 
                      overlay_color: Tuple[int, int, int, int]) -> Image.Image:
        """Create an RGBA overlay of a binary mask on top of a background image."""
        colored_mask = np.zeros((*bin_mask.shape, 4), dtype=np.uint8)
        colored_mask[bin_mask == 1] = overlay_color
        
        mask_img = Image.fromarray(colored_mask, mode='RGBA')
        
        result = background_img.convert('RGBA')
        result = Image.alpha_composite(result, mask_img)
        
        return result.convert('RGB')
    
    def _make_black_bg_overlay(self, gt_mask: np.ndarray, pred_mask: np.ndarray, 
                               target_size: Tuple[int, int]) -> Image.Image:
        """Create a black-background comparison image of GT vs prediction."""
        h, w = target_size
        
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        result[gt_mask == 1] = [0, 255, 0]
        
        result[pred_mask == 1] = [255, 0, 0]
        
        overlap = (gt_mask == 1) & (pred_mask == 1)
        result[overlap] = [255, 255, 0]
        
        return Image.fromarray(result)
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Run evaluation loop over a DataLoader and collect metrics."""
        print("Starting evaluation...")
        
        dataset = data_loader.dataset
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {data_loader.batch_size}")
        print(f"Total batches: {len(data_loader)}")
        
        try:
            first_sample = dataset[0]
            print(f"First sample loaded successfully: {type(first_sample)}")
        except Exception as e:
            print(f"Error loading first sample: {e}")
            raise
        
        with torch.no_grad():
            for batch_idx, (images, masks, image_names) in enumerate(tqdm(data_loader, desc="Evaluating")):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                
                if batch_idx == 0:  # 只在第一个batch打印
                    print(f"Model output type: {type(outputs)}")
                    if isinstance(outputs, dict):
                        print(f"Model output keys: {list(outputs.keys())}")
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                            else:
                                print(f"  {key}: type={type(value)}")
                    elif isinstance(outputs, torch.Tensor):
                        print(f"Model output tensor: shape={outputs.shape}, dtype={outputs.dtype}")
                    else:
                        print(f"Model output: {outputs}")
                
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
                        # 如果都不存在，取第一个键的值
                        first_key = list(outputs.keys())[0]
                        print(f"Warning: Using first output key '{first_key}' instead of 'logits'")
                        preds = outputs[first_key]
                else:
                    preds = outputs
                
                if not isinstance(preds, torch.Tensor):
                    raise ValueError(f"Expected tensor output, got {type(preds)}")
                
                if len(preds.shape) != 4:
                    raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {preds.shape}")
                
                if preds.shape[1] != 2:
                    print(f"Warning: Expected 2 classes, got {preds.shape[1]}. Using first channel as foreground.")
                    if preds.shape[1] == 1:
                        # If single-channel logit map is provided, interpret as foreground logit/prob.
                        preds = torch.cat([1 - preds, preds], dim=1)
                    else:
                        preds = preds[:, :2]
                
                preds_probs = torch.softmax(preds, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                for b in range(images.size(0)):
                    foreground_probs = preds_probs[b, 1]  # (H, W)
                    pred_mask = (foreground_probs >= self.pred_threshold).astype(np.uint8)
                    gt_mask = masks_np[b].astype(np.uint8)
                    
                    pred_2class = np.zeros_like(pred_mask)
                    pred_2class[pred_mask == 1] = 1
                    
                    gt_2class = np.zeros_like(gt_mask)
                    gt_2class[gt_mask == 1] = 1
                    
                    self.running_score.update(pred_2class, gt_2class)
                    self.boundary_iou.update(pred_2class, gt_2class)
                    
                    # 计算边界带IoU
                    band_iou = self._compute_band_iou(pred_2class, gt_2class)
                    self.band_ious.append(band_iou)
                
                if batch_idx * data_loader.batch_size < self.num_samples:
                    for b in range(images.size(0)):
                        if len(self.comparison_images) >= self.num_samples:
                            break
                        
                        foreground_probs = preds_probs[b, 1]  # (H, W)
                        pred_mask_np = (foreground_probs >= self.pred_threshold).astype(np.uint8)
                        gt_mask_np = masks_np[b].astype(np.uint8)
                        
                        img_np = images[b].cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        pil_orig = Image.fromarray(img_np)
                        w, h = pil_orig.size
                        
                        panel_original = pil_orig
                        
                        panel_overlay = self._make_overlay(
                            background_img=pil_orig,
                            bin_mask=pred_mask_np,
                            overlay_color=(0, 0, 255, 120)
                        )
                        
                        panel_black = self._make_black_bg_overlay(
                            gt_mask=gt_mask_np,
                            pred_mask=pred_mask_np,
                            target_size=(w, h)
                        )
                        
                        combined = Image.new("RGB", (w * 3, h))
                        combined.paste(panel_original, (0, 0))
                        combined.paste(panel_overlay, (w, 0))
                        combined.paste(panel_black, (2 * w, 0))
                        
                        self.comparison_images.append(combined)
                        
                        save_path = self.output_dir / "comparison_images" / f"test_sample_{len(self.comparison_images)}_{image_names[b]}"
                        combined.save(save_path)
                
                self.total_inference_time += inference_time
                self.total_frames += images.size(0)
        
        # 计算最终指标
        return self._compute_final_metrics()
    
    def _compute_final_metrics(self) -> Dict:
        """
        Compute final region-, boundary-, and binary-classification metrics.

        Includes mIoU, per-class IoU, boundary IoU, band IoU, Dice, precision,
        recall, F1, accuracy, and simple throughput/memory statistics.
        """
        # 获取基础指标
        scores = self.running_score.get_scores()
        boundary_scores = self.boundary_iou.get_scores()
        
        band_iou = np.mean(self.band_ious) if len(self.band_ious) > 0 else 0
        
        cm = self.running_score.confusion_matrix
        tp = cm[1, 1]  # 真正例：预测为1，实际为1
        fp = cm[0, 1]  # 假正例：预测为1，实际为0
        fn = cm[1, 0]  # 假负例：预测为0，实际为1
        tn = cm[0, 0]  # 真负例：预测为0，实际为0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        dice_cm = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        p_sum = tp + fp  # 预测为正例的总数
        gt_sum = tp + fn  # 真实为正例的总数
        dice_sum = 2 * tp / (p_sum + gt_sum) if (p_sum + gt_sum) > 0 else 0
        dice = dice_sum
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        fps = self.total_frames / self.total_inference_time if self.total_inference_time > 0 else 0
        
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32，4字节
        
        metrics = {
            "miou": scores['miou'],  # 平均交并比
            "macc": scores['macc'],  # 平均准确率
            "overall_acc": scores['overall_acc'],  # 整体准确率
            "fwiou": scores['fwiou'],  # 频率加权IoU
            "binary_iou": scores['iou_per_class'][1],  # 前景类IoU
            "foreground_iou": scores['iou_per_class'][1],  # 前景类IoU
            "background_iou": scores['iou_per_class'][0],  # 背景类IoU
            "iou_per_class": scores['iou_per_class'].tolist(),  # 每类IoU
            
            # 边界IoU指标
            "mbiou": boundary_scores['mbiou'],  # 平均边界IoU
            "biou_per_class": boundary_scores['biou_per_class'],  # 每类边界IoU
            "mhd": boundary_scores['mhd'], # 平均Hausdorff距离
            "hausdorff_per_class": boundary_scores['hausdorff_per_class'], # 每类Hausdorff距离
            "band_iou": band_iou, # 边界带IoU
            "hd95": boundary_scores.get('hd95', boundary_scores['mhd']),  # HD95（若底层未提供，则使用mhd近似）
            
            # 二分类指标
            "precision": precision,  # 精确度
            "recall": recall,  # 召回率
            "f1_score": f1_score,  # F1分数
            "dice": dice,  # Dice系数
            "accuracy": accuracy,  # 准确率
            
            # 混淆矩阵元素
            "tp": int(tp),  # 真正例
            "fp": int(fp),  # 假正例
            "fn": int(fn),  # 假负例
            "tn": int(tn),  # 真负例
            
            # 性能指标
            "fps": fps,  # 每秒帧数
            "total_inference_time_ms": self.total_inference_time * 1000,  # 总推理时间(毫秒)
            "total_frames": self.total_frames,  # 总帧数
            "model_size_mb": round(model_size_mb, 2),  # 模型大小(MB)
            "total_parameters": total_params  # 总参数数量
        }
        
        return metrics

    def _compute_band_iou(self, pred_mask, gt_mask):
        """
        计算边界带IoU - 在边界周围一定宽度的带状区域内计算IoU
        Args:
            pred_mask: 预测掩码 (H, W)
            gt_mask: 真实掩码 (H, W)
        Returns:
            边界带IoU值
        """
        import cv2
        
        # 提取真实边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gt_boundary = cv2.morphologyEx(gt_mask, cv2.MORPH_GRADIENT, kernel)
        
        # 生成边界带掩码 - 在边界周围扩展self.band_width像素
        band_mask = np.zeros_like(gt_mask)
        if gt_boundary.sum() > 0:
            # 距离变换
            dist = cv2.distanceTransform((1 - gt_boundary).astype(np.uint8), cv2.DIST_L2, 3)
            # 边界带 = 距离小于band_width的区域
            band_mask = (dist <= self.band_width).astype(np.uint8)
        
        # 在边界带内计算IoU
        pred_band = pred_mask * band_mask
        gt_band = gt_mask * band_mask
        
        # 计算IoU
        intersection = (pred_band * gt_band).sum()
        union = pred_band.sum() + gt_band.sum() - intersection
        
        if union > 0:
            return intersection / union
        else:
            return 0.0


# 添加一个辅助函数用于递归转换嵌套结构中的numpy类型
def _convert_numpy_to_python(obj):
    """递归地将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate BG-SegNet model on yolo_seg test set")
    parser.add_argument("--config", type=str, default="/root/autodl-tmp/BG-SegNet/project/bgsegnet/config.yaml", 
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="/root/autodl-tmp/BG-SegNet/best.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/BG-SegNet/yolo_seg/yolo_seg",
                       help="Path to yolo_seg dataset root")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of comparison images to save")
    parser.add_argument("--pred_threshold", type=float, default=0.5,
                       help="Prediction threshold for binary segmentation")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CSegNetEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        pred_threshold=args.pred_threshold,
        num_samples=args.num_samples
    )
    
    # 创建数据集和数据加载器
    dataset = YOLOSegDataset(
        data_root=args.data_root,
        split="test",
        img_size=(640, 640)
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 执行评估
    metrics = evaluator.evaluate(data_loader)
    
    # 保存指标到JSON文件
    output_file = Path(args.output_dir) / "evaluation_metrics.json"
    
    # 确保所有numpy数组和numpy标量都转换为Python原生类型，以便JSON序列化
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
        elif isinstance(value, np.number):
            metrics[key] = value.item()  # 将numpy标量转换为Python原生类型
        elif isinstance(value, list):
            # 递归处理列表中的numpy类型
            metrics[key] = _convert_numpy_to_python(value)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_file}")
    print(f"Comparison images saved to: {args.output_dir}/comparison_images/")
    
    # 打印主要指标
    print(f"\nKey Metrics:")
    
    # 分割性能指标
    print(f"\n分割性能指标:")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"前景IoU: {metrics['foreground_iou']:.4f}")
    print(f"背景IoU: {metrics['background_iou']:.4f}")
    print(f"每类IoU: {[f'{x:.4f}' for x in metrics['iou_per_class']]}")
    print(f"频率加权IoU: {metrics['fwiou']:.4f}")
    print(f"平均准确率: {metrics['macc']:.4f}")
    print(f"整体准确率: {metrics['overall_acc']:.4f}")
    
    # 边界IoU指标
    print(f"\n边界评估指标:")
    print(f"平均边界IoU (mbiou): {metrics['mbiou']:.4f}")
    print(f"每类边界IoU: {[f'{x:.4f}' for x in metrics['biou_per_class']]}")
    print(f"平均Hausdorff距离 (mhd): {metrics['mhd']:.4f}")
    print(f"每类Hausdorff距离: {[f'{x:.4f}' for x in metrics['hausdorff_per_class']]}")
    print(f"边界带IoU: {metrics['band_iou']:.4f}")
    print(f"HD95: {metrics['hd95']:.4f}")
    
    # 二分类指标
    print(f"\n二分类指标:")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print(f"Dice系数: {metrics['dice']:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    
    # 性能指标
    print(f"\n性能指标:")
    print(f"FPS: {metrics['fps']:.2f}")
    print(f"模型大小: {metrics['model_size_mb']:.2f} MB")
    print(f"总参数量: {metrics['total_parameters']:,}")

    # 按论文表格格式输出一行：Param  & FPS & Dice & mIoU & Precision & Recall & band_IoU
    params_m = metrics["total_parameters"] / 1e6
    print("\n表格行（Param  & FPS & Dice & mIoU & Precision & Recall & band_IoU）：")
    print(f"{params_m:.2f}M & {metrics['fps']:.2f} & {metrics['dice']:.4f} & {metrics['miou']:.4f} "
          f"& {metrics['precision']:.4f} & {metrics['recall']:.4f} & {metrics['band_iou']:.4f}")


if __name__ == "__main__":
    main() 