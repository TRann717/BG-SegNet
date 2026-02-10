#!/usr/bin/env python
"""
BG-SegNet semantic segmentation training script.
"""
import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 添加src到路径
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_mask import get_dataloader
from src.models.cseg_net import build_model
from src.engine.trainer import Trainer
from src.utils.logger import TBLogger, increment_path


def set_seed(seed, cudnn_benchmark=False):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
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


def main():
    parser = argparse.ArgumentParser(description='BG-SegNet Segmentation Training')
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/BG-SegNet/project/bgsegnet/config.yaml', help='config file')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    seed = config['train']['seed']
    cudnn_benchmark = config['train'].get('cudnn_benchmark', False)
    print(f"Setting seed: {seed} (cudnn_benchmark: {cudnn_benchmark})")
    set_seed(seed, cudnn_benchmark=cudnn_benchmark)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Creating dataloaders...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    print(f"Train dataset: {len(train_loader.dataset)} samples")
    print(f"Val dataset: {len(val_loader.dataset)} samples")
    
    assert len(train_loader.dataset) > 0, "Training dataset is empty!"
    assert len(val_loader.dataset) > 0, "Validation dataset is empty!"
    
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    print(f"\nModel Configuration:")
    print(f"  - Backbone: SAM2 ({'frozen' if config['model']['freeze_backbone'] else 'trainable'})")
    print(f"  - Neck: {config['model']['neck']['type']}")
    print(f"  - Classes: {config['data']['num_classes']}")
    print(f"  - Input size: {config['data']['img_size']}")
    print(f"  - Crop size: {config['data']['aug']['crop_size']}")
    
    # Create Ultralytics-style run directory: runs/exp, runs/exp2, ...
    base_dir = Path(config['log']['outdir']) / config['experiment']
    save_dir = increment_path(base_dir)
    print(f"Save dir: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger: TensorBoard + CSV + sample images.
    logger = TBLogger(save_dir, config, config_path=args.config)

    print("Creating trainer...")
    trainer = Trainer(model, config, device, save_dir=save_dir, logger=logger)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'scores' in checkpoint:
            metric = trainer.save_best_by
            trainer.best_score = checkpoint['scores'].get(metric, checkpoint['scores'].get('miou', 0))
            trainer.best_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        start_epoch = 1
    
    if args.eval:
        print("Evaluation mode")
        val_loss, val_scores = trainer.validate(val_loader)
        print(f"Validation - Loss: {val_loss:.4f}")
        print(f"mIoU: {val_scores['miou']:.4f}")
        print(f"mAcc: {val_scores['macc']:.4f}")
        print(f"Overall Acc: {val_scores['overall_acc']:.4f}")
        
        class_names = config['data']['class_names']
        for i, (name, iou) in enumerate(zip(class_names, val_scores['iou_per_class'])):
            print(f"  {name}: {iou:.4f}")
        return
    
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Training config:")
    print(f"  - Epochs: {config['train']['epochs']}")
    print(f"  - Batch size: {config['train']['batch_size']}")
    print(f"  - Accumulation steps: {config['train']['accum_steps']}")
    print(f"  - Learning rate: {config['train']['lr']}")
    print(f"  - Warmup iters: {config['train'].get('warmup_iters', 0)}")
    print(f"  - AMP: {config['train']['amp']}")
    print(f"  - Gradient clipping: {config['train']['grad_clip']}")
    print("="*60 + "\n")
    trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    
    print("Training completed!")
    logger.close()


if __name__ == '__main__':
    main()