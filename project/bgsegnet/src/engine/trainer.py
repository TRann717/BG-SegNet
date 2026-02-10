"""
Generic trainer for medical image segmentation with boundary-aware losses and open-set support.
"""
import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .metrics import RunningScore, BoundaryIoU
from ..losses.seg_losses import CompoundLoss
from typing import Optional
# from ..losses.seg_losses_simple import SimpleCompoundLoss as CompoundLoss


class Trainer:
    """Trainer for segmentation models with curriculum boundary weighting and gate control."""
    
    def __init__(self, model, config, device='cuda', save_dir: Optional[Path] = None, logger=None):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        # Training configuration
        self.epochs = config['train']['epochs']
        self.batch_size = config['train']['batch_size']
        self.accum_steps = config['train']['accum_steps']
        self.amp = config['train']['amp']
        self.grad_clip = config['train']['grad_clip']
        self.warmup_iters = config['train'].get('warmup_iters', 0)
        self.base_lr = config['train']['lr']
        
        # Optimizer and LR scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Warmup configuration
        self.current_iter = 0
        self.warmup_factor = 0.1 if self.warmup_iters > 0 else 1.0
        
        # Segmentation loss (compound region + boundary objectives)
        self.criterion = CompoundLoss(config)
        
        # Automatic mixed precision (AMP)
        self.scaler = GradScaler('cuda') if self.amp else None
        
        # Region-based segmentation metrics
        self.train_metrics = RunningScore(
            num_classes=config['data']['num_classes'],
            ignore_index=config['data']['ignore_index']
        )
        self.val_metrics = RunningScore(
            num_classes=config['data']['num_classes'],
            ignore_index=config['data']['ignore_index']
        )
        
        # Best validation checkpoint tracking
        self.save_best_by = config['train'].get('save_best_by', 'miou')
        self.best_score = 0
        self.best_epoch = 0
        
        # Output / checkpoint directory
        if save_dir is None:
            self.output_dir = Path(config['log']['outdir']) / config['experiment']
        else:
            self.output_dir = Path(save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.output_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Epoch at which to unfreeze boundary gate gradients (optional)
        self.gate_unfreeze_epoch = config['model']['boundary'].get('gate', {}).get('grad_unfreeze_epoch', None)
        
        # Performance-driven progressive unfreezing of boundary gates
        self.performance_driven_unfreeze = config['model']['boundary'].get('gate', {}).get('performance_driven', False)
        self.unfreeze_patience = config['model']['boundary'].get('gate', {}).get('unfreeze_patience', 2)
        self.gate_unfrozen = False
        self.val_no_improve_count = 0
        self.prev_val_score = 0
        self.grad_release_eta = config['model']['boundary'].get('gate', {}).get('grad_release_eta', 0.3)
    
    def _build_optimizer(self):
        """Build optimizer with differential learning rates for boundary branches."""
        base_lr = self.config['train']['lr']
        wd = self.config['train']['weight_decay']
        
        # Parameter groups: boundary-related parameters use a smaller learning rate.
        boundary_params = []
        main_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'boundary' in name:
                boundary_params.append(param)
            else:
                main_params.append(param)
        
        param_groups = []
        if len(main_params) > 0:
            param_groups.append({'params': main_params, 'lr': base_lr, 'weight_decay': wd})
        if len(boundary_params) > 0:
            param_groups.append({'params': boundary_params, 'lr': base_lr * 0.5, 'weight_decay': wd, 'name': 'boundary'})
        
        if self.config['train']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(param_groups)
        elif self.config['train']['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config['train']['optimizer']}")
        
        print(f"[Optimizer] Main params: {len(main_params)}, Boundary params: {len(boundary_params)}")
        print(f"[Optimizer] Main LR: {base_lr:.6f}, Boundary LR: {base_lr * 0.5:.6f}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config['train']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif self.config['train']['scheduler'] == 'poly':
            lambda_func = lambda epoch: (1 - epoch / self.epochs) ** 0.9
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_func)
        else:
            scheduler = None
        
        return scheduler
    
    def _adjust_learning_rate_warmup(self):
        """Adjust learning rate with linear warmup across all parameter groups."""
        if self.current_iter < self.warmup_iters:
            alpha = self.current_iter / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            warmup_factor = 1.0
        
        for param_group in self.optimizer.param_groups:
            if param_group.get('name') == 'boundary':
                param_group['lr'] = self.base_lr * 0.5 * warmup_factor
            else:
                param_group['lr'] = self.base_lr * warmup_factor
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        print(f"\nStarting epoch {epoch}/{self.epochs}, {num_batches} batches")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Optional epoch-based unfreezing of boundary gate gradients.
        if self.gate_unfreeze_epoch is not None and epoch == int(self.gate_unfreeze_epoch):
            if hasattr(self.model, "set_gate_detach"):
                self.model.set_gate_detach(False)
                print(f"[Info] Unfroze boundary gate gradients at epoch {epoch} (detach -> False)")
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        
        for i, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Update iteration count and apply LR warmup.
            if self.warmup_iters > 0 and self.current_iter < self.warmup_iters:
                self._adjust_learning_rate_warmup()
            self.current_iter += 1
            
            # Gradient accumulation
            if i % self.accum_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass
            if self.amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, masks)
                    loss = loss / self.accum_steps
                    
                    assert not torch.isnan(loss) and not torch.isinf(loss), \
                        f"Invalid loss detected: {loss.item()}, loss_dict: {loss_dict}"
                    
                
                # Backward pass with AMP scaling.
                self.scaler.scale(loss).backward()
                
                # Optimizer step when accumulation is complete.
                if (i + 1) % self.accum_steps == 0:
                    # Gradient clipping after unscaling.
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    # Optimizer step and scaler update.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, masks)
                loss = loss / self.accum_steps
                
                assert not torch.isnan(loss) and not torch.isinf(loss), \
                    f"Invalid loss detected: {loss.item()}, loss_dict: {loss_dict}"
               
                
                loss.backward()
                
                if (i + 1) % self.accum_steps == 0:
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * self.accum_steps
            
            # Compute hard predictions and update metrics.
            preds = outputs['logits'].argmax(dim=1)
            self.train_metrics.update(preds, masks)
            
            # Update progress bar with current loss and LR.
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                'loss': f'{loss.item()*self.accum_steps:.4f}',
                'lr': f'{current_lr:.6f}',
                'iter': f'{self.current_iter}/{self.warmup_iters}' if self.current_iter < self.warmup_iters else ''
            })

            # Log step-wise training metrics to TensorBoard every `log_every` steps.
            if (self.logger is not None) and (i % self.logger.log_every == 0):
                step_scalars = {'train/loss_step': loss.item()*self.accum_steps}
                self.logger.add_scalar_dict(step_scalars, self.current_iter)
                self.logger.add_lr(self.optimizer, self.current_iter)
                with torch.no_grad():
                    preds_step = outputs['logits'].argmax(dim=1)
                    if i == 0 and self.logger.img_log:
                        self.logger.add_images_batch(images, masks, preds_step, tag=f"train/epoch{epoch}", step=self.current_iter)
                        self.logger.save_image_samples(images, masks, preds_step, epoch, tag='train')
        
        # Handle partially accumulated gradients at end of epoch.
        if (len(train_loader) % self.accum_steps) != 0:
            if self.amp:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
        
        # Epoch-level metrics
        avg_loss = total_loss / num_batches
        scores = self.train_metrics.get_scores()
        
        print(f"Epoch {epoch} training completed - Avg Loss: {avg_loss:.4f}, mIoU: {scores['miou']:.4f}")
        if self.logger is not None:
            payload = {
                "train/loss": avg_loss,
                "train/miou": float(scores['miou']),
                "train/macc": float(scores['macc']),
                "train/overall_acc": float(scores['overall_acc']),
            }
            self.logger.add_scalar_dict(payload, epoch)
        return avg_loss, scores
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validation loop computing region and boundary metrics."""
        self.model.eval()
        self.val_metrics.reset()
        biou = BoundaryIoU(num_classes=self.config['data']['num_classes']) if getattr(self.model, 'use_boundary', False) else None
        
        total_loss = 0
        num_batches = len(val_loader)
        
        print(f"\nStarting validation with {num_batches} batches...")
        
        pbar = tqdm(val_loader, desc='Validation')
        
        # For open-set training with OV head and unknown threshold, collect a buffer
        # of pixel-level OV logits to automatically calibrate the decision threshold.
        need_ov_cal = (hasattr(self.model, 'ood_scorer')
                      and getattr(self.model, 'use_ov', False)
                      and self.config.get('openset', {}).get('enable', False)
                      and self.config.get('openset', {}).get('thresh') is None
                      and float(self.model.ood_scorer.thresh.item()) < 0)
        calib_buf = [] if need_ov_cal else None
        max_calib_batches = 5
        
        first = True
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with optional AMP for faster validation.
            if self.amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, masks)
            
            # Collect OV logits from the first few mini-batches for open-set calibration.
            if calib_buf is not None and 'ov_logits_pix' in outputs and len(calib_buf) < max_calib_batches:
                calib_buf.append(outputs['ov_logits_pix'].detach())
            
            total_loss += loss.item()
            
            # Compute hard predictions and update metrics.
            preds = outputs['logits'].argmax(dim=1)
            self.val_metrics.update(preds, masks)
            if biou is not None:
                biou.update(preds, masks)
            # Log one representative batch of validation images and predictions.
            if first and (self.logger is not None) and self.logger.img_log:
                self.logger.add_images_batch(images, masks, preds, tag="val/sample", step=self.current_iter)
                current_epoch = self.current_iter // len(val_loader) if hasattr(self, 'current_iter') else 0
                self.logger.save_image_samples(images, masks, preds, current_epoch, tag='val')
                first = False
        
        # If needed, perform one-shot calibration of the open-set threshold from OV logits.
        if calib_buf is not None and len(calib_buf) > 0:
            with torch.amp.autocast('cuda', enabled=self.amp):
                calib_batch = torch.cat(calib_buf, dim=0)  # aggregate a few mini-batches
                tau = self.model.ood_scorer.calibrate([calib_batch], target_fpr=0.05)
                print(f"[OpenSet/OV] Auto-calibrated threshold τ = {tau:.4f}")
        
        avg_loss = total_loss / num_batches
        scores = self.val_metrics.get_scores()
        if biou is not None:
            scores.update(biou.get_scores())
        
        return avg_loss, scores
    
    def save_checkpoint(self, epoch, scores, is_best=False):
        """Save training checkpoints (last and best model)."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scores': scores,
            'config': self.config
        }
        
        torch.save(checkpoint, self.weights_dir / 'last.pth')
        
        if is_best:
            torch.save(checkpoint, self.weights_dir / 'best.pth')
            score_value = scores.get(self.save_best_by, scores.get('miou', 0))
            print(f"Saved best model at epoch {epoch} with {self.save_best_by}: {score_value:.4f}")
    
    def fit(self, train_loader, val_loader, start_epoch=1):
        """Main training loop over epochs with periodic validation and checkpointing."""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Starting from epoch: {start_epoch}")
        
        # Curriculum learning schedule for boundary-loss weighting.
        bd_warmup_epochs = self.config['train'].get('bd_warmup_epochs', 0)
        
        for epoch in range(start_epoch, self.epochs + 1):
            # Linearly ramp the boundary-loss scalar up to its maximum value.
            if hasattr(self.criterion, "set_boundary_scalar") and bd_warmup_epochs > 0:
                s = min(1.0, max(0.0, epoch / float(max(1, bd_warmup_epochs))))
                self.criterion.set_boundary_scalar(s)
                if epoch <= bd_warmup_epochs:
                    print(f"[Curriculum] Boundary loss scalar: {s:.3f} (warmup epoch {epoch}/{bd_warmup_epochs})")
            
            train_loss, train_scores = self.train_epoch(train_loader, epoch)
            
            if epoch % self.config['train']['val_freq'] == 0:
                val_loss, val_scores = self.validate(val_loader)
                
                print(f"\nEpoch {epoch}/{self.epochs}")
                print(f"Train - Loss: {train_loss:.4f}, mIoU: {train_scores['miou']:.4f}")
                val_info = f"Val   - Loss: {val_loss:.4f}, mIoU: {val_scores['miou']:.4f}"
                if 'mbiou' in val_scores:
                    val_info += f", bIoU: {val_scores['mbiou']:.4f}"
                print(val_info)
                
                # Save best checkpoint according to configured key metric.
                current_score = val_scores.get(self.save_best_by, val_scores['miou'])
                is_best = current_score > self.best_score
                if is_best:
                    self.best_score = current_score
                    self.best_epoch = epoch
                
                # Performance-driven progressive gate unfreezing.
                if self.performance_driven_unfreeze and not self.gate_unfrozen:
                    if current_score <= self.prev_val_score:
                        self.val_no_improve_count += 1
                    else:
                        self.val_no_improve_count = 0
                    
                    # Unfreeze when validation performance has plateaued for several runs.
                    if self.val_no_improve_count >= self.unfreeze_patience:
                        if hasattr(self.model, "set_gate_detach"):
                            self.model.set_gate_detach(False)
                            self.gate_unfrozen = True
                            print(f"[Info] Performance-driven gate unfreeze at epoch {epoch}")
                            print(f"       (No improvement for {self.val_no_improve_count} validations)")
                            print(f"       Gradient release eta: {self.grad_release_eta}")
                    
                    self.prev_val_score = current_score
                
                self.save_checkpoint(epoch, val_scores, is_best)

                # Log validation metrics and learning rates to TensorBoard and CSV.
                if self.logger is not None:
                    lr_payload = {f"lr/group{i}": g.get("lr", 0.0) for i, g in enumerate(self.optimizer.param_groups)}
                    payload = {
                        "val/loss": val_loss,
                        "val/miou": float(val_scores['miou']),
                        "val/macc": float(val_scores['macc']),
                        "val/overall_acc": float(val_scores['overall_acc']),
                        "val/mbiou": float(val_scores.get('mbiou', 0.0)),
                    } | lr_payload
                    # 写标量
                    self.logger.add_scalar_dict(payload, epoch)
                    # 写 CSV
                    row = {
                        "train/loss": train_loss,
                        "train/miou": float(train_scores['miou']),
                        "train/macc": float(train_scores['macc']),
                        "train/overall_acc": float(train_scores['overall_acc']),
                    } | payload
                    self.logger.write_csv_row(epoch, row)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
        
        print(f"\nTraining completed!")
        print(f"Best {self.save_best_by}: {self.best_score:.4f} at epoch {self.best_epoch}")