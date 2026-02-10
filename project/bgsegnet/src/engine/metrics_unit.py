"""
Lightweight segmentation evaluation metrics used during training/validation.
"""
import numpy as np
import torch


class RunningScore:
    """Online segmentation metrics based on a confusion matrix."""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))  # confusion matrix
    
    def update(self, preds, labels):
        """
        Update confusion matrix in a vectorized way.

        Args:
            preds: predicted labels, shape (B, H, W) or (H, W)
            labels: ground-truth labels, shape (B, H, W) or (H, W)
        """
        preds = preds.detach().cpu().numpy() if torch.is_tensor(preds) else preds
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        
        mask = labels != self.ignore_index
        preds = preds[mask]
        labels = labels[mask]
        
        # Vectorized confusion matrix with np.bincount (much faster than Python loops).
        n = self.num_classes
        cm = np.bincount(n * labels + preds, minlength=n*n).reshape(n, n)
        self.confusion_matrix += cm
    
    def get_scores(self):
        """Compute standard region-based segmentation metrics."""
        eps = 1e-6
        
        diag = np.diag(self.confusion_matrix)
        row_sum = self.confusion_matrix.sum(axis=1)
        col_sum = self.confusion_matrix.sum(axis=0)
        
        # Intersection-over-Union (IoU) per class
        iou = diag / (row_sum + col_sum - diag + eps)
        
        # Mean IoU across all classes
        miou = np.nanmean(iou)
        
        # Pixel accuracy per class
        acc = diag / (row_sum + eps)
        macc = np.nanmean(acc)
        
        # Global pixel accuracy
        total = self.confusion_matrix.sum()
        if total > 0:
            overall_acc = diag.sum() / total
        else:
            overall_acc = 0
        
        # Frequency weighted IoU
        freq = row_sum / (total + eps)
        fwiou = (freq * iou).sum()
        
        return {
            'miou': miou,
            'macc': macc,
            'overall_acc': overall_acc,
            'fwiou': fwiou,
            'iou_per_class': iou,
            'acc_per_class': acc
        }
    
    def reset(self):
        """Reset internal confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


class BoundaryIoU:
    """Boundary-based metrics: boundary IoU and Hausdorff distance."""
    
    def __init__(self, num_classes, dilation_ratio=0.02):
        self.num_classes = num_classes
        self.dilation_ratio = dilation_ratio
        self.reset()
    
    def update(self, preds, labels):
        """Update boundary-based metrics for a batch of predictions."""
        import cv2
        
        preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        if len(preds.shape) == 3:
            for pred, label in zip(preds, labels):
                self._update_single(pred, label)
        else:
            self._update_single(preds, labels)
    
    def _update_single(self, pred, label):
        """Update boundary IoU and Hausdorff distance for a single prediction."""
        import cv2
        
        h, w = pred.shape
        kernel_size = int(self.dilation_ratio * max(h, w))
        if kernel_size < 3:
            kernel_size = 3
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Compute boundary IoU for each foreground class (skip background = 0).
        for c in range(1, self.num_classes):
            pred_c = (pred == c).astype(np.uint8)
            label_c = (label == c).astype(np.uint8)
            
            # Morphological gradient to approximate object boundaries.
            pred_boundary = cv2.morphologyEx(pred_c, cv2.MORPH_GRADIENT, kernel)
            label_boundary = cv2.morphologyEx(label_c, cv2.MORPH_GRADIENT, kernel)
            
            # Boundary IoU
            intersection = (pred_boundary * label_boundary).sum()
            union = pred_boundary.sum() + label_boundary.sum() - intersection
            
            if union > 0:
                iou = intersection / union
                self.boundary_ious[c-1].append(iou)
                
                # Compute symmetric Hausdorff distance if both boundaries exist.
                if pred_boundary.sum() > 0 and label_boundary.sum() > 0:
                    min_points_required = 5
                    if pred_boundary.sum() < min_points_required or label_boundary.sum() < min_points_required:
                        continue
                        
                    # Distance transforms for Hausdorff computation.
                    # From prediction boundary to ground-truth boundary.
                    dist_transform_gt = cv2.distanceTransform((1 - label_boundary).astype(np.uint8), cv2.DIST_L2, 3)
                    max_dist_pred_to_gt = np.max(dist_transform_gt * pred_boundary) if pred_boundary.sum() > 0 else 0
                    
                    # From ground-truth boundary to prediction boundary.
                    dist_transform_pred = cv2.distanceTransform((1 - pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
                    max_dist_gt_to_pred = np.max(dist_transform_pred * label_boundary) if label_boundary.sum() > 0 else 0
                    
                    # Hausdorff distance: maximal bidirectional surface distance.
                    hausdorff_dist = max(max_dist_pred_to_gt, max_dist_gt_to_pred)
                    self.hausdorff_dists[c-1].append(hausdorff_dist)
    
    def get_scores(self):
        """Return mean boundary IoU and mean Hausdorff distance."""
        biou_per_class = []
        hausdorff_per_class = []
        
        for ious in self.boundary_ious:
            if len(ious) > 0:
                biou_per_class.append(np.mean(ious))
            else:
                biou_per_class.append(0)
        
        for dists in self.hausdorff_dists:
            if len(dists) > 0:
                hausdorff_per_class.append(np.mean(dists))
            else:
                hausdorff_per_class.append(0)
        
        mbiou = np.mean(biou_per_class) if len(biou_per_class) > 0 else 0
        mhd = np.mean(hausdorff_per_class) if len(hausdorff_per_class) > 0 else 0
        
        return {
            'mbiou': mbiou,
            'biou_per_class': biou_per_class,
            'mhd': mhd,
            'hausdorff_per_class': hausdorff_per_class
        }
    
    def reset(self):
        """Reset stored boundary IoU and Hausdorff statistics."""
        self.boundary_ious = [[] for _ in range(self.num_classes - 1)]  # foreground classes only
        self.hausdorff_dists = [[] for _ in range(self.num_classes - 1)]