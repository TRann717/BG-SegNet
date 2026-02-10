"""
Segmentation evaluation metrics including IoU, Dice, precision/recall, and boundary-based measures.
"""
import numpy as np
import torch


class RunningScore:
    """Online segmentation metrics built from a confusion matrix."""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))  # confusion matrix
    
    def update(self, preds, labels):
        """
        Update confusion matrix in a vectorized way.
        Args:
            preds: predicted labels (B, H, W) or (H, W)
            labels: ground-truth labels (B, H, W) or (H, W)
        """
        preds = preds.detach().cpu().numpy() if torch.is_tensor(preds) else preds
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        
        mask = labels != self.ignore_index
        preds = preds[mask]
        labels = labels[mask]
        
        # Vectorized confusion matrix with np.bincount (substantially faster than loops).
        n = self.num_classes
        cm = np.bincount(n * labels + preds, minlength=n*n).reshape(n, n)
        self.confusion_matrix += cm
    
    def get_scores(self):
        """
        Compute region-based segmentation metrics (IoU, Dice, precision, recall).

        For binary segmentation, the foreground lesion class is assumed to be class index 1.
        """
        eps = 1e-6

        cm = self.confusion_matrix.astype(np.float64)
        diag = np.diag(cm)
        row_sum = cm.sum(axis=1)  # ground-truth pixels per class
        col_sum = cm.sum(axis=0)  # predicted pixels per class

        # Intersection-over-Union (IoU) per class
        iou = diag / (row_sum + col_sum - diag + eps)
        miou = float(np.nanmean(iou)) if iou.size > 0 else 0.0

        # Pixel accuracy per class
        acc = diag / (row_sum + eps)
        macc = float(np.nanmean(acc)) if acc.size > 0 else 0.0

        # Global pixel accuracy
        total = cm.sum()
        overall_acc = float(diag.sum() / total) if total > 0 else 0.0

        # Frequency weighted IoU
        freq = row_sum / (total + eps)
        fwiou = float((freq * iou).sum()) if iou.size > 0 else 0.0

        # Dice / precision / recall per class
        dice_per_class = (2.0 * diag) / (row_sum + col_sum + eps)
        precision_per_class = diag / (col_sum + eps)
        recall_per_class = diag / (row_sum + eps)

        # Binary segmentation: report metrics for foreground lesion class = 1.
        # Multi-class: still expose per-class metrics; aggregates are simple means.
        if len(dice_per_class) > 1:
            dice = float(dice_per_class[1])
            precision = float(precision_per_class[1])
            recall = float(recall_per_class[1])
        else:
            dice = float(dice_per_class[0]) if dice_per_class.size > 0 else 0.0
            precision = float(precision_per_class[0]) if precision_per_class.size > 0 else 0.0
            recall = float(recall_per_class[0]) if recall_per_class.size > 0 else 0.0

        return {
            "miou": miou,
            "macc": macc,
            "overall_acc": overall_acc,
            "fwiou": fwiou,
            "iou_per_class": iou,
            "acc_per_class": acc,
            "dice": dice,
            "dice_per_class": dice_per_class,
            "precision": precision,
            "precision_per_class": precision_per_class,
            "recall": recall,
            "recall_per_class": recall_per_class,
        }
    
    def reset(self):
        """Reset internal confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


class BoundaryIoU:
    """Boundary IoU, Hausdorff distance, and HD95 for contour accuracy."""
    
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
        """Update boundary IoU, Hausdorff distance, and HD95 for a single mask."""
        import cv2
        
        # Determine structuring element size relative to lesion extent.
        h, w = pred.shape
        kernel_size = int(self.dilation_ratio * max(h, w))
        if kernel_size < 3:
            kernel_size = 3
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Compute boundary metrics for each foreground class (skip background = 0).
        for c in range(1, self.num_classes):
            pred_c = (pred == c).astype(np.uint8)
            label_c = (label == c).astype(np.uint8)
            
            # Approximate object contour via morphological gradient.
            pred_boundary = cv2.morphologyEx(pred_c, cv2.MORPH_GRADIENT, kernel)
            label_boundary = cv2.morphologyEx(label_c, cv2.MORPH_GRADIENT, kernel)
            
            # Boundary IoU
            intersection = (pred_boundary * label_boundary).sum()
            union = pred_boundary.sum() + label_boundary.sum() - intersection
            
            if union > 0:
                iou = intersection / union
                self.boundary_ious[c-1].append(iou)
                
                # Compute symmetric Hausdorff distance if boundaries are non-empty.
                if pred_boundary.sum() > 0 and label_boundary.sum() > 0:
                    min_points_required = 5
                    if pred_boundary.sum() < min_points_required or label_boundary.sum() < min_points_required:
                        continue
                        
                    # From prediction boundary to ground-truth boundary.
                    dist_transform_gt = cv2.distanceTransform((1 - label_boundary).astype(np.uint8), cv2.DIST_L2, 3)
                    max_dist_pred_to_gt = np.max(dist_transform_gt * pred_boundary) if pred_boundary.sum() > 0 else 0
                    
                    # From ground-truth boundary to prediction boundary.
                    dist_transform_pred = cv2.distanceTransform((1 - pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
                    max_dist_gt_to_pred = np.max(dist_transform_pred * label_boundary) if label_boundary.sum() > 0 else 0
                    
                    # Hausdorff distance: maximal bidirectional surface distance.
                    hausdorff_dist = max(max_dist_pred_to_gt, max_dist_gt_to_pred)
                    self.hausdorff_dists[c-1].append(hausdorff_dist)
                    
                    # Collect distance samples to compute HD95 (95th percentile).
                    # Distances from prediction boundary to ground-truth boundary.
                    dists_pred_to_gt = []
                    pred_boundary_points = np.argwhere(pred_boundary > 0)
                    for p in pred_boundary_points:
                        if p[0] < dist_transform_gt.shape[0] and p[1] < dist_transform_gt.shape[1]:
                            dists_pred_to_gt.append(dist_transform_gt[p[0], p[1]])
                    
                    # Distances from ground-truth boundary to prediction boundary.
                    dists_gt_to_pred = []
                    gt_boundary_points = np.argwhere(label_boundary > 0)
                    for g in gt_boundary_points:
                        if g[0] < dist_transform_pred.shape[0] and g[1] < dist_transform_pred.shape[1]:
                            dists_gt_to_pred.append(dist_transform_pred[g[0], g[1]])
                    
                    # 95th percentile Hausdorff distance (HD95).
                    if len(dists_pred_to_gt) > 0 and len(dists_gt_to_pred) > 0:
                        hd95_pred_to_gt = np.percentile(dists_pred_to_gt, 95) if len(dists_pred_to_gt) > 0 else 0
                        hd95_gt_to_pred = np.percentile(dists_gt_to_pred, 95) if len(dists_gt_to_pred) > 0 else 0
                        hd95 = max(hd95_pred_to_gt, hd95_gt_to_pred)
                        self.hd95_dists[c-1].append(hd95)
    
    def get_scores(self):
        """Return mean boundary IoU, mean Hausdorff distance, and HD95."""
        biou_per_class = []
        hausdorff_per_class = []
        hd95_per_class = []
        
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
        
        for dists in self.hd95_dists:
            if len(dists) > 0:
                hd95_per_class.append(np.mean(dists))
            else:
                hd95_per_class.append(0)
        
        mbiou = np.mean(biou_per_class) if len(biou_per_class) > 0 else 0
        mhd = np.mean(hausdorff_per_class) if len(hausdorff_per_class) > 0 else 0
        hd95 = np.mean(hd95_per_class) if len(hd95_per_class) > 0 else 0
        
        return {
            'mbiou': mbiou,
            'biou_per_class': biou_per_class,
            'mhd': mhd,
            'hausdorff_per_class': hausdorff_per_class,
            'hd95': hd95,
            'hd95_per_class': hd95_per_class
        }
    
    def reset(self):
        """Reset stored boundary-level statistics."""
        self.boundary_ious = [[] for _ in range(self.num_classes - 1)]  # foreground classes only
        self.hausdorff_dists = [[] for _ in range(self.num_classes - 1)]
        self.hd95_dists = [[] for _ in range(self.num_classes - 1)]