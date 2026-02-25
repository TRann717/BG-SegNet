"""
YOLO-format polygon-based medical image segmentation dataset loader.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


def yolo_poly_to_mask(poly_coords, img_w, img_h):
    """Convert YOLO-normalized polygon coordinates to a binary lesion mask."""
    poly = np.array(poly_coords).reshape(-1, 2)
    poly[:, 0] *= img_w
    poly[:, 1] *= img_h
    poly = poly.astype(np.int32)
    
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask


class YOLOSegDataset(Dataset):
    """Semantic segmentation dataset with YOLO-format polygon annotations."""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',  # train, val, test
                 img_size: tuple = (640, 640),
                 crop_size: tuple = (512, 512),
                 num_classes: int = 2,
                 augment: bool = True):
        
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')
        
        self.img_dir = self.data_root / split / 'images'
        self.label_dir = self.data_root / split / 'labels'
        
        self.img_files = sorted(list(self.img_dir.glob('*.png'))) + \
                        sorted(list(self.img_dir.glob('*.jpg')))
        
        print(f"Found {len(self.img_files)} images in {split} set")
        
       
        self.transform = self._get_transform()
    
    def _get_transform(self):
        """Build data augmentation pipeline for dermoscopic / wound segmentation."""
        if self.augment:
            transform = A.Compose([
                A.LongestMaxSize(max_size=max(self.img_size[0], self.crop_size[0]+100), p=1.0),
                A.PadIfNeeded(min_height=self.crop_size[0], min_width=self.crop_size[1], 
                             border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(self.crop_size[0], self.crop_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                ToTensorV2(transpose_mask=False),  # convert HWC→CHW without normalization
            ])
        else:
            transform = A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                ToTensorV2(transpose_mask=False),  # convert HWC→CHW without normalization
            ])
        return transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        
        assert image is not None, f"Failed to read image: {img_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        assert h > 0 and w > 0, f"Invalid image size: {h}x{w} for {img_path}"
        
        label_path = self.label_dir / (img_path.stem + '.txt')
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6: 
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                poly_mask = yolo_poly_to_mask(coords, w, h)
                # For semantic segmentation: background = 0, lesion labels start from 1.
                mask[poly_mask > 0] = class_id + 1
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.long()
        
        return {
            'image': image,
            'mask': mask,
            'img_path': str(img_path)
        }


def worker_init_fn(worker_id):
    """Initialize worker-specific random seed for reproducible data loading."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config, split='train'):
    """Build DataLoader for YOLO-format polygon-based segmentation datasets."""
    dataset = YOLOSegDataset(
        data_root=config['data']['dataset_root'],
        split=split,
        img_size=tuple(config['data']['img_size']),
        crop_size=tuple(config['data']['aug']['crop_size']) if split == 'train' else tuple(config['data']['img_size']),
        num_classes=config['data']['num_classes'],
        augment=(split == 'train')
    )
    
   
    num_workers = config['data'].get('workers', 4)
    loader_kwargs = {
        'pin_memory': config['data'].get('pin_memory', True),
        'prefetch_factor': config['data'].get('prefetch_factor', 2) if num_workers > 0 else None,
        'persistent_workers': config['data'].get('persistent_workers', True) if num_workers > 0 else False,
        'worker_init_fn': worker_init_fn  
    }
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['train']['batch_size'] if split == 'train' else 2,  # Validation can use smaller batch
        shuffle=(split == 'train'),
        num_workers=num_workers,
        drop_last=(split == 'train'),
        **loader_kwargs
    )
    
    return dataloader
