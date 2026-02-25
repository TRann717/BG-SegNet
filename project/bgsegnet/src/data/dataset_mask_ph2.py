"""
Mask-based medical image segmentation dataset loader.

Supports:
1) Generic layout: <dataset_root>/<split>/{images,labels}
2) Raw PH2 dermoscopic dataset layout: <dataset_root>/PH2 Dataset images/IMDxxx/...
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MaskSegDataset(Dataset):
    """Semantic segmentation dataset with precomputed lesion masks."""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',  # train, val, test
                 img_size: tuple = (640, 640),
                 crop_size: tuple = (512, 512),
                 num_classes: int = 2,
                 augment: bool = True,
                 aug_config: dict = None,
                 cv_num_folds: int = 1,
                 cv_fold_index: int = 0,
                 cv_seed: int = 42):
        
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')
        self.aug_config = aug_config or {}  # augmentation hyperparameters
        self.cv_num_folds = max(int(cv_num_folds), 1)
        self.cv_fold_index = int(cv_fold_index)
        self.cv_seed = int(cv_seed)
        
        # Support two directory layouts:
        # 1) Generic: <dataset_root>/<split>/{images,labels}
        # 2) PH2 raw: <dataset_root>/PH2 Dataset images/IMDxxx/... (one folder per lesion)
        self.img_files = []
        self.mask_files = None  # 对于 PH2，我们会显式保存掩码路径

        split_img_dir = self.data_root / split / 'images'
        split_label_dir = self.data_root / split / 'labels'
        ph2_root = self.data_root / 'PH2 Dataset images'

        if split_img_dir.exists():
            # Generic ISIC / YOLO-style directory.
            self.img_dir = split_img_dir
            self.label_dir = split_label_dir

            self.img_files = sorted(list(self.img_dir.glob('*.png'))) + \
                             sorted(list(self.img_dir.glob('*.jpg')))

            # Filter out non-image files (e.g., LICENSE.txt, ATTRIBUTION.txt).
            self.img_files = [f for f in self.img_files if f.stem not in ['LICENSE', 'ATTRIBUTION']]

        elif ph2_root.exists():
            # PH2 raw layout: parse IMD subfolders and split into train/val.
            self.img_dir = ph2_root
            self.label_dir = ph2_root  # 掩码也在 IMDxxx 子目录中
            self._build_ph2_file_list()
        else:
            raise FileNotFoundError(
                f"Dataset root not found or invalid: {self.data_root}. "
                f"Neither '{split}/images' nor 'PH2 Dataset images' exists."
            )

        print(f"Found {len(self.img_files)} images in {split} set.")
        
        # 设置数据增强
        self.transform = self._get_transform()

    def _build_ph2_file_list(self):
        """Build image/mask pairs from raw PH2 structure and split into train/val folds."""
        ph2_root = self.data_root / 'PH2 Dataset images'
        if not ph2_root.exists():
            raise FileNotFoundError(f"PH2 root directory does not exist: {ph2_root}")

        all_img_paths = []
        all_mask_paths = []

        # Each IMDxxx subdirectory corresponds to a single dermoscopic lesion.
        imd_dirs = [d for d in ph2_root.iterdir() if d.is_dir() and d.name.startswith('IMD')]
        imd_dirs = sorted(imd_dirs, key=lambda p: p.name)

        for imd_dir in imd_dirs:
            lesion_id = imd_dir.name  # e.g., IMD003

            derm_dirs = list(imd_dir.glob(f"{lesion_id}_Dermoscopic_Image*"))
            # Lesion mask subfolder: usually IMDxxx_lesion*
            lesion_dirs = list(imd_dir.glob(f"{lesion_id}_lesion*"))

            if not derm_dirs or not lesion_dirs:
                # Skip samples that do not match the expected PH2 pattern.
                continue

            derm_dir = derm_dirs[0]
            lesion_dir = lesion_dirs[0]

            img_candidates = sorted(
                list(derm_dir.glob("*.bmp")) +
                list(derm_dir.glob("*.png")) +
                list(derm_dir.glob("*.jpg")) +
                list(derm_dir.glob("*.jpeg"))
            )
            mask_candidates = sorted(
                list(lesion_dir.glob("*.bmp")) +
                list(lesion_dir.glob("*.png")) +
                list(lesion_dir.glob("*.jpg")) +
                list(lesion_dir.glob("*.jpeg"))
            )

            if not img_candidates or not mask_candidates:
                # Skip if either dermoscopic image or lesion mask is missing.
                continue

            img_path = img_candidates[0]
            mask_path = mask_candidates[0]

            all_img_paths.append(img_path)
            all_mask_paths.append(mask_path)

        if not all_img_paths:
            raise RuntimeError(
                f"No PH2 image/mask pairs found under {ph2_root}. "
                f"Please verify that the dataset was extracted correctly."
            )

        num_samples = len(all_img_paths)
        indices = np.arange(num_samples)
        rng = np.random.RandomState(self.cv_seed)
        rng.shuffle(indices)

        if self.cv_num_folds > 1:
            folds = np.array_split(indices, self.cv_num_folds)
            fold_idx = max(0, min(self.cv_num_folds - 1, self.cv_fold_index))
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(self.cv_num_folds) if i != fold_idx])

            if self.split == 'train':
                sel_idx = train_idx
            elif self.split == 'val':
                sel_idx = test_idx
            else:
                sel_idx = indices
        else:
            train_ratio = 0.8
            train_end = int(num_samples * train_ratio)

            if self.split == 'train':
                sel_idx = indices[:train_end]
            elif self.split == 'val':
                sel_idx = indices[train_end:]
            else:
                sel_idx = indices

        self.img_files = [all_img_paths[i] for i in sel_idx]
        self.mask_files = [all_mask_paths[i] for i in sel_idx]
    
    def _get_transform(self):
        """Build data augmentation pipeline for medical image segmentation."""
        if self.augment:
            pipeline = self.aug_config.get('pipeline', 'legacy')

            hflip = self.aug_config.get('hflip', True)
            vflip = self.aug_config.get('vflip', False)
            color_jitter = self.aug_config.get('color_jitter', [0.2, 0.2, 0.2, 0.1])  # [brightness, contrast, saturation, hue]
            rotate_deg = self.aug_config.get('rotate_deg', 0)
            scale_range = self.aug_config.get('scale_range', None)  # [min_scale, max_scale]
            blur_prob = self.aug_config.get('blur_prob', 0.1)

            flip_prob = float(self.aug_config.get('flip_prob', 0.5))
            rotate_prob = float(self.aug_config.get('rotate_prob', 0.5))
            color_jitter_prob = float(self.aug_config.get('color_jitter_prob', 0.5))
            rrc_prob = float(self.aug_config.get('rrc_prob', 0.1))
            
            if pipeline == 'litemamba':
                # LiteMamba-style augmentation used for PH2 dermoscopic segmentation.
                transforms = []
                transforms.append(A.Resize(self.img_size[0], self.img_size[1], p=1.0))

                if scale_range is not None and len(scale_range) == 2 and rrc_prob > 0:
                    min_scale, max_scale = float(scale_range[0]), float(scale_range[1])
                    transforms.append(
                        A.RandomResizedCrop(
                            size=(self.img_size[0], self.img_size[1]),
                            scale=(min_scale, max_scale),
                            p=rrc_prob,
                        )
                    )

                if rotate_deg > 0 and rotate_prob > 0:
                    transforms.append(
                        A.Rotate(limit=rotate_deg, p=rotate_prob, border_mode=cv2.BORDER_CONSTANT)
                    )

                if hflip:
                    transforms.append(A.HorizontalFlip(p=flip_prob))
                if vflip:
                    transforms.append(A.VerticalFlip(p=flip_prob))

                if len(color_jitter) >= 4 and any(float(x) > 0 for x in color_jitter) and color_jitter_prob > 0:
                    brightness, contrast, saturation, hue = color_jitter[0], color_jitter[1], color_jitter[2], color_jitter[3]
                    transforms.append(
                        A.ColorJitter(
                            brightness=brightness,
                            contrast=contrast,
                            saturation=saturation,
                            hue=hue,
                            p=color_jitter_prob,
                        )
                    )
                if blur_prob > 0:
                    transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=blur_prob))

                transforms.append(ToTensorV2(transpose_mask=False))
                transform = A.Compose(transforms)
            else:
                # Legacy augmentation pipeline (kept for backward-compatible experiments).
                transforms = []

                if scale_range is not None and len(scale_range) == 2:
                    min_scale, max_scale = scale_range[0], scale_range[1]
                    transforms.append(A.Affine(
                        scale={'x': (min_scale, max_scale), 'y': (min_scale, max_scale)},
                        p=1.0,
                        mode=cv2.BORDER_CONSTANT
                    ))
                
                transforms.append(A.LongestMaxSize(max_size=max(self.img_size[0], self.crop_size[0]+100), p=1.0))

                transforms.append(A.PadIfNeeded(min_height=self.crop_size[0], min_width=self.crop_size[1], 
                                 border_mode=cv2.BORDER_CONSTANT))

                transforms.append(A.RandomCrop(self.crop_size[0], self.crop_size[1]))

                if hflip:
                    transforms.append(A.HorizontalFlip(p=flip_prob))

                if vflip:
                    transforms.append(A.VerticalFlip(p=flip_prob))

                if rotate_deg > 0:
                    transforms.append(A.Rotate(limit=rotate_deg, p=rotate_prob, border_mode=cv2.BORDER_CONSTANT))
                
                if len(color_jitter) >= 4:
                    brightness, contrast, saturation, hue = color_jitter[0], color_jitter[1], color_jitter[2], color_jitter[3]
                    transforms.append(A.ColorJitter(
                        brightness=brightness, 
                        contrast=contrast, 
                        saturation=saturation, 
                        hue=hue, 
                        p=color_jitter_prob
                    ))
                
                if blur_prob > 0:
                    transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=blur_prob))
                
                transforms.append(ToTensorV2(transpose_mask=False))  # convert HWC→CHW without normalization
                
                transform = A.Compose(transforms)
        else:
            transform = A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                ToTensorV2(transpose_mask=False),  # convert HWC→CHW without normalization
            ])
        return transform
    
    def _load_mask(self, mask_path: Path, img_h: int, img_w: int) -> np.ndarray:
        """Load mask image and convert to a binary lesion label map."""
        if not mask_path.exists():
            return np.zeros((img_h, img_w), dtype=np.uint8)
        
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            return np.zeros((img_h, img_w), dtype=np.uint8)
        
        if mask_img.shape[:2] != (img_h, img_w):
            mask_img = cv2.resize(mask_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        mask = np.zeros_like(mask_img, dtype=np.uint8)
        mask[mask_img > 0] = 1  # foreground (lesion) = 1, background = 0
        
        return mask
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        
        assert image is not None, f"Failed to read image: {img_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        assert h > 0 and w > 0, f"Invalid image size: {h}x{w} for {img_path}"
        
        if self.mask_files is not None and len(self.mask_files) == len(self.img_files):
            mask_path = self.mask_files[idx]
        else:
            # Standard mask filename patterns for ISIC/YOLO and PH2-style preprocessing.
            candidates = [
                self.label_dir / f"{img_path.stem}_segmentation.png",
                self.label_dir / f"{img_path.stem}.png",
                self.label_dir / f"{img_path.stem}.bmp",
                self.label_dir / f"{img_path.stem}_lesion.png",
                self.label_dir / f"{img_path.stem}_lesion.bmp",
            ]
            mask_path = None
            for p in candidates:
                if p.exists():
                    mask_path = p
                    break
            
            if mask_path is None:
                mask_path = self.label_dir / f"{img_path.stem}.png"
        
        mask = self._load_mask(mask_path, h, w)
        
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
    """Build DataLoader for PH2-style or generic mask-based segmentation datasets."""
    aug_config = config['data'].get('aug', {}) if split == 'train' else {}

    cv_num_folds = config['data'].get('cv_num_folds', 1)
    cv_fold_index = config['data'].get('cv_fold_index', 0)
    cv_seed = config['data'].get('cv_seed', 42)
    
    dataset = MaskSegDataset(
        data_root=config['data']['dataset_root'],
        split=split,
        img_size=tuple(config['data']['img_size']),
        crop_size=tuple(config['data']['aug']['crop_size']) if split == 'train' else tuple(config['data']['img_size']),
        num_classes=config['data']['num_classes'],
        augment=(split == 'train'),
        aug_config=aug_config,
        cv_num_folds=cv_num_folds,
        cv_fold_index=cv_fold_index,
        cv_seed=cv_seed
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

