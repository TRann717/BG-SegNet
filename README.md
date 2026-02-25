# BG-SegNet: Enhancing Boundary Precision in Medical Image Segmentation with a Lightweight Framework

This repository provides the **official implementation** of the manuscript  
**"BG-SegNet: Enhancing Boundary Precision in Medical Image Segmentation with a Lightweight Framework"**,  
submitted to *The Visual Computer*. The code, pretrained weights, and evaluation results in this repository are **directly related** to the above manuscript. If you use this code or models in your research, please consider citing our paper (see the [Citation](#citation) section).

Medical image segmentation is crucial for clinical analysis, yet it faces challenges such as ill-defined boundaries and the need for real-time processing. This paper introduces BG-SegNet, a lightweight framework that integrates a frozen SAM2 encoder with an explicit boundary-aware gating mechanism. BG-SegNet aggregates multi-scale features, predicts boundary strength and orientation fields, and employs them as spatial gating signals for orientation-aware anisotropic convolutions. The framework achieves high accuracy and efficiency, demonstrating superior performance on a custom wound segmentation dataset and competitive results on the PH2 dermoscopic lesion dataset. Our approach offers a promising solution for real-time, boundary-precise medical image segmentation.

![Overall Model Architecture](figures/model.png)

![Experimental Results](figures/results.png)

## Datasets

### Custom Wound Segmentation Dataset

The primary dataset used in this work follows the **YOLO segmentation format** (`yolo_seg`).

#### Directory Structure

```
/path/to/yolo_seg
  ├── train
  │   ├── images
  │   │   ├── xxx1.jpg / .png
  │   │   └── ...
  │   └── labels
  │       ├── xxx1.txt
  │       └── ...
  ├── val
  │   ├── images
  │   └── labels
  └── test
      ├── images
      └── labels
```

#### Label Format

Each label file `xxx.txt` contains polygon annotations in YOLO format:
```
class_id x1 y1 x2 y2 x3 y3 ...
```
- Normalized coordinates (0–1)
- At least 3 vertices per polygon
- `class_id = 0` for background, `class_id = 1` for foreground (wound/lesion)

The dataset loader (`src/data/dataset_mask.py`) converts YOLO polygons to dense binary masks automatically.

### PH2 Dermoscopic Lesion Dataset

We also evaluate on the PH2 dataset. See `train_ph2_5fold.py` for 5-fold cross-validation setup.

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA recommended)
- Dependencies:
  ```bash
  pip install torch torchvision numpy opencv-python albumentations \
              tqdm tensorboard Pillow hydra-core omegaconf scipy \
              scikit-image scikit-learn
  ```
- We recommend creating a fresh virtual environment and installing dependencies via:
  ```bash
  pip install -r BG-SegNet\project\requirements.txt 
  ```

### SAM2 Weights

1. Download **SAM2.1 Hiera B+** weights (https://huggingface.co/facebook/sam2.1-hiera-base-plus):
   - Model checkpoint: `sam2.1_hiera_base_plus.pt`
   - Place in a directory accessible by `model.sam2_path` in `config.yaml`

2. SAM2 Configuration:
   - Ensure Hydra config files are available under `third_party/sam2/sam2/configs/`
   - Update `model.sam2_path` in `config.yaml` to point to the SAM2 directory

---

## Usage

Configuration files (e.g., `config.yaml` for the wound dataset and `config_ph2_best.yaml` for PH2) are included in this repository and can be used to reproduce the main experiments reported in the manuscript.

### Training

**Basic training on YOLO segmentation dataset:**

```bash
cd /path/to/project/bgsegnet
python train.py --config config.yaml
```

**Resume from checkpoint:**

```bash
python train.py --config config.yaml --resume runs/<experiment>/weights/last.pth
```

**Validation only:**

```bash
python train.py --config config.yaml --eval
```

**5-fold cross-validation (PH2):**

```bash
python train_ph2_5fold.py --config config_ph2_best.yaml --ph2_root /path/to/PH2
```

### Evaluation

**Evaluate on test set:**

```bash
python -m src.evaluate_csegnet \
  --config config.yaml \
  --checkpoint best.pth \
  --data_root /path/to/yolo_seg \
  --output_dir evaluation_results \
  --batch_size 16
```

**Single-image inference:**

```bash
python infer_single_image.py \
  --config config.yaml \
  --checkpoint best.pth \
  --image /path/to/input.png \
  --out /path/to/output_mask.png \
  --threshold 0.5
```

## Pretrained Weights and Evaluation Results

- **Pretrained checkpoint**: `best.pth` (trained on the custom wound segmentation dataset used in the paper).
- **Evaluation results**: `evaluation_results/`
  - `evaluation_metrics.json`: quantitative metrics corresponding to the main results in the manuscript.
  - `comparison_images/`: qualitative comparisons on the test set.

To reproduce the test-time evaluation using the provided checkpoint:

```bash
cd project/bgsegnet
python -m src.evaluate_csegnet \
  --config config.yaml \
  --checkpoint best.pth \
  --data_root /path/to/yolo_seg \
  --output_dir ../../evaluation_results_reproduced \
  --batch_size 16
```

## Code Structure

```
bgsegnet/
  ├── config.yaml                 # Main configuration for YOLO-formatted wound dataset
  ├── config_ph2_best.yaml        # PH2 configuration (5-fold CV)
  ├── train.py                    # Main training script for wound dataset
  ├── train_ph2_5fold.py          # 5-fold CV training for PH2
  ├── infer_single_image.py       # Single-image inference
  └── src/
      ├── config/loader.py        # YAML config loader
      ├── data/
      │   ├── dataset_mask.py     # YOLO polygon → mask dataset (wound)
      │   └── dataset_mask_ph2.py # PH2 mask dataset
      ├── engine/
      │   ├── trainer.py          # Training loop with boundary curriculum (wound)
      │   ├── trainer_ph2.py      # Training loop for PH2
      │   ├── metrics.py          # Comprehensive evaluation metrics
      │   └── metrics_unit.py     
      ├── losses/
      │   └── seg_losses.py       # Segmentation + unified soft boundary loss
      ├── models/
      │   ├── sam2_loader.py      # SAM2 loading & feature extraction
      │   ├── cseg_net.py         # BG-SegNet main model
      │   ├── neck/ppm.py         # UPerNet (PPM + FPN) neck
      │   └── heads/
      │       ├── decoder_head.py  # Main decoder + gating module
      │       ├── boundary_head.py # Boundary prediction head
      │       └── boundary_gate.py # Boundary-gated refinement module
      └── utils/logger.py         # TensorBoard & CSV logging
```

## Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@article{Li2025BGSegNet,
  title   = {BG-SegNet: Enhancing Boundary Precision in Medical Image Segmentation with a Lightweight Framework},
  author  = {Bo Li and Ran Tian and Others},
  journal = {The Visual Computer},
  year    = {2025},
  note    = {Under review},
  % doi   = {10.XXXX/XXXXX}  % To be updated after acceptance
}
```
