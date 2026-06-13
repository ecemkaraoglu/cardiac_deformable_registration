# Cardiac Deformable Image Registration
### Multi-Dataset Evaluation of a CNN-Transformer Hybrid Framework

This repository contains the reproduction and multi-dataset evaluation of the **DIR-MRVIT** method for deformable cardiac image registration. The original model was proposed in:

> Lu, X., Zhao, H., Chen, H., Yang, D., Zhang, S., & Xie, Q. (2026). *Deformable image registration using multi-resolution vision Transformer for cardiac motion estimation*. Physics in Medicine & Biology, 71, 025004. https://doi.org/10.1088/1361-6560/ae365a

The original model code is available at: https://github.com/xslu-scuec/DIR-MRVIT

---

## Overview

This work reproduces the DIR-MRVIT deformable registration framework and evaluates it on three publicly available cardiac MRI datasets:

| Dataset | Pairs | Task |
|---------|-------|------|
| **ACDC17** — MICCAI 2017 Automatic Cardiac Diagnosis Challenge | 100 | 5-fold cross-validation |
| **M&Ms20** — MICCAI 2020 Multi-Centre Multi-Vendor Cardiac Segmentation | 150 | 5-fold cross-validation |
| **CMRxM22** — MICCAI 2022 Extreme Cardiac MRI under Respiratory Motion | ~69 | Generalisability evaluation using M&Ms20-trained model |

The model performs unsupervised deformable registration between end-diastolic (ED, fixed) and end-systolic (ES, moving) cardiac MRI frames.

---

## Repository Structure

```
cardiac_deformable_registration/
│
├── Code/
│   ├── GP_TF.py                    # Main model: 3-level Laplacian pyramid + CPTB (original, unchanged)
│   ├── TransBlock.py               # Convolutional Projection Transformer Block (original, unchanged)
│   ├── Functions.py                # Grid generation, dataset utilities (original, unchanged)
│   ├── Train_GPTF_disp.py          # Original training script (reference only, unchanged)
│   ├── Test_GPTF_disp.py           # Original test script (reference only, unchanged)
│   │
│   ├── train_acdc.py               # ACDC17 training pipeline (written for this project)
│   ├── cross_validation_acdc.py    # ACDC17 5-fold stratified cross-validation (written for this project)
│   ├── cross_validation_mms.py     # M&Ms20 5-fold cross-validation (written for this project)
│   │
│   ├── preprocess_mms.py           # M&Ms20 offline preprocessing (written for this project)
│   ├── preprocess_cmrxm22.py       # CMRxM22 preprocessing + IQA filtering (written for this project)
│   │
│   ├── evaluate_cmrxm22.py         # CMRxM22 evaluation (written for this project)
│   ├── eval_acdc_test.py           # Independent ACDC17 test evaluation (written for this project)
│   └── eval_mms_test.py            # Independent M&Ms20 test evaluation (written for this project)
│
├── Models_cv_full/                 # ACDC17 trained models (5-fold)
│   ├── fold{0-4}_lvl3_best.pth     # Best checkpoint per fold
│   ├── cv_pooled_results.npy       # Pooled test results across all folds
│   └── eval_acdc_independent.npy   # Independent verification results
│
├── Models_mms_full/                # M&Ms20 trained models (5-fold)
│   ├── fold{0-4}_lvl3_best.pth     # Best checkpoint per fold
│   ├── cv_pooled_results_mms.npy   # Pooled test results across all folds
│   └── eval_mms_independent.npy    # Independent verification results
│
└── Results/
    └── cmrxm22/
        └── cmrxm22_results.csv     # Per-case CMRxM22 evaluation results
```

---

## Requirements

### Hardware
- GPU with CUDA support (tested on NVIDIA RTX 5070 Ti, CUDA 12.8)
- Minimum 8 GB VRAM recommended

### Software
- Python 3.12
- CUDA 12.8

### Python Dependencies

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install nibabel==5.4.2 numpy==2.4.4 scipy==1.17.1 simpleitk==2.5.4
```

Full dependency list:
```
filelock==3.25.2
fsspec==2026.2.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
nibabel==5.4.2
numpy==2.4.4
packaging==26.2
pillow==12.1.1
scipy==1.17.1
setuptools==70.2.0
simpleitk==2.5.4
sympy==1.14.0
torch==2.11.0+cu128
torchvision==0.26.0+cu128
typing_extensions==4.15.0
xformers==0.0.35
```

---

## Datasets

All three datasets are publicly available. Download and place them as follows:

### ACDC17
- **Download:** https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- Place training data in: `Data/training/patient001/`, `Data/training/patient002/`, ...
- Expected files per patient: `patient001_frame01.nii.gz`, `patient001_frame01_gt.nii.gz`, etc.

### M&Ms20
- **Download:** https://www.ub.edu/mnms/
- Required files per patient: `<code>_sa.nii.gz`, `<code>_sa_gt.nii.gz`
- CSV file: `Data/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv`
- Place raw data in: `Data/MMs_training/`
- **Run preprocessing before training** (see Preprocessing section)

### CMRxM22
- **Download:** https://zenodo.org/records/6362258
- Place extracted folder (containing `data/` and `IQA.csv`) at: `Data/CMRxM22_raw/`
- **Run preprocessing before evaluation** (see Preprocessing section)

---

## Preprocessing

### M&Ms20 (required before training)
```bash
python Code/preprocess_mms.py
```
Resamples to 1.25x1.25x8 mm, crops to 96x96x16 centred on the LV using anchored cropping, and saves to `Data/MMs_preprocessed/`.

### CMRxM22 (required before evaluation)
```bash
python Code/preprocess_cmrxm22.py \
    --raw_dir Data/CMRxM22_raw \
    --out_dir Data/CMRxM22_preprocessed
```
Filters non-diagnostic cases via IQA scores, resamples, crops, remaps labels to M&Ms convention, and generates `pairs.txt`.

---

## Training

> **Pre-trained checkpoints for all folds are included in this repository.** Training is only needed to reproduce the process from scratch.

### ACDC17 — 5-fold Cross-Validation
```bash
python Code/cross_validation_acdc.py
```
Trains a separate model per fold using stratified split (equal pathology groups per fold). Checkpoints saved to `Models_cv_full/fold{i}_lvl3_best.pth`.

### M&Ms20 — 5-fold Cross-Validation
```bash
python Code/cross_validation_mms.py
```
Trains a separate model per fold (105/15/30 train/val/test split). Checkpoints saved to `Models_mms_full/fold{i}_lvl3_best.pth`.

---

## Evaluation

### ACDC17
```bash
python Code/eval_acdc_test.py
```

### M&Ms20
```bash
python Code/eval_mms_test.py
```

### CMRxM22
```bash
python Code/evaluate_cmrxm22.py \
    --data_dir Data/CMRxM22_preprocessed \
    --model_dir Models_mms_full \
    --all_folds \
    --out_dir Results/cmrxm22
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{lu2026dir,
  title={Deformable image registration using multi-resolution vision Transformer for cardiac motion estimation},
  author={Lu, Xuesong and Zhao, Huaqiu and Chen, Hong and Yang, Dandan and Zhang, Su and Xie, Qinlan},
  journal={Physics in Medicine \& Biology},
  volume={71},
  pages={025004},
  year={2026},
  publisher={IOP Publishing}
}
```
