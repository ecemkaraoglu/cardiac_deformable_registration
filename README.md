# Cardiac Deformable Image Registration
### Multi-Dataset Evaluation of a CNN-Transformer Hybrid Framework

This repository contains the reproduction and multi-dataset evaluation of the **DIR-MRVIT** method for deformable cardiac image registration, applied to three public cardiac MRI datasets (ACDC17, M&Ms20, CMRxM22).

The original model was proposed in:

> Lu, X., Zhao, H., Chen, H., Yang, D., Zhang, S., & Xie, Q. (2026). *Deformable image registration using multi-resolution vision Transformer for cardiac motion estimation*. Physics in Medicine & Biology, 71, 025004. https://doi.org/10.1088/1361-6560/ae365a

The original model code is available at: https://github.com/xslu-scuec/DIR-MRVIT

---

## Introduction

Cardiac MR imaging is the main modality for assessing cardiac function. Estimating the motion of the heart between end-diastole (ED) and end-systole (ES) is clinically valuable, but doing it by hand is slow and varies between observers. **Deformable image registration** finds a spatial mapping between two images so that automatic motion measurement becomes possible.

This project reproduces the DIR-MRVIT method — a hybrid CNN-Transformer network with a three-level Laplacian pyramid — and applies it to three different cardiac MRI datasets. The original repository only targets M&Ms20; this work extends the pipeline to ACDC17 and CMRxM22 as well, and compares the results directly to the values reported in the paper's Tables 2–4.

The original model code (`GP_TF.py`, `TransBlock.py`, `Functions.py`) is used as published. The preprocessing pipelines, cross-validation drivers, and evaluation scripts were written from scratch and adapted to each dataset.

---

## Key Findings

Across all three datasets, the reproduced metrics are close to the paper. On the myocardium Dice, this work **passes the paper on M&Ms20 and CMRxM22**:

| Dataset  | Metric          | This work             | Paper             |
|----------|-----------------|-----------------------|-------------------|
| ACDC17   | LV Dice         | 0.887 ± 0.069         | 0.917 ± 0.043     |
| ACDC17   | MYO Dice        | 0.752 ± 0.065         | 0.789 ± 0.055     |
| M&Ms20   | LV Dice         | 0.877 ± 0.050         | 0.884 ± 0.038     |
| M&Ms20   | **MYO Dice**    | **0.790 ± 0.070** ✅   | 0.729 ± 0.057     |
| CMRxM22  | LV Dice         | 0.857 ± 0.051         | 0.892 ± 0.027     |
| CMRxM22  | **MYO Dice**    | **0.758 ± 0.077** ✅   | 0.703 ± 0.050     |

(✅ = beats paper. Full table including ASSD, HD95, and Jacobian metrics is in the project report.)

The remaining gaps come from the unspecified training schedule in the paper, random training variation, and minor preprocessing choices (e.g. independent vs. anchored cropping). See the full report for the detailed discussion.

---

## Quick Demo

The repository includes a unified demo script that runs on a small set of sample patients shipped with the repository. **No dataset download required.**

```bash
git clone https://github.com/ecemkaraoglu/cardiac_deformable_registration.git
cd cardiac_deformable_registration
pip install -r requirements.txt
python Code/run_test.py
```

Runs all three datasets on the included sample patients (10 ACDC + 10 M&Ms + 18 CMRxM22) using the pre-trained checkpoints. Takes about 1–2 minutes on a CUDA GPU.

Useful options:

```bash
python Code/run_test.py --dataset acdc      # run only ACDC
python Code/run_test.py --dataset mms       # run only M&Ms
python Code/run_test.py --dataset cmrxm22   # run only CMRxM22
python Code/run_test.py --cpu               # force CPU (no GPU needed)
```

The script prints per-patient metrics, a summary table with paper comparison, and saves warped images and deformation fields to `Results/test_output/`.

### Random qualitative figure

For a visual demo on top of the numbers, run:

```bash
python Code/demo_figure.py            # GPU if available, else CPU
python Code/demo_figure.py --cpu      # force CPU
```

This picks **one random patient from each dataset** (ACDC, M&Ms, CMRxM22) and produces a 3×4 figure showing the moving image (ES), fixed image (ED), the warped image predicted by the model, and the difference map — with overlaid LV-endocardium (orange) and LV-epicardium (green) contours. The figure pops up in a matplotlib window. Re-running the script picks a different random patient each time.

**Note:** Sample results are computed on a small subset and may differ slightly from the full-dataset numbers reported above. The full results in the table use 100/150/69 patients respectively with five-fold cross-validation.

---

## Repository Structure

```
cardiac_deformable_registration/
│
├── Code/
│   ├── GP_TF.py                    # Main model: 3-level Laplacian pyramid + CPTB (original, unchanged)
│   ├── TransBlock.py               # Convolutional Projection Transformer Block (original, unchanged)
│   ├── Functions.py                # Grid generation, dataset utilities (original, unchanged)
│   ├── Train_GPTF_disp.py          # Original training script (reference, unchanged)
│   ├── Test_GPTF_disp.py           # Original test script (reference, unchanged)
│   │
│   ├── cross_validation_acdc.py    # ACDC17 5-fold stratified cross-validation
│   ├── cross_validation_mms.py     # M&Ms20 5-fold cross-validation
│   │
│   ├── preprocess_mms.py           # M&Ms20 preprocessing (resample, anchored crop, normalize)
│   ├── preprocess_cmrxm22.py       # CMRxM22 preprocessing (IQA filter, resample, label remap)
│   │
│   ├── evaluate_cmrxm22.py         # CMRxM22 zero-shot evaluation (M&Ms-trained model)
│   ├── eval_acdc_test.py           # Independent ACDC17 verification on saved checkpoints
│   ├── eval_mms_test.py            # Independent M&Ms20 verification on saved checkpoints
│   │
│   ├── run_test.py                 # Unified demo script (all three datasets, sample data)
│   ├── demo_figure.py              # Random qualitative figure (one patient per dataset)
│   └── figure_a.py                 # Qualitative figure generator (used for the project report)
│
├── Data/
│   ├── sample_data/                # 10 ACDC + 10 M&Ms + 18 CMRxM22 sample patients for the demo
│   │   ├── acdc/
│   │   ├── mms/
│   │   └── cmrxm22/
│   └── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
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
├── Results/
│   └── cmrxm22/
│       └── cmrxm22_results.csv     # Per-case CMRxM22 evaluation results
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Requirements

### Hardware

- GPU with CUDA support recommended (tested on NVIDIA RTX 5070 Ti, CUDA 12.8)
- Minimum 8 GB VRAM for training
- The demo script runs on CPU as well (use `--cpu`)

### Software

- Python 3.12
- CUDA 12.8 (for GPU mode)

### Installation

Copy and paste the commands below into your terminal:

**GPU (CUDA 12.8) — recommended:**

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**CPU only — slower, but no CUDA required:**

```bash
pip install torch==2.11.0 torchvision==0.26.0
pip install -r requirements.txt
```

For other CUDA versions, see https://pytorch.org/get-started/locally/ for the right `torch` install command, then run `pip install -r requirements.txt`.

---

## Datasets

The demo uses the included sample data and needs no download. The full datasets are only needed to reproduce the training or full-dataset evaluation.

### ACDC17

- Download: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- Place training data in `Data/training/patient001/`, `Data/training/patient002/`, ...
- Expected files per patient: `patient001_frame01.nii.gz`, `patient001_frame01_gt.nii.gz`, etc.

### M&Ms20

- Download: https://www.ub.edu/mnms/
- Required files per patient: `<code>_sa.nii.gz`, `<code>_sa_gt.nii.gz`
- Place raw data in `Data/MMs_training/`
- CSV file: `Data/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv` (already included)
- Run `preprocess_mms.py` before training (see below)

### CMRxM22

- Download: https://zenodo.org/records/6362258
- Place extracted folder (containing `data/` and `IQA.csv`) at `Data/CMRxM22_raw/`
- Run `preprocess_cmrxm22.py` before evaluation (see below)

---

## Preprocessing

ACDC17 preprocessing (resampling, cropping, normalization) is integrated into `cross_validation_acdc.py` because the dataset is small enough to preprocess at the start of each run. M&Ms20 and CMRxM22 use heavier preprocessing (4D timeseries extraction, IQA filtering, label remapping) so they have separate scripts that run once before training/evaluation.

> **Design note.** ACDC preprocessing is performed inline by each ACDC script (training, evaluation, and demo) rather than as a separate one-shot step that writes to disk. This is for historical reasons — ACDC was the first dataset added and its preprocessing is light, so a dedicated `preprocess_acdc.py` script was never separated out. M&Ms20 and CMRxM22 do use disk-based preprocessing because of their heavier transformations (4D extraction, IQA filtering, label remapping). All three datasets apply the same operations during preprocessing (resample → crop → normalize); only the location of the code differs. Sample ACDC patients in `Data/sample_data/acdc/` are therefore stored as raw NIfTI files, while M&Ms20 and CMRxM22 samples are stored as preprocessed NIfTI files.

### M&Ms20

```bash
python Code/preprocess_mms.py
```

Resamples to 1.25 × 1.25 × 8 mm, crops to 96 × 96 × 16 centred on the LV using anchored cropping (ED-based crop box applied to both ED and ES), and saves to `Data/MMs_preprocessed/`.

### CMRxM22

```bash
python Code/preprocess_cmrxm22.py --raw_dir Data/CMRxM22_raw --out_dir Data/CMRxM22_preprocessed
```

Filters out non-diagnostic cases via IQA scores, resamples, crops, remaps labels to the M&Ms convention, and writes a `pairs.txt` for evaluation.

---

## Training

> **Pre-trained checkpoints for all folds are already included** in `Models_cv_full/` and `Models_mms_full/`. Training is only needed to reproduce the process from scratch.

### ACDC17 — 5-fold Cross-Validation

```bash
python Code/cross_validation_acdc.py
```

Stratified 5-fold split (4 patients from each pathology group per fold's test set). Checkpoints saved to `Models_cv_full/fold{i}_lvl3_best.pth`. Takes overnight on a single GPU.

### M&Ms20 — 5-fold Cross-Validation

```bash
python Code/cross_validation_mms.py
```

5-fold random split (105 train / 15 val / 30 test). Checkpoints saved to `Models_mms_full/fold{i}_lvl3_best.pth`.

---

## Evaluation (full datasets)

These scripts run on the full datasets (not the sample data) and reproduce the values from the report.

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
python Code/evaluate_cmrxm22.py --data_dir Data/CMRxM22_preprocessed --model_dir Models_mms_full --all_folds --out_dir Results/cmrxm22
```

The `--all_folds` flag averages the metrics across all five M&Ms-trained fold models for a more stable estimate than a single randomly chosen fold (which is what the paper reports).

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

---

## Acknowledgements

This project was completed for the EE634 Digital Image Processing course at Middle East Technical University. The original DIR-MRVIT model code is the work of Lu et al.; this repository only contains the reproduction effort and the multi-dataset extensions.