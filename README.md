# Cardiac Deformable Image Registration

Deformable image registration estimates a dense spatial mapping between two images of the same anatomy. In cardiac MRI, this mapping describes how the heart muscle moves between end-diastole (ED) and end-systole (ES), and supports automatic motion analysis that would otherwise require manual delineation.

DIR-MRVIT uses a **Laplacian pyramid** with **Convolutional Projection Transformer Blocks (CPTBs)** trained in a **coarse-to-fine** manner to register cardiac MR images between end-diastole and end-systole.

The original paper evaluates DIR-MRVIT on ACDC17, M&Ms20, and CMRxM22, but the released code base only covers M&Ms20. This project adds the missing pieces: a stratified 5-fold cross-validation pipeline for ACDC17, a preprocessing and evaluation pipeline for CMRxM22, and a unified demo that runs the trained models on sample patients from all three datasets. The original model code (`GP_TF.py`, `TransBlock.py`, `Functions.py`) is used as published; everything else in `Code/` was written here.

Reference paper: Lu et al., *Deformable image registration using multi-resolution vision Transformer for cardiac motion estimation*, Phys. Med. Biol. 71, 025004 (2026). [DOI](https://doi.org/10.1088/1361-6560/ae365a). Original repository: https://github.com/xslu-scuec/DIR-MRVIT.

---

## Repository structure

```
cardiac_deformable_registration/
│
├── Code/
│   ├── GP_TF.py                    # 3-level Laplacian pyramid model (original, unchanged)
│   ├── TransBlock.py               # Convolutional Projection Transformer Block (original)
│   ├── Functions.py                # Grid generation, dataset utilities (original)
│   ├── Train_GPTF_disp.py          # Original training script (reference)
│   ├── Test_GPTF_disp.py           # Original test script (reference)
│   │
│   ├── cross_validation_acdc.py    # ACDC17 5-fold stratified cross-validation
│   ├── cross_validation_mms.py     # M&Ms20 5-fold cross-validation
│   │
│   ├── preprocess_mms.py           # M&Ms20 preprocessing
│   ├── preprocess_cmrxm22.py       # CMRxM22 preprocessing (IQA filter, label remap)
│   │
│   ├── evaluate_cmrxm22.py         # CMRxM22 evaluation
│   ├── eval_acdc_test.py           # Independent ACDC17 verification
│   ├── eval_mms_test.py            # Independent M&Ms20 verification
│   │
│   ├── run_test.py                 # Unified demo (all three datasets, sample data)
│   ├── demo_figure.py              # Random qualitative figure (one patient per dataset)
│
├── Data/
│   ├── sample_data/                # Sample patients shipped with the repo for the demo
│   │   ├── acdc/                   # 10 ACDC patients (raw NIfTI)
│   │   ├── mms/                    # 10 M&Ms patients (preprocessed)
│   │   └── cmrxm22/                # 18 CMRxM22 scans (preprocessed)
│   └── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
│
├── Models_cv_full/                 # ACDC17 trained checkpoints (5 folds)
├── Models_mms_full/                # M&Ms20 trained checkpoints (5 folds)
│
├── Results/
│   └── cmrxm22/                    # Per-case CMRxM22 evaluation results
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Hardware requirements

- GPU with CUDA support recommended (tested on NVIDIA RTX 5070 Ti, CUDA 12.8)
- Minimum 8 GB VRAM for training
- The demo scripts also run on CPU (use the `--cpu` flag)
- Python 3.12

For CUDA versions other than 12.8, see https://pytorch.org/get-started/locally/ for the matching `torch` install command, then continue with `pip install -r requirements.txt`.

---

## Getting started

The repository ships with a small set of sample patients so that the trained models can be tried out **without downloading any of the full datasets**.

The five commands below clone the repository, install PyTorch and the remaining dependencies, and run the unified demo on all three datasets. They can be copied and pasted into a terminal as a single block.

**GPU (CUDA 12.8)**

```bash
git clone https://github.com/ecemkaraoglu/cardiac_deformable_registration.git
cd cardiac_deformable_registration
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python Code/run_test.py
```

**CPU only**

```bash
git clone https://github.com/ecemkaraoglu/cardiac_deformable_registration.git
cd cardiac_deformable_registration
pip install torch==2.11.0 torchvision==0.26.0
pip install -r requirements.txt
python Code/run_test.py --cpu
```

The `run_test.py` script runs the trained models on the sample patients (10 ACDC + 10 M&Ms + 18 CMRxM22) and prints per-patient and pooled metrics for each dataset, with the corresponding paper values printed alongside for comparison.

### Running a single dataset

```bash
python Code/run_test.py --dataset acdc      # ACDC only
python Code/run_test.py --dataset mms       # M&Ms only
python Code/run_test.py --dataset cmrxm22   # CMRxM22 only
```

### Qualitative demo

```bash
python Code/demo_figure.py
```

Picks one random patient from each dataset and opens a matplotlib window showing four panels per patient: the moving frame (ES), the fixed frame (ED), the warped frame predicted by the model, and a difference map. Overlaid contours show the LV endocardium (orange) and LV epicardium (green). Re-running picks different random patients.

Both demo scripts use a single trained fold (fold 0) for each dataset. The pooled results in the *Results* section below are averaged across all five folds.

---

## Datasets

Only required to reproduce the full training and evaluation. The demo runs from the included sample data and needs no external download.

### ACDC17

- Download: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- Place patient folders in `Data/training/`, e.g. `Data/training/patient001/`.

### M&Ms20

- Download: https://www.ub.edu/mnms/
- Place raw cases in `Data/MMs_training/`.
- The diagnosis CSV is already included at `Data/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv`.

### CMRxM22

- Download: https://zenodo.org/records/6362258
- Place the extracted folder at `Data/CMRxM22_raw/`.

---

## Preprocessing

ACDC17 preprocessing (resample to 1.25 x 1.25 x 5.0 mm, crop to 96 x 96 x 16, intensity normalization) is performed inline by each ACDC script (`cross_validation_acdc.py`, `eval_acdc_test.py`, `run_test.py`, and `demo_figure.py`). No separate `preprocess_acdc.py` step is needed.

M&Ms20 and CMRxM22 use disk-based preprocessing because of their heavier transformations (4D timeseries extraction, IQA filtering, label remapping). All three datasets apply the same operations conceptually (resample, crop, normalize); only the location of the code differs.

### M&Ms20

```bash
python Code/preprocess_mms.py
```

Extracts ED and ES frames from the 4D series, resamples to 1.25 x 1.25 x 8 mm, applies anchored cropping (ED-based crop box reused for ES), and writes preprocessed NIfTI files to `Data/MMs_preprocessed/`.

### CMRxM22

```bash
python Code/preprocess_cmrxm22.py --raw_dir Data/CMRxM22_raw --out_dir Data/CMRxM22_preprocessed
```

Filters out non-diagnostic cases via the IQA score, resamples, crops, remaps labels into the M&Ms20 convention, and writes a `pairs.txt` file used by the evaluation script.

---

## Training

Pre-trained checkpoints for all folds are already included in `Models_cv_full/` (ACDC17) and `Models_mms_full/` (M&Ms20). The commands below only need to be re-run to reproduce training from scratch.

### ACDC17

```bash
python Code/cross_validation_acdc.py
```

Stratified 5-fold split, 4 patients per pathology group in each test fold. Checkpoints are saved as `Models_cv_full/fold{i}_lvl3_best.pth`.

### M&Ms20

```bash
python Code/cross_validation_mms.py
```

5-fold split with 105 train / 15 validation / 30 test patients per fold. Checkpoints are saved as `Models_mms_full/fold{i}_lvl3_best.pth`.

---

## Evaluation on the full datasets

These scripts reproduce the numbers in the *Results* section below.

```bash
python Code/eval_acdc_test.py        # ACDC17, 5-fold pooled
python Code/eval_mms_test.py         # M&Ms20, 5-fold pooled
python Code/evaluate_cmrxm22.py --data_dir Data/CMRxM22_preprocessed --model_dir Models_mms_full --all_folds --out_dir Results/cmrxm22
```

The `--all_folds` flag averages metrics across all five M&Ms-trained fold models for a more stable estimate than a single randomly chosen fold (which is what the paper reports).

---

## Results

Three evaluation regimes are reported below. The **sample** column shows the metrics produced by `run_test.py` on the small sample subsets shipped with this repository (10 ACDC, 10 M&Ms, 18 CMRxM22), using fold 0 of the trained checkpoints. The **full** column shows the full-dataset results reproduced in this project: 5-fold stratified cross-validation on the 100 ACDC17 patients, 5-fold cross-validation on the 150 M&Ms20 patients (with smoothness weight λ=0.2), and zero-shot transfer to all 69 CMRxM22 cases averaged across the five M&Ms-trained fold models. The **paper** column shows the values reported in Tables 2–4 of the original DIR-MRVIT paper.

### ACDC17

| Metric          | Sample (10)         | Full (100, 5-fold)   | Paper (100, 5-fold) |
|-----------------|---------------------|----------------------|---------------------|
| LV Dice         | 0.870 ± 0.080       | 0.887 ± 0.069        | 0.917 ± 0.043       |
| MYO Dice        | 0.754 ± 0.063       | 0.752 ± 0.065        | 0.789 ± 0.055       |
| Endo HD95 (mm)  | 5.59  ± 2.44        | 5.27  ± 2.43         | 5.62  ± 1.22        |
| Epi HD95 (mm)   | 6.63  ± 2.32        | 4.40  ± 2.01         | 5.51  ± 1.77        |

### M&Ms20

| Metric          | Sample (10)         | Full (150, 5-fold)   | Paper (150, 5-fold) |
|-----------------|---------------------|----------------------|---------------------|
| LV Dice         | 0.893 ± 0.030       | 0.877 ± 0.050        | 0.884 ± 0.038       |
| MYO Dice        | 0.794 ± 0.047       | 0.790 ± 0.070        | 0.729 ± 0.057       |
| Endo HD95 (mm)  | 6.58  ± 1.84        | 7.15                 | 7.40  ± 2.78        |
| Epi HD95 (mm)   | 8.20  ± 0.82        | 8.74                 | 8.16  ± 1.95        |

### CMRxM22 (zero-shot)

| Metric          | Sample (18)         | Full (69, all folds) | Paper (69)          |
|-----------------|---------------------|----------------------|---------------------|
| LV Dice         | 0.857 ± 0.040       | 0.857 ± 0.051        | 0.892 ± 0.027       |
| MYO Dice        | 0.675 ± 0.051       | 0.758 ± 0.077        | 0.703 ± 0.050       |
| Endo HD95 (mm)  | 7.74  ± 2.10        | 7.51                 | 7.82  ± 2.31        |
| Epi HD95 (mm)   | 7.85  ± 0.90        | 8.22                 | 8.07  ± 2.24        |

Additional metrics (ASSD, Jacobian folding percentage), per-fold breakdowns, and the ablation comparing smoothness weights (λ=0.5 vs λ=0.2 on M&Ms20) are reported in the project report.

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

This project was carried out for the EE634 Digital Image Processing course at Middle East Technical University. The DIR-MRVIT model code is the work of Lu et al.; this repository contains the multi-dataset extensions and evaluation pipeline built around it.