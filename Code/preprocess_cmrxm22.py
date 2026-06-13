"""
preprocess_cmrxm22.py

CMRxMotion 2022 (CMRxM22) dataset preprocessing for DIR-MRVIT evaluation.

Dataset source: https://zenodo.org/records/6362258
Download the "CMRxMotion Training Dataset" zip file.

Folder structure expected after extraction:
    <raw_dir>/
        data/
            P001-1/
                P001-1-ED.nii.gz
                P001-1-ED-label.nii.gz
                P001-1-ES.nii.gz
                P001-1-ES-label.nii.gz
            P001-2/
            ...
        IQA.csv

What this script does:
    1. Reads IQA.csv to filter out non-diagnostic quality cases.
    2. For each remaining subject-scan pair, loads ED and ES NIfTI volumes.
    3. Resamples to 1.25 x 1.25 x 8 mm (paper specification for CMRxM22).
    4. Crops to 96 x 96 x 16 centered on LV.
    5. Saves preprocessed images + labels to output directory.

Output structure:
    <out_dir>/
        P001-1/
            P001-1-ED.nii.gz
            P001-1-ED-label.nii.gz
            P001-1-ES.nii.gz
            P001-1-ES-label.nii.gz
        ...
        pairs.txt   <- list of valid (fixed, moving) pairs for evaluation
"""

import os
import csv
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, center_of_mass

# ---------------------------------------------------------------------------
# Constants from paper (Section 3.1)
# ---------------------------------------------------------------------------
TARGET_SPACING = (1.25, 1.25, 8.0)   # mm, for CMRxM22
CROP_SIZE = (96, 96, 16)              # voxels after resampling

# Label values in CMRxMotion segmentation masks
LABEL_LV  = 1  # left ventricle cavity
LABEL_MYO = 2  # left ventricle myocardium
LABEL_RV  = 3  # right ventricle (not used for registration eval)


def load_nifti(path):
    """Load a NIfTI file and return (data, affine, header)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    # Squeeze out any trailing singleton dimensions (e.g. shape (H,W,D,1))
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    return data, img.affine, img.header


def save_nifti(data, affine, path):
    """Save numpy array as NIfTI."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def get_voxel_spacing(header):
    """Extract voxel spacing (dx, dy, dz) from header."""
    pixdim = header.get_zooms()
    return float(pixdim[0]), float(pixdim[1]), float(pixdim[2])


def resample_volume(data, current_spacing, target_spacing, order=1):
    """
    Resample a 3D volume from current_spacing to target_spacing.
    order=1  -> linear interpolation (images)
    order=0  -> nearest-neighbour (labels)
    """
    zoom_factors = tuple(
        c / t for c, t in zip(current_spacing, target_spacing)
    )
    return zoom(data, zoom_factors, order=order)


def crop_around_lv(image, label, crop_size):
    """
    Crop a region of size crop_size centered on the LV centroid.
    If LV label is absent, fall back to image center.
    Pads with zeros if the crop extends beyond the image boundary.
    """
    lv_mask = (label == LABEL_LV)
    if lv_mask.any():
        centroid = np.array(center_of_mass(lv_mask), dtype=int)
    else:
        centroid = np.array(image.shape) // 2

    half = np.array(crop_size) // 2
    starts = centroid - half
    ends   = starts + np.array(crop_size)

    # Clip and record padding needed
    pad_before = np.maximum(0, -starts)
    pad_after  = np.maximum(0, ends - np.array(image.shape))
    starts_clipped = np.maximum(0, starts)
    ends_clipped   = np.minimum(np.array(image.shape), ends)

    img_crop = image[
        starts_clipped[0]:ends_clipped[0],
        starts_clipped[1]:ends_clipped[1],
        starts_clipped[2]:ends_clipped[2],
    ]
    lbl_crop = label[
        starts_clipped[0]:ends_clipped[0],
        starts_clipped[1]:ends_clipped[1],
        starts_clipped[2]:ends_clipped[2],
    ]

    # Pad to exact crop size
    pad_width = list(zip(pad_before.tolist(), pad_after.tolist()))
    img_crop = np.pad(img_crop, pad_width, mode='constant', constant_values=0)
    lbl_crop = np.pad(lbl_crop, pad_width, mode='constant', constant_values=0)

    return img_crop, lbl_crop


def normalize_image(image):
    """Normalise to [0, 1] using percentile clipping."""
    p1, p99 = np.percentile(image, [1, 99])
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)
    return image.astype(np.float32)


def read_iqa_csv(csv_path):
    """
    Parse IQA.csv from CMRxMotion.
    Returns a set of subject-scan keys (e.g. 'P001-1') whose images
    have diagnostic quality (IQA score != 4).

    IQA score: 1=excellent, 2=good, 3=moderate, 4=non-diagnostic
    We keep scores 1-3 (diagnostic quality) for both ED and ES.
    """
    valid = set()
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # Group rows by subject-scan key
        rows_by_key = {}
        for row in reader:
            # Actual columns in CMRxMotion IQA.csv: Image, Label
            # Image example: P001-1-ED or P001-1-ES
            # Label: 1=excellent, 2=good, 3=moderate, 4=non-diagnostic
            filename = row.get('Image', row.get('FileName', row.get('filename', ''))).strip()
            iqa_val  = row.get('Label', row.get('IQA', row.get('iqa', ''))).strip()
            # Extract key: P001-1 from P001-1-ED
            parts = filename.rsplit('-', 1)
            if len(parts) == 2:
                key = parts[0]
                if key not in rows_by_key:
                    rows_by_key[key] = []
                try:
                    rows_by_key[key].append(int(iqa_val))
                except ValueError:
                    pass

    for key, scores in rows_by_key.items():
        # Keep only if both ED and ES are diagnostic (score <= 3)
        if all(s <= 3 for s in scores):
            valid.add(key)

    return valid


def preprocess_case(raw_dir, case_key, out_dir):
    """
    Preprocess one subject-scan case (e.g. P001-1).
    Returns True on success, False if files are missing.
    """
    case_in = os.path.join(raw_dir, 'data', case_key)
    ed_img_path = os.path.join(case_in, f'{case_key}-ED.nii.gz')
    ed_lbl_path = os.path.join(case_in, f'{case_key}-ED-label.nii.gz')
    es_img_path = os.path.join(case_in, f'{case_key}-ES.nii.gz')
    es_lbl_path = os.path.join(case_in, f'{case_key}-ES-label.nii.gz')

    for p in [ed_img_path, ed_lbl_path, es_img_path, es_lbl_path]:
        if not os.path.exists(p):
            print(f'  [SKIP] Missing file: {p}')
            return False

    # Load
    ed_img, affine, header = load_nifti(ed_img_path)
    ed_lbl, _, _           = load_nifti(ed_lbl_path)
    es_img, _, _           = load_nifti(es_img_path)
    es_lbl, _, _           = load_nifti(es_lbl_path)

    spacing = get_voxel_spacing(header)
    print(f'  Original spacing: {spacing}, shape: {ed_img.shape}')

    # Resample images (linear) and labels (nearest-neighbour)
    ed_img_r = resample_volume(ed_img, spacing, TARGET_SPACING, order=1)
    ed_lbl_r = resample_volume(ed_lbl, spacing, TARGET_SPACING, order=0)
    es_img_r = resample_volume(es_img, spacing, TARGET_SPACING, order=1)
    es_lbl_r = resample_volume(es_lbl, spacing, TARGET_SPACING, order=0)

    # Round label values (interpolation may introduce floats)
    ed_lbl_r = np.round(ed_lbl_r).astype(np.uint8)
    es_lbl_r = np.round(es_lbl_r).astype(np.uint8)

    # Remap CMRxM22 labels to M&Ms20 convention so the trained model applies correctly:
    #   CMRxM22: 1=LV, 2=MYO, 3=RV
    #   M&Ms20:  1=LV, 2=RV,  3=MYO  (model was trained with this)
    # Swap label 2 <-> label 3
    for lbl in [ed_lbl_r, es_lbl_r]:
        tmp = lbl.copy()
        lbl[tmp == 2] = 3   # MYO: 2 -> 3
        lbl[tmp == 3] = 2   # RV:  3 -> 2

    # Crop centered on LV (use ED label for consistent centroid)
    ed_img_c, ed_lbl_c = crop_around_lv(ed_img_r, ed_lbl_r, CROP_SIZE)
    es_img_c, es_lbl_c = crop_around_lv(es_img_r, es_lbl_r, CROP_SIZE)

    # Normalize intensities
    ed_img_c = normalize_image(ed_img_c)
    es_img_c = normalize_image(es_img_c)

    # Build output affine with target spacing
    new_affine = np.diag([TARGET_SPACING[0], TARGET_SPACING[1],
                          TARGET_SPACING[2], 1.0])

    # Save
    case_out = os.path.join(out_dir, case_key)
    os.makedirs(case_out, exist_ok=True)
    save_nifti(ed_img_c, new_affine, os.path.join(case_out, f'{case_key}-ED.nii.gz'))
    save_nifti(ed_lbl_c, new_affine, os.path.join(case_out, f'{case_key}-ED-label.nii.gz'))
    save_nifti(es_img_c, new_affine, os.path.join(case_out, f'{case_key}-ES.nii.gz'))
    save_nifti(es_lbl_c, new_affine, os.path.join(case_out, f'{case_key}-ES-label.nii.gz'))

    print(f'  Saved to {case_out}, shape: {ed_img_c.shape}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess CMRxM22 dataset for DIR-MRVIT')
    parser.add_argument('--raw_dir',  required=True,
                        help='Path to extracted CMRxMotion Training Dataset folder '
                             '(the one containing data/ and IQA.csv)')
    parser.add_argument('--out_dir',  required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--skip_iqa', action='store_true',
                        help='Skip IQA filtering and process all cases')
    args = parser.parse_args()

    iqa_csv = os.path.join(args.raw_dir, 'IQA.csv')
    data_dir = os.path.join(args.raw_dir, 'data')

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'data/ folder not found under {args.raw_dir}')

    # Collect all case keys
    all_cases = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('P')
    ])
    print(f'Found {len(all_cases)} cases in {data_dir}')

    # IQA filtering
    if args.skip_iqa or not os.path.exists(iqa_csv):
        if not args.skip_iqa:
            print(f'WARNING: IQA.csv not found at {iqa_csv}, processing all cases.')
        valid_cases = set(all_cases)
    else:
        valid_cases = read_iqa_csv(iqa_csv)
        print(f'After IQA filtering: {len(valid_cases)} diagnostic-quality cases')

    # Process
    os.makedirs(args.out_dir, exist_ok=True)
    success_keys = []
    for key in all_cases:
        if key not in valid_cases:
            print(f'[IQA FILTERED] {key}')
            continue
        print(f'Processing {key} ...')
        if preprocess_case(args.raw_dir, key, args.out_dir):
            success_keys.append(key)

    # Write pairs.txt: fixed=ED, moving=ES (same as ACDC/M&Ms protocol)
    pairs_path = os.path.join(args.out_dir, 'pairs.txt')
    with open(pairs_path, 'w') as f:
        for key in success_keys:
            ed = os.path.join(key, f'{key}-ED.nii.gz')
            ed_lbl = os.path.join(key, f'{key}-ED-label.nii.gz')
            es = os.path.join(key, f'{key}-ES.nii.gz')
            es_lbl = os.path.join(key, f'{key}-ES-label.nii.gz')
            f.write(f'{ed},{ed_lbl},{es},{es_lbl}\n')

    print(f'\nDone. {len(success_keys)} cases preprocessed.')
    print(f'Pairs file written to: {pairs_path}')
    print('\nNext step: run evaluate_cmrxm22.py with your M&Ms20 trained model.')


if __name__ == '__main__':
    main()