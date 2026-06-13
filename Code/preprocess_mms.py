"""
Offline preprocessing for the M&Ms20 dataset (DIR-MRVIT project).

Key design decisions (v2 - fixes from performance analysis):
  1. ANCHORED CROPPING: crop center is computed from ED frame only,
     then the SAME (z0,z1,h0,h1,w0,w1) coordinates are applied to
     both ED and ES. This preserves the relative spatial displacement
     between diastole and systole that the model must learn.
  2. CROP-THEN-NORMALIZE: intensity normalization is done AFTER
     cropping so the [0,1] range covers only the cardiac region,
     not bright artefacts elsewhere in the full FOV.
  3. LV label = 1 in M&Ms (NOT 3 like ACDC).

For each patient:
  1. Read the 4D short-axis volume and its 4D segmentation.
  2. Extract the ED and ES frames using the indices from the CSV.
  3. Resample each 3D frame to 1.25 x 1.25 x 8 mm (paper Section 3.1).
  4. Compute crop box from ED LV centre; apply same box to ES.
  5. Normalize image intensity to [0, 1] AFTER cropping.
  6. Save as <code>_ED.nii.gz / <code>_ED_gt.nii.gz (and ES).

M&Ms label convention:
  Label 1 = LV cavity  (ACDC uses label 3 for LV)
  Label 2 = RV cavity
  Label 3 = LV myocardium
"""

import os, glob, csv, numpy as np
import SimpleITK as sitk
sitk.ProcessObject.GlobalWarningDisplayOff()

# ==================== CONFIG ====================
SRC_DIR    = 'Data/MMs_training'
CSV_PATH   = 'Data/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'
DST_DIR    = 'Data/MMs_preprocessed'

TARGET_SPACING = (1.25, 1.25, 8.0)   # (x, y, z) mm -- paper Section 3.1
CROP_SIZE      = (96, 96, 16)        # (H, W, D)
LV_LABEL       = 1                   # M&Ms LV cavity label
# ================================================

os.makedirs(DST_DIR, exist_ok=True)

def resample_image(image, new_spacing, is_label=False):
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()
    new_size = [
        int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / new_spacing[2]))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )
    return resample.Execute(image)

def compute_crop_box(gt_arr, crop_size=CROP_SIZE):
    """
    Compute crop box centred on LV cavity (label 1) in gt_arr.
    Returns (z0,z1,h0,h1,w0,w1) -- the same box will be applied
    to both ED and ES to preserve inter-frame spatial alignment.
    """
    lv_mask = (gt_arr == LV_LABEL)
    if lv_mask.sum() == 0:
        lv_mask = (gt_arr > 0)        # fallback: any foreground
    coords = np.where(lv_mask)
    if len(coords[0]) > 0:
        center = [int(np.mean(c)) for c in coords]
    else:
        center = [s // 2 for s in gt_arr.shape]

    cz, ch, cw = center
    sz, sh, sw = crop_size[2], crop_size[0], crop_size[1]

    def get_slice(c, size, max_size):
        start = max(0, c - size // 2)
        end   = start + size
        if end > max_size:
            end   = max_size
            start = max(0, end - size)
        return start, end

    z0, z1 = get_slice(cz, sz, gt_arr.shape[0])
    h0, h1 = get_slice(ch, sh, gt_arr.shape[1])
    w0, w1 = get_slice(cw, sw, gt_arr.shape[2])
    return z0, z1, h0, h1, w0, w1

def apply_crop_and_pad(arr, box, crop_size=CROP_SIZE):
    """Apply a pre-computed crop box; pad if volume is smaller than crop."""
    z0, z1, h0, h1, w0, w1 = box
    sz, sh, sw = crop_size[2], crop_size[0], crop_size[1]
    cropped = arr[z0:z1, h0:h1, w0:w1]
    pad_z = sz - cropped.shape[0]
    pad_h = sh - cropped.shape[1]
    pad_w = sw - cropped.shape[2]
    if pad_z > 0 or pad_h > 0 or pad_w > 0:
        cropped = np.pad(cropped, (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        ))
    return cropped

def normalize_crop(img_arr):
    """Normalize to [0,1] after cropping (crop-then-normalize)."""
    img_arr = np.clip(img_arr, 0, None)
    vmin, vmax = img_arr.min(), img_arr.max()
    return (img_arr - vmin) / (vmax - vmin + 1e-8)

def resample_frame(frame_img, frame_gt, spacing_3d):
    """Resample a single 3D frame to TARGET_SPACING. Returns numpy arrays."""
    img_sitk = sitk.GetImageFromArray(frame_img.astype(np.float32))
    gt_sitk  = sitk.GetImageFromArray(frame_gt.astype(np.float32))
    img_sitk.SetSpacing(spacing_3d)
    gt_sitk.SetSpacing(spacing_3d)
    img_sitk = resample_image(img_sitk, TARGET_SPACING, is_label=False)
    gt_sitk  = resample_image(gt_sitk,  TARGET_SPACING, is_label=True)
    img_arr  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    gt_arr   = np.round(sitk.GetArrayFromImage(gt_sitk)).astype(np.int32)
    return img_arr, gt_arr

def read_ed_es_table():
    table = {}
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            table[row['External code']] = (
                int(row['ED']), int(row['ES']), row['Pathology']
            )
    return table

def save_nifti(arr, path, spacing=TARGET_SPACING):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, path)

if __name__ == '__main__':
    table    = read_ed_es_table()
    patients = sorted(glob.glob(os.path.join(SRC_DIR, '*')))
    print(f'Found {len(patients)} patient folders.')
    print('Anchored cropping from ED LV centre (label 1); crop-then-normalize.')

    done, skipped = 0, 0
    for p in patients:
        code = os.path.basename(p)
        if code not in table:
            skipped += 1; continue
        ed_idx, es_idx, pathology = table[code]

        img_path = os.path.join(p, f'{code}_sa.nii.gz')
        gt_path  = os.path.join(p, f'{code}_sa_gt.nii.gz')
        if not (os.path.exists(img_path) and os.path.exists(gt_path)):
            print(f'  SKIP {code}: missing files')
            skipped += 1; continue

        img_sitk_4d = sitk.ReadImage(img_path)
        gt_sitk_4d  = sitk.ReadImage(gt_path)
        sp          = img_sitk_4d.GetSpacing()
        spacing_3d  = (float(sp[0]), float(sp[1]), float(sp[2]))

        img_4d = sitk.GetArrayFromImage(img_sitk_4d).astype(np.float32)  # (T,Z,Y,X)
        gt_4d  = sitk.GetArrayFromImage(gt_sitk_4d).astype(np.int32)

        n_frames = img_4d.shape[0]
        if ed_idx >= n_frames or es_idx >= n_frames or ed_idx == es_idx:
            print(f'  SKIP {code}: bad frame indices ED={ed_idx} ES={es_idx} T={n_frames}')
            skipped += 1; continue

        # --- resample both frames ---
        ed_img_rs, ed_gt_rs = resample_frame(img_4d[ed_idx], gt_4d[ed_idx], spacing_3d)
        es_img_rs, es_gt_rs = resample_frame(img_4d[es_idx], gt_4d[es_idx], spacing_3d)

        # --- compute crop box from ED only (anchored cropping) ---
        box = compute_crop_box(ed_gt_rs)

        # --- apply the SAME box to both ED and ES ---
        ed_img_crop = apply_crop_and_pad(ed_img_rs, box)
        ed_gt_crop  = apply_crop_and_pad(ed_gt_rs,  box)
        es_img_crop = apply_crop_and_pad(es_img_rs, box)
        es_gt_crop  = apply_crop_and_pad(es_gt_rs,  box)

        # --- normalize AFTER cropping ---
        ed_img_norm = normalize_crop(ed_img_crop)
        es_img_norm = normalize_crop(es_img_crop)

        # sanity check: LV must be present in both frames after cropping
        if (ed_gt_crop == LV_LABEL).sum() == 0 or (es_gt_crop == LV_LABEL).sum() == 0:
            print(f'  SKIP {code}: empty LV after cropping')
            skipped += 1; continue

        out_dir = os.path.join(DST_DIR, code)
        os.makedirs(out_dir, exist_ok=True)
        save_nifti(ed_img_norm, os.path.join(out_dir, f'{code}_ED.nii.gz'))
        save_nifti(ed_gt_crop,  os.path.join(out_dir, f'{code}_ED_gt.nii.gz'))
        save_nifti(es_img_norm, os.path.join(out_dir, f'{code}_ES.nii.gz'))
        save_nifti(es_gt_crop,  os.path.join(out_dir, f'{code}_ES_gt.nii.gz'))

        done += 1
        if done % 25 == 0:
            print(f'  processed {done} patients...')

    print(f'\nDone. {done} patients written to {DST_DIR}, {skipped} skipped.')

    index_path = os.path.join(DST_DIR, 'index.csv')
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'pathology'])
        for p in sorted(glob.glob(os.path.join(DST_DIR, '*'))):
            if os.path.isdir(p):
                code = os.path.basename(p)
                if code in table:
                    writer.writerow([code, table[code][2]])
    print(f'Index written to {index_path}')