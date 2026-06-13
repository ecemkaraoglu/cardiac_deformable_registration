"""
eval_acdc_test.py
=================
Independent ACDC test evaluation script.
No training code — only loads saved checkpoints,
evaluates each fold's test set, prints and saves the results.

Usage:
    cd DIR-MRVIT
    python Code/eval_acdc_test.py

Requirements:
    - Models_cv_full/fold{0..4}_lvl3_best.pth   (provided)
    - Data/training/patient*/                    (ACDC dataset)
"""

import os, glob, sys, random, numpy as np, torch
import SimpleITK as sitk

sys.path.insert(0, '.')
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2,
                   SpatialTransform_unit)
from Functions import generate_grid_unit, transform_unit_flow_to_flow_cuda

# ==================== CONFIG ====================
DATAPATH      = 'Data/training'
MODEL_DIR     = 'Models_cv_full'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (5.0, 1.25, 1.25)   # (z, y, x) — ACDC resampling

# ACDC label convention:
#   1 = RV,  2 = MYO,  3 = LV
LV_LABEL  = 3
MYO_LABEL = 2
# ================================================


# ---------------- preprocessing (same as training) ----------------
def resample_image(image, new_spacing, is_label=False):
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()
    new_size = [
        int(round(orig_size[0] * orig_spacing[0] / new_spacing[0])),
        int(round(orig_size[1] * orig_spacing[1] / new_spacing[1])),
        int(round(orig_size[2] * orig_spacing[2] / new_spacing[2])),
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    )
    return resample.Execute(image)


def crop_around_lv(arr, gt_arr, crop_size=(96, 96, 16)):
    lv_mask = (gt_arr == LV_LABEL)
    if lv_mask.sum() == 0:
        lv_mask = (gt_arr > 0)
    coords = np.where(lv_mask)
    center = ([int(np.mean(c)) for c in coords]
              if len(coords[0]) > 0
              else [s // 2 for s in arr.shape])
    cz, ch, cw = center
    sz, sh, sw  = crop_size[2], crop_size[0], crop_size[1]

    def get_slice(c, size, max_size):
        start = max(0, c - size // 2)
        end   = start + size
        if end > max_size:
            end   = max_size
            start = max(0, end - size)
        return start, end

    z0, z1 = get_slice(cz, sz, arr.shape[0])
    h0, h1 = get_slice(ch, sh, arr.shape[1])
    w0, w1 = get_slice(cw, sw, arr.shape[2])
    cropped = arr[z0:z1, h0:h1, w0:w1]
    pad_z = sz - cropped.shape[0]
    pad_h = sh - cropped.shape[1]
    pad_w = sw - cropped.shape[2]
    if pad_z > 0 or pad_h > 0 or pad_w > 0:
        cropped = np.pad(cropped, (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        ))
    return cropped


def load_patient(patient_dir):
    files     = sorted(glob.glob(os.path.join(patient_dir, '*_frame*.nii.gz')))
    img_files = [f for f in files if '_gt' not in f]
    gt_files  = [f for f in files if '_gt' in f]
    if len(img_files) < 2 or len(gt_files) < 2:
        return None
    target_spacing = (1.25, 1.25, 5.0)

    def process(img_path, gt_path):
        img_sitk = sitk.ReadImage(img_path)
        gt_sitk  = sitk.ReadImage(gt_path)
        img_sitk = resample_image(img_sitk, target_spacing, is_label=False)
        gt_sitk  = resample_image(gt_sitk,  target_spacing, is_label=True)
        img_arr  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        gt_arr   = sitk.GetArrayFromImage(gt_sitk).astype(np.int32)
        img_arr  = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
        img_crop = crop_around_lv(img_arr, gt_arr)
        gt_crop  = crop_around_lv(gt_arr,  gt_arr)
        return img_crop, gt_crop

    ed_img, ed_gt = process(img_files[0], gt_files[0])
    es_img, es_gt = process(img_files[1], gt_files[1])
    return (ed_img, es_img, ed_gt, es_gt)


def load_all_patients():
    patients = sorted(glob.glob(os.path.join(DATAPATH, 'patient*')))
    data, names = [], []
    for p in patients:
        rec = load_patient(p)
        if rec is not None:
            data.append(rec)
            names.append(os.path.basename(p))
    print(f'Loaded {len(data)} ACDC patients')
    return data, names


# ---------------- fold split (must match cross_validation_acdc.py) ----------------
def get_stratified_fold_indices(n_patients, fold):
    """
    Stratified 5-fold split — identical to cross_validation_acdc.py.
    ACDC: 5 groups of 20 patients (NOR, MINF, DCM, HCM, RV).
    Each fold test: 4 per group = 20 patients.
    Each fold val:  2 per group = 10 patients.
    """
    rng        = random.Random(SEED)
    group_size = n_patients // N_FOLDS   # 20
    n_groups   = N_FOLDS                 # 5

    groups = []
    for g in range(n_groups):
        indices = list(range(g * group_size, (g + 1) * group_size))
        rng.shuffle(indices)
        groups.append(indices)

    test_per_group = group_size // N_FOLDS   # 4
    val_per_group  = 2

    test_idx = []
    val_idx  = []
    for g in range(n_groups):
        start = fold * test_per_group
        test_idx.extend(groups[g][start:start + test_per_group])
        remaining = [x for x in groups[g] if x not in test_idx]
        val_start = (fold * val_per_group) % len(remaining)
        val_idx.extend([remaining[(val_start + i) % len(remaining)]
                        for i in range(val_per_group)])

    used       = set(test_idx) | set(val_idx)
    train_idx  = [i for g in groups for i in g if i not in used]
    return train_idx, val_idx, test_idx


# ---------------- metrics ----------------
def to_tensor(arr):
    return (torch.from_numpy(arr).float()
            .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))


def dice_bin(pred_bin, target_bin):
    p = pred_bin.astype(np.float32)
    t = target_bin.astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def surface_distances(pred_bin, target_bin, spacing_mm=SPACING_MM):
    """ASSD and HD95 (95th percentile Hausdorff distance)."""
    pred_sitk   = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    target_sitk = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_mm[::-1])
    target_sitk.SetSpacing(spacing_mm[::-1])

    df = sitk.SignedMaurerDistanceMapImageFilter()
    df.SetSquaredDistance(False)
    df.SetUseImageSpacing(True)

    dist_pred   = sitk.GetArrayFromImage(df.Execute(pred_sitk))
    dist_target = sitk.GetArrayFromImage(df.Execute(target_sitk))

    pred_surf   = sitk.GetArrayFromImage(sitk.LabelContour(pred_sitk)).astype(bool)
    tgt_surf    = sitk.GetArrayFromImage(sitk.LabelContour(target_sitk)).astype(bool)

    d_p2t = np.abs(dist_target[pred_surf])
    d_t2p = np.abs(dist_pred[tgt_surf])

    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return 0.0, 0.0

    assd = (d_p2t.sum() + d_t2p.sum()) / (len(d_p2t) + len(d_t2p))
    hd95 = float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))
    return float(assd), hd95


def jacobian_neg_percentage(disp_field):
    flow = transform_unit_flow_to_flow_cuda(disp_field.permute(0, 2, 3, 4, 1).clone())
    flow = flow.squeeze(0).cpu().numpy()
    dvx  = np.gradient(flow[..., 0], axis=(0, 1, 2))
    dvy  = np.gradient(flow[..., 1], axis=(0, 1, 2))
    dvz  = np.gradient(flow[..., 2], axis=(0, 1, 2))
    J = np.zeros(flow.shape[:3] + (3, 3))
    J[..., 0, 0] = dvx[0] + 1;  J[..., 0, 1] = dvx[1];      J[..., 0, 2] = dvx[2]
    J[..., 1, 0] = dvy[0];      J[..., 1, 1] = dvy[1] + 1;  J[..., 1, 2] = dvy[2]
    J[..., 2, 0] = dvz[0];      J[..., 2, 1] = dvz[1];      J[..., 2, 2] = dvz[2] + 1
    return 100.0 * float(np.mean(np.linalg.det(J) <= 0))


def warp_segmentation(seg, disp_field, grid):
    transform  = SpatialTransform_unit().to(DEVICE)
    seg_tensor = (torch.from_numpy(seg).float()
                  .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
    with torch.no_grad():
        warped = transform(seg_tensor,
                           disp_field.permute(0, 2, 3, 4, 1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)   # (z,y,x)


# ---------------- model builder ----------------
def build_model(weights_path):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True,
        imgshape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    patch_lv2  = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True,
        patch_shape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW,
        model_lvl1=model_lvl1, patch_model_lv2=patch_lv2).to(DEVICE)
    patch_lv3  = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(DEVICE)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW,
        model_lvl2=model_lvl2, patch_model=patch_lv3).to(DEVICE)
    model_lvl3.load_state_dict(
        torch.load(weights_path, map_location=DEVICE))
    model_lvl3.eval()
    return model_lvl3


# ---------------- per-fold evaluation ----------------
def evaluate_fold(fold, test_data, test_names):
    weights = os.path.join(MODEL_DIR, f'fold{fold}_lvl3_best.pth')
    print(f'\n  Loading: {weights}')
    model = build_model(weights)

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(
        np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    keys = ['lv_dice', 'myo_dice', 'endo_assd', 'epi_assd',
            'endo_hd', 'epi_hd', 'jac']
    res  = {k: [] for k in keys}

    for i, ((ed_img, es_img, ed_gt, es_gt), name) in enumerate(
            zip(test_data, test_names)):
        fix = to_tensor(ed_img)
        mov = to_tensor(es_img)

        with torch.no_grad():
            disp = model(mov, fix)[0]

        warped = np.round(warp_segmentation(es_gt, disp, grid)).astype(np.int32)

        # Dice
        lv_d  = dice_bin(warped == LV_LABEL,  ed_gt == LV_LABEL)
        myo_d = dice_bin(warped == MYO_LABEL, ed_gt == MYO_LABEL)

        # Surface metrics
        endo_pred = (warped == LV_LABEL).astype(np.uint8)
        endo_gt   = (ed_gt  == LV_LABEL).astype(np.uint8)
        epi_pred  = ((warped == LV_LABEL) | (warped == MYO_LABEL)).astype(np.uint8)
        epi_gt    = ((ed_gt  == LV_LABEL) | (ed_gt  == MYO_LABEL)).astype(np.uint8)

        ea, eh = surface_distances(endo_pred, endo_gt)
        pa, ph = surface_distances(epi_pred,  epi_gt)
        jac    = jacobian_neg_percentage(disp)

        res['lv_dice'].append(lv_d);   res['myo_dice'].append(myo_d)
        res['endo_assd'].append(ea);   res['endo_hd'].append(eh)
        res['epi_assd'].append(pa);    res['epi_hd'].append(ph)
        res['jac'].append(jac)

        print(f'    [{i+1:2d}/{len(test_data)}] {name}  '
              f'LV={lv_d:.3f}  MYO={myo_d:.3f}  '
              f'EndoASSD={ea:.2f}mm  EndoHD95={eh:.2f}mm  '
              f'|J|={jac:.3f}%')

    print(f'\n  Fold {fold} summary:')
    for k in keys:
        v = np.array(res[k])
        unit = ' mm' if 'assd' in k or 'hd' in k else (' %' if 'jac' in k else '')
        print(f'    {k:12s}: {v.mean():.4f} +/- {v.std():.4f}{unit}')

    return res


# ---------------- main ----------------
if __name__ == '__main__':
    print('=' * 62)
    print('  DIR-MRVIT — ACDC17 Independent Test Evaluation')
    print('=' * 62)
    print(f'Device    : {DEVICE}')
    print(f'Model dir : {MODEL_DIR}')
    print(f'HD metric : HD95 (95th percentile)')
    print(f'Spacing   : {SPACING_MM} mm (z,y,x)')

    all_data, all_names = load_all_patients()
    n = len(all_data)

    keys   = ['lv_dice', 'myo_dice', 'endo_assd', 'epi_assd',
              'endo_hd', 'epi_hd', 'jac']
    pooled = {k: [] for k in keys}

    for fold in range(N_FOLDS):
        weights = os.path.join(MODEL_DIR, f'fold{fold}_lvl3_best.pth')
        if not os.path.exists(weights):
            print(f'\nFold {fold}: checkpoint not found, skipping.')
            continue

        print(f'\n{"=" * 62}\n  FOLD {fold}\n{"=" * 62}')
        _, _, te_idx = get_stratified_fold_indices(n, fold)
        test_data  = [all_data[i]  for i in sorted(te_idx)]
        test_names = [all_names[i] for i in sorted(te_idx)]
        print(f'  Test patients ({len(test_data)}): '
              f'{[all_names[i] for i in sorted(te_idx)]}')

        res = evaluate_fold(fold, test_data, test_names)
        for k in keys:
            pooled[k].extend(res[k])

    print(f'\n{"=" * 62}')
    print(f'  5-FOLD CV RESULTS — ACDC17 (100 pairs)')
    print(f'{"=" * 62}')
    for k in keys:
        v    = np.array(pooled[k])
        unit = ' mm' if 'assd' in k or 'hd' in k else (' %' if 'jac' in k else '')
        print(f'  {k:12s}: {v.mean():.3f} +/- {v.std():.3f}{unit}')

    print(f'\n  --- Paper Table 2 (Proposed, ACDC17) ---')
    print('  lv_dice    : 0.917 +/- 0.043')
    print('  myo_dice   : 0.789 +/- 0.055')
    print('  endo_assd  : 0.79  +/- 0.39 mm')
    print('  epi_assd   : 0.88  +/- 0.21 mm')
    print('  endo_hd    : 5.62  +/- 1.22 mm')
    print('  epi_hd     : 5.51  +/- 1.77 mm')
    print('  jac        : 0.409 +/- 0.153 %')

    save_path = os.path.join(MODEL_DIR, 'eval_acdc_independent.npy')
    np.save(save_path, pooled)
    print(f'\n  Results saved to: {save_path}')