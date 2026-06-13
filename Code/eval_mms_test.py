"""
eval_mms_test.py
================
Bağımsız M&Ms20 test evaluation scripti.
Training kodu yok — sadece kaydedilmiş checkpoint'leri yükler,
her fold'un test setini evaluate eder, sonuçları yazdırır ve kaydeder.

Kullanım:
    cd DIR-MRVIT
    python Code/eval_mms_test.py

Gereksinimler:
    - Models_mms_full/fold{0..4}_lvl3_best.pth   (mevcut)
    - Data/MMs_preprocessed/                      (offline preprocessed)

M&Ms label convention:
    1 = LV cavity   (Endo)
    2 = RV cavity
    3 = LV myocardium (MYO)
    Epicardium (Epi) = label 1 | label 3
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
DATAPATH      = 'Data/MMs_preprocessed'
MODEL_DIR     = 'Models_mms_full'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (8.0, 1.25, 1.25)   # (z, y, x) — M&Ms resampling

LV_LABEL  = 1
RV_LABEL  = 2
MYO_LABEL = 3
# ================================================


# ---------------- data loading ----------------
def load_all_patients():
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(DATAPATH, '*'))
        if os.path.isdir(d)
    ])
    data, codes = [], []
    skipped = 0
    for p in patient_dirs:
        code = os.path.basename(p)
        paths = [os.path.join(p, f'{code}_{x}.nii.gz')
                 for x in ['ED', 'ED_gt', 'ES', 'ES_gt']]
        if not all(os.path.exists(x) for x in paths):
            skipped += 1
            continue
        ed_img = sitk.GetArrayFromImage(
            sitk.ReadImage(paths[0])).astype(np.float32)
        ed_gt  = sitk.GetArrayFromImage(
            sitk.ReadImage(paths[1])).astype(np.int32)
        es_img = sitk.GetArrayFromImage(
            sitk.ReadImage(paths[2])).astype(np.float32)
        es_gt  = sitk.GetArrayFromImage(
            sitk.ReadImage(paths[3])).astype(np.int32)
        data.append((ed_img, es_img, ed_gt, es_gt))
        codes.append(code)
    print(f'Loaded {len(data)} M&Ms patients ({skipped} skipped)')
    return data, codes


# ---------------- fold split (must match cross_validation_mms.py) ----------------
def get_fold_indices(n_patients, fold):
    """5-fold split: 105 train / 15 val / 30 test — identical to cross_validation_mms.py."""
    rng   = random.Random(SEED)
    order = list(range(n_patients))
    rng.shuffle(order)
    test_size  = n_patients // N_FOLDS   # 30
    val_size   = n_patients // 10        # 15
    test_start = fold * test_size
    test_idx   = order[test_start:test_start + test_size]
    remaining  = [i for i in order if i not in set(test_idx)]
    val_start  = (fold * val_size) % len(remaining)
    val_idx    = [remaining[(val_start + i) % len(remaining)]
                  for i in range(val_size)]
    used       = set(test_idx) | set(val_idx)
    train_idx  = [i for i in order if i not in used]
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

    pred_surf = sitk.GetArrayFromImage(sitk.LabelContour(pred_sitk)).astype(bool)
    tgt_surf  = sitk.GetArrayFromImage(sitk.LabelContour(target_sitk)).astype(bool)

    d_p2t = np.abs(dist_target[pred_surf])
    d_t2p = np.abs(dist_pred[tgt_surf])

    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return 0.0, 0.0

    assd = (d_p2t.sum() + d_t2p.sum()) / (len(d_p2t) + len(d_t2p))
    hd95 = float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))
    return float(assd), hd95


def jacobian_neg_percentage(disp_field):
    flow = transform_unit_flow_to_flow_cuda(
        disp_field.permute(0, 2, 3, 4, 1).clone())
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
    """Per-label binary warping — no hallucinated labels at boundaries."""
    transform     = SpatialTransform_unit().to(DEVICE)
    disp_permuted = disp_field.permute(0, 2, 3, 4, 1)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    result = np.zeros(seg.shape, dtype=np.int32)
    probs  = np.zeros(seg.shape, dtype=np.float32)
    for label in unique_labels:
        binary = (seg == label).astype(np.float32)
        bt = (torch.from_numpy(binary).float()
              .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            warped = transform(bt, disp_permuted, grid)
        warped_np = warped.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = warped_np > probs
        result[mask] = label
        probs[mask]  = warped_np[mask]
    return result


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
def evaluate_fold(fold, test_data, test_codes):
    weights = os.path.join(MODEL_DIR, f'fold{fold}_lvl3_best.pth')
    print(f'\n  Loading: {weights}')
    model = build_model(weights)

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(
        np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    keys = ['lv_dice', 'myo_dice', 'endo_assd', 'epi_assd',
            'endo_hd', 'epi_hd', 'jac']
    res  = {k: [] for k in keys}

    for i, ((ed_img, es_img, ed_gt, es_gt), code) in enumerate(
            zip(test_data, test_codes)):
        fix = to_tensor(ed_img)
        mov = to_tensor(es_img)

        with torch.no_grad():
            disp = model(mov, fix)[0]

        warped = warp_segmentation(es_gt, disp, grid)

        lv_d  = dice_bin(warped == LV_LABEL,  ed_gt == LV_LABEL)
        myo_d = dice_bin(warped == MYO_LABEL, ed_gt == MYO_LABEL)

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

        print(f'    [{i+1:2d}/{len(test_data)}] {code}  '
              f'LV={lv_d:.3f}  MYO={myo_d:.3f}  '
              f'EndoASSD={ea:.2f}mm  EndoHD95={eh:.2f}mm  '
              f'|J|={jac:.3f}%')

    print(f'\n  Fold {fold} summary:')
    for k in keys:
        v    = np.array(res[k])
        unit = ' mm' if 'assd' in k or 'hd' in k else (' %' if 'jac' in k else '')
        print(f'    {k:12s}: {v.mean():.4f} +/- {v.std():.4f}{unit}')

    return res


# ---------------- main ----------------
if __name__ == '__main__':
    print('=' * 62)
    print('  DIR-MRVIT — M&Ms20 Independent Test Evaluation')
    print('=' * 62)
    print(f'Device    : {DEVICE}')
    print(f'Model dir : {MODEL_DIR}')
    print(f'HD metric : HD95 (95th percentile)')
    print(f'Spacing   : {SPACING_MM} mm (z,y,x)')
    print(f'Labels    : LV={LV_LABEL}, MYO={MYO_LABEL}, RV={RV_LABEL}')

    all_data, all_codes = load_all_patients()
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
        _, _, te_idx = get_fold_indices(n, fold)
        test_data  = [all_data[i]  for i in te_idx]
        test_codes = [all_codes[i] for i in te_idx]
        print(f'  Test patients ({len(test_data)}): {test_codes[:5]}...')

        res = evaluate_fold(fold, test_data, test_codes)
        for k in keys:
            pooled[k].extend(res[k])

    print(f'\n{"=" * 62}')
    print(f'  5-FOLD CV RESULTS — M&Ms20 (150 pairs)')
    print(f'{"=" * 62}')
    for k in keys:
        v    = np.array(pooled[k])
        unit = ' mm' if 'assd' in k or 'hd' in k else (' %' if 'jac' in k else '')
        print(f'  {k:12s}: {v.mean():.3f} +/- {v.std():.3f}{unit}')

    print(f'\n  --- Paper Table 3 (Proposed, M&Ms20) ---')
    print('  lv_dice    : 0.884 +/- 0.038')
    print('  myo_dice   : 0.729 +/- 0.057')
    print('  endo_assd  : 1.88  +/- 0.79 mm')
    print('  epi_assd   : 2.05  +/- 0.83 mm')
    print('  endo_hd    : 7.40  +/- 2.78 mm')
    print('  epi_hd     : 8.16  +/- 1.95 mm')
    print('  jac        : 0.536 +/- 0.321 %')

    save_path = os.path.join(MODEL_DIR, 'eval_mms_independent.npy')
    np.save(save_path, pooled)
    print(f'\n  Results saved to: {save_path}')
