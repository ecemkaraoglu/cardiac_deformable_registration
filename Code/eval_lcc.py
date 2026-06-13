"""
Re-evaluate saved M&Ms fold checkpoints with morphological opening
post-processing on warped segmentation masks.

Opening (erosion -> dilation) removes thin protrusions and isolated noise
voxels that connect to the main structure but visually look like artefacts.
Applied per-label before metric computation.

Usage:
    python Code/eval_lcc.py           # uses Models_mms_test (fold 0 only)
    python Code/eval_lcc.py --full    # uses Models_mms_full (all 5 folds)
"""

import os, sys, glob, argparse, numpy as np, torch, random
import SimpleITK as sitk
from scipy import ndimage
sys.path.insert(0, '.')
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2,
                   SpatialTransform_unit)
from Functions import generate_grid_unit

# ==================== CONFIG ====================
DATAPATH      = 'Data/MMs_preprocessed'
MODEL_DIR_TEST = 'Models_mms_test'
MODEL_DIR_FULL = 'Models_mms_full'

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)
SPACING_MM = (8.0, 1.25, 1.25)   # (z,y,x)

LV_LABEL  = 1
RV_LABEL  = 2
MYO_LABEL = 3

OPENING_ITERATIONS = 1   # 1 = light, 2 = aggressive
# ================================================

def largest_connected_component(binary_mask):
    """Keep only the largest connected component in a binary mask."""
    if binary_mask.sum() == 0:
        return binary_mask
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 1:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(binary_mask.dtype)

def apply_lcc(seg, labels):
    """Apply LCC to each label independently in a multi-label segmentation."""
    result = np.zeros_like(seg)
    for label in labels:
        binary = (seg == label).astype(np.uint8)
        clean  = largest_connected_component(binary)
        result[clean == 1] = label
    return result

def apply_opening(seg, labels, iterations=1):
    """Apply morphological opening to each label independently.
    Erosion then dilation: removes thin protrusions and isolated noise
    while preserving the main structure shape.
    """
    result = np.zeros_like(seg)
    for label in labels:
        binary = (seg == label).astype(np.uint8)
        cleaned = ndimage.binary_opening(binary, iterations=iterations)
        result[cleaned == 1] = label
    return result

def load_all_patients():
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(DATAPATH, '*'))
        if os.path.isdir(d)
    ])
    data, codes = [], []
    for p in patient_dirs:
        code = os.path.basename(p)
        paths = [os.path.join(p, f'{code}_{x}.nii.gz')
                 for x in ['ED', 'ED_gt', 'ES', 'ES_gt']]
        if not all(os.path.exists(x) for x in paths):
            continue
        ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[0])).astype(np.float32)
        ed_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[1])).astype(np.int32)
        es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[2])).astype(np.float32)
        es_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[3])).astype(np.int32)
        data.append((ed_img, es_img, ed_gt, es_gt))
        codes.append(code)
    print(f'Loaded {len(data)} patients')
    return data, codes

def get_fold_indices(n, fold):
    rng   = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)
    test_size  = n // N_FOLDS
    val_size   = n // 10
    test_start = fold * test_size
    test_idx   = order[test_start:test_start + test_size]
    remaining  = [i for i in order if i not in set(test_idx)]
    val_start  = (fold * val_size) % len(remaining)
    val_idx    = [remaining[(val_start + i) % len(remaining)]
                  for i in range(val_size)]
    used       = set(test_idx) | set(val_idx)
    train_idx  = [i for i in order if i not in used]
    return train_idx, val_idx, test_idx

def to_tensor(arr):
    return (torch.from_numpy(arr).float()
            .permute(1,2,0).unsqueeze(0).unsqueeze(0).to(DEVICE))

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
              .permute(1,2,0).unsqueeze(0).unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            warped = transform(bt, disp_permuted, grid)
        warped_np = warped.squeeze().cpu().numpy().transpose(2,0,1)
        mask = warped_np > probs
        result[mask] = label
        probs[mask]  = warped_np[mask]
    return result

def dice_bin(p, t):
    p = p.astype(np.float32); t = t.astype(np.float32)
    inter = (p*t).sum(); denom = p.sum()+t.sum()
    return 1.0 if denom == 0 else 2.0*inter/denom

def surface_distances(pred_bin, target_bin, spacing_mm=SPACING_MM):
    pred_s   = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    target_s = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    pred_s.SetSpacing(spacing_mm[::-1])
    target_s.SetSpacing(spacing_mm[::-1])
    df = sitk.SignedMaurerDistanceMapImageFilter()
    df.SetSquaredDistance(False); df.SetUseImageSpacing(True)
    dist_pred   = sitk.GetArrayFromImage(df.Execute(pred_s))
    dist_target = sitk.GetArrayFromImage(df.Execute(target_s))
    pred_surf   = sitk.GetArrayFromImage(sitk.LabelContour(pred_s)).astype(bool)
    tgt_surf    = sitk.GetArrayFromImage(sitk.LabelContour(target_s)).astype(bool)
    d_p2t = np.abs(dist_target[pred_surf])
    d_t2p = np.abs(dist_pred[tgt_surf])
    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return 0.0, 0.0, 0.0
    assd   = (d_p2t.sum() + d_t2p.sum()) / (len(d_p2t) + len(d_t2p))
    hd_max = float(max(d_p2t.max(), d_t2p.max()))
    hd_95  = float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))
    return float(assd), hd_max, hd_95

def build_model(weights_path):
    m1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2,3,START_CHANNEL,is_train=True,imgshape=imgshape_4,range_flow=RANGE_FLOW).to(DEVICE)
    pl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2,3,START_CHANNEL,is_train=True,patch_shape=imgshape_4,range_flow=RANGE_FLOW).to(DEVICE)
    m2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2,3,START_CHANNEL,is_train=True,imgshape=imgshape_2,range_flow=RANGE_FLOW,
        model_lvl1=m1,patch_model_lv2=pl2).to(DEVICE)
    pl3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2,3,START_CHANNEL,is_train=True,patch_shape=imgshape_2,range_flow=RANGE_FLOW,
        patch_model=pl2).to(DEVICE)
    m3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2,3,START_CHANNEL,is_train=True,imgshape=imgshape,range_flow=RANGE_FLOW,
        model_lvl2=m2,patch_model=pl3).to(DEVICE)
    m3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m3.eval()
    return m3

def evaluate_fold(weights_path, test_data, use_filter):
    model = build_model(weights_path)
    grid  = generate_grid_unit(imgshape)
    grid  = torch.from_numpy(np.reshape(grid,(1,)+grid.shape)).to(DEVICE).float()

    keys = ['lv_dice','myo_dice','endo_assd','epi_assd',
            'endo_hd','epi_hd','endo_hd95','epi_hd95']
    res  = {k: [] for k in keys}

    for ed_img, es_img, ed_gt, es_gt in test_data:
        fix = to_tensor(ed_img); mov = to_tensor(es_img)
        with torch.no_grad():
            disp = model(mov, fix)[0]
        warped = warp_segmentation(es_gt, disp, grid)

        # Optional post-processing: morphological opening
        if use_filter:
            warped = apply_opening(warped, [LV_LABEL, MYO_LABEL],
                                   iterations=OPENING_ITERATIONS)

        res['lv_dice'].append(dice_bin(warped==LV_LABEL,  ed_gt==LV_LABEL))
        res['myo_dice'].append(dice_bin(warped==MYO_LABEL, ed_gt==MYO_LABEL))

        endo_pred = (warped==LV_LABEL).astype(np.uint8)
        endo_gt   = (ed_gt==LV_LABEL).astype(np.uint8)
        epi_pred  = ((warped==LV_LABEL)|(warped==MYO_LABEL)).astype(np.uint8)
        epi_gt    = ((ed_gt==LV_LABEL)|(ed_gt==MYO_LABEL)).astype(np.uint8)

        ea, eh, eh95 = surface_distances(endo_pred, endo_gt)
        pa, ph, ph95 = surface_distances(epi_pred,  epi_gt)
        res['endo_assd'].append(ea); res['endo_hd'].append(eh); res['endo_hd95'].append(eh95)
        res['epi_assd'].append(pa);  res['epi_hd'].append(ph);  res['epi_hd95'].append(ph95)
    return res

def print_results(pooled, label):
    def s(k): v=np.array(pooled[k]); return v.mean(), v.std()
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    lv=s('lv_dice'); myo=s('myo_dice')
    ea=s('endo_assd'); pa=s('epi_assd')
    eh=s('endo_hd');   ph=s('epi_hd')
    eh95=s('endo_hd95'); ph95=s('epi_hd95')
    print(f'  LV   Dice:     {lv[0]:.3f} +/- {lv[1]:.3f}')
    print(f'  MYO  Dice:     {myo[0]:.3f} +/- {myo[1]:.3f}')
    print(f'  Endo ASSD:     {ea[0]:.2f} +/- {ea[1]:.2f} mm')
    print(f'  Epi  ASSD:     {pa[0]:.2f} +/- {pa[1]:.2f} mm')
    print(f'  Endo HD (max): {eh[0]:.2f} +/- {eh[1]:.2f} mm')
    print(f'  Epi  HD (max): {ph[0]:.2f} +/- {ph[1]:.2f} mm')
    print(f'  Endo HD (95%): {eh95[0]:.2f} +/- {eh95[1]:.2f} mm')
    print(f'  Epi  HD (95%): {ph95[0]:.2f} +/- {ph95[1]:.2f} mm')
    print(f'\n  --- Paper Table 3 (Proposed, M&Ms20) ---')
    print(f'  LV Dice: 0.884  MYO Dice: 0.729')
    print(f'  Endo ASSD: 1.88  Epi ASSD: 2.05')
    print(f'  Endo HD: 7.40   Epi HD: 8.16')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                        help='Use Models_mms_full (5 folds) instead of Models_mms_test (fold 0)')
    args = parser.parse_args()

    model_dir = MODEL_DIR_FULL if args.full else MODEL_DIR_TEST
    folds     = list(range(N_FOLDS)) if args.full else [0]

    print(f'Device: {DEVICE}')
    print(f'Model dir: {model_dir}')
    print(f'Folds: {folds}')
    print(f'Filter: morphological opening, iterations={OPENING_ITERATIONS}')

    all_data, _ = load_all_patients()
    n = len(all_data)

    keys = ['lv_dice','myo_dice','endo_assd','epi_assd',
            'endo_hd','epi_hd','endo_hd95','epi_hd95']
    pooled_raw      = {k: [] for k in keys}
    pooled_opening  = {k: [] for k in keys}

    for fold in folds:
        weights = os.path.join(model_dir, f'fold{fold}_lvl3_best.pth')
        if not os.path.exists(weights):
            print(f'  SKIP fold {fold}: no checkpoint at {weights}')
            continue
        _, _, te_idx = get_fold_indices(n, fold)
        test_data = [all_data[i] for i in te_idx]
        print(f'\nFold {fold} — {len(test_data)} test patients...')

        res_raw     = evaluate_fold(weights, test_data, use_filter=False)
        res_opening = evaluate_fold(weights, test_data, use_filter=True)

        for k in keys:
            pooled_raw[k].extend(res_raw[k])
            pooled_opening[k].extend(res_opening[k])

        lv_raw  = np.mean(res_raw['lv_dice']);     hd_raw  = np.mean(res_raw['endo_hd'])
        lv_op   = np.mean(res_opening['lv_dice']); hd_op   = np.mean(res_opening['endo_hd'])
        myo_raw = np.mean(res_raw['myo_dice']);    myo_op  = np.mean(res_opening['myo_dice'])
        print(f'  Raw:     LV={lv_raw:.3f}  MYO={myo_raw:.3f}  Endo HD={hd_raw:.2f}mm')
        print(f'  Opening: LV={lv_op:.3f}  MYO={myo_op:.3f}  Endo HD={hd_op:.2f}mm')

    print_results(pooled_raw,     'WITHOUT post-processing (raw)')
    print_results(pooled_opening, f'WITH morphological opening (iter={OPENING_ITERATIONS})')