"""
5-fold cross validation for DIR-MRVIT on the M&Ms20 dataset.

Requires offline preprocessing to have been run first:
    python Code/preprocess_mms.py

M&Ms label convention (differs from ACDC!):
  Label 1 = LV cavity      (ACDC uses label 3 for LV)
  Label 2 = RV cavity
  Label 3 = LV myocardium

Key design decisions:
  - Per-label binary warping: avoids hallucinated labels at boundaries
  - HD reported as 95th percentile (HD95), consistent with paper's
    actual implementation (strict max inflated by z-axis 8mm resolution)
  - SMOOTH=0.2: better than paper's 0.5 for M&Ms (thicker slices,
    more complex deformations across 6 centers)

TEST_MODE = True  -> only fold 0, short iterations (~20 min)
TEST_MODE = False -> 5 folds, full schedule (overnight)
"""

import os, glob, csv, sys, numpy as np, torch, random
import SimpleITK as sitk
sys.path.insert(0, '.')
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2,
                   SpatialTransform_unit, smoothloss,
                   neg_Jdet_loss, multi_resolution_NCC, NCC)
from Functions import generate_grid, generate_grid_unit, transform_unit_flow_to_flow_cuda

# ==================== CONFIG ====================
DATAPATH   = 'Data/MMs_preprocessed'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR            = 1e-4
SMOOTH        = 0.2   # 0.5 (paper) -> 0.2 better for M&Ms thick slices
ANTIFOLD      = 0
START_CHANNEL = 4
RANGE_FLOW    = 0.4
FREEZE_STEP   = 8000

N_FOLDS = 5
SEED    = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (8.0, 1.25, 1.25)   # (z,y,x) for surface distance metrics

# M&Ms label mapping
LV_LABEL  = 1   # LV cavity
RV_LABEL  = 2   # RV cavity
MYO_LABEL = 3   # LV myocardium

# ---- TOGGLE HERE ----
TEST_MODE = False
# TEST_MODE = True  -> fold 0, short iterations (quick check)
# TEST_MODE = False -> 5 folds, full schedule (overnight)

if TEST_MODE:
    MODEL_DIR     = 'Models_mms_test'
    ITER_LVL1     = 2000
    ITER_LVL2     = 2000
    ITER_LVL3_MAX = 5000
    VAL_EVERY     = 500
    PATIENCE      = 1500
    RUN_FOLDS     = [0]
else:
    MODEL_DIR     = 'Models_mms_full'
    ITER_LVL1     = 15000
    ITER_LVL2     = 15000
    ITER_LVL3_MAX = 35000
    VAL_EVERY     = 500
    PATIENCE      = 5000
    RUN_FOLDS     = list(range(N_FOLDS))
# ================================================

os.makedirs(MODEL_DIR, exist_ok=True)

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
        ed_img_path = os.path.join(p, f'{code}_ED.nii.gz')
        ed_gt_path  = os.path.join(p, f'{code}_ED_gt.nii.gz')
        es_img_path = os.path.join(p, f'{code}_ES.nii.gz')
        es_gt_path  = os.path.join(p, f'{code}_ES_gt.nii.gz')
        if not all(os.path.exists(x) for x in
                   [ed_img_path, ed_gt_path, es_img_path, es_gt_path]):
            skipped += 1
            continue
        ed_img = sitk.GetArrayFromImage(sitk.ReadImage(ed_img_path)).astype(np.float32)
        ed_gt  = sitk.GetArrayFromImage(sitk.ReadImage(ed_gt_path)).astype(np.int32)
        es_img = sitk.GetArrayFromImage(sitk.ReadImage(es_img_path)).astype(np.float32)
        es_gt  = sitk.GetArrayFromImage(sitk.ReadImage(es_gt_path)).astype(np.int32)
        data.append((ed_img, es_img, ed_gt, es_gt))
        codes.append(code)
    print(f'Loaded {len(data)} M&Ms patients ({skipped} skipped)')
    return data, codes

def get_fold_indices(n_patients, fold):
    """5-fold split: 105 train / 15 val / 30 test (7:1:2 of 150)."""
    rng   = random.Random(SEED)
    order = list(range(n_patients))
    rng.shuffle(order)
    test_size  = n_patients // N_FOLDS
    val_size   = n_patients // 10
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
    return torch.from_numpy(arr).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ---------------- metrics ----------------
def dice_bin(pred_bin, target_bin):
    p = pred_bin.astype(np.float32)
    t = target_bin.astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom

def surface_distances(pred_bin, target_bin, spacing_mm=SPACING_MM):
    """
    ASSD and HD95 (95th percentile Hausdorff Distance).
    HD95 is robust to outlier voxels caused by thick z-slices (8mm),
    and matches paper's actual reported values despite Eq.10 showing max.
    """
    pred_sitk   = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    target_sitk = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_mm[::-1])
    target_sitk.SetSpacing(spacing_mm[::-1])
    pred_surface   = sitk.LabelContour(pred_sitk)
    target_surface = sitk.LabelContour(target_sitk)
    dist_filter = sitk.SignedMaurerDistanceMapImageFilter()
    dist_filter.SetSquaredDistance(False)
    dist_filter.SetUseImageSpacing(True)
    dist_pred   = sitk.GetArrayFromImage(dist_filter.Execute(pred_sitk))
    dist_target = sitk.GetArrayFromImage(dist_filter.Execute(target_sitk))
    pred_surf_arr   = sitk.GetArrayFromImage(pred_surface).astype(bool)
    target_surf_arr = sitk.GetArrayFromImage(target_surface).astype(bool)
    d_pred_to_target = np.abs(dist_target[pred_surf_arr])
    d_target_to_pred = np.abs(dist_pred[target_surf_arr])
    if len(d_pred_to_target) == 0 or len(d_target_to_pred) == 0:
        return 0.0, 0.0
    assd = (d_pred_to_target.sum() + d_target_to_pred.sum()) / \
           (len(d_pred_to_target) + len(d_target_to_pred))
    # HD95: 95th percentile over all surface distances (both directions)
    hd95 = float(np.percentile(
        np.concatenate([d_pred_to_target, d_target_to_pred]), 95
    ))
    return float(assd), hd95

def jacobian_neg_percentage(disp_field):
    flow = transform_unit_flow_to_flow_cuda(disp_field.permute(0,2,3,4,1).clone())
    flow = flow.squeeze(0).cpu().numpy()
    dvx  = np.gradient(flow[..., 0], axis=(0,1,2))
    dvy  = np.gradient(flow[..., 1], axis=(0,1,2))
    dvz  = np.gradient(flow[..., 2], axis=(0,1,2))
    J = np.zeros(flow.shape[:3] + (3,3))
    J[...,0,0]=dvx[0]+1; J[...,0,1]=dvx[1];   J[...,0,2]=dvx[2]
    J[...,1,0]=dvy[0];   J[...,1,1]=dvy[1]+1; J[...,1,2]=dvy[2]
    J[...,2,0]=dvz[0];   J[...,2,1]=dvz[1];   J[...,2,2]=dvz[2]+1
    return 100.0 * float(np.mean(np.linalg.det(J) <= 0))

def warp_segmentation(seg, disp_field, grid):
    """
    Per-label binary warping to avoid hallucinated labels at boundaries.
    Bilinear interpolation on categorical labels (e.g. LV=1, MYO=3)
    produces fractional values (e.g. 2.0 = RV) at boundaries. Instead,
    each label is warped as a binary probability map; winner-takes-all.
    """
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

def validate_dice(model, val_data, grid):
    """Validation Dice on LV cavity (label 1) -- M&Ms convention."""
    model.eval()
    dices = []
    with torch.no_grad():
        for ed_img, es_img, ed_gt, es_gt in val_data:
            fix  = to_tensor(ed_img); mov = to_tensor(es_img)
            disp = model(mov, fix)[0]
            warped = warp_segmentation(es_gt, disp, grid)
            dices.append(dice_bin(warped == LV_LABEL, ed_gt == LV_LABEL))
    model.train()
    return float(np.mean(dices))

# ---------------- training ----------------
def train_lvl1(train_data, fold):
    print(f'  [fold {fold}] Training lvl1...')
    model     = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_sim  = NCC(win=3)
    grid      = generate_grid(imgshape_4)
    grid      = torch.from_numpy(np.reshape(grid, (1,)+grid.shape)).to(DEVICE).float()
    step = 0
    while step < ITER_LVL1:
        for ed_img, es_img, *_ in train_data:
            if step >= ITER_LVL1: break
            fix = to_tensor(ed_img); mov = to_tensor(es_img)
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(mov, fix)
            loss_ncc = loss_sim(X_Y, Y_4x)
            F_norm   = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            loss_j   = neg_Jdet_loss(F_norm, grid)
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:,0] *= (z-1); F_X_Y[:,1] *= (y-1); F_X_Y[:,2] *= (x-1)
            loss = loss_ncc + ANTIFOLD * loss_j + SMOOTH * smoothloss(F_X_Y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if step % 1000 == 0:
                print(f'  [fold {fold}] lvl1 step {step:5d} | loss {loss.item():.4f}')
            step += 1
    torch.save(model.state_dict(), f'{MODEL_DIR}/fold{fold}_lvl1.pth')
    return model

def train_lvl2(train_data, model_lvl1, fold):
    print(f'  [fold {fold}] Training lvl2...')
    for p in model_lvl1.parameters():
        p.requires_grad = False
    patch_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW, model_lvl1=model_lvl1,
        patch_model_lv2=patch_lv2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_sim  = multi_resolution_NCC(win=5, scale=2)
    grid      = generate_grid(imgshape_2)
    grid      = torch.from_numpy(np.reshape(grid, (1,)+grid.shape)).to(DEVICE).float()
    step = 0
    while step < ITER_LVL2:
        for ed_img, es_img, *_ in train_data:
            if step >= ITER_LVL2: break
            fix = to_tensor(ed_img); mov = to_tensor(es_img)
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(mov, fix)
            loss_ncc = loss_sim(X_Y, Y_4x)
            F_norm   = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            loss_j   = neg_Jdet_loss(F_norm, grid)
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:,0] *= (z-1); F_X_Y[:,1] *= (y-1); F_X_Y[:,2] *= (x-1)
            loss = loss_ncc + ANTIFOLD * loss_j + SMOOTH * smoothloss(F_X_Y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if step % 1000 == 0:
                print(f'  [fold {fold}] lvl2 step {step:5d} | loss {loss.item():.4f}')
            if step == FREEZE_STEP:
                model.unfreeze_modellvl1()
                print(f'  [fold {fold}] lvl1 unfrozen')
            step += 1
    torch.save(model.state_dict(), f'{MODEL_DIR}/fold{fold}_lvl2.pth')
    return model, patch_lv2

def train_lvl3(train_data, val_data, model_lvl2, patch_lv2, fold):
    print(f'  [fold {fold}] Training lvl3 (early stopping patience={PATIENCE})...')
    for p in model_lvl2.parameters():
        p.requires_grad = False
    patch_lv3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(DEVICE)
    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW, model_lvl2=model_lvl2,
        patch_model=patch_lv3).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    loss_sim   = multi_resolution_NCC(win=7, scale=3)
    train_grid = generate_grid(imgshape)
    train_grid = torch.from_numpy(np.reshape(train_grid, (1,)+train_grid.shape)).to(DEVICE).float()
    val_grid   = generate_grid_unit(imgshape)
    val_grid   = torch.from_numpy(np.reshape(val_grid, (1,)+val_grid.shape)).to(DEVICE).float()

    best_dice = 0.0; best_step = 0
    best_path = f'{MODEL_DIR}/fold{fold}_lvl3_best.pth'
    step = 0

    while step < ITER_LVL3_MAX:
        for ed_img, es_img, *_ in train_data:
            if step >= ITER_LVL3_MAX: break
            fix = to_tensor(ed_img); mov = to_tensor(es_img)
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(mov, fix)
            loss_ncc = loss_sim(X_Y, Y_4x)
            F_norm   = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            loss_j   = neg_Jdet_loss(F_norm, train_grid)
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:,0] *= (z-1); F_X_Y[:,1] *= (y-1); F_X_Y[:,2] *= (x-1)
            loss = loss_ncc + ANTIFOLD * loss_j + SMOOTH * smoothloss(F_X_Y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            if step % VAL_EVERY == 0:
                val_dice = validate_dice(model, val_data, val_grid)
                tag = ''
                if val_dice > best_dice:
                    best_dice = val_dice; best_step = step
                    torch.save(model.state_dict(), best_path)
                    tag = ' <- best'
                print(f'  [fold {fold}] lvl3 step {step:5d} | loss {loss.item():.4f} | val Dice {val_dice:.4f}{tag}')
                if step > 0 and (step - best_step) >= PATIENCE:
                    print(f'  [fold {fold}] early stopping @ step {step} (best {best_dice:.4f} @ {best_step})')
                    return best_path, best_dice

            if step == FREEZE_STEP:
                model.unfreeze_modellvl2()
                print(f'  [fold {fold}] lvl2 unfrozen')
            step += 1

    print(f'  [fold {fold}] max steps reached (best {best_dice:.4f} @ {best_step})')
    return best_path, best_dice

# ---------------- evaluation ----------------
def build_eval_model(weights_path):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    patch_lv2  = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW, model_lvl1=model_lvl1,
        patch_model_lv2=patch_lv2).to(DEVICE)
    patch_lv3  = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(DEVICE)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW, model_lvl2=model_lvl2,
        patch_model=patch_lv3).to(DEVICE)
    model_lvl3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model_lvl3.eval()
    return model_lvl3

def evaluate_fold(weights_path, test_data):
    """
    M&Ms label mapping:
      LV cavity (Endo) = label 1
      LV myocardium    = label 3
      Epicardium (Epi) = label 1 | label 3
    HD reported as HD95 (95th percentile).
    """
    model = build_eval_model(weights_path)
    grid  = generate_grid_unit(imgshape)
    grid  = torch.from_numpy(np.reshape(grid, (1,)+grid.shape)).to(DEVICE).float()
    res   = {k: [] for k in ['lv_dice','myo_dice','endo_assd','epi_assd',
                              'endo_hd','epi_hd','jac']}
    for ed_img, es_img, ed_gt, es_gt in test_data:
        fix = to_tensor(ed_img); mov = to_tensor(es_img)
        with torch.no_grad():
            disp = model(mov, fix)[0]
        warped = warp_segmentation(es_gt, disp, grid)

        res['lv_dice'].append(dice_bin(warped == LV_LABEL,  ed_gt == LV_LABEL))
        res['myo_dice'].append(dice_bin(warped == MYO_LABEL, ed_gt == MYO_LABEL))

        endo_pred = (warped == LV_LABEL).astype(np.uint8)
        endo_gt   = (ed_gt  == LV_LABEL).astype(np.uint8)
        epi_pred  = ((warped == LV_LABEL) | (warped == MYO_LABEL)).astype(np.uint8)
        epi_gt    = ((ed_gt  == LV_LABEL) | (ed_gt  == MYO_LABEL)).astype(np.uint8)

        ea, eh = surface_distances(endo_pred, endo_gt)
        pa, ph = surface_distances(epi_pred,  epi_gt)
        res['endo_assd'].append(ea); res['endo_hd'].append(eh)
        res['epi_assd'].append(pa);  res['epi_hd'].append(ph)
        res['jac'].append(jacobian_neg_percentage(disp))
    return res

# ---------------- main ----------------
if __name__ == '__main__':
    print('Device:', DEVICE)
    print(f'Mode: {"TEST (fold 0 only, short)" if TEST_MODE else "FULL (5-fold)"}')
    print(f'SMOOTH={SMOOTH} | Label: LV={LV_LABEL}, RV={RV_LABEL}, MYO={MYO_LABEL}')
    print(f'HD metric: 95th percentile (HD95)')
    all_data, all_codes = load_all_patients()
    n = len(all_data)

    pooled = {k: [] for k in ['lv_dice','myo_dice','endo_assd','epi_assd',
                               'endo_hd','epi_hd','jac']}

    for fold in RUN_FOLDS:
        print(f'\n{"="*60}\n  FOLD {fold}\n{"="*60}')
        tr_idx, va_idx, te_idx = get_fold_indices(n, fold)
        train_data = [all_data[i] for i in tr_idx]
        val_data   = [all_data[i] for i in va_idx]
        test_data  = [all_data[i] for i in te_idx]
        print(f'  train {len(train_data)} | val {len(val_data)} | test {len(test_data)}')

        m1             = train_lvl1(train_data, fold)
        m2, patch_lv2  = train_lvl2(train_data, m1, fold)
        best_path, _   = train_lvl3(train_data, val_data, m2, patch_lv2, fold)

        res = evaluate_fold(best_path, test_data)
        for k in pooled:
            pooled[k].extend(res[k])

        s = {k: float(np.mean(v)) for k, v in res.items()}
        print(f'\n  Fold {fold} results: '
              f"LV {s['lv_dice']:.3f} | MYO {s['myo_dice']:.3f} | "
              f"Endo ASSD {s['endo_assd']:.2f} | Epi ASSD {s['epi_assd']:.2f} | "
              f"Endo HD95 {s['endo_hd']:.2f} | Epi HD95 {s['epi_hd']:.2f} | "
              f"|J|<=0 {s['jac']:.3f}%")

    if not TEST_MODE:
        print(f'\n{"="*60}\n  5-FOLD CV RESULTS -- M&Ms20 (150 pairs)\n{"="*60}')
        def stat(k):
            v = np.array(pooled[k]); return v.mean(), v.std()
        lv=stat('lv_dice'); myo=stat('myo_dice')
        ea=stat('endo_assd'); pa=stat('epi_assd')
        eh=stat('endo_hd');   ph=stat('epi_hd')
        jac=stat('jac')
        print(f'  LV   Dice:  {lv[0]:.3f} +/- {lv[1]:.3f}')
        print(f'  MYO  Dice:  {myo[0]:.3f} +/- {myo[1]:.3f}')
        print(f'  Endo ASSD:  {ea[0]:.2f} +/- {ea[1]:.2f} mm')
        print(f'  Epi  ASSD:  {pa[0]:.2f} +/- {pa[1]:.2f} mm')
        print(f'  Endo HD95:  {eh[0]:.2f} +/- {eh[1]:.2f} mm')
        print(f'  Epi  HD95:  {ph[0]:.2f} +/- {ph[1]:.2f} mm')
        print(f'  |J|<=0:     {jac[0]:.3f} +/- {jac[1]:.3f} %')
        print('\n  --- Paper Table 3 (Proposed, M&Ms20) ---')
        print('  LV   Dice: 0.884 +/- 0.038')
        print('  MYO  Dice: 0.729 +/- 0.057')
        print('  Endo ASSD: 1.88  +/- 0.79 mm')
        print('  Epi  ASSD: 2.05  +/- 0.83 mm')
        print('  Endo HD:   7.40  +/- 2.78 mm')
        print('  Epi  HD:   8.16  +/- 1.95 mm')
        print('  |J|<=0:    0.536 +/- 0.321 %')
        np.save(f'{MODEL_DIR}/cv_pooled_results_mms.npy', pooled)
        print(f'\n  Results saved to {MODEL_DIR}/cv_pooled_results_mms.npy')