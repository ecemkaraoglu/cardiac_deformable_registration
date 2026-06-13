"""
5-fold stratified cross validation for DIR-MRVIT on ACDC17.

ACDC pathology groups (20 patients per group, sequential):
  NOR:  patient001-020
  MINF: patient021-040
  DCM:  patient041-060
  HCM:  patient061-080
  RV:   patient081-100

Stratified split: each fold's test set contains 4 patients from each group.
Random seed fixed -> reproducible.

TEST_MODE = True  -> fold 0 only, short iterations (~20 min)
TEST_MODE = False -> 5 folds, full iterations (overnight)
"""

import os, glob, sys, numpy as np, torch, random
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
DATAPATH      = 'Data/training'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR            = 1e-4
SMOOTH        = 0.5
ANTIFOLD      = 0
START_CHANNEL = 4
RANGE_FLOW    = 0.4
FREEZE_STEP   = 5000      # unfreeze step for lvl2/lvl3

N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

# ---- TOGGLE HERE ----
TEST_MODE = False
# TEST_MODE = True  -> fold 0, short iterations (~20 min, for testing)
# TEST_MODE = False -> 5 folds, full iterations  (run overnight)

if TEST_MODE:
    MODEL_DIR     = 'Models_cv_test'
    ITER_LVL1     = 2000
    ITER_LVL2     = 2000
    ITER_LVL3_MAX = 5000
    VAL_EVERY     = 500
    PATIENCE      = 1500
    RUN_FOLDS     = [0]
else:
    MODEL_DIR     = 'Models_cv_full'
    ITER_LVL1     = 15000
    ITER_LVL2     = 15000
    ITER_LVL3_MAX = 35000
    VAL_EVERY     = 500
    PATIENCE      = 3000
    RUN_FOLDS     = list(range(N_FOLDS))
# ================================================

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- preprocessing ----------------
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
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return resample.Execute(image)

def crop_around_lv(img_arr, gt_arr, crop_size=(96, 96, 16)):
    lv_mask = (gt_arr == 3)
    if lv_mask.sum() == 0:
        lv_mask = (gt_arr > 0)
    coords = np.where(lv_mask)
    center = [int(np.mean(c)) for c in coords] if len(coords[0]) > 0 else [s // 2 for s in img_arr.shape]
    cz, ch, cw = center
    sz, sh, sw  = crop_size[2], crop_size[0], crop_size[1]

    def get_slice(c, size, max_size):
        start = max(0, c - size // 2)
        end   = start + size
        if end > max_size:
            end   = max_size
            start = max(0, end - size)
        return start, end

    z0, z1 = get_slice(cz, sz, img_arr.shape[0])
    h0, h1 = get_slice(ch, sh, img_arr.shape[1])
    w0, w1 = get_slice(cw, sw, img_arr.shape[2])
    cropped = img_arr[z0:z1, h0:h1, w0:w1]
    pad_z = sz - cropped.shape[0]
    pad_h = sh - cropped.shape[1]
    pad_w = sw - cropped.shape[2]
    if pad_z > 0 or pad_h > 0 or pad_w > 0:
        cropped = np.pad(cropped, (
            (pad_z//2, pad_z - pad_z//2),
            (pad_h//2, pad_h - pad_h//2),
            (pad_w//2, pad_w - pad_w//2)
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
    data = []
    for p in patients:
        rec = load_patient(p)
        if rec is not None:
            data.append(rec)
    print(f'Loaded {len(data)} patients total')
    return data

def get_stratified_fold_indices(n_patients, fold):
    """
    Stratified 5-fold split.
    ACDC: 5 groups of 20 patients (NOR, MINF, DCM, HCM, RV).
    Each fold's test set has 4 patients from each group (total 20).
    Val set has 2 from each group (total 10).
    Train: remaining 70.
    """
    rng = random.Random(SEED)
    group_size = n_patients // N_FOLDS  # 20
    n_groups   = N_FOLDS                # 5

    # Shuffle within each group
    groups = []
    for g in range(n_groups):
        indices = list(range(g * group_size, (g + 1) * group_size))
        rng.shuffle(indices)
        groups.append(indices)

    # Each fold: 4 test + 2 val from each group
    test_per_group = group_size // N_FOLDS   # 4
    val_per_group  = 2

    test_idx  = []
    val_idx   = []
    for g in range(n_groups):
        start     = fold * test_per_group
        test_idx += groups[g][start:start + test_per_group]
        # val: next 2 after test block (circular)
        val_start  = (start + test_per_group) % group_size
        for i in range(val_per_group):
            val_idx.append(groups[g][(val_start + i) % group_size])

    used      = set(test_idx) | set(val_idx)
    train_idx = [i for i in range(n_patients) if i not in used]
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

def surface_distances(pred_bin, target_bin, spacing_mm=(5.0, 1.25, 1.25)):
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
    all_d = np.concatenate([d_pred_to_target, d_target_to_pred])
    hd    = float(np.percentile(all_d, 95))
    return float(assd), hd

def jacobian_neg_percentage(disp_field):
    flow = transform_unit_flow_to_flow_cuda(disp_field.permute(0,2,3,4,1).clone())
    flow = flow.squeeze(0).cpu().numpy()
    dvx  = np.gradient(flow[..., 0], axis=(0,1,2))
    dvy  = np.gradient(flow[..., 1], axis=(0,1,2))
    dvz  = np.gradient(flow[..., 2], axis=(0,1,2))
    J = np.zeros(flow.shape[:3] + (3,3))
    J[...,0,0]=dvx[0]+1; J[...,0,1]=dvx[1];     J[...,0,2]=dvx[2]
    J[...,1,0]=dvy[0];   J[...,1,1]=dvy[1]+1;   J[...,1,2]=dvy[2]
    J[...,2,0]=dvz[0];   J[...,2,1]=dvz[1];     J[...,2,2]=dvz[2]+1
    return 100.0 * float(np.mean(np.linalg.det(J) <= 0))

def warp_segmentation(seg, disp_field, grid):
    transform  = SpatialTransform_unit().to(DEVICE)
    seg_tensor = torch.from_numpy(seg).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        warped = transform(seg_tensor, disp_field.permute(0,2,3,4,1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)

def validate_dice(model, val_data, grid):
    model.eval()
    dices = []
    with torch.no_grad():
        for ed_img, es_img, ed_gt, es_gt in val_data:
            fix  = to_tensor(ed_img); mov = to_tensor(es_img)
            disp = model(mov, fix)[0]
            warped = np.round(warp_segmentation(es_gt, disp, grid)).astype(np.int32)
            dices.append(dice_bin(warped == 3, ed_gt == 3))
    model.train()
    return float(np.mean(dices))

# ---------------- training ----------------
def train_lvl1(train_data, fold):
    print(f'  [fold {fold}] Training lvl1...')
    model     = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
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
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW, model_lvl1=model_lvl1, patch_model_lv2=patch_lv2).to(DEVICE)
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
        range_flow=RANGE_FLOW, model_lvl2=model_lvl2, patch_model=patch_lv3).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    loss_sim   = multi_resolution_NCC(win=7, scale=3)
    train_grid = generate_grid(imgshape)
    train_grid = torch.from_numpy(np.reshape(train_grid, (1,)+train_grid.shape)).to(DEVICE).float()
    val_grid   = generate_grid_unit(imgshape)
    val_grid   = torch.from_numpy(np.reshape(val_grid, (1,)+val_grid.shape)).to(DEVICE).float()

    best_dice  = 0.0
    best_step  = 0
    best_path  = f'{MODEL_DIR}/fold{fold}_lvl3_best.pth'
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
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    patch_lv2  = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW, model_lvl1=model_lvl1, patch_model_lv2=patch_lv2).to(DEVICE)
    patch_lv3  = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(DEVICE)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW, model_lvl2=model_lvl2, patch_model=patch_lv3).to(DEVICE)
    model_lvl3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model_lvl3.eval()
    return model_lvl3

def evaluate_fold(weights_path, test_data):
    model = build_eval_model(weights_path)
    grid  = generate_grid_unit(imgshape)
    grid  = torch.from_numpy(np.reshape(grid, (1,)+grid.shape)).to(DEVICE).float()
    spacing_mm = (5.0, 1.25, 1.25)
    res = {k: [] for k in ['lv_dice','myo_dice','endo_assd','epi_assd','endo_hd','epi_hd','jac']}
    for ed_img, es_img, ed_gt, es_gt in test_data:
        fix = to_tensor(ed_img); mov = to_tensor(es_img)
        with torch.no_grad():
            disp = model(mov, fix)[0]
        warped = np.round(warp_segmentation(es_gt, disp, grid)).astype(np.int32)
        res['lv_dice'].append(dice_bin(warped == 3, ed_gt == 3))
        res['myo_dice'].append(dice_bin(warped == 2, ed_gt == 2))
        endo_pred = (warped == 3).astype(np.uint8);  endo_gt = (ed_gt == 3).astype(np.uint8)
        epi_pred  = ((warped==2)|(warped==3)).astype(np.uint8)
        epi_gt    = ((ed_gt==2)|(ed_gt==3)).astype(np.uint8)
        ea, eh = surface_distances(endo_pred, endo_gt, spacing_mm)
        pa, ph = surface_distances(epi_pred,  epi_gt,  spacing_mm)
        res['endo_assd'].append(ea); res['endo_hd'].append(eh)
        res['epi_assd'].append(pa);  res['epi_hd'].append(ph)
        res['jac'].append(jacobian_neg_percentage(disp))
    return res

# ---------------- main ----------------
if __name__ == '__main__':
    print('Device:', DEVICE)
    print(f'Mode: {"TEST (fold 0 only, short)" if TEST_MODE else "FULL (5-fold)"}')
    all_data = load_all_patients()
    n = len(all_data)

    pooled = {k: [] for k in ['lv_dice','myo_dice','endo_assd','epi_assd','endo_hd','epi_hd','jac']}

    for fold in RUN_FOLDS:
        print(f'\n{"="*60}\n  FOLD {fold}\n{"="*60}')
        tr_idx, va_idx, te_idx = get_stratified_fold_indices(n, fold)
        train_data = [all_data[i] for i in tr_idx]
        val_data   = [all_data[i] for i in va_idx]
        test_data  = [all_data[i] for i in te_idx]
        print(f'  train {len(train_data)} | val {len(val_data)} | test {len(test_data)}')
        print(f'  test indices: {sorted(te_idx)}')

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
              f"Endo HD {s['endo_hd']:.2f} | Epi HD {s['epi_hd']:.2f} | "
              f"|J|<=0 {s['jac']:.3f}%")

    if not TEST_MODE:
        print(f'\n{"="*60}\n  5-FOLD CV RESULTS (100 pairs)\n{"="*60}')
        def stat(k):
            v = np.array(pooled[k]); return v.mean(), v.std()
        lv=stat('lv_dice'); myo=stat('myo_dice')
        ea=stat('endo_assd'); pa=stat('epi_assd')
        eh=stat('endo_hd');   ph=stat('epi_hd')
        jac=stat('jac')
        print(f'  LV   Dice: {lv[0]:.3f} +/- {lv[1]:.3f}')
        print(f'  MYO  Dice: {myo[0]:.3f} +/- {myo[1]:.3f}')
        print(f'  Endo ASSD: {ea[0]:.2f} +/- {ea[1]:.2f} mm')
        print(f'  Epi  ASSD: {pa[0]:.2f} +/- {pa[1]:.2f} mm')
        print(f'  Endo HD:   {eh[0]:.2f} +/- {eh[1]:.2f} mm')
        print(f'  Epi  HD:   {ph[0]:.2f} +/- {ph[1]:.2f} mm')
        print(f'  |J|<=0:    {jac[0]:.3f} +/- {jac[1]:.3f} %')
        print('\n  --- Paper Table 2 (Proposed) ---')
        print('  LV   Dice: 0.917 +/- 0.043')
        print('  MYO  Dice: 0.789 +/- 0.055')
        print('  Endo ASSD: 0.79  +/- 0.39 mm')
        print('  Epi  ASSD: 0.88  +/- 0.21 mm')
        print('  Endo HD:   5.62  +/- 1.22 mm')
        print('  Epi  HD:   5.51  +/- 1.77 mm')
        print('  |J|<=0:    0.409 +/- 0.153 %')
        np.save(f'{MODEL_DIR}/cv_pooled_results.npy', pooled)
        print(f'\n  Results saved to {MODEL_DIR}/cv_pooled_results.npy')