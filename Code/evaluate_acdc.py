import os, glob, sys, numpy as np, torch
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
MODEL_DIR     = 'Models'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

START_CHANNEL = 4
RANGE_FLOW    = 0.4

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)
# ================================================

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
        return None, None, None, None
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
    return ed_img, es_img, ed_gt, es_gt

def to_tensor(arr):
    return torch.from_numpy(arr).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ---------- metrics ----------
def dice_score(pred_bin, target_bin):
    """Dice for two binary masks."""
    p = pred_bin.astype(np.float32)
    t = target_bin.astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom

def surface_distances(pred_bin, target_bin, spacing_mm=(5.0, 1.25, 1.25)):
    """Compute ASSD and 95th-percentile HD between two binary masks."""
    pred_sitk   = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    target_sitk = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_mm[::-1])    # SimpleITK: (x,y,z) = (w,h,d)
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

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    hd = float(np.percentile(all_distances, 95))

    return float(assd), hd

def jacobian_neg_percentage(disp_field):
    """Percentage of voxels with non-positive Jacobian determinant (folding)."""
    # disp_field: (1,3,X,Y,Z) unit-normalized -> convert to voxel flow
    flow = transform_unit_flow_to_flow_cuda(disp_field.permute(0,2,3,4,1).clone())
    flow = flow.squeeze(0).cpu().numpy()  # (X,Y,Z,3)

    # Spatial gradients of displacement
    dx = np.gradient(flow[..., 0], axis=0)
    dy = np.gradient(flow[..., 1], axis=1)
    dz = np.gradient(flow[..., 2], axis=2)

    dvx = np.gradient(flow[..., 0], axis=(0,1,2))
    dvy = np.gradient(flow[..., 1], axis=(0,1,2))
    dvz = np.gradient(flow[..., 2], axis=(0,1,2))

    # Jacobian = I + grad(u); determinant of 3x3 matrix per voxel
    J = np.zeros(flow.shape[:3] + (3,3))
    J[..., 0, 0] = dvx[0] + 1; J[..., 0, 1] = dvx[1];     J[..., 0, 2] = dvx[2]
    J[..., 1, 0] = dvy[0];     J[..., 1, 1] = dvy[1] + 1; J[..., 1, 2] = dvy[2]
    J[..., 2, 0] = dvz[0];     J[..., 2, 1] = dvz[1];     J[..., 2, 2] = dvz[2] + 1

    det = np.linalg.det(J)
    return 100.0 * float(np.mean(det <= 0))

def warp_segmentation(seg, disp_field, grid):
    transform  = SpatialTransform_unit().to(DEVICE)
    seg_tensor = torch.from_numpy(seg).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        warped = transform(seg_tensor, disp_field.permute(0,2,3,4,1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)  # (16,96,96)

def build_model(weights_path):
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
    print(f'Model loaded: {weights_path}')
    return model_lvl3

if __name__ == '__main__':
    print('Device:', DEVICE)

    best_path  = f'{MODEL_DIR}/lvl3_best.pth'
    final_path = f'{MODEL_DIR}/lvl3_final.pth'
    weights    = best_path if os.path.exists(best_path) else final_path
    model      = build_model(weights)

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,)+grid.shape)).to(DEVICE).float()

    patients      = sorted(glob.glob(os.path.join(DATAPATH, 'patient*')))
    test_patients = patients[80:]
    print(f'Testing on {len(test_patients)} patients...\n')

    # Paper labels:
    #   LV   (Dice)  = label 3            (LV blood pool)
    #   MYO  (Dice)  = label 2            (myocardium)
    #   LV-Endo (surface) = label 3       (endocardial contour)
    #   LV-Epi  (surface) = label 2 + 3   (epicardial contour = myo outer boundary)
    lv_dice_list   = []
    myo_dice_list  = []
    endo_assd_list = []
    endo_hd_list   = []
    epi_assd_list  = []
    epi_hd_list    = []
    jac_neg_list   = []

    spacing_mm = (5.0, 1.25, 1.25)  # (z, y, x) after resampling

    for pat_dir in test_patients:
        pat_name = os.path.basename(pat_dir)
        ed_img, es_img, ed_gt, es_gt = load_patient(pat_dir)
        if ed_img is None:
            continue

        fix = to_tensor(ed_img)
        mov = to_tensor(es_img)

        with torch.no_grad():
            outputs    = model(mov, fix)
            disp_field = outputs[0]

        warped_seg = warp_segmentation(es_gt, disp_field, grid)
        warped_seg = np.round(warped_seg).astype(np.int32)

        # --- Dice (LV = label 3, MYO = label 2) ---
        lv_dice  = dice_score(warped_seg == 3, ed_gt == 3)
        myo_dice = dice_score(warped_seg == 2, ed_gt == 2)

        # --- Surface metrics ---
        # LV-Endo: LV blood pool (label 3)
        endo_pred = (warped_seg == 3).astype(np.uint8)
        endo_gt   = (ed_gt == 3).astype(np.uint8)
        # LV-Epi: epicardium = myocardium + blood pool (label 2 or 3)
        epi_pred  = ((warped_seg == 2) | (warped_seg == 3)).astype(np.uint8)
        epi_gt    = ((ed_gt == 2) | (ed_gt == 3)).astype(np.uint8)

        endo_assd, endo_hd = surface_distances(endo_pred, endo_gt, spacing_mm)
        epi_assd,  epi_hd  = surface_distances(epi_pred,  epi_gt,  spacing_mm)

        # --- Jacobian folding percentage ---
        jac_neg = jacobian_neg_percentage(disp_field)

        lv_dice_list.append(lv_dice);     myo_dice_list.append(myo_dice)
        endo_assd_list.append(endo_assd); endo_hd_list.append(endo_hd)
        epi_assd_list.append(epi_assd);   epi_hd_list.append(epi_hd)
        jac_neg_list.append(jac_neg)

        print(f'{pat_name} | LV Dice: {lv_dice:.4f}  MYO Dice: {myo_dice:.4f} | '
              f'Endo ASSD: {endo_assd:.2f}  Epi ASSD: {epi_assd:.2f} | '
              f'Endo HD: {endo_hd:.2f}  Epi HD: {epi_hd:.2f} | '
              f'|J|<=0: {jac_neg:.3f}%')

    print('\n========================= RESULTS =========================')
    print(f'Mean LV   Dice: {np.mean(lv_dice_list):.3f} +/- {np.std(lv_dice_list):.3f}')
    print(f'Mean MYO  Dice: {np.mean(myo_dice_list):.3f} +/- {np.std(myo_dice_list):.3f}')
    print(f'Mean Endo ASSD: {np.mean(endo_assd_list):.2f} +/- {np.std(endo_assd_list):.2f} mm')
    print(f'Mean Epi  ASSD: {np.mean(epi_assd_list):.2f} +/- {np.std(epi_assd_list):.2f} mm')
    print(f'Mean Endo HD:   {np.mean(endo_hd_list):.2f} +/- {np.std(endo_hd_list):.2f} mm')
    print(f'Mean Epi  HD:   {np.mean(epi_hd_list):.2f} +/- {np.std(epi_hd_list):.2f} mm')
    print(f'Mean |J|<=0:    {np.mean(jac_neg_list):.3f} +/- {np.std(jac_neg_list):.3f} %')
    print('\n--- Paper Table 2 (Proposed) ---')
    print('  LV   Dice: 0.917 +/- 0.043')
    print('  MYO  Dice: 0.789 +/- 0.055')
    print('  Endo ASSD: 0.79  +/- 0.39 mm')
    print('  Epi  ASSD: 0.88  +/- 0.21 mm')
    print('  Endo HD:   5.62  +/- 1.22 mm')
    print('  Epi  HD:   5.51  +/- 1.77 mm')
    print('  |J|<=0:    0.409 +/- 0.153 %')