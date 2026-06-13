"""
evaluate_cmrxm22.py

Evaluate DIR-MRVIT on CMRxM22 using a model trained on M&Ms20.
Replicates paper Section 4.3 (zero-shot transfer / generalizability test).

Usage:
    python Code/evaluate_cmrxm22.py \
        --data_dir  Data/CMRxM22_preprocessed \
        --model_dir Models_mms_full \
        --fold 0 \
        --out_dir   Results/cmrxm22
"""

import os, sys, csv, numpy as np, torch
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

# ==================== CONFIG — must match cross_validation_mms.py ====================
START_CHANNEL = 4
RANGE_FLOW    = 0.4

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (8.0, 1.25, 1.25)   # (z, y, x)

# CMRxMotion label convention (differs from M&Ms!):
#   1 = LV blood pool
#   2 = LV myocardium
#   3 = RV blood pool
LV_LABEL  = 1
MYO_LABEL = 2


# ==================== DATA ====================
def load_nifti_np(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)  # (z,y,x)


def to_tensor(arr, device):
    return torch.from_numpy(arr).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(device)


def load_pairs(data_dir):
    pairs = []
    with open(os.path.join(data_dir, 'pairs.txt')) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                pairs.append(tuple(parts))
    return pairs


# ==================== METRICS — copied from cross_validation_mms.py ====================
def dice_bin(pred_bin, target_bin):
    p = pred_bin.astype(np.float32)
    t = target_bin.astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def surface_distances(pred_bin, target_bin, spacing_mm=SPACING_MM):
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
    # HD95: 95th percentile of symmetric surface distances (robust to outliers)
    all_dists = np.concatenate([d_pred_to_target, d_target_to_pred])
    hd95 = float(np.percentile(all_dists, 95))
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


def warp_segmentation(seg, disp_field, grid, device):
    transform     = SpatialTransform_unit().to(device)
    disp_permuted = disp_field.permute(0, 2, 3, 4, 1)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    result = np.zeros(seg.shape, dtype=np.int32)
    probs  = np.zeros(seg.shape, dtype=np.float32)
    for label in unique_labels:
        binary = (seg == label).astype(np.float32)
        bin_tensor = (torch.from_numpy(binary)
                      .float().permute(1,2,0)
                      .unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            warped_bin = transform(bin_tensor, disp_permuted, grid)
        warped_np = warped_bin.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = warped_np > probs
        result[mask] = label
        probs[mask]  = warped_np[mask]
    return result


# ==================== MODEL — exact copy of build_eval_model ====================
def build_eval_model(weights_path, device):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4,
        range_flow=RANGE_FLOW).to(device)
    patch_lv2  = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4,
        range_flow=RANGE_FLOW).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW, model_lvl1=model_lvl1,
        patch_model_lv2=patch_lv2).to(device)
    patch_lv3  = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(device)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW, model_lvl2=model_lvl2,
        patch_model=patch_lv3).to(device)
    model_lvl3.load_state_dict(torch.load(weights_path, map_location=device))
    model_lvl3.eval()
    return model_lvl3


# ==================== MAIN ====================
def run_fold(fold, pairs, model, grid, device, data_dir):
    keys = ['DSC_LV','DSC_MYO','ASSD_LVEndo','ASSD_LVEpi',
            'HD_LVEndo','HD_LVEpi','Jac_neg_pct']
    fold_metrics = []
    for i, (ed_rel, ed_lbl_rel, es_rel, es_lbl_rel) in enumerate(pairs):
        fixed_img  = load_nifti_np(os.path.join(data_dir, ed_rel))
        fixed_lbl  = load_nifti_np(os.path.join(data_dir, ed_lbl_rel)).astype(np.int32)
        moving_img = load_nifti_np(os.path.join(data_dir, es_rel))
        moving_lbl = load_nifti_np(os.path.join(data_dir, es_lbl_rel)).astype(np.int32)

        f = to_tensor(fixed_img,  device)
        m = to_tensor(moving_img, device)

        with torch.no_grad():
            disp = model(m, f)[0]

        warped_lbl = warp_segmentation(moving_lbl, disp, grid, device)

        met = {}
        met['DSC_LV']  = dice_bin(warped_lbl == LV_LABEL,  fixed_lbl == LV_LABEL)
        met['DSC_MYO'] = dice_bin(warped_lbl == MYO_LABEL, fixed_lbl == MYO_LABEL)

        endo_pred = (warped_lbl == LV_LABEL).astype(np.uint8)
        endo_gt   = (fixed_lbl  == LV_LABEL).astype(np.uint8)
        epi_pred  = ((warped_lbl == LV_LABEL) | (warped_lbl == MYO_LABEL)).astype(np.uint8)
        epi_gt    = ((fixed_lbl  == LV_LABEL) | (fixed_lbl  == MYO_LABEL)).astype(np.uint8)

        met['ASSD_LVEndo'], met['HD_LVEndo'] = surface_distances(endo_pred, endo_gt)
        met['ASSD_LVEpi'],  met['HD_LVEpi']  = surface_distances(epi_pred,  epi_gt)
        met['Jac_neg_pct'] = jacobian_neg_percentage(disp)
        met['case'] = os.path.dirname(ed_rel)
        met['fold'] = fold
        fold_metrics.append(met)

        print(f'  [{i+1:3d}/{len(pairs)}] {met["case"]:12s}  '
              f'DSC_LV={met["DSC_LV"]:.3f}  DSC_MYO={met["DSC_MYO"]:.3f}  '
              f'ASSD_Endo={met["ASSD_LVEndo"]:.2f}mm')

    print(f'\n  Fold {fold} summary:')
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        unit = ' mm' if 'ASSD' in k or 'HD' in k else (' %' if 'Jac' in k else '')
        print(f'    {k:20s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}{unit}')
    return fold_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--model_dir',  required=True)
    parser.add_argument('--fold',       type=int, default=0)
    parser.add_argument('--all_folds',  action='store_true',
                        help='Run all 5 folds and average results')
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--gpu',        type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    grid  = generate_grid_unit(imgshape)
    grid  = torch.from_numpy(np.reshape(grid, (1,)+grid.shape)).to(device).float()
    pairs = load_pairs(args.data_dir)
    folds = list(range(5)) if args.all_folds else [args.fold]
    keys  = ['DSC_LV','DSC_MYO','ASSD_LVEndo','ASSD_LVEpi',
             'HD_LVEndo','HD_LVEpi','Jac_neg_pct']

    all_metrics = []
    for fold in folds:
        weights_path = os.path.join(args.model_dir, f'fold{fold}_lvl3_best.pth')
        print(f'\n--- Fold {fold} | {weights_path} ---')
        model = build_eval_model(weights_path, device)
        all_metrics.extend(run_fold(fold, pairs, model, grid, device, args.data_dir))

    print('\n' + '='*62)
    if args.all_folds:
        print(f'OVERALL RESULTS — all 5 folds ({len(all_metrics)} evaluations)')
    else:
        print(f'RESULTS fold {args.fold} — paper Table 4 comparison')
    print('='*62)
    for k in keys:
        vals = [m[k] for m in all_metrics]
        unit = ' mm' if 'ASSD' in k or 'HD' in k else (' %' if 'Jac' in k else '')
        print(f'  {k:20s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}{unit}')
    print('='*62)
    print('\nPaper Table 4 (Proposed):')
    print('  DSC_LV      : 0.892 ± 0.027')
    print('  DSC_MYO     : 0.703 ± 0.050')
    print('  ASSD_LVEndo : 1.78  ± 0.60 mm')
    print('  ASSD_LVEpi  : 1.94  ± 0.69 mm')
    print('  HD_LVEndo   : 7.82  ± 2.31 mm')
    print('  HD_LVEpi    : 8.07  ± 2.24 mm')

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'cmrxm22_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold','case']+keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_metrics)
        writer.writerow({'fold':'MEAN','case':'-',
                         **{k: f'{np.mean([m[k] for m in all_metrics]):.4f}' for k in keys}})
        writer.writerow({'fold':'STD', 'case':'-',
                         **{k: f'{np.std( [m[k] for m in all_metrics]):.4f}' for k in keys}})
    print(f'\nResults saved to: {csv_path}')


if __name__ == '__main__':
    main()