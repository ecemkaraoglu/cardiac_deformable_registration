"""
run_test.py
===========
Unified test script for evaluating DIR-MRVIT on sample data.
Tests all three datasets (ACDC17, M&Ms20, CMRxM22) using pre-trained checkpoints.

For each patient:
  1. Loads ED (fixed) and ES (moving) images
  2. Runs forward pass through the model
  3. Warps the ES segmentation using the predicted deformation field
  4. Computes evaluation metrics (DSC, ASSD, HD95, Jacobian)
  5. Saves warped images and deformation fields to Results/test_output/

Usage:
    python Code/run_test.py
    python Code/run_test.py --dataset acdc
    python Code/run_test.py --dataset mms
    python Code/run_test.py --dataset cmrxm22
    python Code/run_test.py --dataset all
    python Code/run_test.py --cpu          # Force CPU mode
"""

import os, sys, glob, argparse, time
import numpy as np
import torch
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2,
                   SpatialTransform_unit)
from Functions import generate_grid_unit, transform_unit_flow_to_flow_cuda

# ==================== CONFIG ====================
START_CHANNEL = 4
RANGE_FLOW    = 0.4

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

# Paths relative to repository root
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR     = os.path.join(BASE_DIR, 'Data', 'sample_data')
ACDC_MODEL     = os.path.join(BASE_DIR, 'Models_cv_full', 'fold0_lvl3_best.pth')
MMS_MODEL      = os.path.join(BASE_DIR, 'Models_mms_full', 'fold0_lvl3_best.pth')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'Results', 'test_output')
# ================================================


# -------------------- Model --------------------
def build_model(weights_path, device):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True,
        imgshape=imgshape_4, range_flow=RANGE_FLOW).to(device)
    patch_lv2  = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True,
        patch_shape=imgshape_4, range_flow=RANGE_FLOW).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW,
        model_lvl1=model_lvl1, patch_model_lv2=patch_lv2).to(device)
    patch_lv3  = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(device)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW,
        model_lvl2=model_lvl2, patch_model=patch_lv3).to(device)
    model_lvl3.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True))
    model_lvl3.eval()
    return model_lvl3


# -------------------- Preprocessing (ACDC raw) --------------------
def resample_image(image, new_spacing, is_label=False):
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return resample.Execute(image)


def crop_around_center(arr, center, crop_size):
    cz, ch, cw = center
    sz, sh, sw = crop_size[2], crop_size[0], crop_size[1]

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


def preprocess_acdc_patient(patient_dir):
    """Preprocess raw ACDC patient: resample, crop, normalize."""
    files     = sorted(glob.glob(os.path.join(patient_dir, '*_frame*.nii.gz')))
    img_files = [f for f in files if '_gt' not in f]
    gt_files  = [f for f in files if '_gt' in f]
    if len(img_files) < 2 or len(gt_files) < 2:
        return None
    target_spacing = (1.25, 1.25, 5.0)

    def process(img_path, gt_path, center=None):
        img_sitk = sitk.ReadImage(img_path)
        gt_sitk  = sitk.ReadImage(gt_path)
        img_sitk = resample_image(img_sitk, target_spacing, is_label=False)
        gt_sitk  = resample_image(gt_sitk,  target_spacing, is_label=True)
        img_arr  = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        gt_arr   = sitk.GetArrayFromImage(gt_sitk).astype(np.int32)
        img_arr  = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
        if center is None:
            lv_mask = (gt_arr == 3)
            if lv_mask.sum() == 0:
                lv_mask = (gt_arr > 0)
            coords = np.where(lv_mask)
            center = ([int(np.mean(c)) for c in coords]
                      if len(coords[0]) > 0
                      else [s // 2 for s in gt_arr.shape])
        img_crop = crop_around_center(img_arr, center, imgshape)
        gt_crop  = crop_around_center(gt_arr,  center, imgshape)
        return img_crop, gt_crop, center

    ed_img, ed_gt, center = process(img_files[0], gt_files[0])
    es_img, es_gt, _      = process(img_files[1], gt_files[1], center)
    return ed_img, es_img, ed_gt, es_gt


# -------------------- Metrics --------------------
def to_tensor(arr, device):
    return (torch.from_numpy(arr).float()
            .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device))


def dice_bin(pred_bin, target_bin):
    p = pred_bin.astype(np.float32)
    t = target_bin.astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def surface_distances(pred_bin, target_bin, spacing_mm):
    pred_sitk   = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    target_sitk = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_mm[::-1])
    target_sitk.SetSpacing(spacing_mm[::-1])

    df = sitk.SignedMaurerDistanceMapImageFilter()
    df.SetSquaredDistance(False)
    df.SetUseImageSpacing(True)

    dist_pred   = sitk.GetArrayFromImage(df.Execute(pred_sitk))
    dist_target = sitk.GetArrayFromImage(df.Execute(target_sitk))

    pred_surf = sitk.GetArrayFromImage(
        sitk.LabelContour(pred_sitk)).astype(bool)
    tgt_surf  = sitk.GetArrayFromImage(
        sitk.LabelContour(target_sitk)).astype(bool)

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


def warp_segmentation(seg, disp_field, grid, device):
    transform     = SpatialTransform_unit().to(device)
    disp_permuted = disp_field.permute(0, 2, 3, 4, 1)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    result = np.zeros(seg.shape, dtype=np.int32)
    probs  = np.zeros(seg.shape, dtype=np.float32)
    for label in unique_labels:
        binary = (seg == label).astype(np.float32)
        bt = (torch.from_numpy(binary).float()
              .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            warped = transform(bt, disp_permuted, grid)
        warped_np = warped.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = warped_np > probs
        result[mask] = label
        probs[mask]  = warped_np[mask]
    return result


def save_nifti(arr, path, spacing=(1.25, 1.25, 5.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)


# -------------------- Dataset Loaders --------------------
def load_acdc_samples():
    acdc_dir = os.path.join(SAMPLE_DIR, 'acdc')
    if not os.path.isdir(acdc_dir):
        return []
    patients = sorted(glob.glob(os.path.join(acdc_dir, 'patient*')))
    samples = []
    for p in patients:
        result = preprocess_acdc_patient(p)
        if result is not None:
            samples.append((os.path.basename(p), result))
    return samples


def load_mms_samples():
    mms_dir = os.path.join(SAMPLE_DIR, 'mms')
    if not os.path.isdir(mms_dir):
        return []
    patients = sorted([d for d in glob.glob(os.path.join(mms_dir, '*'))
                       if os.path.isdir(d)])
    samples = []
    for p in patients:
        code = os.path.basename(p)
        paths = [os.path.join(p, f'{code}_{x}.nii.gz')
                 for x in ['ED', 'ED_gt', 'ES', 'ES_gt']]
        if not all(os.path.exists(x) for x in paths):
            continue
        ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[0])).astype(np.float32)
        ed_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[1])).astype(np.int32)
        es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[2])).astype(np.float32)
        es_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[3])).astype(np.int32)
        samples.append((code, (ed_img, es_img, ed_gt, es_gt)))
    return samples


def load_cmrxm22_samples():
    cmr_dir = os.path.join(SAMPLE_DIR, 'cmrxm22')
    if not os.path.isdir(cmr_dir):
        return []
    patients = sorted([d for d in glob.glob(os.path.join(cmr_dir, '*'))
                       if os.path.isdir(d)])
    samples = []
    for p in patients:
        code = os.path.basename(p)
        paths = [os.path.join(p, f'{code}-ED.nii.gz'),
                 os.path.join(p, f'{code}-ED-label.nii.gz'),
                 os.path.join(p, f'{code}-ES.nii.gz'),
                 os.path.join(p, f'{code}-ES-label.nii.gz')]
        if not all(os.path.exists(x) for x in paths):
            continue
        ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[0])).astype(np.float32)
        ed_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[1])).astype(np.int32)
        es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[2])).astype(np.float32)
        es_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[3])).astype(np.int32)
        samples.append((code, (ed_img, es_img, ed_gt, es_gt)))
    return samples


# -------------------- Evaluation --------------------
def evaluate_dataset(name, samples, model, device, lv_label, myo_label,
                     spacing_mm, paper_results):
    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(
        np.reshape(grid, (1,) + grid.shape)).to(device).float()

    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    keys = ['lv_dice', 'myo_dice', 'endo_assd', 'epi_assd',
            'endo_hd', 'epi_hd', 'jac']
    results = {k: [] for k in keys}
    times = []

    print(f'\n{"=" * 70}')
    print(f'  {name.upper()} - {len(samples)} patients')
    print(f'  LV label: {lv_label}  |  MYO label: {myo_label}  |  '
          f'Spacing: {spacing_mm} mm')
    print(f'{"=" * 70}')

    for i, (patient_id, (ed_img, es_img, ed_gt, es_gt)) in enumerate(samples):
        fix = to_tensor(ed_img, device)
        mov = to_tensor(es_img, device)

        with torch.no_grad():
            t0 = time.time()
            disp = model(mov, fix)[0]
            t1 = time.time()

        reg_time = t1 - t0
        times.append(reg_time)

        warped_seg = warp_segmentation(es_gt, disp, grid, device)

        # Dice
        lv_d  = dice_bin(warped_seg == lv_label,  ed_gt == lv_label)
        myo_d = dice_bin(warped_seg == myo_label, ed_gt == myo_label)

        # Surface distances
        endo_pred = (warped_seg == lv_label).astype(np.uint8)
        endo_gt   = (ed_gt == lv_label).astype(np.uint8)
        epi_pred  = ((warped_seg == lv_label) |
                     (warped_seg == myo_label)).astype(np.uint8)
        epi_gt    = ((ed_gt == lv_label) |
                     (ed_gt == myo_label)).astype(np.uint8)

        ea, eh = surface_distances(endo_pred, endo_gt, spacing_mm)
        pa, ph = surface_distances(epi_pred,  epi_gt,  spacing_mm)
        jac    = jacobian_neg_percentage(disp)

        results['lv_dice'].append(lv_d)
        results['myo_dice'].append(myo_d)
        results['endo_assd'].append(ea)
        results['epi_assd'].append(pa)
        results['endo_hd'].append(eh)
        results['epi_hd'].append(ph)
        results['jac'].append(jac)

        # Save warped image and deformation field
        transform = SpatialTransform_unit().to(device)
        with torch.no_grad():
            warped_img = transform(
                mov, disp.permute(0, 2, 3, 4, 1), grid)
        warped_np = warped_img.squeeze().cpu().numpy().transpose(2, 0, 1)

        save_nifti(warped_np, os.path.join(out_dir, f'{patient_id}_warped.nii.gz'),
                   spacing_mm)
        save_nifti(warped_seg.astype(np.float32),
                   os.path.join(out_dir, f'{patient_id}_warped_seg.nii.gz'),
                   spacing_mm)

        flow = transform_unit_flow_to_flow_cuda(
            disp.permute(0, 2, 3, 4, 1).clone())
        flow_np = flow.squeeze(0).cpu().numpy().transpose(3, 0, 1, 2)
        flow_sitk = sitk.GetImageFromArray(flow_np.transpose(1, 2, 3, 0),
                                           isVector=True)
        flow_sitk.SetSpacing(spacing_mm)
        sitk.WriteImage(flow_sitk,
                        os.path.join(out_dir, f'{patient_id}_flow.nii.gz'))

        print(f'  [{i+1:2d}/{len(samples)}] {patient_id:12s}  '
              f'LV={lv_d:.3f}  MYO={myo_d:.3f}  '
              f'EndoASSD={ea:.2f}mm  EpiASSD={pa:.2f}mm  '
              f'EndoHD95={eh:.2f}mm  EpiHD95={ph:.2f}mm  '
              f'|J|={jac:.3f}%  time={reg_time:.3f}s')

    # Print summary
    print(f'\n  {"-" * 60}')
    print(f'  {name.upper()} RESULTS ({len(samples)} patients)')
    print(f'  {"-" * 60}')
    print(f'  {"Metric":<20s} {"This work":>20s} {"Paper":>20s}')
    print(f'  {"-" * 60}')

    metric_names = {
        'lv_dice':   'LV Dice',
        'myo_dice':  'MYO Dice',
        'endo_assd': 'Endo ASSD (mm)',
        'epi_assd':  'Epi ASSD (mm)',
        'endo_hd':   'Endo HD95 (mm)',
        'epi_hd':    'Epi HD95 (mm)',
        'jac':       '|J|<=0 (%)',
    }

    for k in keys:
        v = np.array(results[k])
        ours = f'{v.mean():.3f} +/- {v.std():.3f}'
        paper = paper_results.get(k, 'N/A')
        print(f'  {metric_names[k]:<20s} {ours:>20s} {paper:>20s}')

    print(f'  {"Inference time":<20s} {np.mean(times):.3f}s / pair')
    print(f'\n  Outputs saved to: {out_dir}')

    return results


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description='DIR-MRVIT test evaluation on sample data')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['acdc', 'mms', 'cmrxm22', 'all'],
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (slower but no CUDA needed)')
    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('  DIR-MRVIT Test Evaluation')
    print('=' * 70)
    print(f'  Device       : {device}')
    print(f'  Sample data  : {SAMPLE_DIR}')
    print(f'  Output       : {OUTPUT_DIR}')
    print(f'  ACDC model   : {ACDC_MODEL}')
    print(f'  M&Ms model   : {MMS_MODEL}')

    # Paper reference values
    paper_acdc = {
        'lv_dice':   '0.917 +/- 0.043',
        'myo_dice':  '0.789 +/- 0.055',
        'endo_assd': '0.79  +/- 0.39',
        'epi_assd':  '0.88  +/- 0.21',
        'endo_hd':   '5.62  +/- 1.22',
        'epi_hd':    '5.51  +/- 1.77',
        'jac':       '0.409 +/- 0.153',
    }
    paper_mms = {
        'lv_dice':   '0.884 +/- 0.038',
        'myo_dice':  '0.729 +/- 0.057',
        'endo_assd': '1.88  +/- 0.79',
        'epi_assd':  '2.05  +/- 0.83',
        'endo_hd':   '7.40  +/- 2.78',
        'epi_hd':    '8.16  +/- 1.95',
        'jac':       '0.536 +/- 0.321',
    }
    paper_cmr = {
        'lv_dice':   '0.892 +/- 0.027',
        'myo_dice':  '0.703 +/- 0.050',
        'endo_assd': '1.78  +/- 0.60',
        'epi_assd':  '1.94  +/- 0.69',
        'endo_hd':   '7.82  +/- 2.31',
        'epi_hd':    '8.07  +/- 2.24',
        'jac':       '0.303 +/- 0.141',
    }

    run_acdc   = args.dataset in ('acdc', 'all')
    run_mms    = args.dataset in ('mms', 'all')
    run_cmrxm22 = args.dataset in ('cmrxm22', 'all')

    # ---- ACDC17 ----
    if run_acdc:
        samples = load_acdc_samples()
        if len(samples) > 0:
            if not os.path.exists(ACDC_MODEL):
                print(f'\n  ERROR: ACDC model not found: {ACDC_MODEL}')
            else:
                model = build_model(ACDC_MODEL, device)
                evaluate_dataset(
                    'acdc', samples, model, device,
                    lv_label=3, myo_label=2,
                    spacing_mm=(5.0, 1.25, 1.25),
                    paper_results=paper_acdc)
                del model
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        else:
            print('\n  No ACDC sample data found.')

    # ---- M&Ms20 ----
    if run_mms:
        samples = load_mms_samples()
        if len(samples) > 0:
            if not os.path.exists(MMS_MODEL):
                print(f'\n  ERROR: M&Ms model not found: {MMS_MODEL}')
            else:
                model = build_model(MMS_MODEL, device)
                evaluate_dataset(
                    'mms', samples, model, device,
                    lv_label=1, myo_label=3,
                    spacing_mm=(8.0, 1.25, 1.25),
                    paper_results=paper_mms)
                del model
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        else:
            print('\n  No M&Ms sample data found.')

    # ---- CMRxM22 ----
    if run_cmrxm22:
        samples = load_cmrxm22_samples()
        if len(samples) > 0:
            if not os.path.exists(MMS_MODEL):
                print(f'\n  ERROR: M&Ms model not found: {MMS_MODEL}')
            else:
                model = build_model(MMS_MODEL, device)
                # CMRxM22 preprocessed with label remap: LV=1, MYO=3 (M&Ms convention)
                evaluate_dataset(
                    'cmrxm22', samples, model, device,
                    lv_label=1, myo_label=3,
                    spacing_mm=(8.0, 1.25, 1.25),
                    paper_results=paper_cmr)
                del model
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        else:
            print('\n  No CMRxM22 sample data found.')

    print(f'\n{"=" * 70}')
    print('  All tests completed.')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
