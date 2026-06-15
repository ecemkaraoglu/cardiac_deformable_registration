"""
demo_figure.py
==============
Random qualitative demo figure for DIR-MRVIT.

Picks ONE random patient from each of the three sample datasets
(Data/sample_data/{acdc,mms,cmrxm22}/) and produces a 3x4 figure:

    rows    : ACDC / M&Ms / CMRxM22
    columns : Moving (ES) | Fixed (ED) | Warped (estimated) | Difference map

Contours overlaid:
    orange = LV endocardium (inner LV boundary)
    green  = LV epicardium  (outer myocardial boundary, LV + MYO union)

Usage:
    python Code/demo_figure.py            # GPU if available, else CPU
    python Code/demo_figure.py --cpu      # force CPU mode
"""

import os
import sys
import glob
import random
import argparse

import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (
    Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
    Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    SpatialTransform_unit,
)
from Functions import generate_grid_unit

# ==================== CONFIG ====================
START_CHANNEL = 4
RANGE_FLOW    = 0.4

IMGSHAPE   = (96, 96, 16)
IMGSHAPE_2 = (48, 48, 8)
IMGSHAPE_4 = (24, 24, 4)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(BASE_DIR, 'Data', 'sample_data')
ACDC_MODEL = os.path.join(BASE_DIR, 'Models_cv_full',  'fold0_lvl3_best.pth')
MMS_MODEL  = os.path.join(BASE_DIR, 'Models_mms_full', 'fold0_lvl3_best.pth')

ENDO_COLOR = '#ff6b35'   # orange
EPI_COLOR  = '#34c759'   # green
# ================================================


# -------------------- Model --------------------
def build_model(weights_path, device):
    m1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True,
        imgshape=IMGSHAPE_4, range_flow=RANGE_FLOW).to(device)
    pl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True,
        patch_shape=IMGSHAPE_4, range_flow=RANGE_FLOW).to(device)
    m2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=IMGSHAPE_2,
        range_flow=RANGE_FLOW,
        model_lvl1=m1, patch_model_lv2=pl2).to(device)
    pl3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=IMGSHAPE_2,
        range_flow=RANGE_FLOW, patch_model=pl2).to(device)
    m3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=IMGSHAPE,
        range_flow=RANGE_FLOW,
        model_lvl2=m2, patch_model=pl3).to(device)
    m3.load_state_dict(torch.load(weights_path, map_location=device,
                                  weights_only=True))
    m3.eval()
    return m3


# -------------------- Preprocessing for raw ACDC --------------------
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


def load_acdc_patient(patient_dir):
    """Load and preprocess one raw ACDC patient."""
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
        img_crop = crop_around_center(img_arr, center, IMGSHAPE)
        gt_crop  = crop_around_center(gt_arr,  center, IMGSHAPE)
        return img_crop, gt_crop, center

    ed_img, ed_gt, center = process(img_files[0], gt_files[0])
    es_img, es_gt, _      = process(img_files[1], gt_files[1], center)
    return ed_img, es_img, ed_gt, es_gt


def load_preprocessed_patient(patient_dir, dataset):
    """Load a patient from M&Ms or CMRxM22 preprocessed folder."""
    code = os.path.basename(patient_dir)
    if dataset == 'mms':
        paths = [os.path.join(patient_dir, f'{code}_{x}.nii.gz')
                 for x in ['ED', 'ED_gt', 'ES', 'ES_gt']]
    else:  # cmrxm22
        paths = [os.path.join(patient_dir, f'{code}-{x}.nii.gz')
                 for x in ['ED', 'ED-label', 'ES', 'ES-label']]
    if not all(os.path.exists(p) for p in paths):
        return None
    ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[0])).astype(np.float32)
    ed_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[1])).astype(np.int32)
    es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[2])).astype(np.float32)
    es_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[3])).astype(np.int32)
    return ed_img, es_img, ed_gt, es_gt


# -------------------- Warping --------------------
def to_tensor(arr, device):
    return (torch.from_numpy(arr).float()
            .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device))


def warp_segmentation(seg, disp_field, grid, device):
    """Per-label binary warping — no hallucinated labels at boundaries."""
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


def warp_image(img, disp_field, grid, device):
    transform     = SpatialTransform_unit().to(device)
    disp_permuted = disp_field.permute(0, 2, 3, 4, 1)
    mov_t = to_tensor(img, device)
    with torch.no_grad():
        warped = transform(mov_t, disp_permuted, grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)


# -------------------- Random selection --------------------
def pick_random_patient(dataset):
    """Pick one random patient directory from sample_data/<dataset>/."""
    dataset_dir = os.path.join(SAMPLE_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        return None
    candidates = sorted([
        d for d in glob.glob(os.path.join(dataset_dir, '*'))
        if os.path.isdir(d)
    ])
    if not candidates:
        return None
    return random.choice(candidates)


# -------------------- Per-dataset processing --------------------
def process_dataset(dataset, lv_label, myo_label, device):
    """Load random patient, run model, return display arrays + contours."""
    patient_dir = pick_random_patient(dataset)
    if patient_dir is None:
        return None
    patient_id = os.path.basename(patient_dir)

    # Load
    if dataset == 'acdc':
        loaded = load_acdc_patient(patient_dir)
        model_path = ACDC_MODEL
    else:
        loaded = load_preprocessed_patient(patient_dir, dataset)
        model_path = MMS_MODEL
    if loaded is None:
        return None
    ed_img, es_img, ed_gt, es_gt = loaded

    # Simple slice selection: slice with most LV in ED
    lv_per_slice = np.array([(ed_gt[z] == lv_label).sum()
                             for z in range(ed_gt.shape[0])])
    if lv_per_slice.max() == 0:
        print(f'    skipping {patient_id}: no LV in ED')
        return None
    z = int(np.argmax(lv_per_slice))

    # Build and run model
    model = build_model(model_path, device)
    grid  = generate_grid_unit(IMGSHAPE)
    grid  = torch.from_numpy(
        np.reshape(grid, (1,) + grid.shape)).to(device).float()

    fix = to_tensor(ed_img, device)
    mov = to_tensor(es_img, device)
    with torch.no_grad():
        disp = model(mov, fix)[0]

    warped_img = warp_image(es_img, disp, grid, device)
    warped_seg = warp_segmentation(es_gt, disp, grid, device)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'    {patient_id} slice z={z}')

    # Raw contour masks — no filtering
    moving_endo = (es_gt[z] == lv_label)
    moving_epi  = (es_gt[z] == lv_label) | (es_gt[z] == myo_label)
    fixed_endo  = (ed_gt[z] == lv_label)
    fixed_epi   = (ed_gt[z] == lv_label) | (ed_gt[z] == myo_label)
    warped_endo = (warped_seg[z] == lv_label)
    warped_epi  = (warped_seg[z] == lv_label) | (warped_seg[z] == myo_label)

    diff = np.abs(warped_img[z] - ed_img[z])

    return {
        'patient_id': patient_id,
        'moving_img': es_img[z],
        'fixed_img':  ed_img[z],
        'warped_img': warped_img[z],
        'diff_map':   diff,
        'moving_endo': moving_endo, 'moving_epi': moving_epi,
        'fixed_endo':  fixed_endo,  'fixed_epi':  fixed_epi,
        'warped_endo': warped_endo, 'warped_epi': warped_epi,
    }


# -------------------- Plotting --------------------
def draw_panel(ax, image, endo_mask=None, epi_mask=None, cmap='gray',
               vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if epi_mask is not None and epi_mask.any():
        ax.contour(epi_mask.astype(float),  levels=[0.5],
                   colors=[EPI_COLOR],  linewidths=1.6)
    if endo_mask is not None and endo_mask.any():
        ax.contour(endo_mask.astype(float), levels=[0.5],
                   colors=[ENDO_COLOR], linewidths=2.0)
    ax.set_xticks([])
    ax.set_yticks([])


def make_figure(results):
    rows = [r for r in results if r is not None]
    if len(rows) == 0:
        print('No data available for any dataset; aborting.')
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.5 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    col_titles = ['Moving (ES)', 'Fixed (ED)',
                  'Warped (estimated)', 'Difference map']
    row_labels = ['ACDC', 'M&Ms', 'CMRxM22']

    diff_vmax = max(r['diff_map'].max() for r in rows) * 0.8 + 1e-6

    for i, r in enumerate(rows):
        draw_panel(axes[i, 0], r['moving_img'],
                   r['moving_endo'], r['moving_epi'])
        draw_panel(axes[i, 1], r['fixed_img'],
                   r['fixed_endo'],  r['fixed_epi'])
        draw_panel(axes[i, 2], r['warped_img'],
                   r['warped_endo'], r['warped_epi'])
        draw_panel(axes[i, 3], r['diff_map'],
                   cmap='hot', vmin=0, vmax=diff_vmax)

        axes[i, 0].set_ylabel(
            f'{row_labels[i]}\n({r["patient_id"]})',
            rotation=0, labelpad=55, fontsize=11, va='center')

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11)

    legend_handles = [
        Patch(facecolor='none', edgecolor=ENDO_COLOR,
              label='Endocardium (LV-Endo)'),
        Patch(facecolor='none', edgecolor=EPI_COLOR,
              label='Epicardium (LV-Epi)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description='DIR-MRVIT random qualitative demo figure')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (slower but no CUDA needed)')
    args = parser.parse_args()

    device = (torch.device('cpu') if args.cpu
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    print('=' * 60)
    print('  DIR-MRVIT Demo Figure (random sample per dataset)')
    print('=' * 60)
    print(f'  Device     : {device}')
    print(f'  Sample dir : {SAMPLE_DIR}')

    if not os.path.exists(ACDC_MODEL):
        print(f'\n  ERROR: ACDC model not found at {ACDC_MODEL}')
        return
    if not os.path.exists(MMS_MODEL):
        print(f'\n  ERROR: M&Ms model not found at {MMS_MODEL}')
        return

    results = []

    print('\n  Picking random patient: ACDC ...')
    r = process_dataset('acdc', lv_label=3, myo_label=2, device=device)
    if r:
        print(f'    selected: {r["patient_id"]}')
    results.append(r)

    print('\n  Picking random patient: M&Ms ...')
    r = process_dataset('mms', lv_label=1, myo_label=3, device=device)
    if r:
        print(f'    selected: {r["patient_id"]}')
    results.append(r)

    print('\n  Picking random patient: CMRxM22 ...')
    r = process_dataset('cmrxm22', lv_label=1, myo_label=3, device=device)
    if r:
        print(f'    selected: {r["patient_id"]}')
    results.append(r)

    print('\n  Building figure ...')
    make_figure(results)


if __name__ == '__main__':
    main()