"""
Figure A — Qualitative registration results across three datasets.

Three rows (one patient per dataset), four columns each:
  Moving image (ES) | Fixed image (ED) | Warped image | Difference map

Overlays: LV contour (red), MYO contour (green).

All patients use the fold-0 lvl3 checkpoint (footnote: for ACDC and M&Ms this
may include training data; figure is qualitative illustration only).

Run from the project root:
    python Code/figure_a.py

Output: Figures/figure_a.png
"""

import os
import sys

import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = r'C:\Users\ecemk\Desktop\eco\DIR-MRVIT'
CODE_DIR = os.path.join(PROJECT_ROOT, 'Code')
sys.path.insert(0, CODE_DIR)

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
from cross_validation_acdc import load_patient as acdc_load_patient

# ---- Configuration ---------------------------------------------------
START_CHANNEL = 4
RANGE_FLOW    = 0.4

IMGSHAPE   = (96, 96, 16)
IMGSHAPE_2 = (48, 48, 8)
IMGSHAPE_4 = (24, 24, 4)

ACDC_PATIENT = 'patient002'
MMS_PATIENT  = 'A0S9V9'
CMR_PATIENT  = 'P001-1'

ACDC_MODEL = os.path.join(PROJECT_ROOT, 'Models_cv_full',  'fold0_lvl3_best.pth')
MMS_MODEL  = os.path.join(PROJECT_ROOT, 'Models_mms_full', 'fold0_lvl3_best.pth')

# Label conventions — only LV cavity and MYO are visualized.
# ACDC native: LV=3, MYO=2; M&Ms native: LV=1, MYO=3; CMRxM22 native: LV=1, MYO=2
ACDC_LV, ACDC_MYO = 3, 2
MMS_LV,  MMS_MYO  = 1, 3
CMR_LV,  CMR_MYO  = 1, 2

OUT_DIR = os.path.join(PROJECT_ROOT, 'Figures')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ---- Tensor helpers --------------------------------------------------
def to_tensor(arr):
    """numpy (Z, Y, X) -> torch tensor (1, 1, Y, X, Z) on DEVICE."""
    return (
        torch.from_numpy(arr)
        .float()
        .permute(1, 2, 0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


def load_nifti_zyx(path):
    """Read a NIfTI file and return numpy in (Z, Y, X) layout."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


# ---- Per-dataset loaders --------------------------------------------
def load_acdc(patient):
    """ACDC uses on-the-fly preprocessing via cross_validation_acdc."""
    patient_dir = os.path.join(PROJECT_ROOT, 'Data', 'training', patient)
    ed_img, es_img, ed_gt, es_gt = acdc_load_patient(patient_dir)
    return ed_img, ed_gt.astype(np.int32), es_img, es_gt.astype(np.int32)


def load_mms(patient):
    base = os.path.join(PROJECT_ROOT, 'Data', 'MMs_preprocessed', patient)
    ed_img = load_nifti_zyx(os.path.join(base, f'{patient}_ED.nii.gz'))
    ed_gt  = load_nifti_zyx(os.path.join(base, f'{patient}_ED_gt.nii.gz')).astype(np.int32)
    es_img = load_nifti_zyx(os.path.join(base, f'{patient}_ES.nii.gz'))
    es_gt  = load_nifti_zyx(os.path.join(base, f'{patient}_ES_gt.nii.gz')).astype(np.int32)
    return ed_img, ed_gt, es_img, es_gt


def load_cmr(patient):
    """CMRxM22 preprocessing already remapped labels to M&Ms convention.
    However, label files keep CMR-native names; we read them as-is and use
    the CMR label constants for plotting."""
    base = os.path.join(PROJECT_ROOT, 'Data', 'CMRxM22_preprocessed', patient)
    ed_img = load_nifti_zyx(os.path.join(base, f'{patient}-ED.nii.gz'))
    ed_gt  = load_nifti_zyx(os.path.join(base, f'{patient}-ED-label.nii.gz')).astype(np.int32)
    es_img = load_nifti_zyx(os.path.join(base, f'{patient}-ES.nii.gz'))
    es_gt  = load_nifti_zyx(os.path.join(base, f'{patient}-ES-label.nii.gz')).astype(np.int32)
    return ed_img, ed_gt, es_img, es_gt


# ---- Model builder (identical to evaluate_cmrxm22.build_eval_model) -
def build_eval_model(weights_path):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=IMGSHAPE_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    patch_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=IMGSHAPE_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=IMGSHAPE_2,
        range_flow=RANGE_FLOW, model_lvl1=model_lvl1,
        patch_model_lv2=patch_lv2).to(DEVICE)
    patch_lv3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=IMGSHAPE_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(DEVICE)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=IMGSHAPE,
        range_flow=RANGE_FLOW, model_lvl2=model_lvl2,
        patch_model=patch_lv3).to(DEVICE)
    model_lvl3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model_lvl3.eval()
    return model_lvl3


# ---- Warping using SpatialTransform_unit ----------------------------
def warp_image(moving_img_np, disp, grid):
    """Warp the moving image with the unit-normalized displacement field."""
    transform = SpatialTransform_unit().to(DEVICE)
    moving_t = to_tensor(moving_img_np)
    disp_p = disp.permute(0, 2, 3, 4, 1)
    with torch.no_grad():
        warped = transform(moving_t, disp_p, grid)
    # warped is (1, 1, Y, X, Z); convert back to (Z, Y, X)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)


def warp_segmentation(seg, disp, grid):
    """Per-label binary warping (same idea as evaluate_cmrxm22.warp_segmentation)."""
    transform = SpatialTransform_unit().to(DEVICE)
    disp_p = disp.permute(0, 2, 3, 4, 1)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    result = np.zeros(seg.shape, dtype=np.int32)
    probs  = np.zeros(seg.shape, dtype=np.float32)
    for label in unique_labels:
        binary = (seg == label).astype(np.float32)
        bin_t = (
            torch.from_numpy(binary)
            .float()
            .permute(1, 2, 0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            warped_bin = transform(bin_t, disp_p, grid)
        warped_np = warped_bin.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = warped_np > probs
        result[mask] = label
        probs[mask] = warped_np[mask]
    return result


# ---- Registration -----------------------------------------------------
def register(ed_img, es_img, weights_path, grid):
    model = build_eval_model(weights_path)
    fixed  = to_tensor(ed_img)
    moving = to_tensor(es_img)
    with torch.no_grad():
        disp = model(moving, fixed)[0]
    return disp


# ---- Plotting --------------------------------------------------------
def overlay_contours(ax, img, gt, lv_label, myo_label, title=''):
    """Plot the slice with two contours:
      - LV-Endo (pink/orange): boundary of LV cavity (label == lv_label)
      - LV-Epi  (green): boundary of LV cavity + myocardium union
    Matches the convention used in the DIR-MRVIT paper Figure 7.
    """
    ax.imshow(img, cmap='gray')
    if gt is not None:
        endo = (gt == lv_label).astype(np.float32)
        epi  = ((gt == lv_label) | (gt == myo_label)).astype(np.float32)
        ax.contour(epi,  levels=[0.5], colors='#34c759', linewidths=1.0)
        ax.contour(endo, levels=[0.5], colors='#ff6b35', linewidths=1.0)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def difference_map(ax, fixed, warped, title=''):
    diff = np.abs(fixed - warped)
    vmax = max(diff.max(), 1e-6)
    ax.imshow(diff, cmap='hot', vmin=0, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# ---- Per-dataset pipeline -------------------------------------------
def process(name, patient, loader, weights, lv_label, myo_label, grid):
    print(f'\n[{name}] loading patient {patient}...')
    ed_img, ed_gt, es_img, es_gt = loader(patient)
    print(f'  ed_img shape={ed_img.shape}, '
          f'ed_gt unique={np.unique(ed_gt)}, es_gt unique={np.unique(es_gt)}')

    print(f'[{name}] registering with {os.path.basename(weights)}...')
    disp = register(ed_img, es_img, weights, grid)

    print(f'[{name}] warping image and segmentation...')
    warped_img = warp_image(es_img, disp, grid)
    warped_gt  = warp_segmentation(es_gt, disp, grid)

    # Pick the mid-slice for the figure
    z = ed_img.shape[0] // 2
    return {
        'name':       name,
        'patient':    patient,
        'moving':     es_img[z],
        'fixed':      ed_img[z],
        'warped':     warped_img[z],
        'moving_gt':  es_gt[z],
        'fixed_gt':   ed_gt[z],
        'warped_gt':  warped_gt[z],
        'lv':         lv_label,
        'myo':        myo_label,
    }


def main():
    # Unit-normalized grid (same as evaluate_cmrxm22.py)
    grid = generate_grid_unit(IMGSHAPE)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    rows = [
        process('ACDC',    ACDC_PATIENT, load_acdc, ACDC_MODEL, ACDC_LV, ACDC_MYO, grid),
        process('M&Ms',    MMS_PATIENT,  load_mms,  MMS_MODEL,  MMS_LV,  MMS_MYO,  grid),
        process('CMRxM22', CMR_PATIENT,  load_cmr,  MMS_MODEL,  CMR_LV,  CMR_MYO,  grid),
    ]

    col_titles = ['Moving (ES)', 'Fixed (ED)', 'Warped (estimated)', 'Difference map']
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for r, data in enumerate(rows):
        overlay_contours(
            axes[r, 0], data['moving'], data['moving_gt'],
            data['lv'], data['myo'],
            title=col_titles[0] if r == 0 else '',
        )
        overlay_contours(
            axes[r, 1], data['fixed'], data['fixed_gt'],
            data['lv'], data['myo'],
            title=col_titles[1] if r == 0 else '',
        )
        overlay_contours(
            axes[r, 2], data['warped'], data['warped_gt'],
            data['lv'], data['myo'],
            title=col_titles[2] if r == 0 else '',
        )
        difference_map(
            axes[r, 3], data['fixed'], data['warped'],
            title=col_titles[3] if r == 0 else '',
        )
        axes[r, 0].set_ylabel(
            f"{data['name']}\n({data['patient']})",
            fontsize=11, rotation=0, labelpad=42, va='center',
        )

    legend_handles = [
        Patch(facecolor='none', edgecolor='#ff6b35', label='Endocardium (LV-Endo)'),
        Patch(facecolor='none', edgecolor='#34c759', label='Epicardium (LV-Epi)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=2,
        bbox_to_anchor=(0.5, 0.0), fontsize=10, frameon=False,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    plt.show()


if __name__ == '__main__':
    main()