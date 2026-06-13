"""
Figure B — Deformation field visualization for one ACDC case.

Three rows (basal, mid-ventricle, apical slice), two columns each:
  Displacement magnitude heatmap | Quiver plot of the deformation field

Highlights the spatial structure of the model-estimated cardiac motion.

Run from the project root:
    python Code/figure_b.py
"""

import os
import sys

import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt

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
)
from Functions import generate_grid_unit, transform_unit_flow_to_flow_cuda
from cross_validation_acdc import load_patient as acdc_load_patient

# ---- Configuration ---------------------------------------------------
START_CHANNEL = 4
RANGE_FLOW    = 0.4

IMGSHAPE   = (96, 96, 16)
IMGSHAPE_2 = (48, 48, 8)
IMGSHAPE_4 = (24, 24, 4)

PATIENT = 'patient002'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Models_cv_full', 'fold0_lvl3_best.pth')

# Three slices: basal, mid-ventricle, apical
# Volume shape is (Z, Y, X) = (16, 96, 96), so Z=0..15
SLICES = [2, 8, 14]
SLICE_NAMES = ['Basal', 'Mid-ventricle', 'Apical']

# Quiver downsampling: arrow every QUIVER_STRIDE voxels in Y and X
QUIVER_STRIDE = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ---- Helpers ---------------------------------------------------------
def to_tensor(arr):
    """numpy (Z, Y, X) -> torch (1, 1, Y, X, Z) on DEVICE."""
    return (
        torch.from_numpy(arr)
        .float()
        .permute(1, 2, 0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


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


# ---- Main ------------------------------------------------------------
def main():
    # Load patient
    print(f'Loading ACDC {PATIENT}...')
    patient_dir = os.path.join(PROJECT_ROOT, 'Data', 'training', PATIENT)
    ed_img, es_img, ed_gt, es_gt = acdc_load_patient(patient_dir)
    print(f'  ed_img shape: {ed_img.shape}')

    # Run model to get deformation field
    print(f'Loading model {os.path.basename(MODEL_PATH)}...')
    model = build_eval_model(MODEL_PATH)

    fixed  = to_tensor(ed_img)
    moving = to_tensor(es_img)

    print('Running registration...')
    with torch.no_grad():
        disp = model(moving, fixed)[0]   # (1, 3, Y, X, Z), unit-normalized

    # Convert unit-normalized flow to voxel-unit flow
    # transform_unit_flow_to_flow_cuda multiplies each axis by (size-1)/2
    disp_voxel = transform_unit_flow_to_flow_cuda(
        disp.permute(0, 2, 3, 4, 1).clone()
    )
    # disp_voxel shape: (1, Y, X, Z, 3), each entry is voxel displacement
    disp_voxel = disp_voxel.squeeze(0).cpu().numpy()  # (Y, X, Z, 3)
    # Reorder so we have (Z, Y, X, 3) like the image arrays
    disp_voxel = disp_voxel.transpose(2, 0, 1, 3)

    # Per-voxel magnitude
    magnitude = np.linalg.norm(disp_voxel, axis=-1)  # (Z, Y, X)
    print(f'  displacement magnitude: '
          f'min={magnitude.min():.2f}, max={magnitude.max():.2f}, '
          f'mean={magnitude.mean():.2f}')

    # Use a single colormap scale for all three slices, so they're comparable
    vmax = float(magnitude.max())

    # Plot 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))

    for row, (z, name) in enumerate(zip(SLICES, SLICE_NAMES)):
        # ---- Left column: magnitude heatmap ----
        ax_mag = axes[row, 0]
        ax_mag.imshow(ed_img[z], cmap='gray')
        im = ax_mag.imshow(magnitude[z], cmap='viridis',
                           vmin=0, vmax=vmax, alpha=0.6)
        ax_mag.set_title(
            f'{name} (z={z})  -  Displacement magnitude' if row == 0
            else 'Displacement magnitude',
            fontsize=10,
        )
        ax_mag.set_xticks([]); ax_mag.set_yticks([])
        if row == 0:
            cbar = fig.colorbar(im, ax=ax_mag, fraction=0.046, pad=0.04)
            cbar.set_label('voxels', fontsize=9)

        # ---- Right column: quiver plot ----
        ax_quiv = axes[row, 1]
        ax_quiv.imshow(ed_img[z], cmap='gray')

        Y, X = ed_img.shape[1], ed_img.shape[2]
        yy, xx = np.meshgrid(
            np.arange(0, Y, QUIVER_STRIDE),
            np.arange(0, X, QUIVER_STRIDE),
            indexing='ij',
        )
        # Sample displacement at downsampled grid
        # disp_voxel[z, yy, xx] has shape (n, n, 3) with components (y, x, z)
        # For 2D quiver, we want X-direction (right) and Y-direction (down)
        u = disp_voxel[z, yy, xx, 1]   # X component
        v = disp_voxel[z, yy, xx, 0]   # Y component

        ax_quiv.quiver(
            xx, yy, u, v,
            color='#ff6b35', angles='xy', scale_units='xy', scale=1.0,
            width=0.004, headwidth=4, headlength=4,
        )
        ax_quiv.set_title(
            f'{name} (z={z})  -  Deformation field' if row == 0
            else 'Deformation field',
            fontsize=10,
        )
        ax_quiv.set_xticks([]); ax_quiv.set_yticks([])
        # Row label
        ax_mag.set_ylabel(name, fontsize=11, rotation=0, labelpad=40, va='center')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
