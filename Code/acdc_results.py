"""
acdc_results.py
===============
Run the DIR-MRVIT model on the ACDC sample data and do two things:

  1. Compute the registration metrics (LV/MYO Dice, endo/epi ASSD, endo/epi
     HD95, non-positive Jacobian %) for every sample patient, each evaluated
     with the fold model whose test set it belongs to, and print a per-patient
     table plus a mean +/- std summary.

  2. Draw the result for one random patient as a 2x3 figure: the ES / predicted
     ED / ED images on top, and the ES / predicted ED / ED contours
     (LV endocardium and epicardium) on the bottom.

The predicted ED segmentation shown in the figure is the same warped ES
segmentation used to compute the metrics, so the picture and the numbers match.

Usage:
    python Code/acdc_results.py
    python Code/acdc_results.py --patient patient021
    python Code/acdc_results.py --slice 6 --seed 7
    python Code/acdc_results.py --no-show     # metrics only, no figure
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
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2,
                   SpatialTransform_unit)
from Functions import generate_grid_unit, transform_unit_flow_to_flow_cuda

# ==================== CONFIG ====================
HERE      = os.path.dirname(__file__)
ACDC_DIR  = os.path.join(HERE, "..", "Data", "sample_data", "acdc")
MODEL_DIR = os.path.join(HERE, "..", "Models_cv_full")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (5.0, 1.25, 1.25)   # (z, y, x), ACDC resampling
TARGET_SPACING = (1.25, 1.25, 5.0)

LV_LABEL  = 3
MYO_LABEL = 2

# Contour colors
ENDO_COLOR = (1.0, 0.55, 0.10)   # LV endocardium (orange)
EPI_COLOR  = (0.20, 0.85, 0.30)  # LV epicardium  (green)
# ================================================


# ---------------- preprocessing (same as training) ----------------
def resample_image(image, new_spacing, is_label=False):
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [int(round(orig_size[i] * orig_spacing[i] / new_spacing[i])) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(new_spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(image.GetDirection())
    r.SetOutputOrigin(image.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return r.Execute(image)


def crop_around_center(arr, center, crop_size):
    cz, ch, cw = center
    sz, sh, sw = crop_size[2], crop_size[0], crop_size[1]

    def get_slice(c, size, max_size):
        start = c - size // 2
        start = max(0, min(start, max_size - size)) if max_size >= size else 0
        return start, start + size

    z0, z1 = get_slice(cz, sz, arr.shape[0])
    h0, h1 = get_slice(ch, sh, arr.shape[1])
    w0, w1 = get_slice(cw, sw, arr.shape[2])
    cropped = arr[z0:z1, h0:h1, w0:w1]
    pad_z = sz - cropped.shape[0]
    pad_h = sh - cropped.shape[1]
    pad_w = sw - cropped.shape[2]
    if pad_z > 0 or pad_h > 0 or pad_w > 0:
        cropped = np.pad(cropped,
                         ((0, max(0, pad_z)), (0, max(0, pad_h)), (0, max(0, pad_w))),
                         mode="constant")
    return cropped


def preprocess_acdc_patient(patient_dir):
    """Return ES image, ED image, ES gt, ED gt on the 96x96x16 grid.

    img_files[0] = ED (frame01), img_files[1] = ES (later frame).
    """
    files = sorted(glob.glob(os.path.join(patient_dir, "*_frame*.nii.gz")))
    img_files = [f for f in files if "_gt" not in f]
    gt_files = [f for f in files if "_gt" in f]
    if len(img_files) < 2 or len(gt_files) < 2:
        return None

    def process(img_path, gt_path, center=None):
        img_sitk = resample_image(sitk.ReadImage(img_path), TARGET_SPACING, is_label=False)
        gt_sitk = resample_image(sitk.ReadImage(gt_path), TARGET_SPACING, is_label=True)
        img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        gt_arr = sitk.GetArrayFromImage(gt_sitk).astype(np.int32)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
        if center is None:
            lv_mask = (gt_arr == LV_LABEL)
            if lv_mask.sum() == 0:
                lv_mask = (gt_arr > 0)
            coords = np.where(lv_mask)
            center = ([int(np.mean(c)) for c in coords]
                      if len(coords[0]) > 0 else [s // 2 for s in gt_arr.shape])
        return crop_around_center(img_arr, center, imgshape), \
            crop_around_center(gt_arr, center, imgshape), center

    ed_img, ed_gt, center = process(img_files[0], gt_files[0])
    es_img, es_gt, _ = process(img_files[1], gt_files[1], center)
    return es_img, ed_img, es_gt, ed_gt


# ---------------- fold assignment (same split as training) ----------------
def patient_to_fold(patient_index, n_patients=100):
    """Return the fold whose TEST set contains this patient index (0-based).

    Reproduces the stratified split used in cross_validation_acdc.py.
    """
    rng = random.Random(SEED)
    group_size = n_patients // N_FOLDS   # 20
    groups = []
    for g in range(N_FOLDS):
        indices = list(range(g * group_size, (g + 1) * group_size))
        rng.shuffle(indices)
        groups.append(indices)

    test_per_group = group_size // N_FOLDS   # 4
    for fold in range(N_FOLDS):
        test_idx = []
        for g in range(N_FOLDS):
            start = fold * test_per_group
            test_idx.extend(groups[g][start:start + test_per_group])
        if patient_index in test_idx:
            return fold
    return 0  # fallback


def patient_index_from_name(name):
    """patient001 -> 0, patient021 -> 20, etc."""
    return int(name.replace("patient", "")) - 1


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
    pred_sitk = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    target_sitk = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_mm[::-1])
    target_sitk.SetSpacing(spacing_mm[::-1])

    df = sitk.SignedMaurerDistanceMapImageFilter()
    df.SetSquaredDistance(False)
    df.SetUseImageSpacing(True)

    dist_pred = sitk.GetArrayFromImage(df.Execute(pred_sitk))
    dist_target = sitk.GetArrayFromImage(df.Execute(target_sitk))

    pred_surf = sitk.GetArrayFromImage(sitk.LabelContour(pred_sitk)).astype(bool)
    tgt_surf = sitk.GetArrayFromImage(sitk.LabelContour(target_sitk)).astype(bool)

    d_p2t = np.abs(dist_target[pred_surf])
    d_t2p = np.abs(dist_pred[tgt_surf])
    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return 0.0, 0.0

    assd = (d_p2t.sum() + d_t2p.sum()) / (len(d_p2t) + len(d_t2p))
    hd95 = float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))
    return float(assd), hd95


def jacobian_neg_percentage(disp_field):
    flow = transform_unit_flow_to_flow_cuda(disp_field.permute(0, 2, 3, 4, 1).clone())
    flow = flow.squeeze(0).cpu().numpy()
    dvx = np.gradient(flow[..., 0], axis=(0, 1, 2))
    dvy = np.gradient(flow[..., 1], axis=(0, 1, 2))
    dvz = np.gradient(flow[..., 2], axis=(0, 1, 2))
    J = np.zeros(flow.shape[:3] + (3, 3))
    J[..., 0, 0] = dvx[0] + 1; J[..., 0, 1] = dvx[1];     J[..., 0, 2] = dvx[2]
    J[..., 1, 0] = dvy[0];     J[..., 1, 1] = dvy[1] + 1; J[..., 1, 2] = dvy[2]
    J[..., 2, 0] = dvz[0];     J[..., 2, 1] = dvz[1];     J[..., 2, 2] = dvz[2] + 1
    return 100.0 * float(np.mean(np.linalg.det(J) <= 0))


def warp_segmentation(seg, disp_field, grid):
    transform = SpatialTransform_unit().to(DEVICE)
    seg_tensor = (torch.from_numpy(seg).float()
                  .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
    with torch.no_grad():
        warped = transform(seg_tensor, disp_field.permute(0, 2, 3, 4, 1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)   # (z, y, x)


# ---------------- model builder ----------------
def build_model(weights_path):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True,
        imgshape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    patch_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True,
        patch_shape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW,
        model_lvl1=model_lvl1, patch_model_lv2=patch_lv2).to(DEVICE)
    patch_lv3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=patch_lv2).to(DEVICE)
    model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW,
        model_lvl2=model_lvl2, patch_model=patch_lv3).to(DEVICE)
    model_lvl3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model_lvl3.eval()
    return model_lvl3


def warp_image(img, disp_field, grid):
    """Warp an intensity image with the deformation field (bilinear)."""
    transform = SpatialTransform_unit().to(DEVICE)
    img_tensor = (torch.from_numpy(img).float()
                  .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
    with torch.no_grad():
        warped = transform(img_tensor, disp_field.permute(0, 2, 3, 4, 1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)   # (z, y, x)


def run_one_patient(patient_dir, model, grid):
    """Run the model and return metrics + arrays needed for plotting."""
    es_img, ed_img, es_gt, ed_gt = preprocess_acdc_patient(patient_dir)

    fix = to_tensor(ed_img)
    mov = to_tensor(es_img)
    with torch.no_grad():
        disp = model(mov, fix)[0]

    warped = np.round(warp_segmentation(es_gt, disp, grid)).astype(np.int32)
    warped_img = warp_image(es_img, disp, grid)   # predicted ED image

    lv_d = dice_bin(warped == LV_LABEL, ed_gt == LV_LABEL)
    myo_d = dice_bin(warped == MYO_LABEL, ed_gt == MYO_LABEL)
    endo_pred = (warped == LV_LABEL).astype(np.uint8)
    endo_gt = (ed_gt == LV_LABEL).astype(np.uint8)
    epi_pred = ((warped == LV_LABEL) | (warped == MYO_LABEL)).astype(np.uint8)
    epi_gt = ((ed_gt == LV_LABEL) | (ed_gt == MYO_LABEL)).astype(np.uint8)
    ea, eh = surface_distances(endo_pred, endo_gt)
    pa, ph = surface_distances(epi_pred, epi_gt)
    jac = jacobian_neg_percentage(disp)

    metrics = dict(lv_dice=lv_d, myo_dice=myo_d, endo_assd=ea, epi_assd=pa,
                   endo_hd=eh, epi_hd=ph, jac=jac)
    arrays = dict(es_img=es_img, ed_img=ed_img, es_gt=es_gt, ed_gt=ed_gt,
                  warped_seg=warped, warped_img=warped_img)
    return metrics, arrays


# ---------------- plotting ----------------
def draw_contours(ax, base_img, seg, title):
    ax.imshow(base_img, cmap="gray")
    lv_mask = (seg == LV_LABEL).astype(float)
    epi_mask = ((seg == LV_LABEL) | (seg == MYO_LABEL)).astype(float)
    if lv_mask.sum() > 0:
        ax.contour(lv_mask, levels=[0.5], colors=[ENDO_COLOR], linewidths=1.5)
    if epi_mask.sum() > 0:
        ax.contour(epi_mask, levels=[0.5], colors=[EPI_COLOR], linewidths=1.5)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def show_patient(patient, arrays, metrics, slice_idx, save_path=None):
    es_img = np.transpose(arrays["es_img"], (1, 2, 0))
    ed_img = np.transpose(arrays["ed_img"], (1, 2, 0))
    es_gt = np.transpose(arrays["es_gt"], (1, 2, 0))
    ed_gt = np.transpose(arrays["ed_gt"], (1, 2, 0))
    warped = np.transpose(arrays["warped_seg"], (1, 2, 0))
    warped_img = np.transpose(arrays["warped_img"], (1, 2, 0))

    if slice_idx is None:
        areas = [(ed_gt[:, :, z] > 0).sum() for z in range(ed_gt.shape[2])]
        slice_idx = int(np.argmax(areas))
    z = max(0, min(slice_idx, ed_img.shape[2] - 1))
    print(f"Showing slice {z}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    axes[0, 0].imshow(es_img[:, :, z], cmap="gray"); axes[0, 0].set_title("ES image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(warped_img[:, :, z], cmap="gray"); axes[0, 1].set_title("predicted ED image"); axes[0, 1].axis("off")
    axes[0, 2].imshow(ed_img[:, :, z], cmap="gray"); axes[0, 2].set_title("ED image"); axes[0, 2].axis("off")

    draw_contours(axes[1, 0], es_img[:, :, z], es_gt[:, :, z], "ES GT")
    draw_contours(axes[1, 1], warped_img[:, :, z], warped[:, :, z], "predicted ED GT")
    draw_contours(axes[1, 2], ed_img[:, :, z], ed_gt[:, :, z], "ED GT")

    legend = [
        Line2D([0], [0], color=ENDO_COLOR, lw=2, label="LV endocardium"),
        Line2D([0], [0], color=EPI_COLOR, lw=2, label="LV epicardium (LV+MYO)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10, frameon=False)

    fig.suptitle(f"{patient}", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.subplots_adjust(hspace=0.18)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default=None, help="Patient to plot, e.g. patient021.")
    parser.add_argument("--slice", type=int, default=None, help="Slice index to plot.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for patient choice.")
    parser.add_argument("--no-show", action="store_true", help="Compute metrics only, no figure.")
    parser.add_argument("--save", default=None, help="Save the figure to this path instead of showing it.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    patients = sorted([os.path.basename(p) for p in glob.glob(os.path.join(ACDC_DIR, "patient*"))])
    if not patients:
        sys.exit("No ACDC sample patients found.")

    keys = ["lv_dice", "myo_dice", "endo_assd", "epi_assd", "endo_hd", "epi_hd", "jac"]
    units = {"endo_assd": "mm", "epi_assd": "mm", "endo_hd": "mm", "epi_hd": "mm", "jac": "%"}

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    # cache models per fold so we load each checkpoint once
    model_cache = {}
    all_metrics = []
    plot_cache = {}

    header = f"{'patient':12s} {'fold':>4s} {'LV':>7s} {'MYO':>7s} {'eASSD':>7s} {'pASSD':>7s} {'eHD':>7s} {'pHD':>7s} {'Jac%':>7s}"
    print("\n" + "=" * len(header))
    print("ACDC sample-data metrics (each patient on its own fold model)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in patients:
        idx = patient_index_from_name(name)
        fold = patient_to_fold(idx)
        if fold not in model_cache:
            wpath = os.path.join(MODEL_DIR, f"fold{fold}_lvl3_best.pth")
            model_cache[fold] = build_model(wpath)
        metrics, arrays = run_one_patient(os.path.join(ACDC_DIR, name), model_cache[fold], grid)
        all_metrics.append(metrics)
        plot_cache[name] = (arrays, metrics)
        print(f"{name:12s} {fold:>4d} "
              f"{metrics['lv_dice']:7.3f} {metrics['myo_dice']:7.3f} "
              f"{metrics['endo_assd']:7.2f} {metrics['epi_assd']:7.2f} "
              f"{metrics['endo_hd']:7.2f} {metrics['epi_hd']:7.2f} "
              f"{metrics['jac']:7.3f}")

    print("-" * len(header))

    # Reference numbers (mean +/- std):
    #   FULL  = full 100-patient ACDC result (Models_cv_full/eval_acdc_independent.npy)
    #   PAPER = DIR-MRVIT paper, Table 2. HD column reports HD95.
    FULL = {
        "lv_dice": (0.887, 0.069), "myo_dice": (0.752, 0.065),
        "endo_assd": (1.509, 1.033), "epi_assd": (1.082, 0.512),
        "endo_hd": (5.270, 2.428), "epi_hd": (4.399, 2.012),
        "jac": (0.129, 0.163),
    }
    PAPER = {
        "lv_dice": (0.917, 0.043), "myo_dice": (0.789, 0.055),
        "endo_assd": (0.79, 0.39), "epi_assd": (0.88, 0.21),
        "endo_hd": (5.62, 1.22), "epi_hd": (5.51, 1.77),
        "jac": (0.409, 0.153),
    }
    nice = {
        "lv_dice": "LV Dice", "myo_dice": "MYO Dice",
        "endo_assd": "Endo ASSD (mm)", "epi_assd": "Epi ASSD (mm)",
        "endo_hd": "Endo HD (mm)", "epi_hd": "Epi HD (mm)",
        "jac": "Jacobian (%)",
    }

    n = len(all_metrics)
    chdr = f"{'Metric':18s} {'Sample (' + str(n) + ')':>16s} {'Full (100)':>16s} {'Paper':>16s}"
    print("\n" + "=" * len(chdr))
    print("ACDC comparison")
    print("=" * len(chdr))
    print(chdr)
    print("-" * len(chdr))
    for k in keys:
        vals = [m[k] for m in all_metrics]
        s_m, s_s = np.mean(vals), np.std(vals)
        f_m, f_s = FULL[k]
        p_m, p_s = PAPER[k]
        print(f"{nice[k]:18s} "
              f"{s_m:6.3f} +/- {s_s:5.3f} "
              f"{f_m:6.3f} +/- {f_s:5.3f} "
              f"{p_m:6.3f} +/- {p_s:5.3f}")
    print("=" * len(chdr) + "\n")

    if args.no_show:
        return

    plot_patient = args.patient or random.choice(patients)
    if plot_patient not in plot_cache:
        # patient given but maybe not in sample; run it now
        idx = patient_index_from_name(plot_patient)
        fold = patient_to_fold(idx)
        if fold not in model_cache:
            model_cache[fold] = build_model(os.path.join(MODEL_DIR, f"fold{fold}_lvl3_best.pth"))
        metrics, arrays = run_one_patient(os.path.join(ACDC_DIR, plot_patient), model_cache[fold], grid)
        plot_cache[plot_patient] = (arrays, metrics)

    print(f"Plotting patient: {plot_patient}")
    arrays, metrics = plot_cache[plot_patient]
    show_patient(plot_patient, arrays, metrics, args.slice, args.save)


if __name__ == "__main__":
    main()