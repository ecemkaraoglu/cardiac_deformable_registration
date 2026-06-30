"""
cmr_results.py
==============
Run DIR-MRVIT on the CMRxM22 sample data and do two things:

  1. Compute the registration metrics (LV/MYO Dice, endo/epi ASSD, endo/epi
     HD, Jacobian %) for every sample case and print a per-case table plus a
     three-column comparison (sample / full / paper).

  2. Draw the result for one random case as a 2x3 figure: the ES / predicted
     ED / ED images on top, and the ES / predicted ED / ED contours
     (LV endocardium and epicardium) on the bottom.

CMRxM22 is NOT trained on. Following the paper, a model trained on M&Ms20 is
applied directly. Each case is evaluated with all five M&Ms fold models and the
metrics are averaged, which is more stable than a single random fold.

Usage:
    python Code/cmr_results.py
    python Code/cmr_results.py --case P001-1
    python Code/cmr_results.py --slice 6 --seed 7
    python Code/cmr_results.py --fold 0       # single fold instead of averaging
    python Code/cmr_results.py --no-show      # metrics only, no figure
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
HERE       = os.path.dirname(__file__)
SAMPLE_DIR = os.path.join(HERE, "..", "Data", "sample_data", "cmrxm22")
MODEL_DIR  = os.path.join(HERE, "..", "Models_mms_full")   # M&Ms-trained models
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (8.0, 1.25, 1.25)   # (z, y, x), same as M&Ms

# CMRxM22 label convention (differs from M&Ms!):  LV = 1, MYO = 2, RV = 3
LV_LABEL  = 1
MYO_LABEL = 2
RV_LABEL  = 3

ENDO_COLOR = (1.0, 0.55, 0.10)
EPI_COLOR  = (0.20, 0.85, 0.30)
# ================================================


# ---------------- data loading ----------------
def load_case(case_dir):
    """CMRxM22 sample data is already 96x96x16, normalized. File names use -ED / -ED-label."""
    code = os.path.basename(case_dir)
    paths = {
        "ed_img": os.path.join(case_dir, f"{code}-ED.nii.gz"),
        "ed_gt":  os.path.join(case_dir, f"{code}-ED-label.nii.gz"),
        "es_img": os.path.join(case_dir, f"{code}-ES.nii.gz"),
        "es_gt":  os.path.join(case_dir, f"{code}-ES-label.nii.gz"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths["ed_img"])).astype(np.float32)
    ed_gt = sitk.GetArrayFromImage(sitk.ReadImage(paths["ed_gt"])).astype(np.int32)
    es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths["es_img"])).astype(np.float32)
    es_gt = sitk.GetArrayFromImage(sitk.ReadImage(paths["es_gt"])).astype(np.int32)
    return es_img, ed_img, es_gt, ed_gt


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


def warp_label(seg, disp_field, grid):
    """Per-label binary warping: warp each label separately, take arg-max."""
    transform = SpatialTransform_unit().to(DEVICE)
    labels = [LV_LABEL, MYO_LABEL, RV_LABEL]
    probs = np.zeros(seg.shape, dtype=np.float32)
    out = np.zeros(seg.shape, dtype=np.int32)
    for lab in labels:
        binary = (seg == lab).astype(np.float32)
        bt = (torch.from_numpy(binary).float()
              .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            warped = transform(bt, disp_field.permute(0, 2, 3, 4, 1), grid)
        warped_np = warped.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = warped_np > probs
        out[mask] = lab
        probs[mask] = warped_np[mask]
    return out


def warp_image(img, disp_field, grid):
    transform = SpatialTransform_unit().to(DEVICE)
    img_tensor = (torch.from_numpy(img).float()
                  .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
    with torch.no_grad():
        warped = transform(img_tensor, disp_field.permute(0, 2, 3, 4, 1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)


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


def run_one_case(case_dir, models, grid):
    """Evaluate a case with one or more fold models; average the metrics.

    Returns averaged metrics and the arrays from the first model (for plotting).
    """
    es_img, ed_img, es_gt, ed_gt = load_case(case_dir)
    fix = to_tensor(ed_img)
    mov = to_tensor(es_img)

    per_model = []
    first_arrays = None
    for model in models:
        with torch.no_grad():
            disp = model(mov, fix)[0]
        warped = warp_label(es_gt, disp, grid)

        lv_d = dice_bin(warped == LV_LABEL, ed_gt == LV_LABEL)
        myo_d = dice_bin(warped == MYO_LABEL, ed_gt == MYO_LABEL)
        endo_pred = (warped == LV_LABEL).astype(np.uint8)
        endo_gt = (ed_gt == LV_LABEL).astype(np.uint8)
        epi_pred = ((warped == LV_LABEL) | (warped == MYO_LABEL)).astype(np.uint8)
        epi_gt = ((ed_gt == LV_LABEL) | (ed_gt == MYO_LABEL)).astype(np.uint8)
        ea, eh = surface_distances(endo_pred, endo_gt)
        pa, ph = surface_distances(epi_pred, epi_gt)
        jac = jacobian_neg_percentage(disp)
        per_model.append(dict(lv_dice=lv_d, myo_dice=myo_d, endo_assd=ea, epi_assd=pa,
                              endo_hd=eh, epi_hd=ph, jac=jac))
        if first_arrays is None:
            warped_img = warp_image(es_img, disp, grid)
            first_arrays = dict(es_img=es_img, ed_img=ed_img, es_gt=es_gt, ed_gt=ed_gt,
                                warped_seg=warped, warped_img=warped_img)

    keys = ["lv_dice", "myo_dice", "endo_assd", "epi_assd", "endo_hd", "epi_hd", "jac"]
    metrics = {k: float(np.mean([m[k] for m in per_model])) for k in keys}
    return metrics, first_arrays


# ---------------- plotting ----------------
def draw_contours(ax, base_img, seg, title):
    ax.imshow(base_img, cmap="gray")
    epi_mask = ((seg == LV_LABEL) | (seg == MYO_LABEL)).astype(float)
    lv_mask = (seg == LV_LABEL).astype(float)
    # draw epicardium first, then endocardium on top so LV stays visible
    if epi_mask.sum() > 0:
        ax.contour(epi_mask, levels=[0.5], colors=[EPI_COLOR], linewidths=1.5)
    if lv_mask.sum() > 0:
        ax.contour(lv_mask, levels=[0.5], colors=[ENDO_COLOR], linewidths=1.8)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def show_case(case, arrays, slice_idx):
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

    fig.suptitle(f"{case}", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.subplots_adjust(hspace=0.18)
    plt.show()


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="Case to plot, e.g. P001-1.")
    parser.add_argument("--slice", type=int, default=None, help="Slice index to plot.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for case choice.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Use a single M&Ms fold instead of averaging all five.")
    parser.add_argument("--no-show", action="store_true", help="Compute metrics only, no figure.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cases = sorted([os.path.basename(p) for p in glob.glob(os.path.join(SAMPLE_DIR, "*"))
                    if os.path.isdir(p)])
    if not cases:
        sys.exit("No CMRxM22 sample cases found.")

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    keys = ["lv_dice", "myo_dice", "endo_assd", "epi_assd", "endo_hd", "epi_hd", "jac"]

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    models = []
    for f in folds:
        wpath = os.path.join(MODEL_DIR, f"fold{f}_lvl3_best.pth")
        models.append(build_model(wpath))

    all_metrics = []
    plot_cache = {}

    header = f"{'case':10s} {'LV':>7s} {'MYO':>7s} {'eASSD':>7s} {'pASSD':>7s} {'eHD':>7s} {'pHD':>7s} {'Jac%':>7s}"
    mode = f"single M&Ms fold {folds[0]}" if args.fold is not None else "all 5 M&Ms folds averaged"
    print("\n" + "=" * len(header))
    print(f"CMRxM22 sample-data metrics (M&Ms-trained model, {mode})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for code in cases:
        metrics, arrays = run_one_case(os.path.join(SAMPLE_DIR, code), models, grid)
        all_metrics.append(metrics)
        plot_cache[code] = arrays
        print(f"{code:10s} "
              f"{metrics['lv_dice']:7.3f} {metrics['myo_dice']:7.3f} "
              f"{metrics['endo_assd']:7.2f} {metrics['epi_assd']:7.2f} "
              f"{metrics['endo_hd']:7.2f} {metrics['epi_hd']:7.2f} "
              f"{metrics['jac']:7.3f}")
    print("-" * len(header))

    # Reference numbers (mean +/- std):
    #   FULL  = full CMRxM22 result (Results/cmrxm22/cmrxm22_results.csv, fold-pooled)
    #   PAPER = DIR-MRVIT paper, Table 4. HD column reports HD95.
    FULL = {
        "lv_dice": (0.854, 0.067), "myo_dice": (0.756, 0.085),
        "endo_assd": (1.944, 0.863), "epi_assd": (2.629, 0.824),
        "endo_hd": (7.498, 2.123), "epi_hd": (9.393, 2.649),
        "jac": (1.046, 0.505),
    }
    PAPER = {
        "lv_dice": (0.892, 0.027), "myo_dice": (0.703, 0.050),
        "endo_assd": (1.78, 0.60), "epi_assd": (1.94, 0.69),
        "endo_hd": (7.82, 2.31), "epi_hd": (8.07, 2.24),
        "jac": (0.303, 0.141),
    }
    nice = {
        "lv_dice": "LV Dice", "myo_dice": "MYO Dice",
        "endo_assd": "Endo ASSD (mm)", "epi_assd": "Epi ASSD (mm)",
        "endo_hd": "Endo HD (mm)", "epi_hd": "Epi HD (mm)",
        "jac": "Jacobian (%)",
    }

    n = len(all_metrics)
    chdr = f"{'Metric':18s} {'Sample (' + str(n) + ')':>16s} {'Full':>16s} {'Paper':>16s}"
    print("\n" + "=" * len(chdr))
    print("CMRxM22 comparison")
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

    plot_case = args.case or random.choice(cases)
    if plot_case not in plot_cache:
        metrics, arrays = run_one_case(os.path.join(SAMPLE_DIR, plot_case), models, grid)
        plot_cache[plot_case] = arrays

    print(f"Plotting case: {plot_case}")
    show_case(plot_case, plot_cache[plot_case], args.slice)


if __name__ == "__main__":
    main()
