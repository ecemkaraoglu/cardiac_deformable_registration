"""
mms_results.py
==============
Run the DIR-MRVIT model on the M&Ms20 sample data and do two things:

  1. Compute the registration metrics (LV/MYO Dice, endo/epi ASSD, endo/epi
     HD, Jacobian %) for every sample subject and print a per-subject table
     plus a three-column comparison (sample / full 150 / paper).

  2. Draw the result for one random subject as a 2x3 figure: the ES / predicted
     ED / ED images on top, and the ES / predicted ED / ED contours
     (LV endocardium and epicardium) on the bottom.

Fold assignment:
  M&Ms folds are defined over the full 150-subject list (shuffled with SEED).
  If the full preprocessed set is present (Data/MMs_preprocessed/), each sample
  subject is evaluated with the fold model whose test set it belongs to. If only
  the sample data is available, a single fold model is used (set with --fold),
  and this is stated in the output.

Usage:
    python Code/mms_results.py
    python Code/mms_results.py --subject A0S9V9
    python Code/mms_results.py --slice 6 --seed 7
    python Code/mms_results.py --fold 0          # fold model when full set absent
    python Code/mms_results.py --no-show         # metrics only, no figure
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
SAMPLE_DIR = os.path.join(HERE, "..", "Data", "sample_data", "mms")
FULL_DIR   = os.path.join(HERE, "..", "Data", "MMs_preprocessed")
MODEL_DIR  = os.path.join(HERE, "..", "Models_mms_full")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

SPACING_MM = (8.0, 1.25, 1.25)   # (z, y, x), M&Ms resampling

# M&Ms label convention:  LV = 1, RV = 2, MYO = 3
LV_LABEL  = 1
RV_LABEL  = 2
MYO_LABEL = 3

# Contour colors
ENDO_COLOR = (1.0, 0.55, 0.10)   # LV endocardium (orange)
EPI_COLOR  = (0.20, 0.85, 0.30)  # LV epicardium  (green)
# ================================================


# ---------------- data loading ----------------
def load_subject(subject_dir):
    """M&Ms sample/preprocessed data is already 96x96x16, normalized. Just read."""
    code = os.path.basename(subject_dir)
    paths = [os.path.join(subject_dir, f"{code}_{x}.nii.gz")
             for x in ["ED", "ED_gt", "ES", "ES_gt"]]
    if not all(os.path.exists(p) for p in paths):
        return None
    ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[0])).astype(np.float32)
    ed_gt = sitk.GetArrayFromImage(sitk.ReadImage(paths[1])).astype(np.int32)
    es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[2])).astype(np.float32)
    es_gt = sitk.GetArrayFromImage(sitk.ReadImage(paths[3])).astype(np.int32)
    return es_img, ed_img, es_gt, ed_gt


# ---------------- fold assignment (matches cross_validation_mms.py) ----------------
def build_fold_map():
    """Map each full-set subject code to the fold whose TEST set contains it.

    Returns {} if the full preprocessed set is not present.
    """
    if not os.path.isdir(FULL_DIR):
        return {}
    codes = sorted(os.path.basename(d) for d in glob.glob(os.path.join(FULL_DIR, "*"))
                   if os.path.isdir(d))
    n = len(codes)
    if n == 0:
        return {}
    rng = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)
    test_size = n // N_FOLDS
    fold_of = {}
    for fold in range(N_FOLDS):
        start = fold * test_size
        # last fold takes the remainder too
        end = start + test_size if fold < N_FOLDS - 1 else n
        for i in order[start:end]:
            fold_of[codes[i]] = fold
    return fold_of


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
    labels = [LV_LABEL, RV_LABEL, MYO_LABEL]
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


def run_one_subject(subject_dir, model, grid):
    es_img, ed_img, es_gt, ed_gt = load_subject(subject_dir)

    fix = to_tensor(ed_img)
    mov = to_tensor(es_img)
    with torch.no_grad():
        disp = model(mov, fix)[0]

    warped = warp_label(es_gt, disp, grid)
    warped_img = warp_image(es_img, disp, grid)

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


def show_subject(subject, arrays, slice_idx, save_path=None):
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

    fig.suptitle(f"{subject}", fontsize=12)
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
    parser.add_argument("--subject", default=None, help="Subject to plot, e.g. A0S9V9.")
    parser.add_argument("--slice", type=int, default=None, help="Slice index to plot.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for subject choice.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold model to use when the full set is absent (default 0).")
    parser.add_argument("--no-show", action="store_true", help="Compute metrics only, no figure.")
    parser.add_argument("--save", default=None, help="Save the figure to this path instead of showing it.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    subjects = sorted([os.path.basename(p) for p in glob.glob(os.path.join(SAMPLE_DIR, "*"))
                       if os.path.isdir(p)])
    if not subjects:
        sys.exit("No M&Ms sample subjects found.")

    fold_map = build_fold_map()
    use_full_mapping = len(fold_map) > 0

    keys = ["lv_dice", "myo_dice", "endo_assd", "epi_assd", "endo_hd", "epi_hd", "jac"]

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    model_cache = {}

    def get_model(fold):
        if fold not in model_cache:
            wpath = os.path.join(MODEL_DIR, f"fold{fold}_lvl3_best.pth")
            model_cache[fold] = build_model(wpath)
        return model_cache[fold]

    all_metrics = []
    plot_cache = {}

    header = f"{'subject':12s} {'fold':>4s} {'LV':>7s} {'MYO':>7s} {'eASSD':>7s} {'pASSD':>7s} {'eHD':>7s} {'pHD':>7s} {'Jac%':>7s}"
    print("\n" + "=" * len(header))
    if use_full_mapping:
        print("M&Ms sample-data metrics (each subject on its own fold model)")
    else:
        print(f"M&Ms sample-data metrics (single fold model: fold {args.fold})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for code in subjects:
        fold = fold_map.get(code, args.fold) if use_full_mapping else args.fold
        model = get_model(fold)
        metrics, arrays = run_one_subject(os.path.join(SAMPLE_DIR, code), model, grid)
        all_metrics.append(metrics)
        plot_cache[code] = arrays
        print(f"{code:12s} {fold:>4d} "
              f"{metrics['lv_dice']:7.3f} {metrics['myo_dice']:7.3f} "
              f"{metrics['endo_assd']:7.2f} {metrics['epi_assd']:7.2f} "
              f"{metrics['endo_hd']:7.2f} {metrics['epi_hd']:7.2f} "
              f"{metrics['jac']:7.3f}")
    print("-" * len(header))

    # Reference numbers (mean +/- std):
    #   FULL  = full 150-subject M&Ms result (Models_mms_full/eval_mms_independent.npy)
    #   PAPER = DIR-MRVIT paper, Table 3. HD column reports HD95.
    FULL = {
        "lv_dice": (0.877, 0.050), "myo_dice": (0.790, 0.070),
        "endo_assd": (1.620, 0.923), "epi_assd": (2.081, 0.736),
        "endo_hd": (7.153, 2.769), "epi_hd": (8.744, 2.277),
        "jac": (0.275, 0.195),
    }
    PAPER = {
        "lv_dice": (0.884, 0.038), "myo_dice": (0.729, 0.057),
        "endo_assd": (1.88, 0.79), "epi_assd": (2.05, 0.83),
        "endo_hd": (7.40, 2.78), "epi_hd": (8.16, 1.95),
        "jac": (0.536, 0.321),
    }
    nice = {
        "lv_dice": "LV Dice", "myo_dice": "MYO Dice",
        "endo_assd": "Endo ASSD (mm)", "epi_assd": "Epi ASSD (mm)",
        "endo_hd": "Endo HD (mm)", "epi_hd": "Epi HD (mm)",
        "jac": "Jacobian (%)",
    }

    n = len(all_metrics)
    chdr = f"{'Metric':18s} {'Sample (' + str(n) + ')':>16s} {'Full (150)':>16s} {'Paper':>16s}"
    print("\n" + "=" * len(chdr))
    print("M&Ms comparison")
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

    plot_subject = args.subject or random.choice(subjects)
    if plot_subject not in plot_cache:
        fold = fold_map.get(plot_subject, args.fold) if use_full_mapping else args.fold
        metrics, arrays = run_one_subject(os.path.join(SAMPLE_DIR, plot_subject), get_model(fold), grid)
        plot_cache[plot_subject] = arrays

    print(f"Plotting subject: {plot_subject}")
    show_subject(plot_subject, plot_cache[plot_subject], args.slice, args.save)


if __name__ == "__main__":
    main()