"""
external_results.py
===================
Run the DIR-MRVIT registration model on an EXTERNAL case supplied as 2D JPEG
(or PNG) images. The script converts the images to the project's data format,
runs a trained model, and reports the registration metrics and a figure -- all
in one command. It lets a new case be tried without touching the datasets that
ship with the repository.

--------------------------------------------------------------------------
INPUT
--------------------------------------------------------------------------
One folder containing up to four images (JPEG or PNG). File names are matched
case-insensitively; common spellings (ED-gt, ED_label, ed.jpeg, ...) are all
accepted:

    ED.jpg      end-diastole image          (REQUIRED)
    ES.jpg      end-systole  image          (REQUIRED)
    ES_gt.jpg   end-systole  segmentation   (REQUIRED)
    ED_gt.jpg   end-diastole segmentation   (OPTIONAL)

Why these requirements:
    - ED and ES images are the two inputs the model registers.
    - ES_gt is the segmentation the model WARPS with the estimated field to
      produce the predicted ED segmentation. Without it there is nothing to
      propagate, so it is required.
    - ED_gt is only the comparison target for the metrics. If it is missing,
      the model still runs and the figure is drawn, but the metrics (Dice,
      ASSD, HD) cannot be computed.

--------------------------------------------------------------------------
DATASET FLAG
--------------------------------------------------------------------------
Each of the three datasets in this project uses a different trained model and
a different label convention. Choose which one to apply with --dataset:

    --dataset acdc   ACDC17 model  (Models_cv_full),  labels LV=3, MYO=2, RV=1
    --dataset mms    M&Ms20 model  (Models_mms_full), labels LV=1, MYO=3, RV=2
    --dataset cmr    M&Ms20 model  (Models_mms_full), labels LV=1, MYO=2, RV=3

The label convention is the one the SEGMENTATION images use. Make sure the GT
images you supply follow the convention of the dataset you select, otherwise
the metrics will measure the wrong structures. The script prints the label
values it found so the mapping can be checked.

--------------------------------------------------------------------------
NOTES / LIMITATIONS
--------------------------------------------------------------------------
    - The model is 3D (96 x 96 x 16). A single 2D slice has no through-plane
      information, so the slice is repeated to fill the volume. In-plane
      registration still works, but through-plane (z) metrics are synthetic.
    - A segmentation stored as JPEG is lossy (compression blurs the label
      edges). The script clusters the gray levels back into labels, but PNG
      is strongly preferred for segmentation images.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Put this file in the Code/ folder and run it from the project root. The
--in_dir folder must contain the case images (ED, ES, ES_gt, and optionally
ED_gt). Pick the model with --dataset (acdc / mms / cmr).

    # M&Ms model:
    python Code/external_results.py --in_dir path/to/case --dataset mms

    # ACDC model:
    python Code/external_results.py --in_dir path/to/case --dataset acdc

    # CMR setting (M&Ms model), average all five folds:
    python Code/external_results.py --in_dir path/to/case --dataset cmr --fold all

    # choose a specific fold model, and save the figure instead of showing it:
    python Code/external_results.py --in_dir path/to/case --dataset mms --fold 2 --save out.png

    # metrics only, no figure window:
    python Code/external_results.py --in_dir path/to/case --dataset mms --no-show

    # pick which slice of the 16-slice volume to draw:
    python Code/external_results.py --in_dir path/to/case --dataset mms --slice 8

On Windows use backslashes in paths, e.g. Code\\external_results.py.
"""

import os
import sys
import argparse

import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
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
HERE   = os.path.dirname(__file__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_CHANNEL = 4
RANGE_FLOW    = 0.4

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

TARGET_HW = (96, 96)
N_SLICES  = 16

# Per-dataset settings: model directory, spacing, and label convention.
DATASETS = {
    "acdc": {
        "model_dir": os.path.join(HERE, "..", "Models_cv_full"),
        "spacing":   (5.0, 1.25, 1.25),   # (z, y, x)
        "labels":    {"LV": 3, "MYO": 2, "RV": 1},
    },
    "mms": {
        "model_dir": os.path.join(HERE, "..", "Models_mms_full"),
        "spacing":   (8.0, 1.25, 1.25),
        "labels":    {"LV": 1, "MYO": 3, "RV": 2},
    },
    "cmr": {
        "model_dir": os.path.join(HERE, "..", "Models_mms_full"),   # M&Ms-trained
        "spacing":   (8.0, 1.25, 1.25),
        "labels":    {"LV": 1, "MYO": 2, "RV": 3},
    },
}

ENDO_COLOR = (1.0, 0.55, 0.10)   # LV endocardium
EPI_COLOR  = (0.20, 0.85, 0.30)  # LV epicardium

EXTS = [".jpg", ".jpeg", ".png"]
PATTERNS = {
    "ed_img": ["ed", "ed_image", "ed-img"],
    "es_img": ["es", "es_image", "es-img"],
    "es_gt":  ["es_gt", "es-gt", "es_label", "es-label", "esgt", "es_seg"],
    "ed_gt":  ["ed_gt", "ed-gt", "ed_label", "ed-label", "edgt", "ed_seg"],
}
# ================================================


# ---------------- file finding + JPEG conversion ----------------
def find_file(folder, keys):
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in EXTS]
    for key in keys:
        for f in files:
            if os.path.splitext(f)[0].lower() == key:
                return os.path.join(folder, f)
    for key in keys:
        for f in files:
            if key in os.path.splitext(f)[0].lower():
                return os.path.join(folder, f)
    return None


def load_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def resize_slice(arr, is_label):
    mode = Image.NEAREST if is_label else Image.BILINEAR
    img = Image.fromarray(arr).resize((TARGET_HW[1], TARGET_HW[0]), mode)
    return np.asarray(img).astype(np.float32)


def to_volume(slice2d):
    return np.repeat(slice2d[np.newaxis, :, :], N_SLICES, axis=0)


def convert_image(path):
    arr = resize_slice(load_gray(path), is_label=False)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return to_volume(arr).astype(np.float32)


def convert_gt(path, name):
    """Grayscale GT -> integer-label volume, robust to JPEG compression.

    Handles three cases:
      1. Clean label image with only values {0,1,2,3}  -> used directly.
      2. Labels encoded as gray = label * 60  (0/60/120/180). JPEG cannot move
         a pixel across these wide bands, and the original label numbers are
         recovered as round(gray / 60). This is the recommended way to store a
         JPEG segmentation, because it keeps the dataset's LV/MYO/RV numbering.
      3. Any other gray-level encoding -> the dominant bands are clustered and
         mapped to 0,1,2,... by brightness (a best effort; the printed mapping
         should be checked against the dataset's label convention).
    """
    arr = resize_slice(load_gray(path), is_label=True)
    rounded = np.round(arr).astype(int)
    uniq = np.unique(rounded)
    total = arr.size

    # Case 1: already clean labels.
    allowed = {0, 1, 2, 3}
    if set(int(v) for v in uniq).issubset(allowed):
        print(f"  [{name}] distinct values: {[int(v) for v in uniq]} -> used as labels directly.")
        return to_volume(rounded.astype(np.int32)).astype(np.int32)

    # Case 2: gray = label * 60 encoding. Check that every dominant gray value
    # is close to a multiple of 60 within tolerance.
    dominant = [int(v) for v in uniq if (rounded == v).sum() / total > 0.002]
    def near_mult60(v):
        return abs(v - round(v / 60.0) * 60) <= 12
    if dominant and all(near_mult60(v) for v in dominant) and max(dominant) <= 200:
        # Which labels are actually present (as multiples of 60).
        present = sorted(set(int(round(v / 60.0)) for v in dominant))
        present_grays = np.array([p * 60 for p in present])
        # Snap every pixel to the nearest PRESENT band, so JPEG transition
        # pixels between two bands do not create a third, non-existent label.
        idx = np.abs(arr[..., None] - present_grays[None, None, :]).argmin(axis=-1)
        label = np.array(present, dtype=np.int32)[idx]
        print(f"  [{name}] detected gray = label*60 encoding; recovered labels {present}.")
        return to_volume(label).astype(np.int32)

    # Case 3: unknown gray-level encoding -> cluster into bands.
    hist = np.bincount(rounded.ravel(), minlength=256).astype(float)
    smooth = np.convolve(hist, np.ones(11) / 11.0, mode="same")
    centers = []
    for v in range(256):
        if smooth[v] <= 0:
            continue
        lo = max(0, v - 5); hi = min(256, v + 6)
        if smooth[v] == smooth[lo:hi].max() and hist[max(0, v - 8):v + 9].sum() / total > 0.01:
            centers.append(v)
    merged = []
    for c in sorted(centers):
        if merged and c - merged[-1] < 20:
            continue
        merged.append(c)
    if not merged:
        merged = [0]
    centers_arr = np.array(merged)
    clean_centers = [int(c) for c in centers_arr]
    nearest = np.abs(rounded[..., None] - centers_arr[None, None, :]).argmin(axis=-1)
    order = np.argsort(centers_arr)
    center_to_label = {int(centers_arr[o]): i for i, o in enumerate(order)}
    label = np.zeros_like(rounded, dtype=np.int32)
    for idx, c in enumerate(centers_arr):
        label[nearest == idx] = center_to_label[int(c)]
    label[label > 3] = 0
    print(f"  [{name}] JPEG is lossy; clustered gray levels into bands at {clean_centers}.")
    print(f"  [{name}] gray band -> label (by brightness): "
          f"{ {int(c): center_to_label[int(c)] for c in centers_arr} }")
    print(f"  [{name}] NOTE: this brightness order may not match the dataset's "
          f"LV/MYO/RV numbering; verify, or store the GT as gray = label*60.")
    return to_volume(label).astype(np.int32)


def load_case(in_dir):
    paths = {k: find_file(in_dir, keys) for k, keys in PATTERNS.items()}
    print("Found files:")
    for k, p in paths.items():
        print(f"  {k:7s}: {os.path.basename(p) if p else '(missing)'}")

    if paths["ed_img"] is None or paths["es_img"] is None:
        sys.exit("ERROR: ED and ES images are both required (ED.jpg and ES.jpg).")
    if paths["es_gt"] is None:
        sys.exit("ERROR: ES_gt is required. It is the segmentation that gets warped "
                 "to produce the prediction; without it there is nothing to propagate.")

    ed_img = convert_image(paths["ed_img"])
    es_img = convert_image(paths["es_img"])
    es_gt = convert_gt(paths["es_gt"], "ES_gt")
    ed_gt = convert_gt(paths["ed_gt"], "ED_gt") if paths["ed_gt"] else None
    return es_img, ed_img, es_gt, ed_gt


# ---------------- metrics ----------------
def to_tensor(arr):
    return (torch.from_numpy(arr).float()
            .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))


def dice_bin(pred_bin, target_bin):
    p = pred_bin.astype(np.float32); t = target_bin.astype(np.float32)
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * (p * t).sum() / denom


def surface_distances(pred_bin, target_bin, spacing_mm):
    ps = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    ts = sitk.GetImageFromArray(target_bin.astype(np.uint8))
    ps.SetSpacing(spacing_mm[::-1]); ts.SetSpacing(spacing_mm[::-1])
    df = sitk.SignedMaurerDistanceMapImageFilter()
    df.SetSquaredDistance(False); df.SetUseImageSpacing(True)
    dp = sitk.GetArrayFromImage(df.Execute(ps))
    dt = sitk.GetArrayFromImage(df.Execute(ts))
    ps_surf = sitk.GetArrayFromImage(sitk.LabelContour(ps)).astype(bool)
    ts_surf = sitk.GetArrayFromImage(sitk.LabelContour(ts)).astype(bool)
    d_p2t = np.abs(dt[ps_surf]); d_t2p = np.abs(dp[ts_surf])
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


def warp_label(seg, disp_field, grid, labels):
    transform = SpatialTransform_unit().to(DEVICE)
    probs = np.zeros(seg.shape, dtype=np.float32)
    out = np.zeros(seg.shape, dtype=np.int32)
    for lab in [labels["LV"], labels["RV"], labels["MYO"]]:
        binary = (seg == lab).astype(np.float32)
        bt = (torch.from_numpy(binary).float()
              .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            warped = transform(bt, disp_field.permute(0, 2, 3, 4, 1), grid)
        wn = warped.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = wn > probs
        out[mask] = lab; probs[mask] = wn[mask]
    return out


def warp_image(img, disp_field, grid):
    transform = SpatialTransform_unit().to(DEVICE)
    it = (torch.from_numpy(img).float()
          .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
    with torch.no_grad():
        warped = transform(it, disp_field.permute(0, 2, 3, 4, 1), grid)
    return warped.squeeze().cpu().numpy().transpose(2, 0, 1)


# ---------------- model ----------------
def build_model(weights_path):
    m1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    p2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4, range_flow=RANGE_FLOW).to(DEVICE)
    m2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2, range_flow=RANGE_FLOW,
        model_lvl1=m1, patch_model_lv2=p2).to(DEVICE)
    p3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2, range_flow=RANGE_FLOW,
        patch_model=p2).to(DEVICE)
    m3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape, range_flow=RANGE_FLOW,
        model_lvl2=m2, patch_model=p3).to(DEVICE)
    m3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m3.eval()
    return m3


# ---------------- plotting ----------------
def draw_contours(ax, base_img, seg, title, labels):
    ax.imshow(base_img, cmap="gray")
    epi = ((seg == labels["LV"]) | (seg == labels["MYO"])).astype(float)
    lv = (seg == labels["LV"]).astype(float)
    if epi.sum() > 0:
        ax.contour(epi, levels=[0.5], colors=[EPI_COLOR], linewidths=1.5)
    if lv.sum() > 0:
        ax.contour(lv, levels=[0.5], colors=[ENDO_COLOR], linewidths=1.8)
    ax.set_title(title, fontsize=12); ax.axis("off")


def show_figure(name, es_img, ed_img, warped_img, es_gt, warped_seg, ed_gt,
                labels, slice_idx, save_path):
    es_img = np.transpose(es_img, (1, 2, 0))
    ed_img = np.transpose(ed_img, (1, 2, 0))
    warped_img = np.transpose(warped_img, (1, 2, 0))
    es_gt = np.transpose(es_gt, (1, 2, 0))
    warped_seg = np.transpose(warped_seg, (1, 2, 0))
    ed_gt_t = np.transpose(ed_gt, (1, 2, 0)) if ed_gt is not None else None

    if slice_idx is None:
        areas = [(es_gt[:, :, z] > 0).sum() for z in range(es_gt.shape[2])]
        slice_idx = int(np.argmax(areas))
    z = max(0, min(slice_idx, ed_img.shape[2] - 1))
    print(f"Showing slice {z}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes[0, 0].imshow(es_img[:, :, z], cmap="gray"); axes[0, 0].set_title("ES image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(warped_img[:, :, z], cmap="gray"); axes[0, 1].set_title("predicted ED image"); axes[0, 1].axis("off")
    axes[0, 2].imshow(ed_img[:, :, z], cmap="gray"); axes[0, 2].set_title("ED image"); axes[0, 2].axis("off")

    draw_contours(axes[1, 0], es_img[:, :, z], es_gt[:, :, z], "ES GT", labels)
    draw_contours(axes[1, 1], warped_img[:, :, z], warped_seg[:, :, z], "predicted ED GT", labels)
    if ed_gt_t is not None:
        draw_contours(axes[1, 2], ed_img[:, :, z], ed_gt_t[:, :, z], "ED GT", labels)
    else:
        axes[1, 2].imshow(ed_img[:, :, z], cmap="gray")
        axes[1, 2].set_title("ED GT (not provided)"); axes[1, 2].axis("off")

    legend = [Line2D([0], [0], color=ENDO_COLOR, lw=2, label="LV endocardium"),
              Line2D([0], [0], color=EPI_COLOR, lw=2, label="LV epicardium (LV+MYO)")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96]); fig.subplots_adjust(hspace=0.18)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Run a trained DIR-MRVIT model on an external JPEG/PNG case.")
    parser.add_argument("--in_dir", required=True,
                        help="Folder with the case images (ED, ES, ES_gt, and optionally ED_gt).")
    parser.add_argument("--dataset", required=True, choices=["acdc", "mms", "cmr"],
                        help="Which trained model and label convention to use.")
    parser.add_argument("--fold", default="0",
                        help="Fold model index 0-4, or 'all' to average all five folds "
                             "(default 0). 'all' is intended for --dataset cmr.")
    parser.add_argument("--slice", type=int, default=None, help="Slice index of the volume to draw.")
    parser.add_argument("--save", default=None, help="Save the figure to this path instead of showing it.")
    parser.add_argument("--no-show", action="store_true", help="Compute metrics only, no figure.")
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        sys.exit(f"Input folder not found: {args.in_dir}")

    cfg = DATASETS[args.dataset]
    labels = cfg["labels"]
    spacing = cfg["spacing"]
    model_dir = cfg["model_dir"]
    name = os.path.basename(os.path.normpath(args.in_dir))

    print(f"\nExternal case: {name}")
    print(f"Dataset setting: {args.dataset}  "
          f"(labels LV={labels['LV']}, MYO={labels['MYO']}, RV={labels['RV']})")

    es_img, ed_img, es_gt, ed_gt = load_case(args.in_dir)

    # which fold model(s)
    if str(args.fold).lower() == "all":
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [int(args.fold)]

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    models = []
    for f in folds:
        wpath = os.path.join(model_dir, f"fold{f}_lvl3_best.pth")
        if not os.path.exists(wpath):
            sys.exit(f"Model checkpoint not found: {wpath}")
        models.append(build_model(wpath))
    mode = f"fold {folds[0]}" if len(folds) == 1 else "average of all 5 folds"
    print(f"Using {args.dataset} model ({mode}).")

    # register (moving = ES, fixed = ED); average metrics across models
    fix = to_tensor(ed_img); mov = to_tensor(es_img)
    per_model = []
    first = None
    for model in models:
        with torch.no_grad():
            disp = model(mov, fix)[0]
        warped_seg = warp_label(es_gt, disp, grid, labels)
        if first is None:
            first = (warped_seg, warp_image(es_img, disp, grid))
        if ed_gt is not None:
            lv_d = dice_bin(warped_seg == labels["LV"], ed_gt == labels["LV"])
            myo_d = dice_bin(warped_seg == labels["MYO"], ed_gt == labels["MYO"])
            ea, eh = surface_distances((warped_seg == labels["LV"]).astype(np.uint8),
                                       (ed_gt == labels["LV"]).astype(np.uint8), spacing)
            epi_p = ((warped_seg == labels["LV"]) | (warped_seg == labels["MYO"])).astype(np.uint8)
            epi_g = ((ed_gt == labels["LV"]) | (ed_gt == labels["MYO"])).astype(np.uint8)
            pa, ph = surface_distances(epi_p, epi_g, spacing)
            jac = jacobian_neg_percentage(disp)
            per_model.append((lv_d, myo_d, ea, pa, eh, ph, jac))

    warped_seg, warped_img = first

    if ed_gt is not None:
        arr = np.array(per_model)
        m = arr.mean(axis=0)
        print("\n" + "=" * 46)
        print(f"Metrics for external case: {name}")
        print("=" * 46)
        print(f"  LV Dice        : {m[0]:.3f}")
        print(f"  MYO Dice       : {m[1]:.3f}")
        print(f"  Endo ASSD (mm) : {m[2]:.2f}")
        print(f"  Epi ASSD (mm)  : {m[3]:.2f}")
        print(f"  Endo HD (mm)   : {m[4]:.2f}")
        print(f"  Epi HD (mm)    : {m[5]:.2f}")
        print(f"  Jacobian (%)   : {m[6]:.3f}")
        print("=" * 46)
        print("Note: the input is a single 2D slice repeated into a 3D volume,")
        print("so treat these as in-plane values; through-plane (z) is synthetic.\n")
    else:
        print("\n" + "=" * 60)
        print("ED_gt was not provided, so the metrics cannot be computed.")
        print("The deformation field was still estimated and the ES")
        print("segmentation was warped; the figure below shows the result.")
        print("=" * 60 + "\n")

    if args.no_show:
        return
    show_figure(name, es_img, ed_img, warped_img, es_gt, warped_seg, ed_gt,
                labels, args.slice, args.save)


if __name__ == "__main__":
    main()