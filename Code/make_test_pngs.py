"""
make_test_pngs.py
=================
Export one slice of a sample case (NIfTI) to four PNG images
(ED, ES, ES_gt, ED_gt), so that external_results.py can be tested end to end
with real cardiac data.

It picks the fullest slice (the one with the most segmentation) and writes:
    ED.png, ES.png, ES_gt.png, ED_gt.png
into the output folder.

PNG is lossless, so the two images keep their grayscale values and the two
segmentations keep their exact label values (0/1/2/3). external_results.py
reads these clean labels directly, so the dataset's LV/MYO/RV convention is
preserved. (PNG is the preferred format for segmentation; JPEG would blur the
label edges.)

This helper understands the file naming of all three datasets:
    M&Ms:  <code>_ED.nii.gz / _ES.nii.gz / _ED_gt.nii.gz / _ES_gt.nii.gz
    CMR:   <code>-ED.nii.gz / -ES.nii.gz / -ED-label.nii.gz / -ES-label.nii.gz
    ACDC:  <code>_frameNN.nii.gz (+ _gt), ED = first frame, ES = last frame

Usage (run from the project root, this file lives in Code/):

    # M&Ms sample case -> PNGs, then evaluate with the M&Ms model:
    python Code/make_test_pngs.py --case_dir Data/sample_data/mms/A0S9V9 --out_dir test_mms
    python Code/external_results.py --in_dir test_mms --dataset mms

    # ACDC sample case:
    python Code/make_test_pngs.py --case_dir Data/sample_data/acdc/patient001 --out_dir test_acdc
    python Code/external_results.py --in_dir test_acdc --dataset acdc

    # CMR sample case:
    python Code/make_test_pngs.py --case_dir Data/sample_data/cmrxm22/P001-1 --out_dir test_cmr
    python Code/external_results.py --in_dir test_cmr --dataset cmr

On Windows use backslashes in paths, e.g. Code\\make_test_pngs.py.
"""

import os
import glob
import argparse

import numpy as np
import SimpleITK as sitk
from PIL import Image

sitk.ProcessObject.GlobalWarningDisplayOff()


def read_vol(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))   # (Z, H, W)


def find(case_dir, suffixes):
    for suf in suffixes:
        hits = glob.glob(os.path.join(case_dir, f"*{suf}"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True, help="Folder of one sample case (NIfTI files).")
    ap.add_argument("--out_dir", required=True, help="Where to write the 4 PNGs.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # match the project's naming across datasets:
    #   M&Ms:  _ED / _ES / _ED_gt / _ES_gt
    #   CMR:   -ED / -ES / -ED-label / -ES-label
    #   ACDC:  _frameNN (ED = lowest frame, ES = highest frame), _frameNN_gt
    ed_p = find(args.case_dir, ["_ED.nii.gz", "-ED.nii.gz"])
    es_p = find(args.case_dir, ["_ES.nii.gz", "-ES.nii.gz"])
    ed_gt_p = find(args.case_dir, ["_ED_gt.nii.gz", "-ED-label.nii.gz", "_ED_label.nii.gz"])
    es_gt_p = find(args.case_dir, ["_ES_gt.nii.gz", "-ES-label.nii.gz", "_ES_label.nii.gz"])

    # ACDC fallback: frames are named *_frameNN.nii.gz with *_frameNN_gt.nii.gz
    if not all([ed_p, es_p, ed_gt_p, es_gt_p]):
        gts = sorted(glob.glob(os.path.join(args.case_dir, "*_frame*_gt.nii.gz")))
        imgs = sorted(f for f in glob.glob(os.path.join(args.case_dir, "*_frame*.nii.gz"))
                      if not f.endswith("_gt.nii.gz"))
        if len(imgs) >= 2 and len(gts) >= 2:
            ed_p, es_p = imgs[0], imgs[-1]          # ED = first frame, ES = last frame
            ed_gt_p, es_gt_p = gts[0], gts[-1]

    if not all([ed_p, es_p, ed_gt_p, es_gt_p]):
        raise SystemExit(f"Could not find all four NIfTI files in {args.case_dir}.\n"
                         f"  ED={ed_p}\n  ES={es_p}\n  ED_gt={ed_gt_p}\n  ES_gt={es_gt_p}")

    ed, es = read_vol(ed_p), read_vol(es_p)
    ed_gt, es_gt = read_vol(ed_gt_p), read_vol(es_gt_p)

    # pick the fullest slice from the ED segmentation
    areas = [(ed_gt[z] > 0).sum() for z in range(ed_gt.shape[0])]
    z = int(np.argmax(areas))
    print(f"Using slice {z} of {ed_gt.shape[0]}.")

    def save_img(arr2d, path):
        a = arr2d.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0
        Image.fromarray(a.astype(np.uint8)).save(path)

    def save_lbl(arr2d, path):
        # PNG is lossless, so the label values (0/1/2/3) are stored as-is and
        # read back exactly. external_results.py reads these clean labels
        # directly, preserving the dataset's LV/MYO/RV convention.
        Image.fromarray(arr2d.astype(np.uint8)).save(path)

    save_img(ed[z],    os.path.join(args.out_dir, "ED.png"))
    save_img(es[z],    os.path.join(args.out_dir, "ES.png"))
    save_lbl(ed_gt[z], os.path.join(args.out_dir, "ED_gt.png"))
    save_lbl(es_gt[z], os.path.join(args.out_dir, "ES_gt.png"))

    print(f"Wrote ED.png, ES.png, ED_gt.png, ES_gt.png to {args.out_dir}")
    print("Label values in ED_gt:", sorted(np.unique(ed_gt[z]).tolist()))


if __name__ == "__main__":
    main()
