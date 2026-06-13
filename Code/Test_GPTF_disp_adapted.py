"""
Test_GPTF_disp_adapted.py
==========================
Orijinal Test_GPTF_disp.py'nin bizim data formatina uyarlanmis versiyonu.

Orijinalden degistirilen tek sey: data loading loop.
Model kurma, forward pass, spatial transform, flow kaydetme — hepsi orijinal ile ayni.

Orijinal script: pat001..150 / info.txt / p001_fr01.nii.gz formatini bekliyordu.
Bu script   : Data/MMs_preprocessed/<code>/<code>_ED.nii.gz formatini okuyor.

Kullanim:
    cd DIR-MRVIT
    python Code/Test_GPTF_disp_adapted.py ^
        --modelpath Models_mms_full/fold0_lvl3_best.pth ^
        --savepath  Results/test_gptf_original ^
        --datapath  Data/MMs_preprocessed ^
        --start_channel 4
"""

import os, glob, sys
from argparse import ArgumentParser
import numpy as np
import SimpleITK as sitk
import torch
import time

sys.path.insert(0, '.')
sitk.ProcessObject.GlobalWarningDisplayOff()

from Functions import generate_grid_unit, transform_unit_flow_to_flow
from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   SpatialTransform_unit, SpatialTransformNearest_unit,
                   smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2)

# ---- argument parser (orijinal ile ayni) ----
parser = ArgumentParser()
parser.add_argument("--modelpath", type=str, dest="modelpath",
                    default='Models_mms_full/fold0_lvl3_best.pth',
                    help="Pre-trained model path")
parser.add_argument("--savepath", type=str, dest="savepath",
                    default='Results/test_gptf_original',
                    help="Path for saving output images")
parser.add_argument("--start_channel", type=int, dest="start_channel",
                    default=4,
                    help="Number of start channels (4 for our models)")
parser.add_argument("--datapath", type=str, dest="datapath",
                    default='Data/MMs_preprocessed',
                    help="Path to preprocessed M&Ms data")
parser.add_argument("--n_patients", type=int, dest="n_patients",
                    default=10,
                    help="Number of patients to test (default 10, -1 for all)")
opt = parser.parse_args()

modelpath     = opt.modelpath
savepath      = opt.savepath
datapath      = opt.datapath
start_channel = opt.start_channel
n_patients    = opt.n_patients

if not os.path.isdir(savepath):
    os.makedirs(savepath)


# ---- helper functions (orijinalden kopyalandi, degistirilmedi) ----
def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = sum(m1 * m2)
    return (2. * intersection + smooth) / (sum(m1) + sum(m2) + smooth)

def compute_label_dice(gt, pred):
    return DSC(gt, pred)

def save_image(img, ref_img, name):
    out = sitk.GetImageFromArray(img)
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    out.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(out, name)

def save_flow(img, ref_img, name):
    out = sitk.GetImageFromArray(img, isVector=True)
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    out.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(out, name)


# ---- test fonksiyonu (model kurma orijinal ile ayni) ----
def test():
    imgshape   = (96, 96, 16)
    imgshape_4 = (96 / 4, 96 / 4, 16 / 4)
    imgshape_2 = (96 / 2, 96 / 2, 16 / 2)
    range_flow = 0.4

    # --- MODEL KURMA (orijinal Test_GPTF_disp.py ile birebir ayni) ---
    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, start_channel, is_train=True,
        imgshape=imgshape_4, range_flow=range_flow).cuda(0)

    patch_model_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, start_channel, is_train=True,
        patch_shape=imgshape_4, range_flow=range_flow).cuda(0)

    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, start_channel, is_train=True,
        imgshape=imgshape_2, range_flow=range_flow,
        model_lvl1=model_lvl1,
        patch_model_lv2=patch_model_lv2).cuda(0)

    patch_model_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, start_channel, is_train=True,
        patch_shape=imgshape_4, range_flow=range_flow).cuda(0)

    patch_model = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, start_channel, is_train=True,
        patch_shape=imgshape_2, range_flow=range_flow,
        patch_model=patch_model_lv2).cuda(0)

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, start_channel, is_train=False,
        imgshape=imgshape, range_flow=range_flow,
        model_lvl2=model_lvl2, patch_model=patch_model).cuda(0)

    transform1 = SpatialTransform_unit().cuda(0)
    transform2 = SpatialTransformNearest_unit().cuda(0)

    model.load_state_dict(torch.load(modelpath))
    model.eval()
    transform1.eval()
    transform2.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(
        np.reshape(grid, (1,) + grid.shape)).cuda(0).float()

    device = torch.device("cuda:0")

    # --- DATA LOADING (bizim formata uyarlanmis) ---
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(datapath, '*'))
        if os.path.isdir(d)
    ])
    if n_patients > 0:
        patient_dirs = patient_dirs[:n_patients]

    print(f'Testing {len(patient_dirs)} patients from {datapath}')
    print(f'Model: {modelpath}')
    print(f'Saving to: {savepath}\n')

    dice_lv_list  = []
    dice_myo_list = []
    time_list     = []

    for patient_dir in patient_dirs:
        code = os.path.basename(patient_dir)

        ed_img_path  = os.path.join(patient_dir, f'{code}_ED.nii.gz')
        ed_gt_path   = os.path.join(patient_dir, f'{code}_ED_gt.nii.gz')
        es_img_path  = os.path.join(patient_dir, f'{code}_ES.nii.gz')
        es_gt_path   = os.path.join(patient_dir, f'{code}_ES_gt.nii.gz')

        if not all(os.path.exists(p) for p in
                   [ed_img_path, ed_gt_path, es_img_path, es_gt_path]):
            print(f'  SKIP {code}: missing files')
            continue

        print(f'--- Patient {code} start Registration ---')

        # fixed = ED, moving = ES (paper Section 3.2 ile ayni)
        fixed_img    = sitk.ReadImage(ed_img_path)
        fixed_label  = sitk.ReadImage(ed_gt_path)
        moving_img   = sitk.ReadImage(es_img_path)
        moving_label = sitk.ReadImage(es_gt_path)

        # orijinal scriptteki permute ile ayni: (B,1,H,W,D) -> permute(0,1,4,3,2)
        input_fixed = sitk.GetArrayFromImage(fixed_img)[np.newaxis, np.newaxis, ...]
        input_fixed = torch.from_numpy(input_fixed).to(device).float().permute(0, 1, 4, 3, 2)

        input_moving = sitk.GetArrayFromImage(moving_img)[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float().permute(0, 1, 4, 3, 2)

        input_moving_label = sitk.GetArrayFromImage(moving_label)[np.newaxis, np.newaxis, ...]
        input_moving_label = torch.from_numpy(input_moving_label).to(device).float().permute(0, 1, 4, 3, 2)

        # --- FORWARD PASS (orijinal ile ayni) ---
        with torch.no_grad():
            start = time.time()
            F_X_Y = model(input_moving, input_fixed)
            X_Y = transform1(
                input_moving, F_X_Y.permute(0, 2, 3, 4, 1), grid
            ).permute(0, 1, 4, 3, 2).data.cpu().numpy()[0, 0, :, :, :]
            end = time.time()

            X_Y_label = transform2(
                input_moving_label, F_X_Y.permute(0, 2, 3, 4, 1), grid
            ).permute(0, 1, 4, 3, 2).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.permute(0, 4, 3, 2, 1).data.cpu().numpy()[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        reg_time = end - start
        time_list.append(reg_time)

        # --- KAYDETME (orijinal ile ayni) ---
        save_flow(F_X_Y_cpu,  fixed_img, os.path.join(savepath, f'{code}_flow.nii.gz'))
        save_image(X_Y,       fixed_img, os.path.join(savepath, f'{code}_warped.nii.gz'))
        save_image(X_Y_label, fixed_label, os.path.join(savepath, f'{code}_label.nii.gz'))

        # --- DSC hesaplama (orijinal DSC fonksiyonu ile) ---
        fixed_gt_arr = sitk.GetArrayFromImage(fixed_label)
        X_Y_label_round = np.round(X_Y_label).astype(np.int32)

        # M&Ms label: 1=LV, 3=MYO
        dice_lv  = compute_label_dice(
            (fixed_gt_arr == 1).astype(np.float32),
            (X_Y_label_round == 1).astype(np.float32))
        dice_myo = compute_label_dice(
            (fixed_gt_arr == 3).astype(np.float32),
            (X_Y_label_round == 3).astype(np.float32))

        dice_lv_list.append(dice_lv)
        dice_myo_list.append(dice_myo)

        print(f'  Time: {reg_time:.3f}s | LV DSC: {dice_lv:.4f} | MYO DSC: {dice_myo:.4f}')

        del F_X_Y_cpu, X_Y, X_Y_label

    # --- OZET ---
    print('\n' + '=' * 50)
    print(f'  Results ({len(dice_lv_list)} patients)')
    print('=' * 50)
    print(f'  Mean LV  DSC : {np.mean(dice_lv_list):.4f} +/- {np.std(dice_lv_list):.4f}')
    print(f'  Mean MYO DSC : {np.mean(dice_myo_list):.4f} +/- {np.std(dice_myo_list):.4f}')
    print(f'  Mean time    : {np.mean(time_list):.3f}s per pair')
    print(f'\n  Warped images saved to: {savepath}')
    print("Finished")


if __name__ == '__main__':
    test()