import os
from argparse import ArgumentParser
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import torch
import glob
import time

from Functions import generate_grid_unit, transform_unit_flow_to_flow, load_4D
# from LapIRN.Code.GlobalPatchmodel import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
#     Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit,SpatialTransformNearest_unit,Miccai2020_LDR_laplacian_unit_add_lvl1,Miccai2020_LDR_laplacian_unit_add_lvl2
# from LapIRN.Code.GlobalPatchmodel import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
#     Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
#     SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, \
#     Miccai2020_LDR_laplacian_unit_add_lvl1, \
#     Miccai2020_LDR_laplacian_unit_add_lvl2
# from LapIRN.Code.Global_Patch_pyconv_featurefusion import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
#     Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
#     SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, \
#     Miccai2020_LDR_laplacian_unit_add_lvl1, \
#     Miccai2020_LDR_laplacian_unit_add_lvl2
from GP_TF import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, \
    Miccai2020_LDR_laplacian_unit_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_add_lvl2


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath",
                    default='F:/MultiVm/model/fold0/MM20_disp_gptf_reg_1stagelvl3_31840.pth',#'../Model/LapIRN_disp_fea7.pth',
                    help="Pre-trained Model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='F:/MultiVm/results/mm20/fold0',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7, #7
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='F:/MultiVm/data/MM20/norm01',
                    help="data path for training images")
opt = parser.parse_args()

modelpath = opt.modelpath
savepath = opt.savepath
datapath = opt.datapath

if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = sum(m1 * m2)
    return  (2. * intersection + smooth) / (sum(m1) + sum(m2) + smooth)

def compute_label_dice(gt, pred):
    dice = DSC(gt, pred)
    return dice

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

def test():
    '''Global ca and Patch 、 Global ca and Patch ca'''
    imgshape_4 = (96 / 4, 96 / 4, 16 / 4)
    imgshape_2 = (96 / 2, 96 / 2, 16 / 2)

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                             range_flow=range_flow).cuda(0)
    patch_model_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, start_channel,
        is_train=True, patch_shape=imgshape_4,
        range_flow=range_flow).cuda(0)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel,
                                                             is_train=True, imgshape=imgshape_2,
                                                             range_flow=range_flow,
                                                             model_lvl1=model_lvl1,
                                                             patch_model_lv2=patch_model_lv2).cuda(0)

    patch_model_lv2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, start_channel,
        is_train=True, patch_shape=imgshape_4,
        range_flow=range_flow).cuda(0)

    patch_model = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, patch_shape=imgshape_2,
                                                         range_flow=range_flow, patch_model=patch_model_lv2).cuda(0)

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                        range_flow=range_flow, model_lvl2=model_lvl2,
                                                        patch_model=patch_model).cuda(0)

    transform1 = SpatialTransform_unit().cuda(0)
    transform2 = SpatialTransformNearest_unit().cuda(0)

    model.load_state_dict(torch.load(modelpath))
    model.eval()
    transform1.eval()
    transform2.eval()
    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda(0).float()

    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")

    for pat in range(121, 151):  # patient 001 002 ....
        patient_path = 'pat' + str('%03d' % pat)

        print('---------------Patient-' + str(pat) + ' start Registration--------------')

        patient_file = os.path.join('F:/MultiVm/data/MM20/cropandsz16', patient_path)
        infos = {}
        for line in open(os.path.join(patient_file, 'info.txt')):
            label, value = line.split(':')
            infos[label] = value.rstrip('\n').lstrip(' ')
        systole_frame = int(infos['ES'])
        diastole_frame = int(infos['ED'])
        # fix image
        fix_file = os.path.join(datapath, patient_path,
                                'p' + str('%03d' % pat) + '_fr' + str('%02d' % diastole_frame) + '.nii.gz')
        # mov image
        mov_file = os.path.join(datapath, patient_path,
                                'p' + str('%03d' % pat) + '_fr' + str('%02d' % systole_frame) + '.nii.gz')

        fixed_img = sitk.ReadImage(fix_file)
        input_fixed = sitk.GetArrayFromImage(fixed_img)[np.newaxis, np.newaxis, ...]
        input_fixed = torch.from_numpy(input_fixed).to(device).float().permute(0, 1, 4, 3, 2)

        moving_img = sitk.ReadImage(mov_file)
        input_moving = sitk.GetArrayFromImage(moving_img)[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float().permute(0, 1, 4, 3, 2)

        f_label_dir = os.path.join('F:/MultiVm/data/MM20/cropandsz16', patient_path)
        fixed_label = sitk.ReadImage(os.path.join(f_label_dir,'p' + str('%03d' % pat) + '_fr' + str('%02d' % diastole_frame) + '_gt.nii.gz'))

        moving_label = sitk.ReadImage(os.path.join(f_label_dir,'p' + str('%03d' % pat) + '_fr' + str('%02d' % systole_frame) + '_gt.nii.gz'))
        input_moving_label = sitk.GetArrayFromImage(moving_label)[np.newaxis, np.newaxis, ...]
        input_moving_label = torch.from_numpy(input_moving_label).to(device).float().permute(0, 1, 4, 3, 2)

        with torch.no_grad():
            start = time.time()
            F_X_Y = model(input_moving, input_fixed)
            X_Y = transform1(input_moving, F_X_Y.permute(0, 2, 3, 4, 1), grid).permute(0, 1, 4, 3, 2).data.cpu().numpy()[0, 0, :, :, :]
            end  = time.time()
            # print('RegOneTime: ',end - start,file=f)
            X_Y_label = transform2(input_moving_label, F_X_Y.permute(0, 2, 3, 4, 1), grid).permute(0, 1, 4, 3, 2).data.cpu().numpy()[0, 0, :, :, :]
            F_X_Y_cpu = F_X_Y.permute(0, 4, 3, 2, 1).data.cpu().numpy()[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            save_flow(F_X_Y_cpu, fixed_img, savepath+"/"+patient_path+'_lvm_flow.nii.gz')
            save_image(X_Y, fixed_img, savepath+"/"+patient_path+'_lvm_warpped.nii.gz')
            save_image(X_Y_label, fixed_label, savepath + "/" + patient_path+ '_lvm_label.nii.gz')

            del F_X_Y_cpu, X_Y, X_Y_label

    print("Finished")


if __name__ == '__main__':
    imgshape = (96,96,16)
    range_flow = 0.4
    test()
