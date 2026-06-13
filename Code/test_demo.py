import sys, numpy as np, torch
sys.path.insert(0, '.')

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2)
import SimpleITK as sitk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)
sc = 4

model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2,3,sc,is_train=True,imgshape=imgshape_4,range_flow=0.4).to(device)
patch_lv2  = Miccai2020_LDR_laplacian_unit_add_lvl2(2,3,sc,is_train=True,patch_shape=imgshape_4,range_flow=0.4).to(device)
model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2,3,sc,is_train=True,imgshape=imgshape_2,range_flow=0.4,model_lvl1=model_lvl1,patch_model_lv2=patch_lv2).to(device)
patch_lv3  = Miccai2020_LDR_laplacian_unit_add_lvl1(2,3,sc,is_train=True,patch_shape=imgshape_2,range_flow=0.4,patch_model=patch_lv2).to(device)
model_lvl3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2,3,sc,is_train=True,imgshape=imgshape,range_flow=0.4,model_lvl2=model_lvl2,patch_model=patch_lv3).to(device)

print('Model kuruldu, parametre sayisi:', sum(p.numel() for p in model_lvl3.parameters()))

ed = sitk.GetArrayFromImage(sitk.ReadImage('../Data/P001-1-ED.nii.gz'))
es = sitk.GetArrayFromImage(sitk.ReadImage('../Data/P001-1-ES.nii.gz'))

fix = torch.from_numpy(ed).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(device)
mov = torch.from_numpy(es).float().permute(1,2,0).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model_lvl3(mov, fix)

print('Forward pass OK!')
print('Deformation field shape:', outputs[0].shape)
print('Warped image shape:', outputs[1].shape)