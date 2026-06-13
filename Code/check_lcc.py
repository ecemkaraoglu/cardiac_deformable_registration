"""
Diagnostic: count how many warped segmentations have multiple connected
components per label. If most have only 1 component, LCC genuinely has
nothing to remove.
"""
import os, sys, glob, numpy as np, torch, random
import SimpleITK as sitk
from scipy import ndimage
sys.path.insert(0, '.')
sitk.ProcessObject.GlobalWarningDisplayOff()

from GP_TF import (Miccai2020_LDR_laplacian_unit_disp_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl2,
                   Miccai2020_LDR_laplacian_unit_disp_add_lvl3,
                   Miccai2020_LDR_laplacian_unit_add_lvl1,
                   Miccai2020_LDR_laplacian_unit_add_lvl2,
                   SpatialTransform_unit)
from Functions import generate_grid_unit

DATAPATH      = 'Data/MMs_preprocessed'
MODEL_DIR     = 'Models_mms_full'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
START_CHANNEL = 4
RANGE_FLOW    = 0.4
N_FOLDS       = 5
SEED          = 42

imgshape   = (96, 96, 16)
imgshape_2 = (48, 48, 8)
imgshape_4 = (24, 24, 4)

LV_LABEL  = 1
MYO_LABEL = 3


def load_all_patients():
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(DATAPATH, '*'))
        if os.path.isdir(d)
    ])
    data, codes = [], []
    for p in patient_dirs:
        code = os.path.basename(p)
        paths = [os.path.join(p, f'{code}_{x}.nii.gz')
                 for x in ['ED', 'ED_gt', 'ES', 'ES_gt']]
        if not all(os.path.exists(x) for x in paths):
            continue
        ed_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[0])).astype(np.float32)
        ed_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[1])).astype(np.int32)
        es_img = sitk.GetArrayFromImage(sitk.ReadImage(paths[2])).astype(np.float32)
        es_gt  = sitk.GetArrayFromImage(sitk.ReadImage(paths[3])).astype(np.int32)
        data.append((ed_img, es_img, ed_gt, es_gt))
        codes.append(code)
    return data, codes


def get_fold_indices(n, fold):
    rng   = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)
    test_size  = n // N_FOLDS
    val_size   = n // 10
    test_start = fold * test_size
    test_idx   = order[test_start:test_start + test_size]
    return test_idx


def to_tensor(arr):
    return (torch.from_numpy(arr).float()
            .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))


def warp_segmentation(seg, disp_field, grid):
    transform     = SpatialTransform_unit().to(DEVICE)
    disp_permuted = disp_field.permute(0, 2, 3, 4, 1)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    result = np.zeros(seg.shape, dtype=np.int32)
    probs  = np.zeros(seg.shape, dtype=np.float32)
    for label in unique_labels:
        binary = (seg == label).astype(np.float32)
        bt = (torch.from_numpy(binary).float()
              .permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            warped = transform(bt, disp_permuted, grid)
        warped_np = warped.squeeze().cpu().numpy().transpose(2, 0, 1)
        mask = warped_np > probs
        result[mask] = label
        probs[mask] = warped_np[mask]
    return result


def build_model(weights_path):
    m1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    pl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_4,
        range_flow=RANGE_FLOW).to(DEVICE)
    m2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape_2,
        range_flow=RANGE_FLOW, model_lvl1=m1, patch_model_lv2=pl2).to(DEVICE)
    pl3 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, START_CHANNEL, is_train=True, patch_shape=imgshape_2,
        range_flow=RANGE_FLOW, patch_model=pl2).to(DEVICE)
    m3 = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(
        2, 3, START_CHANNEL, is_train=True, imgshape=imgshape,
        range_flow=RANGE_FLOW, model_lvl2=m2, patch_model=pl3).to(DEVICE)
    m3.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m3.eval()
    return m3


def main():
    all_data, codes = load_all_patients()
    n = len(all_data)
    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(DEVICE).float()

    # Counters
    stats_lv = {'1': 0, '2': 0, '3+': 0}
    stats_myo = {'1': 0, '2': 0, '3+': 0}
    largest_extras = []   # (case, label, total_components, fraction_in_extra)
    total = 0

    for fold in range(N_FOLDS):
        weights = os.path.join(MODEL_DIR, f'fold{fold}_lvl3_best.pth')
        if not os.path.exists(weights):
            continue
        model = build_model(weights)
        test_idx = get_fold_indices(n, fold)
        for idx in test_idx:
            ed_img, es_img, ed_gt, es_gt = all_data[idx]
            code = codes[idx]
            fix = to_tensor(ed_img); mov = to_tensor(es_img)
            with torch.no_grad():
                disp = model(mov, fix)[0]
            warped = warp_segmentation(es_gt, disp, grid)

            for label, label_name, stats in [(LV_LABEL, 'LV', stats_lv),
                                              (MYO_LABEL, 'MYO', stats_myo)]:
                binary = (warped == label).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                _, n_components = ndimage.label(binary)
                if n_components == 1:
                    stats['1'] += 1
                elif n_components == 2:
                    stats['2'] += 1
                else:
                    stats['3+'] += 1
                if n_components > 1:
                    # How much of the volume is in the non-largest components?
                    labeled, _ = ndimage.label(binary)
                    sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
                    largest = sizes.max()
                    extra_fraction = (sizes.sum() - largest) / sizes.sum()
                    largest_extras.append(
                        (code, label_name, n_components, extra_fraction, sizes.sum() - largest)
                    )
            total += 1

    print(f'\n{"="*70}')
    print(f'Total evaluations: {total // 2} cases ({total} label-checks)')
    print(f'{"="*70}')
    print(f'\nLV connected components:')
    for k, v in stats_lv.items():
        pct = 100.0 * v / max(1, sum(stats_lv.values()))
        print(f'  {k} component(s): {v:4d}  ({pct:.1f}%)')
    print(f'\nMYO connected components:')
    for k, v in stats_myo.items():
        pct = 100.0 * v / max(1, sum(stats_myo.values()))
        print(f'  {k} component(s): {v:4d}  ({pct:.1f}%)')

    print(f'\nMulti-component cases (top 10 by extra voxel count):')
    largest_extras.sort(key=lambda x: -x[4])
    for code, lbl, nc, frac, vox in largest_extras[:10]:
        print(f'  {code:8s}  {lbl:3s}  {nc} components, '
              f'{vox:5.0f} extra voxels ({100*frac:.2f}% of total)')


if __name__ == '__main__':
    main()
