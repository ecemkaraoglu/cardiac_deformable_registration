import numpy as np
import csv
import statistics

print("=" * 60)
print("ACDC — median selection")
print("=" * 60)
acdc = np.load(r'C:\Users\ecemk\Desktop\eco\DIR-MRVIT\Models_cv_full\cv_pooled_results.npy',
               allow_pickle=True).item()
lv = [float(x) for x in acdc['lv_dice']]
print(f'LV Dice: mean={np.mean(lv):.4f}, median={np.median(lv):.4f}')
print(f'         min={min(lv):.4f}, max={max(lv):.4f}')
sorted_lv = sorted(range(len(lv)), key=lambda i: lv[i])
median_idx = sorted_lv[len(lv) // 2]
print(f'Median index: {median_idx} (LV Dice = {lv[median_idx]:.4f})')
print(f'MYO at that index: {float(acdc["myo_dice"][median_idx]):.4f}')

print()
print("=" * 60)
print("M&Ms — median selection")
print("=" * 60)
mms = np.load(r'C:\Users\ecemk\Desktop\eco\DIR-MRVIT\Models_mms_full\cv_pooled_results_mms.npy',
              allow_pickle=True).item()
lv = [float(x) for x in mms['lv_dice']]
print(f'LV Dice: mean={np.mean(lv):.4f}, median={np.median(lv):.4f}')
print(f'         min={min(lv):.4f}, max={max(lv):.4f}')
sorted_lv = sorted(range(len(lv)), key=lambda i: lv[i])
median_idx = sorted_lv[len(lv) // 2]
print(f'Median index: {median_idx} (LV Dice = {lv[median_idx]:.4f})')
print(f'MYO at that index: {float(mms["myo_dice"][median_idx]):.4f}')

print()
print("=" * 60)
print("CMRxM22 — median selection")
print("=" * 60)
with open(r'C:\Users\ecemk\Desktop\eco\DIR-MRVIT\Results\cmrxm22\cmrxm22_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Group by case, average across folds
case_dice = {}
for r in rows:
    case = r['case']
    if case not in case_dice:
        case_dice[case] = []
    case_dice[case].append(float(r['DSC_LV']))

case_avg = {c: sum(vs) / len(vs) for c, vs in case_dice.items()}
sorted_cases = sorted(case_avg.items(), key=lambda x: x[1])
print(f'Total cases: {len(case_avg)}')
lv_values = list(case_avg.values())
print(f'LV Dice (5-fold avg): mean={np.mean(lv_values):.4f}, median={np.median(lv_values):.4f}')
median_case, median_dice = sorted_cases[len(sorted_cases) // 2]
print(f'Median case: {median_case} (LV Dice = {median_dice:.4f})')