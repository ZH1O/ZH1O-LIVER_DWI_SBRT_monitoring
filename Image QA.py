import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import binary_erosion, generate_binary_structure
from collections import defaultdict
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

config = {
    "font.family": 'Times New Roman',
    "mathtext.fontset": 'cm',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
plt.rc('font', family='Times New Roman')
plt.rcParams['text.usetex'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

B_VALUE_MAP = {0: 0, 1: 30, 2: 50, 3: 100, 4: 150, 5: 300, 6: 400, 7: 500}
QIBA_SNR_THRESHOLD_B0 = 7
QIBA_SNR_THRESHOLD_HIGH_B = 3

PERCENTILE_LOW = 10
PERCENTILE_HIGH = 90


def find_patient_dirs(base_paths):
    patient_dirs = []
    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            dir_name = os.path.basename(root)
            if '=' in dir_name:
                has_valid_fraction = any(
                    '=' in d and (os.path.isdir(os.path.join(root, d, 'DWI_registration')) or
                                  os.path.isdir(os.path.join(root, d, 'RT')))
                    for d in dirs
                )
                if has_valid_fraction:
                    patient_dirs.append(root)
    return patient_dirs


def get_fractions(patient_dir):
    fractions = []
    for d in os.listdir(patient_dir):
        full_path = os.path.join(patient_dir, d)
        if os.path.isdir(full_path) and '=' in d:
            try:
                frac_num = int(d.split('=')[-1])
                fractions.append((frac_num, full_path))
            except ValueError:
                continue
    fractions.sort(key=lambda x: x[0])
    return fractions


def find_tumor_mask(fraction_path):
    candidates = [
        os.path.join(fraction_path, 'RTst_registration', 'Tumor.nii.gz'),
        os.path.join(fraction_path, 'RTst_registration', 'RTst_registration', 'Tumor.nii.gz'),
        os.path.join(fraction_path, 'DWI_RTst', 'Tumor.nii.gz'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def erode_mask(mask, voxels=1, connectivity=3):
    if voxels <= 0:
        return mask
    struct = generate_binary_structure(3, connectivity)
    return binary_erosion(mask, structure=struct, iterations=voxels)


def resample_to_reference(image, reference_image, is_mask=False):
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    resampled = sitk.Resample(image, reference_image, sitk.Transform(), interpolator, 0.0, image.GetPixelID())
    return resampled


def analyze_patient_snr_core_parenchyma(patient_path, erosion_voxels=1):
    try:
        patient_name = os.path.basename(patient_path).split('=')[0]
    except Exception:
        patient_name = os.path.basename(patient_path)

    fractions_list = get_fractions(patient_path)
    if not fractions_list:
        return None, "No fraction"

    results = []

    for frac_num, frac_dir in fractions_list:
        frac_name = os.path.basename(frac_dir)
        liver_path = os.path.join(frac_dir, 'DWI_RTst', 'Liver.nii.gz')
        tumor_path = find_tumor_mask(frac_dir)
        dwi_dir = os.path.join(frac_dir, 'DWI_registration')

        if not os.path.exists(liver_path) or not tumor_path or not os.path.exists(dwi_dir):
            continue

        try:
            b0_path = os.path.join(dwi_dir, '0.nii.gz')
            if not os.path.exists(b0_path):
                continue
            b0_img = sitk.ReadImage(b0_path, sitk.sitkFloat64)
            b0_array = sitk.GetArrayFromImage(b0_img)

            liver_img = sitk.ReadImage(liver_path, sitk.sitkFloat64)
            liver_img = resample_to_reference(liver_img, b0_img, is_mask=True)
            liver_mask = sitk.GetArrayFromImage(liver_img) > 0.5

            tumor_img = sitk.ReadImage(tumor_path, sitk.sitkFloat64)
            tumor_img = resample_to_reference(tumor_img, b0_img, is_mask=True)
            tumor_mask = sitk.GetArrayFromImage(tumor_img) > 0.5

            normal_liver_mask = liver_mask & (~tumor_mask)
            normal_liver_eroded = erode_mask(normal_liver_mask, voxels=erosion_voxels)

            if np.sum(normal_liver_eroded) < 7:
                continue

            nl_b0_signals = b0_array[normal_liver_eroded]
            p_low = np.percentile(nl_b0_signals, PERCENTILE_LOW)
            p_high = np.percentile(nl_b0_signals, PERCENTILE_HIGH)

            core_parenchyma_mask = normal_liver_eroded & (b0_array >= p_low) & (b0_array <= p_high)

            if np.sum(core_parenchyma_mask) < 30:
                continue

        except Exception:
            continue
        for b_idx in range(8):
            b_val = B_VALUE_MAP.get(b_idx, b_idx)
            dwi_path = os.path.join(dwi_dir, f'{b_idx}.nii.gz')

            if not os.path.exists(dwi_path):
                continue

            try:
                dwi_img = sitk.ReadImage(dwi_path, sitk.sitkFloat64)
                dwi_array = sitk.GetArrayFromImage(dwi_img)
            except Exception:
                continue

            signal_core = dwi_array[core_parenchyma_mask]
            core_mean = np.mean(signal_core)
            core_std = np.std(signal_core)
            snr_core = core_mean / core_std if core_std > 0 else np.nan

            results.append({
                'patient': patient_name,
                'fraction': frac_name,
                'fraction_number': frac_num,
                'b_index': b_idx,
                'b_value': b_val,
                'core_mean': core_mean,
                'core_std': core_std,
                'snr_core': snr_core,
                'core_voxels': np.sum(core_parenchyma_mask)
            })

    if not results:
        return None, "No data"

    return results, "ok"


def print_fraction_pass_rate(all_results, save_csv=None):

    if not all_results:
        print("No result")
        return []

    fraction_groups = defaultdict(list)
    for r in all_results:
        key = (r['patient'], r['fraction_number'])
        fraction_groups[key].append(r)

    n_total_fractions = len(fraction_groups)
    n_pass_fractions = 0
    fraction_pass_details = []

    for (patient, frac_num), records in fraction_groups.items():
        all_pass = True
        failed_b_values = []
        fraction_name = records[0]['fraction'] 
        for r in records:
            thresh = QIBA_SNR_THRESHOLD_B0 if r['b_value'] <= 10 else QIBA_SNR_THRESHOLD_HIGH_B
            if np.isnan(r['snr_core']) or r['snr_core'] < thresh:
                all_pass = False
                failed_b_values.append(r['b_value'])

        if all_pass:
            n_pass_fractions += 1

        fraction_pass_details.append({
            'patient': patient,
            'fraction': fraction_name,
            'fraction_number': frac_num,
            'all_pass': all_pass,
            'n_b_values': len(records),
            'failed_b_values': failed_b_values
        })

    pass_rate = n_pass_fractions / n_total_fractions * 100 if n_total_fractions > 0 else 0

    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"  All fraction: {n_total_fractions}")
    print(f"  passed fraction: {n_pass_fractions}")
    print(f"  pass rate: {n_pass_fractions}/{n_total_fractions} ({pass_rate:.1f}%)")
    print(f"{'='*70}")

    failed_fractions = [d for d in fraction_pass_details if not d['all_pass']]
    if failed_fractions:
        print(f"\n  No pass:")
        print(f"  {'patient':<30} {'fraction':<30} {'bvalue'}")
        print(f"  {'-'*70}")
        for d in failed_fractions[:10]:
            failed_str = ', '.join([str(b) for b in d['failed_b_values']])
            print(f"  {d['patient']:<30} {d['fraction']:<30} {failed_str}")

    if save_csv:
        df = pd.DataFrame(fraction_pass_details)
        df['failed_b_values'] = df['failed_b_values'].apply(lambda x: ','.join(map(str, x)) if x else '')
        df.to_csv(save_csv, index=False)
        print(f"\n  QA saved to: {save_csv}")

    return fraction_pass_details


def run_analysis(base_paths, erosion_voxels=1):
    patient_dirs = find_patient_dirs(base_paths)

    all_results = []
    failed_cases = 0

    for i, p_dir in enumerate(patient_dirs):
        results, status = analyze_patient_snr_core_parenchyma(p_dir, erosion_voxels=erosion_voxels)

        if results:
            all_results.extend(results)
        else:
            failed_cases += 1
            if failed_cases <= 5:
                print(f"  [Error] {os.path.basename(p_dir)}: {status}")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(patient_dirs)}")

    print(f"\n all: {len(patient_dirs) - failed_cases} patients")
    print(f"Fail: {failed_cases} patients")
    return all_results


def print_report(all_results):
    if not all_results:
        print("No data")
        return

    by_bvalue = defaultdict(list)
    for r in all_results:
        by_bvalue[r['b_value']].append(r)


    for b_val in sorted(by_bvalue.keys()):
        res = by_bvalue[b_val]

        core_means = [r['core_mean'] for r in res]
        core_stds = [r['core_std'] for r in res]
        snrs_core = [r['snr_core'] for r in res if not np.isnan(r['snr_core'])]

        thresh = QIBA_SNR_THRESHOLD_B0 if b_val <= 10 else QIBA_SNR_THRESHOLD_HIGH_B
        n_pass = np.sum(np.array(snrs_core) >= thresh) if snrs_core else 0
        pass_rate_str = f"{n_pass}/{len(snrs_core)} ({n_pass/len(snrs_core)*100:.1f}%)" if snrs_core else "N/A"

def plot_results(all_results, save_path=None):
    if not all_results:
        return
    by_bvalue = defaultdict(list)
    for r in all_results:
        by_bvalue[r['b_value']].append(r)
    b_vals = sorted(by_bvalue.keys())


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    snr_data = [[r['snr_core'] for r in by_bvalue[b] if not np.isnan(r['snr_core'])] for b in b_vals]
    bp = ax.boxplot(snr_data, labels=[str(b) for b in b_vals], patch_artist=True, widths=0.6)
    colors = ['#6E86A5', '#D4A574', '#7BA47B', '#B87D8B', '#A58674', '#86A56E', '#A56E86', '#6EA5A5']
    for patch, color in zip(bp['boxes'], colors[:len(b_vals)]):
        patch.set_facecolor('white')

    ax.set_xlabel('$b$-value (s/mm²)', fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_title('', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')


    ax = axes[1]
    pass_rates_core = []
    for b in b_vals:
        core_s = [r['snr_core'] for r in by_bvalue[b] if not np.isnan(r['snr_core'])]
        thresh = 10 if b <= 1 else 3
        rate_core = (np.sum(np.array(core_s) >= thresh) / len(core_s) * 100) if core_s else 0
        pass_rates_core.append(rate_core)

    x = np.arange(len(b_vals))
    width = 0.6
    ax.bar(x, pass_rates_core, width, color='#7BA47B', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in b_vals])
    ax.set_xlabel('$b$-value (s/mm²)', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('', fontsize=12)
    ax.set_ylim(50, 105)

    for i, rate in enumerate(pass_rates_core):
        ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=12, fontweight='normal')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    base_paths = [
        r'XXX'
    ]
    EROSION_VOXELS = 3

    QA_CSV_PATH = r'XXX'

    all_results = run_analysis(base_paths, erosion_voxels=EROSION_VOXELS)
    print(f"\n obtain {len(all_results)} recode。")

    print_report(all_results)
    print_fraction_pass_rate(all_results, save_csv=QA_CSV_PATH)
    plot_results(all_results)
