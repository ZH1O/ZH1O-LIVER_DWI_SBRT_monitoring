import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',
    "mathtext.fontset": 'cm',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
plt.rc('font', family='Times New Roman')
plt.rcParams['text.usetex'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ALPHA_BETA = 10.0 

COLOR_RESPONSE = '#2E86AB'
COLOR_NONRESPONSE = '#F18F01'
COLOR_RESPONSE_DARK = '#1A5C7A'
COLOR_NONRESPONSE_DARK = '#B8650A'
COLOR_MIXED = '#B0B0B0'
COLOR_CASE_GOOD = '#E41A1C'
COLOR_CASE_POOR = '#377EB8'

def traversalDir_FirstDir(path):
    return [os.path.join(path, file) for file in os.listdir(path)
            if os.path.exists(os.path.join(path, file))]

def load_qa_pass_dict(csv_path):

    if not csv_path or not os.path.exists(csv_path):
        print("  No QA，All fraction will used。")
        return None

    df = pd.read_csv(csv_path)
    qa_dict = {}
    for _, row in df.iterrows():
        patient = str(row['patient'])
        frac_num = int(row['fraction_number'])
        all_pass = row['all_pass'] if isinstance(row['all_pass'], bool) else (str(row['all_pass']).lower() == 'true')
        qa_dict[(patient, frac_num)] = all_pass

    n_pass = sum(1 for v in qa_dict.values() if v)
    print(f"  load QA: {len(qa_dict)} fraction，{n_pass} pass。")
    return qa_dict

def extract_adc_dose_means_4class(base_path, parameter, func, multi=1,
                                   qa_dict=None, alpha_beta=10.0):

    out = [{}, {}, {}, {}]
    for path1 in traversalDir_FirstDir(base_path):
        for patient in traversalDir_FirstDir(path1):
            try:
                cls = int(patient.split('=')[-1])
                total_N = int(patient.split('=')[-2])
                name = os.path.basename(patient).split('=')[0]
                adc_out = np.full(total_N, np.nan)
                dose_out = np.full(total_N, np.nan)
                for fraction in traversalDir_FirstDir(patient):
                    try:
                        fraction_N = int(fraction.split('=')[-1])
                        if qa_dict is not None:
                            key = (name, fraction_N)
                            if key in qa_dict and not qa_dict[key]:
                                continue

                        adc_path = os.path.join(fraction, 'parameter_map',
                                                parameter + '.nii.gz')
                        mask_path = os.path.join(fraction, 'RTst_registration',
                                                 'Tumor.nii.gz')
                        dose_path = os.path.join(fraction, 'RT', 'Dose.nii.gz')

                        mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat64)
                        mask_array = mask_img > 0.5
                        mask_flat = sitk.GetArrayFromImage(mask_array).reshape(-1,)

                        ADC = sitk.GetArrayFromImage(
                            sitk.ReadImage(adc_path, sitk.sitkFloat64)).reshape(-1,)
                        roi_adc = ADC[np.where(mask_flat)] * multi
                        if roi_adc.size > 0:
                            adc_out[fraction_N - 1] = func(roi_adc)

                        Dose = sitk.GetArrayFromImage(
                            sitk.ReadImage(dose_path, sitk.sitkFloat64)).reshape(-1,)
                        roi_dose = Dose[np.where(mask_flat)]
                        if roi_dose.size > 0:
                            dose_out[fraction_N - 1] = round(np.percentile(roi_dose, 90), 2)
                    except Exception:
                        continue

                for i in range(total_N):
                    if np.isnan(dose_out[i]) and i > 0:
                        dose_out[i] = dose_out[i - 1]

                eqd2_per_frac = dose_out * (alpha_beta + dose_out) / (alpha_beta + 2)
                cum_eqd2 = np.cumsum(np.nan_to_num(eqd2_per_frac))
                cum_eqd2 = np.insert(cum_eqd2[:-1], 0, 0)

                for i in range(total_N):
                    if np.isnan(adc_out[i]):
                        cum_eqd2[i] = np.nan
                    if i > 0 and np.isnan(dose_out[i - 1]):
                        cum_eqd2[i] = np.nan

                out[cls][name] = {
                    "adc": adc_out,
                    "dose": dose_out,
                    "cum_eqd2": cum_eqd2 
                }
            except Exception:
                continue
    return out

def compute_bands_25_75(good_vals, poor_vals):
    if len(good_vals) < 3 or len(poor_vals) < 3:
        return None
    good_lower = np.percentile(good_vals, 25)
    good_upper = np.percentile(good_vals, 75)
    poor_lower = np.percentile(poor_vals, 25)
    poor_upper = np.percentile(poor_vals, 75)
    good_median = np.median(good_vals)
    poor_median = np.median(poor_vals)
    overlap = not (good_upper < poor_lower or poor_upper < good_lower)
    if overlap:
        separation = -max(0, good_upper - poor_lower, poor_upper - good_lower)
    else:
        separation = max(poor_lower - good_upper, good_lower - poor_upper)
    return {
        'good_band': (good_lower, good_upper),
        'poor_band': (poor_lower, poor_upper),
        'good_median': good_median,
        'poor_median': poor_median,
        'overlap': overlap,
        'percentile_used': 25,
        'separation': separation
    }

def compute_filled_bands(good_vals, poor_vals):
    if len(good_vals) < 3 or len(poor_vals) < 3:
        return None

    good_median = np.median(good_vals)
    poor_median = np.median(poor_vals)
    good_25 = np.percentile(good_vals, 25)
    good_75 = np.percentile(good_vals, 75)
    poor_25 = np.percentile(poor_vals, 25)
    poor_75 = np.percentile(poor_vals, 75)

    if good_75 < poor_25:
        good_is_upper = False
    elif poor_75 < good_25:
        good_is_upper = True
    else:
        good_is_upper = good_75 > poor_75

    if good_75 < poor_25:
        for step in range(1, 26):
            if 75 + step > 100 or 25 - step < 0:
                break
            good_upper_exp = np.percentile(good_vals, 75 + step)
            poor_lower_exp = np.percentile(poor_vals, 25 - step)
            if good_upper_exp >= poor_lower_exp:
                split_line = (good_upper_exp + poor_lower_exp) / 2.0
                return {'good_band': (good_25, split_line),
                        'poor_band': (split_line, poor_75),
                        'good_median': good_median, 'poor_median': poor_median,
                        'overlap': False, 'separation': 0.0,
                        'split_line': split_line, 'expanded': True}
        split_line = (good_75 + poor_25) / 2.0
        return {'good_band': (good_25, split_line),
                'poor_band': (split_line, poor_75),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'expanded': True}

    if poor_75 < good_25:
        for step in range(1, 26):
            if 25 - step < 0 or 75 + step > 100:
                break
            good_lower_exp = np.percentile(good_vals, 25 - step)
            poor_upper_exp = np.percentile(poor_vals, 75 + step)
            if poor_upper_exp >= good_lower_exp:
                split_line = (poor_upper_exp + good_lower_exp) / 2.0
                return {'good_band': (split_line, good_75),
                        'poor_band': (poor_25, split_line),
                        'good_median': good_median, 'poor_median': poor_median,
                        'overlap': False, 'separation': 0.0,
                        'split_line': split_line, 'expanded': True}
        split_line = (poor_75 + good_25) / 2.0
        return {'good_band': (split_line, good_75),
                'poor_band': (poor_25, split_line),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'expanded': True}

    if good_is_upper:
        for step in range(1, 26):
            good_lower = np.percentile(good_vals, 25 + step)
            poor_upper = np.percentile(poor_vals, 75 - step)
            if poor_upper <= good_lower:
                split_line = (poor_upper + good_lower) / 2.0
                return {'good_band': (split_line, good_75),
                        'poor_band': (poor_25, split_line),
                        'good_median': good_median, 'poor_median': poor_median,
                        'overlap': False, 'separation': 0.0,
                        'split_line': split_line, 'shrunk': True}
        split_line = (good_median + poor_median) / 2.0
        return {'good_band': (split_line, good_75),
                'poor_band': (poor_25, split_line),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'shrunk': True}
    else:
        for step in range(1, 26):
            poor_lower = np.percentile(poor_vals, 25 + step)
            good_upper = np.percentile(good_vals, 75 - step)
            if good_upper <= poor_lower:
                split_line = (good_upper + poor_lower) / 2.0
                return {'good_band': (good_25, split_line),
                        'poor_band': (split_line, poor_75),
                        'good_median': good_median, 'poor_median': poor_median,
                        'overlap': False, 'separation': 0.0,
                        'split_line': split_line, 'shrunk': True}
        split_line = (good_median + poor_median) / 2.0
        return {'good_band': (good_25, split_line),
                'poor_band': (split_line, poor_75),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'shrunk': True}

def _extract_patient_bin_means(group_dict, condition_func):
    patient_means = {}
    for name, arr_dict in group_dict.items():
        vals = []
        for d, a in zip(arr_dict['cum_eqd2'], arr_dict['adc']):
            if not np.isnan(d) and not np.isnan(a) and condition_func(d):
                vals.append(a)
        if len(vals) > 0:
            patient_means[name] = np.mean(vals)
    return patient_means



def bin_by_cumulative_dose(good_group, poor_group, dose_edges, min_samples=2):
    n_intervals = len(dose_edges)

    dose_bin_labels = ["0"]
    dose_bin_labels.append(f"(0,{dose_edges[0]}]")
    for i in range(n_intervals - 1):
        dose_bin_labels.append(f"({dose_edges[i]},{dose_edges[i+1]}]")

    dose_bins_data = {}

    # Bin "0"
    good_means = _extract_patient_bin_means(good_group, lambda d: d == 0)
    poor_means = _extract_patient_bin_means(poor_group, lambda d: d == 0)
    if len(good_means) >= min_samples and len(poor_means) >= min_samples:
        dose_bins_data["0"] = {
            'good_vals': np.array(list(good_means.values())),
            'poor_vals': np.array(list(poor_means.values()))
        }

    label = dose_bin_labels[1]
    good_means = _extract_patient_bin_means(
        good_group, lambda d, hi=dose_edges[0]: 0 < d <= hi)
    poor_means = _extract_patient_bin_means(
        poor_group, lambda d, hi=dose_edges[0]: 0 < d <= hi)
    if len(good_means) >= min_samples and len(poor_means) >= min_samples:
        dose_bins_data[label] = {
            'good_vals': np.array(list(good_means.values())),
            'poor_vals': np.array(list(poor_means.values()))
        }

    for i in range(n_intervals - 1):
        label = dose_bin_labels[i + 2]
        low = dose_edges[i]
        high = dose_edges[i + 1]
        good_means = _extract_patient_bin_means(
            good_group, lambda d, lo=low, hi=high: lo < d <= hi)
        poor_means = _extract_patient_bin_means(
            poor_group, lambda d, lo=low, hi=high: lo < d <= hi)
        if len(good_means) >= min_samples and len(poor_means) >= min_samples:
            dose_bins_data[label] = {
                'good_vals': np.array(list(good_means.values())),
                'poor_vals': np.array(list(poor_means.values()))
            }

    return dose_bins_data, dose_bin_labels

def compute_dose_bands(good_group, poor_group, dose_edges, min_samples=2):
    dose_bins_data, dose_bin_labels = bin_by_cumulative_dose(
        good_group, poor_group, dose_edges, min_samples)

    dose_bands_25_75 = {}
    dose_bands_filled = {}
    dose_raw_data = {}

    for label in dose_bin_labels:
        if label not in dose_bins_data:
            continue
        data = dose_bins_data[label]
        good_vals = data['good_vals']
        poor_vals = data['poor_vals']

        dose_raw_data[label] = {
            'good_vals': good_vals,
            'poor_vals': poor_vals,
            'n_good': len(good_vals),
            'n_poor': len(poor_vals)
        }

        bands_25_75 = compute_bands_25_75(good_vals, poor_vals)
        bands_filled = compute_filled_bands(good_vals, poor_vals)

        if bands_25_75 is not None:
            dose_bands_25_75[label] = bands_25_75
        if bands_filled is not None:
            dose_bands_filled[label] = bands_filled

    return dose_bands_25_75, dose_bands_filled, dose_raw_data, dose_bin_labels

def _sort_bin_labels(labels):
    def sort_key(label):
        if label == "0":
            return (0, 0)
        clean = label.replace("(", "").replace("]", "").replace("[", "").replace(")", "")
        parts = clean.split(',')
        first_num = float(parts[0])
        return (first_num, 1)
    return sorted(labels, key=sort_key)


def _get_valid_labels(bands_dict, all_labels):
    valid = [l for l in all_labels if l in bands_dict]
    return _sort_bin_labels(valid)



def _get_patient_bin_values(patient_data, dose_edges, dose_bin_labels):
    cum_eqd2 = patient_data['cum_eqd2']
    adc = patient_data['adc']
    values = {}
    for label in dose_bin_labels:
        if label == "0":
            vals = [a for d, a in zip(cum_eqd2, adc)
                    if not np.isnan(d) and not np.isnan(a) and d == 0]
        elif label.startswith("(0,"):
            hi = float(label.split(",")[1].rstrip("]"))
            vals = [a for d, a in zip(cum_eqd2, adc)
                    if not np.isnan(d) and not np.isnan(a) and 0 < d <= hi]
        else:
            parts = label.replace("(", "").replace("]", "").split(",")
            lo = float(parts[0])
            hi = float(parts[1])
            vals = [a for d, a in zip(cum_eqd2, adc)
                    if not np.isnan(d) and not np.isnan(a) and lo < d <= hi]
        values[label] = np.mean(vals) if vals else np.nan
    return values

def print_dose_bin_summary(dose_raw_data, dose_bin_labels):
    sorted_labels = _sort_bin_labels(dose_bin_labels)
    total_r, total_nr = 0, 0
    for label in sorted_labels:
        if label in dose_raw_data:
            n_r = dose_raw_data[label]['n_good']
            n_nr = dose_raw_data[label]['n_poor']
            total_r += n_r
            total_nr += n_nr
            print(f"  {label:<22} {n_r:<14} {n_nr:<18} {n_r + n_nr:<10}")
        else:
            print(f"  {label:<22} {'data':<14} {'data':<18} {'-':<10}")
    print(f"  {'-'*64}")
    print(f"  {'合计':<22} {total_r:<14} {total_nr:<18} {total_r + total_nr:<10}")
    print(f"{'='*80}")



def plot_dose_combined_figure(bands_25_75, bands_filled, raw_data,
                               dose_bin_labels, good_group, poor_group,
                               ylims_dict, param_labels, param_names,
                               dose_edges,
                               good_case_idx=0, poor_case_idx=0,
                               save_path=None):

    n_params = len(param_names)

    sorted_labels = _sort_bin_labels(dose_bin_labels)

    fig, axes = plt.subplots(n_params, 2, figsize=(19, 4.5 * n_params))

    global_handles = {}
    global_labels_order = []

    for row_idx, param_name in enumerate(param_names):
        param_label = param_labels[row_idx]
        ylim = ylims_dict.get(param_name, None)

        param_bands_25_75 = bands_25_75[param_name]
        param_bands_filled = bands_filled[param_name]
        param_raw = raw_data[param_name]
        param_good = good_group[param_name]
        param_poor = poor_group[param_name]

        valid_labels = _get_valid_labels(param_bands_25_75, sorted_labels)
        if len(valid_labels) < 4:
            continue
        n_bins = len(valid_labels)
        bin_indices = np.arange(n_bins)

        ax_left = axes[row_idx, 0]

        good_25 = np.full(n_bins, np.nan)
        good_75 = np.full(n_bins, np.nan)
        good_med = np.full(n_bins, np.nan)
        poor_25 = np.full(n_bins, np.nan)
        poor_75 = np.full(n_bins, np.nan)
        poor_med = np.full(n_bins, np.nan)
        overlaps = np.full(n_bins, False)

        for i, label in enumerate(valid_labels):
            b = param_bands_25_75[label]
            good_25[i] = b['good_band'][0]
            good_75[i] = b['good_band'][1]
            good_med[i] = b['good_median']
            poor_25[i] = b['poor_band'][0]
            poor_75[i] = b['poor_band'][1]
            poor_med[i] = b['poor_median']
            overlaps[i] = b['overlap']

        L1 = np.minimum(good_25, poor_25)
        L2 = np.maximum(good_25, poor_25)
        L3 = np.minimum(good_75, poor_75)
        L4 = np.maximum(good_75, poor_75)

        zone1_upper = np.where(overlaps, L2, np.minimum(L2, L3))
        zone3_lower = np.where(overlaps, L3, np.maximum(L2, L3))

        good_mid = (good_25 + good_75) / 2.0
        poor_mid = (poor_25 + poor_75) / 2.0
        response_is_lower = good_mid < poor_mid

        global_zone1_is_response = response_is_lower[0] if not np.isnan(response_is_lower[0]) else True
        global_zone3_is_response = not response_is_lower[0] if not np.isnan(response_is_lower[0]) else False

        color1 = COLOR_RESPONSE if global_zone1_is_response else COLOR_NONRESPONSE
        ax_left.fill_between(bin_indices, L1, zone1_upper,
                             alpha=0.3, color=color1, edgecolor='none')

        color3 = COLOR_RESPONSE if global_zone3_is_response else COLOR_NONRESPONSE
        ax_left.fill_between(bin_indices, zone3_lower, L4,
                             alpha=0.3, color=color3, edgecolor='none')

        mixed_lower = np.where(overlaps, L2, np.nan)
        mixed_upper = np.where(overlaps, L3, np.nan)
        if np.any(overlaps):
            ax_left.fill_between(bin_indices, mixed_lower, mixed_upper,
                                 alpha=0.3, color=COLOR_MIXED, edgecolor='none',
                                 label='Mixed zone')

        ax_left.plot(bin_indices, good_med, linestyle='dashed',
                     color=COLOR_RESPONSE_DARK, linewidth=3.5,
                     label='Response median')
        ax_left.plot(bin_indices, poor_med, linestyle='dashed',
                     color=COLOR_NONRESPONSE_DARK, linewidth=3.5,
                     label='Non-response median')

        for i, label in enumerate(valid_labels):
            if label in param_raw:
                gv = param_raw[label]['good_vals']
                pv = param_raw[label]['poor_vals']
                if len(gv) > 0:
                    jitter = np.random.normal(i - 0.15, 0.06, size=len(gv))
                    ax_left.scatter(jitter, gv, alpha=0.3, s=20, color=COLOR_RESPONSE,
                                    edgecolors='none', zorder=3)
                if len(pv) > 0:
                    jitter = np.random.normal(i + 0.15, 0.06, size=len(pv))
                    ax_left.scatter(jitter, pv, alpha=0.3, s=20, color=COLOR_NONRESPONSE,
                                    edgecolors='none', zorder=3)

        ax_left.set_xlabel('Cumulative $EQD_2$ [Gy]', fontsize=18, fontfamily='Times New Roman')
        ax_left.set_ylabel(param_label, fontsize=18, fontfamily='Times New Roman')
        ax_left.set_xticks(bin_indices)
        ax_left.set_xticklabels(valid_labels, fontsize=14, rotation=30)
        for label in ax_left.get_xticklabels() + ax_left.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(18)
        if ylim is not None:
            ax_left.set_ylim(ylim)
        for spine in ax_left.spines.values():
            spine.set_linewidth(0.8)

        for h, l in zip(*ax_left.get_legend_handles_labels()):
            if l not in global_handles:
                global_handles[l] = h
                global_labels_order.append(l)

        ax_right = axes[row_idx, 1]

        valid_filled = _get_valid_labels(param_bands_filled, sorted_labels)
        n_fb = len(valid_filled)
        fb_indices = np.arange(n_fb)

        good_lower = np.full(n_fb, np.nan)
        good_upper = np.full(n_fb, np.nan)
        poor_lower = np.full(n_fb, np.nan)
        poor_upper = np.full(n_fb, np.nan)

        for i, label in enumerate(valid_filled):
            b = param_bands_filled[label]
            good_lower[i] = b['good_band'][0]
            good_upper[i] = b['good_band'][1]
            poor_lower[i] = b['poor_band'][0]
            poor_upper[i] = b['poor_band'][1]

        good_mid_f = (good_lower + good_upper) / 2.0
        poor_mid_f = (poor_lower + poor_upper) / 2.0
        global_good_is_upper = good_mid_f[0] > poor_mid_f[0]

        if global_good_is_upper:
            upper_color = COLOR_RESPONSE
            upper_label = 'Response band'
            lower_color = COLOR_NONRESPONSE
            lower_label = 'Non-response band'
        else:
            upper_color = COLOR_NONRESPONSE
            upper_label = 'Non-response band'
            lower_color = COLOR_RESPONSE
            lower_label = 'Response band'

        upper_lower = np.full(n_fb, np.nan)
        upper_upper = np.full(n_fb, np.nan)
        lower_lower = np.full(n_fb, np.nan)
        lower_upper = np.full(n_fb, np.nan)

        for i in range(n_fb):
            if np.isnan(good_lower[i]) or np.isnan(poor_lower[i]):
                continue
            if good_mid_f[i] > poor_mid_f[i]:
                upper_lower[i] = good_lower[i]
                upper_upper[i] = good_upper[i]
                lower_lower[i] = poor_lower[i]
                lower_upper[i] = poor_upper[i]
            else:
                upper_lower[i] = poor_lower[i]
                upper_upper[i] = poor_upper[i]
                lower_lower[i] = good_lower[i]
                lower_upper[i] = good_upper[i]

        ax_right.fill_between(fb_indices, upper_lower, upper_upper,
                              alpha=0.3, color=upper_color, edgecolor='none',
                              label=upper_label)
        ax_right.fill_between(fb_indices, lower_lower, lower_upper,
                              alpha=0.3, color=lower_color, edgecolor='none',
                              label=lower_label)

        good_names = list(param_good.keys())
        poor_names = list(param_poor.keys())

        good_case_name = good_names[min(good_case_idx, len(good_names) - 1)]
        poor_case_name = poor_names[min(poor_case_idx, len(poor_names) - 1)]

        print(f"  [{param_name}] Response case: {good_case_name} (idx={good_case_idx})")
        print(f"  [{param_name}] Non-response case: {poor_case_name} (idx={poor_case_idx})")

        good_case_data = param_good[good_case_name]
        poor_case_data = param_poor[poor_case_name]

        good_bin_vals = _get_patient_bin_values(good_case_data, dose_edges, valid_filled)
        poor_bin_vals = _get_patient_bin_values(poor_case_data, dose_edges, valid_filled)

        good_x, good_y = [], []
        for i, label in enumerate(valid_filled):
            v = good_bin_vals.get(label, np.nan)
            if not np.isnan(v):
                good_x.append(i)
                good_y.append(v)

        poor_x, poor_y = [], []
        for i, label in enumerate(valid_filled):
            v = poor_bin_vals.get(label, np.nan)
            if not np.isnan(v):
                poor_x.append(i)
                poor_y.append(v)

        if len(good_x) > 0:
            ax_right.plot(good_x, good_y, 'o-', color=COLOR_RESPONSE_DARK,
                          linewidth=2.5, markersize=10,
                          label='Response case', zorder=5)

        if len(poor_x) > 0:
            ax_right.plot(poor_x, poor_y, 's-', color=COLOR_NONRESPONSE_DARK,
                          linewidth=2.5, markersize=10,
                          label='Non-response case', zorder=5)

        ax_right.set_xlabel('Cumulative $EQD_2$ [Gy]', fontsize=18, fontfamily='Times New Roman')
        ax_right.set_ylabel(param_label, fontsize=18, fontfamily='Times New Roman')
        ax_right.set_xticks(fb_indices)
        ax_right.set_xticklabels(valid_filled, fontsize=14, rotation=30)
        for label in ax_right.get_xticklabels() + ax_right.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(18)
        if ylim is not None:
            ax_right.set_ylim(ylim)
        for spine in ax_right.spines.values():
            spine.set_linewidth(0.8)

        for h, l in zip(*ax_right.get_legend_handles_labels()):
            if l not in global_handles:
                global_handles[l] = h
                global_labels_order.append(l)

    desired_order = ['Response band', 'Non-response band',
                     'Response median', 'Non-response median', 'Mixed zone',
                     'Response case', 'Non-response case']
    ordered_h = [global_handles[k] for k in desired_order if k in global_handles]
    ordered_l = [k for k in desired_order if k in global_handles]

    n_items = len(ordered_l)
    ncol = 4
    nrow = (n_items + ncol - 1) // ncol

    reordered_h = []
    reordered_l = []
    for i in range(n_items):
        idx = (i % nrow) * ncol + (i // nrow)
        reordered_h.append(ordered_h[idx])
        reordered_l.append(ordered_l[idx])

    fig.legend(reordered_h, reordered_l, loc='lower center',
               fontsize=18, ncol=ncol, frameon=True,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.show()
    return fig, axes
if __name__ == "__main__":
    
    base_path = r'XXX'
    param_names = ['ADC', 'Dslow', 'Dfast', 'Ffast']
    param_labels = [r"$ADC \; [\times 10^{-3}\ \text{mm}^2/\text{s}]$",
                    r"$D \; [\times 10^{-3}\ \text{mm}^2/\text{s}]$",
                    r"$D* \; [\times 10^{-3}\ \text{mm}^2/\text{s}]$",
                    r"$f \;[\%]$"]
    multi_list = [1000, 1000, 1000, 100]
    
    QA_CSV_PATH = r'XXX'
    
    DOSE_EDGES = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    
    GOOD_CASE_IDX = 5
    POOR_CASE_IDX = 7
    
    qa_dict = load_qa_pass_dict(QA_CSV_PATH)
    
    all_params_bands_25_75 = {}
    all_params_bands_filled = {}
    all_params_raw_data = {}
    all_params_good_group = {}
    all_params_poor_group = {}
    all_dose_bin_labels = None
    
    for param_name, param_label, multi in zip(param_names, param_labels, multi_list):
        print(f"\n{'='*60}")
        print(f" parameter: {param_label} ({param_name})")
        print(f"{'='*60}")
    
        out_all = extract_adc_dose_means_4class(
            base_path, param_name, np.mean, multi=multi,
            qa_dict=qa_dict, alpha_beta=ALPHA_BETA
        )
        good_group = {**out_all[0], **out_all[1]}
        poor_group = {**out_all[2], **out_all[3]}
    
        bands_25_75, bands_filled, raw_data, dose_bin_labels = compute_dose_bands(
            good_group, poor_group, DOSE_EDGES, min_samples=2)
    
        print_dose_bin_summary(raw_data, dose_bin_labels)
    
        all_params_bands_25_75[param_name] = bands_25_75
        all_params_bands_filled[param_name] = bands_filled
        all_params_raw_data[param_name] = raw_data
        all_params_good_group[param_name] = good_group
        all_params_poor_group[param_name] = poor_group
        if all_dose_bin_labels is None:
            all_dose_bin_labels = dose_bin_labels
    
    YLIM_MANUAL = {
        'ADC':   (0.8, 1.9),
        'Dslow': (0.9, 1.15),
        'Dfast': (46, 50),
        'Ffast': (8, 25),
    }
    
    plot_dose_combined_figure(
        all_params_bands_25_75,
        all_params_bands_filled,
        all_params_raw_data,
        all_dose_bin_labels,
        all_params_good_group,
        all_params_poor_group,
        YLIM_MANUAL,
        param_labels,
        param_names,
        DOSE_EDGES,
        good_case_idx=GOOD_CASE_IDX,
        poor_case_idx=POOR_CASE_IDX,
        save_path=r'XXX'
    )