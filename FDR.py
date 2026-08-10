import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

ALPHA_BETA = 10.0

def traversalDir_FirstDir(path):
    return [os.path.join(path, file) for file in os.listdir(path)
            if os.path.exists(os.path.join(path, file))]


def load_qa_pass_dict(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        print("  No QA record found. All fractions will be used.")
        return None

    df = pd.read_csv(csv_path)
    qa_dict = {}
    for _, row in df.iterrows():
        patient = str(row['patient'])
        frac_num = int(row['fraction_number'])
        all_pass = row['all_pass'] if isinstance(row['all_pass'], bool) else (str(row['all_pass']).lower() == 'true')
        qa_dict[(patient, frac_num)] = all_pass

    n_pass = sum(1 for v in qa_dict.values() if v)
    print(f"  Loaded QA records: {len(qa_dict)} fractions, {n_pass} passed.")
    return qa_dict


def extract_adc_means_4class(base_path, parameter, func, multi=1, qa_dict=None):
    out = [{}, {}, {}, {}]
    for path1 in traversalDir_FirstDir(base_path):
        for patient in traversalDir_FirstDir(path1):
            try:
                cls = int(patient.split('=')[-1])
                total_N = int(patient.split('=')[-2])
                name = os.path.basename(patient).split('=')[0]
                sub_out = np.full(total_N, np.nan)
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

                        mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat64)
                        mask_array = mask_img > 0.5
                        mask_flat = sitk.GetArrayFromImage(mask_array).reshape(-1,)

                        ADC = sitk.GetArrayFromImage(
                            sitk.ReadImage(adc_path, sitk.sitkFloat64)).reshape(-1,)
                        roi = ADC[np.where(mask_flat)] * multi
                        if roi.size > 0:
                            sub_out[fraction_N - 1] = func(roi)
                    except Exception:
                        continue
                out[cls][name] = sub_out
            except Exception:
                continue
    return out


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
                        mask_array = sitk.GetArrayFromImage(mask_img) > 0.5
                        mask_flat = mask_array.reshape(-1,)

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


def fraction_statistical_analysis(good_group, poor_group, max_fraction=8, alpha=0.05):
    all_patients = {}
    for name, arr in good_group.items():
        all_patients[name] = {'values': arr, 'label': 1}
    for name, arr in poor_group.items():
        all_patients[name] = {'values': arr, 'label': 0}

    frac_good = {}
    frac_poor = {}
    for frac in range(1, max_fraction + 1):
        frac_good[frac] = []
        frac_poor[frac] = []
        for name, info in all_patients.items():
            if frac <= len(info['values']) and not np.isnan(info['values'][frac - 1]):
                if info['label'] == 1:
                    frac_good[frac].append(info['values'][frac - 1])
                else:
                    frac_poor[frac].append(info['values'][frac - 1])

    between_pvals = {}
    between_stats = {}
    for frac in range(1, max_fraction + 1):
        g = frac_good[frac]
        p = frac_poor[frac]
        if len(g) >= 3 and len(p) >= 3:
            stat, pval = mannwhitneyu(g, p, alternative='two-sided')
            between_pvals[frac] = pval
            between_stats[frac] = {'U': stat, 'n_good': len(g), 'n_poor': len(p)}
        else:
            between_pvals[frac] = np.nan
            between_stats[frac] = {'U': np.nan, 'n_good': len(g), 'n_poor': len(p)}

    frac_list = sorted(between_pvals.keys())
    pvals_list = [between_pvals[f] for f in frac_list]
    valid_mask = ~np.isnan(pvals_list)
    valid_pvals = np.array(pvals_list)[valid_mask]
    if len(valid_pvals) > 0:
        _, fdr_between, _, _ = multipletests(valid_pvals, alpha=alpha, method='fdr_bh')
        fdr_between_dict = {f: fdr_between[i] for i, f in enumerate(np.array(frac_list)[valid_mask])}
    else:
        fdr_between_dict = {}

    within_pvals_good = {}
    within_pvals_poor = {}
    within_stats_good = {}
    within_stats_poor = {}

    good_names = list(good_group.keys())
    poor_names = list(poor_group.keys())

    for frac in range(2, max_fraction + 1):
        paired_good = []
        for name in good_names:
            arr = good_group[name]
            if (frac <= len(arr) and 1 <= len(arr)
                and not np.isnan(arr[0]) and not np.isnan(arr[frac - 1])):
                paired_good.append((arr[0], arr[frac - 1]))
        if len(paired_good) >= 5:
            base_vals = [x[0] for x in paired_good]
            curr_vals = [x[1] for x in paired_good]
            stat, pval = wilcoxon(base_vals, curr_vals, alternative='two-sided')
            within_pvals_good[frac] = pval
            within_stats_good[frac] = {'W': stat, 'n_pairs': len(paired_good)}
        else:
            within_pvals_good[frac] = np.nan
            within_stats_good[frac] = {'W': np.nan, 'n_pairs': len(paired_good)}

        paired_poor = []
        for name in poor_names:
            arr = poor_group[name]
            if (frac <= len(arr) and 1 <= len(arr)
                and not np.isnan(arr[0]) and not np.isnan(arr[frac - 1])):
                paired_poor.append((arr[0], arr[frac - 1]))
        if len(paired_poor) >= 3:
            base_vals = [x[0] for x in paired_poor]
            curr_vals = [x[1] for x in paired_poor]
            stat, pval = wilcoxon(base_vals, curr_vals, alternative='two-sided')
            within_pvals_poor[frac] = pval
            within_stats_poor[frac] = {'W': stat, 'n_pairs': len(paired_poor)}
        else:
            within_pvals_poor[frac] = np.nan
            within_stats_poor[frac] = {'W': np.nan, 'n_pairs': len(paired_poor)}

    frac_list_w = sorted(within_pvals_good.keys())
    all_pvals_w = [within_pvals_good[f] for f in frac_list_w] + [within_pvals_poor[f] for f in frac_list_w]
    valid_mask_w = ~np.isnan(all_pvals_w)
    valid_pvals_w = np.array(all_pvals_w)[valid_mask_w]
    if len(valid_pvals_w) > 0:
        _, fdr_w, _, _ = multipletests(valid_pvals_w, alpha=alpha, method='fdr_bh')
        fdr_w_good = {}
        fdr_w_poor = {}
        w_idx = 0
        for f in frac_list_w:
            if not np.isnan(within_pvals_good[f]):
                fdr_w_good[f] = fdr_w[w_idx]
                w_idx += 1
            else:
                fdr_w_good[f] = np.nan
        for f in frac_list_w:
            if not np.isnan(within_pvals_poor[f]):
                fdr_w_poor[f] = fdr_w[w_idx]
                w_idx += 1
            else:
                fdr_w_poor[f] = np.nan
    else:
        fdr_w_good = {}
        fdr_w_poor = {}

    return {
        'between': {'pvals': between_pvals, 'fdr': fdr_between_dict, 'stats': between_stats},
        'within_good': {'pvals': within_pvals_good, 'fdr': fdr_w_good, 'stats': within_stats_good},
        'within_poor': {'pvals': within_pvals_poor, 'fdr': fdr_w_poor, 'stats': within_stats_poor}
    }


def dose_statistical_analysis(good_group, poor_group, dose_edges, min_samples=3, alpha=0.05):

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

    dose_labels = ["0"]
    dose_labels.append(f"(0,{dose_edges[0]}]")
    for i in range(len(dose_edges) - 1):
        dose_labels.append(f"({dose_edges[i]},{dose_edges[i+1]}]")

    bin_good = {}
    bin_poor = {}

    bin_good["0"] = _extract_patient_bin_means(good_group, lambda d: d == 0)
    bin_poor["0"] = _extract_patient_bin_means(poor_group, lambda d: d == 0)

    label = dose_labels[1]
    bin_good[label] = _extract_patient_bin_means(
        good_group, lambda d, hi=dose_edges[0]: 0 < d <= hi)
    bin_poor[label] = _extract_patient_bin_means(
        poor_group, lambda d, hi=dose_edges[0]: 0 < d <= hi)

    for i in range(len(dose_edges) - 1):
        label = dose_labels[i + 2]
        low = dose_edges[i]
        high = dose_edges[i + 1]
        bin_good[label] = _extract_patient_bin_means(
            good_group, lambda d, lo=low, hi=high: lo < d <= hi)
        bin_poor[label] = _extract_patient_bin_means(
            poor_group, lambda d, lo=low, hi=high: lo < d <= hi)

    valid_labels = []
    for label in dose_labels:
        if label in bin_good and label in bin_poor:
            if len(bin_good[label]) >= min_samples and len(bin_poor[label]) >= min_samples:
                valid_labels.append(label)

    print(f"  Valid bins: {len(valid_labels)}/{len(dose_labels)}")
    for label in dose_labels:
        ng = len(bin_good.get(label, {}))
        np_ = len(bin_poor.get(label, {}))
        mark = " ✓" if label in valid_labels else ""
        print(f"    {label:<20} good={ng}, poor={np_}{mark}")

    between_pvals = {}
    between_stats = {}
    for label in valid_labels:
        g = np.array(list(bin_good[label].values()))
        p = np.array(list(bin_poor[label].values()))
        if len(g) >= 3 and len(p) >= 3:
            stat, pval = mannwhitneyu(g, p, alternative='two-sided')
            between_pvals[label] = pval
            between_stats[label] = {'U': stat, 'n_good': len(g), 'n_poor': len(p)}
        else:
            between_pvals[label] = np.nan
            between_stats[label] = {'U': np.nan, 'n_good': len(g), 'n_poor': len(p)}

    pvals_list = [between_pvals[l] for l in valid_labels]
    valid_mask = ~np.isnan(pvals_list)
    valid_pvals = np.array(pvals_list)[valid_mask]
    if len(valid_pvals) > 0:
        _, fdr_between, _, _ = multipletests(valid_pvals, alpha=alpha, method='fdr_bh')
        fdr_between_dict = {l: fdr_between[i] for i, l in enumerate(np.array(valid_labels)[valid_mask])}
    else:
        fdr_between_dict = {}

    within_pvals_good = {}
    within_pvals_poor = {}
    within_stats_good = {}
    within_stats_poor = {}

    base_good = bin_good.get("0", {})
    base_poor = bin_poor.get("0", {})

    for label in valid_labels:
        if label == "0":
            continue

        paired_good = []
        for name in base_good:
            if name in bin_good[label]:
                paired_good.append((base_good[name], bin_good[label][name]))
        if len(paired_good) >= 5:
            stat, pval = wilcoxon([x[0] for x in paired_good],
                                   [x[1] for x in paired_good],
                                   alternative='two-sided')
            within_pvals_good[label] = pval
            within_stats_good[label] = {'W': stat, 'n_pairs': len(paired_good)}
        else:
            within_pvals_good[label] = np.nan
            within_stats_good[label] = {'W': np.nan, 'n_pairs': len(paired_good)}

        paired_poor = []
        for name in base_poor:
            if name in bin_poor[label]:
                paired_poor.append((base_poor[name], bin_poor[label][name]))
        if len(paired_poor) >= 3:
            stat, pval = wilcoxon([x[0] for x in paired_poor],
                                   [x[1] for x in paired_poor],
                                   alternative='two-sided')
            within_pvals_poor[label] = pval
            within_stats_poor[label] = {'W': stat, 'n_pairs': len(paired_poor)}
        else:
            within_pvals_poor[label] = np.nan
            within_stats_poor[label] = {'W': np.nan, 'n_pairs': len(paired_poor)}

    within_labels = [l for l in valid_labels if l != "0"]
    pvals_w = [within_pvals_good[l] for l in within_labels] + [within_pvals_poor[l] for l in within_labels]
    valid_mask_w = ~np.isnan(pvals_w)
    valid_pvals_w = np.array(pvals_w)[valid_mask_w]
    if len(valid_pvals_w) > 0:
        _, fdr_w, _, _ = multipletests(valid_pvals_w, alpha=alpha, method='fdr_bh')
        fdr_w_good = {}
        fdr_w_poor = {}
        w_idx = 0
        for label in within_labels:
            if not np.isnan(within_pvals_good[label]):
                fdr_w_good[label] = fdr_w[w_idx]
                w_idx += 1
            else:
                fdr_w_good[label] = np.nan
        for label in within_labels:
            if not np.isnan(within_pvals_poor[label]):
                fdr_w_poor[label] = fdr_w[w_idx]
                w_idx += 1
            else:
                fdr_w_poor[label] = np.nan
    else:
        fdr_w_good = {}
        fdr_w_poor = {}

    return {
        'between': {'pvals': between_pvals, 'fdr': fdr_between_dict, 'stats': between_stats},
        'within_good': {'pvals': within_pvals_good, 'fdr': fdr_w_good, 'stats': within_stats_good},
        'within_poor': {'pvals': within_pvals_poor, 'fdr': fdr_w_poor, 'stats': within_stats_poor}
    }


def print_fraction_stats(results, param_name):
    between = results['between']
    w_good = results['within_good']
    w_poor = results['within_poor']

    print(f"\n{'='*100}")
    print(f"  Fraction Analysis: {param_name}")
    print(f"{'='*100}")

    print(f"\n  1. Between-group (good vs poor, Mann-Whitney U, FDR corrected)")
    header = f"  {'Fraction':<8} {'n_good':<10} {'n_poor':<10} {'p value':<12} {'FDR':<12} {'Significance':<10}"
    print(header)
    print(f"  {'-'*62}")
    for frac in sorted(between['pvals'].keys()):
        p = between['pvals'][frac]
        fdr = between['fdr'].get(frac, np.nan)
        if not np.isnan(p):
            sig = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'ns'))
            print(f"  {frac:<8} {between['stats'][frac]['n_good']:<10} "
                  f"{between['stats'][frac]['n_poor']:<10} {p:<12.4f} {fdr:<12.4f} {sig:<10}")
        else:
            print(f"  {frac:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print(f"\n  2. Within-group - Response (vs fraction 1, Wilcoxon, FDR corrected)")
    header = f"  {'Fraction':<8} {'n_pairs':<10} {'p value':<12} {'FDR':<12} {'Significance':<10}"
    print(header)
    print(f"  {'-'*52}")
    for frac in sorted(w_good['pvals'].keys()):
        p = w_good['pvals'][frac]
        fdr = w_good['fdr'].get(frac, np.nan)
        if not np.isnan(p):
            sig = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'ns'))
            print(f"  {frac:<8} {w_good['stats'][frac]['n_pairs']:<10} "
                  f"{p:<12.4f} {fdr:<12.4f} {sig:<10}")
        else:
            print(f"  {frac:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print(f"\n  3. Within-group - Non-response (vs fraction 1, Wilcoxon, FDR corrected)")
    header = f"  {'Fraction':<8} {'n_pairs':<10} {'p value':<12} {'FDR':<12} {'Significance':<10}"
    print(header)
    print(f"  {'-'*52}")
    for frac in sorted(w_poor['pvals'].keys()):
        p = w_poor['pvals'][frac]
        fdr = w_poor['fdr'].get(frac, np.nan)
        if not np.isnan(p):
            sig = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'ns'))
            print(f"  {frac:<8} {w_poor['stats'][frac]['n_pairs']:<10} "
                  f"{p:<12.4f} {fdr:<12.4f} {sig:<10}")
        else:
            print(f"  {frac:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")


def _sort_dose_label(label):
    if label == "0":
        return -1
    clean = label.replace("(", "").replace("]", "").replace("[", "").replace(")", "")
    return float(clean.split(",")[0])


def print_dose_stats(results, param_name):
    between = results['between']
    w_good = results['within_good']
    w_poor = results['within_poor']

    print(f"\n{'='*100}")
    print(f"  Dose Analysis (Cumulative EQD2, alpha/beta = 10 Gy, D10): {param_name}")
    print(f"{'='*100}")

    print(f"\n  1. Between-group (good vs poor, Mann-Whitney U, FDR corrected)")
    header = f"  {'Dose bin(EQD2)':<20} {'n_good':<10} {'n_poor':<10} {'p value':<12} {'FDR':<12} {'Significance':<10}"
    print(header)
    print(f"  {'-'*74}")
    for label in sorted(between['pvals'].keys(), key=_sort_dose_label):
        p = between['pvals'][label]
        fdr = between['fdr'].get(label, np.nan)
        if not np.isnan(p):
            sig = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'ns'))
            print(f"  {label:<20} {between['stats'][label]['n_good']:<10} "
                  f"{between['stats'][label]['n_poor']:<10} {p:<12.4f} {fdr:<12.4f} {sig:<10}")
        else:
            print(f"  {label:<20} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print(f"\n  2. Within-group - Response (vs bin 0, Wilcoxon, FDR corrected)")
    header = f"  {'Dose bin(EQD2)':<20} {'n_pairs':<10} {'p value':<12} {'FDR':<12} {'Significance':<10}"
    print(header)
    print(f"  {'-'*64}")
    for label in sorted(w_good['pvals'].keys(), key=_sort_dose_label):
        p = w_good['pvals'][label]
        fdr = w_good['fdr'].get(label, np.nan)
        if not np.isnan(p):
            sig = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'ns'))
            print(f"  {label:<20} {w_good['stats'][label]['n_pairs']:<10} "
                  f"{p:<12.4f} {fdr:<12.4f} {sig:<10}")
        else:
            print(f"  {label:<20} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print(f"\n  3. Within-group - Non-response (vs bin 0, Wilcoxon, FDR corrected)")
    header = f"  {'Dose bin(EQD2)':<20} {'n_pairs':<10} {'p value':<12} {'FDR':<12} {'Significance':<10}"
    print(header)
    print(f"  {'-'*64}")
    for label in sorted(w_poor['pvals'].keys(), key=_sort_dose_label):
        p = w_poor['pvals'][label]
        fdr = w_poor['fdr'].get(label, np.nan)
        if not np.isnan(p):
            sig = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'ns'))
            print(f"  {label:<20} {w_poor['stats'][label]['n_pairs']:<10} "
                  f"{p:<12.4f} {fdr:<12.4f} {sig:<10}")
        else:
            print(f"  {label:<20} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")


if __name__ == "__main__":

    base_path = r'XXX'
    param_names = ['ADC', 'Dslow', 'Dfast', 'Ffast']
    param_labels = [r"ADC", r"D", r"D*", r"f"]
    multi_list = [1000, 1000, 1000, 100]

    QA_CSV_PATH = r'XXX'

    DOSE_EDGES = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    MAX_FRACTION = 9

    qa_dict = load_qa_pass_dict(QA_CSV_PATH)

    for param_name, param_label, multi in zip(param_names, param_labels, multi_list):
        print(f"\n{'#'*100}")
        print(f"#  Parameter: {param_label} ({param_name})")
        print(f"{'#'*100}")

        out_frac = extract_adc_means_4class(base_path, param_name, np.mean, multi=multi, qa_dict=qa_dict)
        good_frac = {**out_frac[0], **out_frac[1]}
        poor_frac = {**out_frac[2], **out_frac[3]}

        frac_results = fraction_statistical_analysis(good_frac, poor_frac, MAX_FRACTION)
        print_fraction_stats(frac_results, param_name)

        out_dose = extract_adc_dose_means_4class(
            base_path, param_name, np.mean, multi=multi,
            qa_dict=qa_dict, alpha_beta=ALPHA_BETA
        )
        good_dose = {**out_dose[0], **out_dose[1]}
        poor_dose = {**out_dose[2], **out_dose[3]}

        dose_results = dose_statistical_analysis(good_dose, poor_dose, DOSE_EDGES)
        print_dose_stats(dose_results, param_name)

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)
