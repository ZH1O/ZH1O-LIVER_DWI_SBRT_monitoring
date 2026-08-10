import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
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

PALETTE = sns.cubehelix_palette(4, rot=-.5, gamma=0.5, dark=0.1)

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


def extract_data_4class(base_path, parameter, func, multi=1, qa_dict=None):
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

                        adc_path = os.path.join(fraction, 'parameter_map', parameter + '.nii.gz')
                        mask_path = os.path.join(fraction, 'RTst_registration', 'Tumor.nii.gz')

                        mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat64)
                        mask_array = mask_img > 0.5
                        mask_flat = sitk.GetArrayFromImage(mask_array).reshape(-1,)

                        ADC = sitk.GetArrayFromImage(sitk.ReadImage(adc_path, sitk.sitkFloat64)).reshape(-1,)
                        roi = ADC[np.where(mask_flat)] * multi
                        if roi.size > 0:
                            sub_out[fraction_N - 1] = func(roi)
                    except Exception:
                        continue
                out[cls][name] = sub_out
            except Exception:
                continue
    return out


def extract_data_dose_4class(base_path, parameter, func, multi=1, qa_dict=None,
                              alpha_beta=10.0):
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

                        adc_path = os.path.join(fraction, 'parameter_map', parameter + '.nii.gz')
                        mask_path = os.path.join(fraction, 'RTst_registration', 'Tumor.nii.gz')
                        dose_path = os.path.join(fraction, 'RT', 'Dose.nii.gz')

                        mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat64)
                        mask_array = sitk.GetArrayFromImage(mask_img) > 0.5
                        mask_flat = mask_array.reshape(-1,)

                        ADC = sitk.GetArrayFromImage(sitk.ReadImage(adc_path, sitk.sitkFloat64)).reshape(-1,)
                        roi_adc = ADC[np.where(mask_flat)] * multi
                        if roi_adc.size > 0:
                            adc_out[fraction_N - 1] = func(roi_adc)

                        Dose = sitk.GetArrayFromImage(sitk.ReadImage(dose_path, sitk.sitkFloat64)).reshape(-1,)
                        roi_dose = Dose[np.where(mask_flat)]
                        if roi_dose.size > 0:
                            dose_out[fraction_N - 1] = round(np.mean(roi_dose), 2)
                    except Exception:
                        continue

                for i in range(total_N):
                    if np.isnan(dose_out[i]) and i > 0:
                        dose_out[i] = dose_out[i - 1]

                cum_dose = np.cumsum(np.nan_to_num(dose_out))
                cum_dose = np.insert(cum_dose[:-1], 0, 0)

                eqd2_per_frac = dose_out * (alpha_beta + dose_out) / (alpha_beta + 2)
                cum_eqd2 = np.cumsum(np.nan_to_num(eqd2_per_frac))
                cum_eqd2 = np.insert(cum_eqd2[:-1], 0, 0)

                for i in range(total_N):
                    if np.isnan(adc_out[i]):
                        cum_dose[i] = np.nan
                        cum_eqd2[i] = np.nan
                    if i > 0 and np.isnan(dose_out[i - 1]):
                        cum_dose[i] = np.nan
                        cum_eqd2[i] = np.nan

                out[cls][name] = {
                    "adc": adc_out,
                    "dose": dose_out,
                    "cum_dose": cum_dose,
                    "cum_eqd2": cum_eqd2
                }
            except Exception:
                continue
    return out


def auc_ci_delong(y_true, y_score):
    n1, n0 = np.sum(y_true == 1), np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan
    sp, sn = y_score[y_true == 1], y_score[y_true == 0]
    auc_val = (sum(1 for s in sp for t in sn if s > t) +
               0.5 * sum(1 for s in sp for t in sn if s == t)) / (n1 * n0)
    def placebo(scores, comp, is_pos):
        r = np.zeros(len(scores))
        for i, s in enumerate(scores):
            r[i] = (sum(1 for c in comp if (is_pos and s > c) or (not is_pos and c > s)) +
                    0.5 * sum(1 for c in comp if s == c)) / len(comp)
        return r
    vp, vn = placebo(sp, sn, True), placebo(sn, sp, False)
    se = np.sqrt(np.var(vp, ddof=1) / n1 + np.var(vn, ddof=1) / n0)
    z = norm.ppf(0.975)
    return auc_val, max(0, auc_val - z * se), min(1, auc_val + z * se)


def wilson_ci(p, n):
    if n == 0 or np.isnan(p):
        return np.nan, np.nan
    z = 1.96
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return max(0, center - margin), min(1, center + margin)


def delong_test(y_true, y_score1, y_score2):
    n1, n0 = np.sum(y_true == 1), np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan, np.nan
    sp1, sn1 = y_score1[y_true == 1], y_score1[y_true == 0]
    sp2, sn2 = y_score2[y_true == 1], y_score2[y_true == 0]
    def auc(sp, sn):
        return (sum(1 for s in sp for t in sn if s > t) +
                0.5 * sum(1 for s in sp for t in sn if s == t)) / (len(sp) * len(sn))
    a1, a2 = auc(sp1, sn1), auc(sp2, sn2)
    def placebo(scores, comp, is_pos):
        r = np.zeros(len(scores))
        for i, s in enumerate(scores):
            r[i] = (sum(1 for c in comp if (is_pos and s > c) or (not is_pos and c > s)) +
                    0.5 * sum(1 for c in comp if s == c)) / len(comp)
        return r
    v1p, v1n = placebo(sp1, sn1, True), placebo(sn1, sp1, False)
    v2p, v2n = placebo(sp2, sn2, True), placebo(sn2, sp2, False)
    cp = np.cov(v1p, v2p, ddof=1)[0, 1] / n1
    cn = np.cov(v1n, v2n, ddof=1)[0, 1] / n0
    v1 = np.var(v1p, ddof=1) / n1 + np.var(v1n, ddof=1) / n0
    v2 = np.var(v2p, ddof=1) / n1 + np.var(v2n, ddof=1) / n0
    vd = v1 + v2 - 2 * (cp + cn)
    z = (a1 - a2) / np.sqrt(max(vd, 1e-10))
    p = 2 * (1 - norm.cdf(abs(z)))
    return a1, a2, p, z

def compute_filled_bands(good_vals, poor_vals, verbose=False):
    if len(good_vals) < 3 or len(poor_vals) < 3:
        return None

    good_median = np.median(good_vals)
    poor_median = np.median(poor_vals)
    good_25 = np.percentile(good_vals, 25)
    good_75 = np.percentile(good_vals, 75)
    poor_25 = np.percentile(poor_vals, 25)
    poor_75 = np.percentile(poor_vals, 75)

    static_mean_median = (good_median + poor_median) / 2.0
    if good_75 < poor_25:
        static_mean_boundary = (good_75 + poor_25) / 2.0
    elif poor_75 < good_25:
        static_mean_boundary = (poor_75 + good_25) / 2.0
    else:
        static_mean_boundary = (good_median + poor_median) / 2.0

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
                        'split_line': split_line, 'expanded': True,
                        'good_is_upper': False,
                        'static_mean_median': static_mean_median,
                        'static_mean_boundary': static_mean_boundary}
        split_line = (good_75 + poor_25) / 2.0
        return {'good_band': (good_25, split_line),
                'poor_band': (split_line, poor_75),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'expanded': True,
                'good_is_upper': False,
                'static_mean_median': static_mean_median,
                'static_mean_boundary': static_mean_boundary}

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
                        'split_line': split_line, 'expanded': True,
                        'good_is_upper': True,
                        'static_mean_median': static_mean_median,
                        'static_mean_boundary': static_mean_boundary}
        split_line = (poor_75 + good_25) / 2.0
        return {'good_band': (split_line, good_75),
                'poor_band': (poor_25, split_line),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'expanded': True,
                'good_is_upper': True,
                'static_mean_median': static_mean_median,
                'static_mean_boundary': static_mean_boundary}

    if good_is_upper:
        for step in range(1, 26):
            if 25 + step > 100 or 75 - step < 0:
                break
            good_lower = np.percentile(good_vals, 25 + step)
            poor_upper = np.percentile(poor_vals, 75 - step)
            if poor_upper <= good_lower:
                split_line = (poor_upper + good_lower) / 2.0
                return {'good_band': (split_line, good_75),
                        'poor_band': (poor_25, split_line),
                        'good_median': good_median, 'poor_median': poor_median,
                        'overlap': False, 'separation': 0.0,
                        'split_line': split_line, 'shrunk': True,
                        'good_is_upper': True,
                        'static_mean_median': static_mean_median,
                        'static_mean_boundary': static_mean_boundary}
        split_line = (good_median + poor_median) / 2.0
        return {'good_band': (split_line, good_75),
                'poor_band': (poor_25, split_line),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'shrunk': True,
                'good_is_upper': True,
                'static_mean_median': static_mean_median,
                'static_mean_boundary': static_mean_boundary}
    else:
        for step in range(1, 26):
            if 25 + step > 100 or 75 - step < 0:
                break
            poor_lower = np.percentile(poor_vals, 25 + step)
            good_upper = np.percentile(good_vals, 75 - step)
            if good_upper <= poor_lower:
                split_line = (good_upper + poor_lower) / 2.0
                return {'good_band': (good_25, split_line),
                        'poor_band': (split_line, poor_75),
                        'good_median': good_median, 'poor_median': poor_median,
                        'overlap': False, 'separation': 0.0,
                        'split_line': split_line, 'shrunk': True,
                        'good_is_upper': False,
                        'static_mean_median': static_mean_median,
                        'static_mean_boundary': static_mean_boundary}
        split_line = (good_median + poor_median) / 2.0
        return {'good_band': (good_25, split_line),
                'poor_band': (split_line, poor_75),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0,
                'split_line': split_line, 'shrunk': True,
                'good_is_upper': False,
                'static_mean_median': static_mean_median,
                'static_mean_boundary': static_mean_boundary}

def classify_patient_raw(value, bands):
    if bands is None:
        return 0.5
    gl, gu = bands['good_band']
    pl, pu = bands['poor_band']
    split_line = bands.get('split_line', (gu + pl) / 2.0)

    if gl <= value <= gu:
        if gu - gl > 0:
            dist = abs(value - split_line) / (gu - gl)
            return 0.5 + 0.5 * dist
        return 0.75
    if pl <= value <= pu:
        if pu - pl > 0:
            dist = abs(value - split_line) / (pu - pl)
            return 0.5 - 0.5 * dist
        return 0.25
    if value < min(gl, pl):
        return 0.0 if gl > pl else 1.0
    return 1.0 if gl > pl else 0.0

def fraction_loocv(good_group, poor_group, max_fraction=8, verbose=False):
    all_patients = {}
    for name, arr in good_group.items():
        all_patients[name] = {'values': arr, 'label': 1}
    for name, arr in poor_group.items():
        all_patients[name] = {'values': arr, 'label': 0}
    pnames = list(all_patients.keys())

    frac_metrics = []
    frac_predictions = {}
    frac_raw_values = {}

    for frac in range(1, max_fraction + 1):
        valid = [n for n in pnames if frac <= len(all_patients[n]['values'])
                 and not np.isnan(all_patients[n]['values'][frac - 1])]
        n_valid = len(valid)
        n_good = sum(1 for n in valid if all_patients[n]['label'] == 1)
        n_poor = sum(1 for n in valid if all_patients[n]['label'] == 0)

        all_good = np.array([all_patients[n]['values'][frac - 1]
                              for n in valid if all_patients[n]['label'] == 1])
        all_poor = np.array([all_patients[n]['values'][frac - 1]
                              for n in valid if all_patients[n]['label'] == 0])

        frac_raw_values[frac] = {
            'good_vals': all_good,
            'poor_vals': all_poor
        }

        if n_valid < 6 or n_good < 3 or n_poor < 3:
            frac_metrics.append({'Fraction': frac, 'n': n_valid,
                                 'AUC': np.nan, 'AUC_CI_low': np.nan, 'AUC_CI_up': np.nan,
                                 'Accuracy': np.nan, 'Accuracy_CI_low': np.nan, 'Accuracy_CI_up': np.nan,
                                 'Sensitivity': np.nan, 'Sensitivity_CI_low': np.nan, 'Sensitivity_CI_up': np.nan,
                                 'Specificity': np.nan, 'Specificity_CI_low': np.nan, 'Specificity_CI_up': np.nan,
                                 'split_line': np.nan, 'static_mean': np.nan,
                                 'good_is_upper': np.nan})
            continue

        global_bands = compute_filled_bands(all_good, all_poor, verbose=False)
        if global_bands is None:
            global_good_is_upper = np.nan
        else:
            global_good_is_upper = global_bands['good_is_upper']

        yt, yp = [], []
        split_lines = []
        static_means = []

        for test_name in valid:
            train = [n for n in valid if n != test_name]
            tg = np.array([all_patients[n]['values'][frac - 1] for n in train if all_patients[n]['label'] == 1])
            tp_ = np.array([all_patients[n]['values'][frac - 1] for n in train if all_patients[n]['label'] == 0])
            tv = all_patients[test_name]['values'][frac - 1]
            tl = all_patients[test_name]['label']
            if len(tg) < 3 or len(tp_) < 3:
                continue
            bands = compute_filled_bands(tg, tp_, verbose=False)
            if bands is None:
                continue
            split_lines.append(bands['split_line'])
            static_means.append(bands.get('static_mean_median', np.nan))

            raw_prob = classify_patient_raw(tv, bands)

            fold_good_is_upper = bands['good_is_upper']
            if not np.isnan(global_good_is_upper) and fold_good_is_upper != global_good_is_upper:
                raw_prob = 1.0 - raw_prob

            yt.append(tl)
            yp.append(raw_prob)

        if len(yt) == 0:
            frac_metrics.append({'Fraction': frac, 'n': n_valid,
                                 'AUC': np.nan, 'AUC_CI_low': np.nan, 'AUC_CI_up': np.nan,
                                 'Accuracy': np.nan, 'Accuracy_CI_low': np.nan, 'Accuracy_CI_up': np.nan,
                                 'Sensitivity': np.nan, 'Sensitivity_CI_low': np.nan, 'Sensitivity_CI_up': np.nan,
                                 'Specificity': np.nan, 'Specificity_CI_low': np.nan, 'Specificity_CI_up': np.nan,
                                 'split_line': np.nan, 'static_mean': np.nan,
                                 'good_is_upper': np.nan})
            continue

        yt = np.array(yt)
        yp = np.array(yp)

        auc, ci_l, ci_u = auc_ci_delong(yt, yp)

        y_pred = (yp >= 0.5).astype(int)
        tp_cnt = np.sum((yt == 1) & (y_pred == 1))
        tn_cnt = np.sum((yt == 0) & (y_pred == 0))
        n = len(yt)
        np_ = int(np.sum(yt == 1))
        nn_ = int(np.sum(yt == 0))
        acc = (tp_cnt + tn_cnt) / n
        sen = tp_cnt / np_ if np_ > 0 else np.nan
        spe = tn_cnt / nn_ if nn_ > 0 else np.nan
        acl, acu = wilson_ci(acc, n)
        sl, su = wilson_ci(sen, np_) if np_ > 0 else (np.nan, np.nan)
        spl, spu = wilson_ci(spe, nn_) if nn_ > 0 else (np.nan, np.nan)

        avg_split = np.mean(split_lines) if split_lines else np.nan
        avg_static = np.mean(static_means) if static_means else np.nan

        if verbose:
            dir_str = "good up" if global_good_is_upper else "good down"
            print(f"  Fraction {frac}: AUC={auc:.3f}, direction={dir_str}, "
                  f"split_line={avg_split:.4f}")

        frac_metrics.append({'Fraction': frac, 'n': n_valid,
                             'AUC': auc, 'AUC_CI_low': ci_l, 'AUC_CI_up': ci_u,
                             'Accuracy': acc, 'Accuracy_CI_low': acl, 'Accuracy_CI_up': acu,
                             'Sensitivity': sen, 'Sensitivity_CI_low': sl, 'Sensitivity_CI_up': su,
                             'Specificity': spe, 'Specificity_CI_low': spl, 'Specificity_CI_up': spu,
                             'split_line': avg_split, 'static_mean': avg_static,
                             'good_is_upper': global_good_is_upper})
        frac_predictions[frac] = (yt, yp)

    return pd.DataFrame(frac_metrics), frac_predictions, frac_raw_values

def dose_bin_loocv(good_group, poor_group, dose_edges, min_samples=3, verbose=False):
    dose_labels = ["0"]
    dose_labels.append(f"(0,{dose_edges[0]}]")
    for i in range(len(dose_edges) - 1):
        dose_labels.append(f"({dose_edges[i]},{dose_edges[i+1]}]")

    bin_data = {l: {'good_means': {}, 'poor_means': {}} for l in dose_labels}

    for name, ad in good_group.items():
        if len(ad['adc']) > 0 and not np.isnan(ad['adc'][0]):
            bin_data["0"]['good_means'][name] = ad['adc'][0]
    for name, ad in poor_group.items():
        if len(ad['adc']) > 0 and not np.isnan(ad['adc'][0]):
            bin_data["0"]['poor_means'][name] = ad['adc'][0]

    label = dose_labels[1]
    for name, ad in good_group.items():
        vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                if not np.isnan(d) and not np.isnan(a) and 0 < d <= dose_edges[0]]
        if vals:
            bin_data[label]['good_means'][name] = np.mean(vals)
    for name, ad in poor_group.items():
        vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                if not np.isnan(d) and not np.isnan(a) and 0 < d <= dose_edges[0]]
        if vals:
            bin_data[label]['poor_means'][name] = np.mean(vals)

    for i in range(len(dose_edges) - 1):
        label = dose_labels[i + 2]
        low = dose_edges[i]
        high = dose_edges[i + 1]
        for name, ad in good_group.items():
            vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                    if not np.isnan(d) and not np.isnan(a) and low < d <= high]
            if vals:
                bin_data[label]['good_means'][name] = np.mean(vals)
        for name, ad in poor_group.items():
            vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                    if not np.isnan(d) and not np.isnan(a) and low < d <= high]
            if vals:
                bin_data[label]['poor_means'][name] = np.mean(vals)

    bin_metrics = {}
    bin_predictions = {}
    bin_raw_values = {}

    for label in dose_labels:
        good_means = bin_data[label]['good_means']
        poor_means = bin_data[label]['poor_means']

        ng = len(good_means)
        np_ = len(poor_means)

        bin_raw_values[label] = {
            'good_vals': np.array(list(good_means.values())),
            'poor_vals': np.array(list(poor_means.values()))
        }

        if ng < min_samples or np_ < min_samples:
            bin_metrics[label] = None
            continue

        all_vals = np.array(list(good_means.values()) + list(poor_means.values()))
        all_labels = np.array([1] * ng + [0] * np_)
        n = len(all_vals)

        global_bands = compute_filled_bands(
            np.array(list(good_means.values())),
            np.array(list(poor_means.values())),
            verbose=False)
        if global_bands is None:
            global_good_is_upper = np.nan
        else:
            global_good_is_upper = global_bands['good_is_upper']

        yt, yp = [], []
        split_lines = []
        static_means = []

        for test_idx in range(n):
            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False
            tg = all_vals[mask & (all_labels == 1)]
            tp_vals = all_vals[mask & (all_labels == 0)]
            if len(tg) < 3 or len(tp_vals) < 3:
                continue
            bands = compute_filled_bands(tg, tp_vals, verbose=False)
            if bands is None:
                continue
            split_lines.append(bands['split_line'])
            static_means.append(bands.get('static_mean_median', np.nan))

            raw_prob = classify_patient_raw(all_vals[test_idx], bands)

            fold_good_is_upper = bands['good_is_upper']
            if not np.isnan(global_good_is_upper) and fold_good_is_upper != global_good_is_upper:
                raw_prob = 1.0 - raw_prob

            yt.append(all_labels[test_idx])
            yp.append(raw_prob)

        if len(yt) == 0:
            bin_metrics[label] = None
            continue

        yt, yp = np.array(yt), np.array(yp)

        auc, ci_l, ci_u = auc_ci_delong(yt, yp)

        y_pred = (yp >= 0.5).astype(int)
        tp_cnt = np.sum((yt == 1) & (y_pred == 1))
        tn_cnt = np.sum((yt == 0) & (y_pred == 0))
        nn = len(yt)
        npos = int(np.sum(yt == 1))
        nneg = int(np.sum(yt == 0))
        acc = (tp_cnt + tn_cnt) / nn
        sen = tp_cnt / npos if npos > 0 else np.nan
        spe = tn_cnt / nneg if nneg > 0 else np.nan
        acl, acu = wilson_ci(acc, nn)
        sl, su = wilson_ci(sen, npos) if npos > 0 else (np.nan, np.nan)
        spl, spu = wilson_ci(spe, nneg) if nneg > 0 else (np.nan, np.nan)

        avg_split = np.mean(split_lines) if split_lines else np.nan
        avg_static = np.mean(static_means) if static_means else np.nan

        if verbose:
            dir_str = "good up" if global_good_is_upper else "good down"
            print(f"  Bin {label}: AUC={auc:.3f}, direction={dir_str}, "
                  f"split_line={avg_split:.4f}")

        bin_metrics[label] = {'AUC': auc, 'AUC_CI_low': ci_l, 'AUC_CI_up': ci_u,
                              'Accuracy': acc, 'Accuracy_CI_low': acl, 'Accuracy_CI_up': acu,
                              'Sensitivity': sen, 'Sensitivity_CI_low': sl, 'Sensitivity_CI_up': su,
                              'Specificity': spe, 'Specificity_CI_low': spl, 'Specificity_CI_up': spu,
                              'n': nn, 'split_line': avg_split, 'static_mean': avg_static,
                              'good_is_upper': global_good_is_upper}
        bin_predictions[label] = (yt, yp)

    return bin_metrics, bin_predictions, dose_labels, bin_raw_values

def pooled_loocv(good_group, poor_group, verbose=False):
    all_patients = {}
    for name, arr in good_group.items():
        all_patients[name] = {'values': arr, 'label': 1}
    for name, arr in poor_group.items():
        all_patients[name] = {'values': arr, 'label': 0}
    pnames = list(all_patients.keys())

    all_good_vals = np.concatenate([all_patients[n]['values'][~np.isnan(all_patients[n]['values'])]
                                     for n in pnames if all_patients[n]['label'] == 1])
    all_poor_vals = np.concatenate([all_patients[n]['values'][~np.isnan(all_patients[n]['values'])]
                                     for n in pnames if all_patients[n]['label'] == 0])
    global_bands = compute_filled_bands(all_good_vals, all_poor_vals, verbose=False)
    if global_bands is None:
        global_good_is_upper = np.nan
    else:
        global_good_is_upper = global_bands['good_is_upper']

    yt_all, yp_all = [], []
    split_lines = []
    static_means = []

    for test_name in pnames:
        tg, tp_ = [], []
        for n in pnames:
            if n == test_name:
                continue
            valid = all_patients[n]['values'][~np.isnan(all_patients[n]['values'])]
            if all_patients[n]['label'] == 1:
                tg.extend(valid.tolist())
            else:
                tp_.extend(valid.tolist())
        tg, tp_ = np.array(tg), np.array(tp_)
        tv = all_patients[test_name]['values'][~np.isnan(all_patients[test_name]['values'])]
        tl = all_patients[test_name]['label']
        if len(tg) < 3 or len(tp_) < 3:
            continue
        bands = compute_filled_bands(tg, tp_, verbose=False)
        if bands is None:
            continue
        split_lines.append(bands['split_line'])
        static_means.append(bands.get('static_mean_median', np.nan))

        for v in tv:
            raw_prob = classify_patient_raw(v, bands)
            fold_good_is_upper = bands['good_is_upper']
            if not np.isnan(global_good_is_upper) and fold_good_is_upper != global_good_is_upper:
                raw_prob = 1.0 - raw_prob
            yt_all.append(tl)
            yp_all.append(raw_prob)

    if len(yt_all) == 0:
        return None, None, None

    yt_all, yp_all = np.array(yt_all), np.array(yp_all)

    auc, ci_l, ci_u = auc_ci_delong(yt_all, yp_all)

    y_pred = (yp_all >= 0.5).astype(int)
    tp_cnt = np.sum((yt_all == 1) & (y_pred == 1))
    tn_cnt = np.sum((yt_all == 0) & (y_pred == 0))
    n = len(yt_all)
    np_ = int(np.sum(yt_all == 1))
    nn_ = int(np.sum(yt_all == 0))
    acc = (tp_cnt + tn_cnt) / n
    sen = tp_cnt / np_ if np_ > 0 else np.nan
    spe = tn_cnt / nn_ if nn_ > 0 else np.nan
    acl, acu = wilson_ci(acc, n)
    sl, su = wilson_ci(sen, np_) if np_ > 0 else (np.nan, np.nan)
    spl, spu = wilson_ci(spe, nn_) if nn_ > 0 else (np.nan, np.nan)

    avg_split = np.mean(split_lines) if split_lines else np.nan
    avg_static = np.mean(static_means) if static_means else np.nan

    if verbose:
        dir_str = "good up" if global_good_is_upper else "good down"
        print(f"  Static model: AUC={auc:.3f}, direction={dir_str}")

    pooled = {'AUC': auc, 'AUC_CI_low': ci_l, 'AUC_CI_up': ci_u,
              'Accuracy': acc, 'Accuracy_CI_low': acl, 'Accuracy_CI_up': acu,
              'Sensitivity': sen, 'Sensitivity_CI_low': sl, 'Sensitivity_CI_up': su,
              'Specificity': spe, 'Specificity_CI_low': spl, 'Specificity_CI_up': spu,
              'n': n, 'split_line': avg_split, 'static_mean': avg_static,
              'good_is_upper': global_good_is_upper}
    return pooled, yt_all, yp_all

def print_decision_values(all_frac_metrics, all_bin_metrics, all_pooled,
                           param_names, all_dose_labels, max_fraction=8):
    print(f"\n{'='*120}")
    print(f"{'='*120}")

    for param in param_names:
        for frac in range(1, max_fraction + 1):
            df = all_frac_metrics[param]
            v = df[df['Fraction'] == frac]
            if len(v) > 0 and not np.isnan(v['AUC'].values[0]):
                dir_str = "good up" if v['good_is_upper'].values[0] else "good down"
                print(f"  {frac:<12} {v['AUC'].values[0]:<10.3f} {dir_str:<16} "
                      f"{v['split_line'].values[0]:<14.4f} {v['static_mean'].values[0]:<14.4f}")

        for label in all_dose_labels:
            bm = all_bin_metrics[param]
            if label in bm and bm[label]:
                dir_str = "good up" if bm[label].get('good_is_upper') else "good down"
                print(f"  {label:<12} {bm[label]['AUC']:<10.3f} {dir_str:<16} "
                      f"{bm[label]['split_line']:<14.4f} {bm[label]['static_mean']:<14.4f}")


def batch_analyze(base_path, param_names, dose_edges, max_fraction=8, verbose=False,
                   qa_csv=None, alpha_beta=10.0):
    qa_dict = load_qa_pass_dict(qa_csv)

    all_frac_metrics = {}
    all_frac_predictions = {}
    all_frac_raw_values = {}
    all_bin_metrics = {}
    all_bin_predictions = {}
    all_bin_raw_values = {}
    all_pooled = {}
    all_pooled_predictions = {}
    all_dose_labels = None

    for param in param_names:
        print(f"\n  Analyzing: {param} ...")

        out_frac = extract_data_4class(base_path, param, np.mean, multi=1, qa_dict=qa_dict)
        good_frac = {**out_frac[0], **out_frac[1]}
        poor_frac = {**out_frac[2], **out_frac[3]}

        out_dose = extract_data_dose_4class(base_path, param, np.mean, multi=1,
                                             qa_dict=qa_dict, alpha_beta=alpha_beta)
        good_dose = {**out_dose[0], **out_dose[1]}
        poor_dose = {**out_dose[2], **out_dose[3]}

        frac_metrics, frac_predictions, frac_raw = fraction_loocv(good_frac, poor_frac, max_fraction, verbose=verbose)
        all_frac_metrics[param] = frac_metrics
        all_frac_predictions[param] = frac_predictions
        all_frac_raw_values[param] = frac_raw

        bin_metrics, bin_predictions, dose_labels, bin_raw = dose_bin_loocv(good_dose, poor_dose, dose_edges, verbose=verbose)
        all_bin_metrics[param] = bin_metrics
        all_bin_predictions[param] = bin_predictions
        all_bin_raw_values[param] = bin_raw
        if all_dose_labels is None:
            all_dose_labels = dose_labels

        pooled, yt_pooled, yp_pooled = pooled_loocv(good_frac, poor_frac, verbose=verbose)
        all_pooled[param] = pooled
        all_pooled_predictions[param] = (yt_pooled, yp_pooled)

    return (all_frac_metrics, all_frac_predictions, all_frac_raw_values,
            all_bin_metrics, all_bin_predictions, all_bin_raw_values,
            all_pooled, all_pooled_predictions,
            all_dose_labels)


def plot_auc_comparison(all_frac_metrics, all_bin_metrics, all_pooled,
                         param_names, all_dose_labels, max_fraction=8,
                         plot_name=None, save_path=None):
    if plot_name is None:
        plot_name = param_names
    colors = PALETTE
    markers = ['o', 's', '^', 'D']
    n_params = len(param_names)
    offsets = np.linspace(-0., 0., n_params)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    for p_idx, param in enumerate(param_names):
        df = all_frac_metrics[param]
        fracs = df['Fraction'].values.astype(float)
        aucs = df['AUC'].values.copy()
        x_plot = fracs + offsets[p_idx]
        ax.plot(x_plot, aucs, marker=markers[p_idx], linestyle='-',
                color=colors[p_idx], linewidth=5, markersize=12,
                label=plot_name[p_idx], alpha=0.9)
    ax.axhline(y=0.5, color='grey', linestyle=':', linewidth=2.5)
    ax.set_xlabel('Fraction', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('AUC', fontsize=18, fontfamily='Times New Roman')
    ax.set_xticks(range(1, max_fraction + 1))
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=24, loc='upper center', ncol=4, frameon=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax = axes[1]
    n_bins = len(all_dose_labels)
    bin_idx = np.arange(n_bins)
    for p_idx, param in enumerate(param_names):
        bin_metrics = all_bin_metrics[param]
        auc_vals = []
        valid_idx = []
        for i, label in enumerate(all_dose_labels):
            if label in bin_metrics and bin_metrics[label]:
                auc_vals.append(bin_metrics[label]['AUC'])
                valid_idx.append(i)
        if len(valid_idx) > 0:
            x_plot = np.array(valid_idx) + offsets[p_idx]
            ax.plot(x_plot, auc_vals, marker=markers[p_idx], linestyle='-',
                    color=colors[p_idx], linewidth=5, markersize=12,
                    label=plot_name[p_idx], alpha=0.9)
    ax.axhline(y=0.5, color='grey', linestyle=':', linewidth=2.5)
    ax.set_xlabel('Cumulative $EQD_2$ [Gy]', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('AUC', fontsize=18, fontfamily='Times New Roman')
    ax.set_xticks(bin_idx)
    ax.set_xticklabels(all_dose_labels, fontsize=18, rotation=30)
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=24, loc='upper center', ncol=4, frameon=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.show()
    return fig, axes

def _format_value_ci(value, ci_low, ci_up, width=12):
    if np.isnan(value):
        return f"{'N/A':^{width}}", f"{'N/A':^{width}}"
    line1 = f"{value:.2f}".center(width)
    line2 = f"[{ci_low:.2f},{ci_up:.2f}]".center(width)
    return line1, line2


def _print_metric_block_fraction(metric_name, all_metrics, all_pooled, param_names,
                                  param_labels, max_fraction=8):
    col_width = 14
    param_width = 8
    header = f"{'':<{param_width}}"
    header += f"{'Pooled':^{col_width}}"
    for frac in range(1, max_fraction + 1):
        header += f"{f'Frac {frac}':^{col_width}}"
    print(header)
    sep = "-" * (param_width + col_width * (max_fraction + 1))
    print(sep)
    for p_idx, p in enumerate(param_names):
        label = param_labels[p_idx] if param_labels else p
        line1 = f"{label:<{param_width}}"
        line2 = f"{'':<{param_width}}"
        pooled = all_pooled.get(p)
        if pooled and not np.isnan(pooled.get(metric_name, np.nan)):
            v = pooled[metric_name]
            cl = pooled[f'{metric_name}_CI_low']
            cu = pooled[f'{metric_name}_CI_up']
            l1, l2 = _format_value_ci(v, cl, cu, col_width)
        else:
            l1 = f"{'N/A':^{col_width}}"
            l2 = f"{'N/A':^{col_width}}"
        line1 += l1
        line2 += l2
        df = all_metrics[p]
        for frac in range(1, max_fraction + 1):
            v = df[df['Fraction'] == frac]
            if len(v) > 0 and not np.isnan(v[metric_name].values[0]):
                val = v[metric_name].values[0]
                cl = v[f'{metric_name}_CI_low'].values[0]
                cu = v[f'{metric_name}_CI_up'].values[0]
                l1, l2 = _format_value_ci(val, cl, cu, col_width)
            else:
                l1 = f"{'N/A':^{col_width}}"
                l2 = f"{'N/A':^{col_width}}"
            line1 += l1
            line2 += l2
        print(line1)
        print(line2)
        if p_idx < len(param_names) - 1:
            print(sep)


def _print_metric_block_dose(metric_name, all_bin_metrics, all_pooled, param_names,
                              param_labels, all_dose_labels):
    col_width = 14
    param_width = 8
    header = f"{'':<{param_width}}"
    header += f"{'Pooled':^{col_width}}"
    for label in all_dose_labels:
        header += f"{label:^{col_width}}"
    print(header)
    sep = "-" * (param_width + col_width * (len(all_dose_labels) + 1))
    print(sep)
    for p_idx, p in enumerate(param_names):
        label = param_labels[p_idx] if param_labels else p
        line1 = f"{label:<{param_width}}"
        line2 = f"{'':<{param_width}}"
        pooled = all_pooled.get(p)
        if pooled and not np.isnan(pooled.get(metric_name, np.nan)):
            v = pooled[metric_name]
            cl = pooled[f'{metric_name}_CI_low']
            cu = pooled[f'{metric_name}_CI_up']
            l1, l2 = _format_value_ci(v, cl, cu, col_width)
        else:
            l1 = f"{'N/A':^{col_width}}"
            l2 = f"{'N/A':^{col_width}}"
        line1 += l1
        line2 += l2
        bm = all_bin_metrics[p]
        for dlabel in all_dose_labels:
            if dlabel in bm and bm[dlabel]:
                m = bm[dlabel]
                l1, l2 = _format_value_ci(m[metric_name],
                                           m[f'{metric_name}_CI_low'],
                                           m[f'{metric_name}_CI_up'],
                                           col_width)
            else:
                l1 = f"{'N/A':^{col_width}}"
                l2 = f"{'N/A':^{col_width}}"
            line1 += l1
            line2 += l2
        print(line1)
        print(line2)
        if p_idx < len(param_names) - 1:
            print(sep)


def print_summary_table(all_frac_metrics, all_bin_metrics, all_pooled,
                         param_names, all_dose_labels, max_fraction=8,
                         param_labels=None, save_csv_dir=None):
    if param_labels is None:
        param_labels = param_names

    print(f"\n{'=' * (8 + 14 * (max_fraction + 1))}")
    print(f"  Analysis by Fraction (Pooled = static model)")
    print(f"{'=' * (8 + 14 * (max_fraction + 1))}")
    frac_csv_data = []
    for metric_name in ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']:
        print(f"\n  --- {metric_name} ---")
        _print_metric_block_fraction(metric_name, all_frac_metrics, all_pooled,
                                      param_names, param_labels, max_fraction)
        for p_idx, p in enumerate(param_names):
            label = param_labels[p_idx] if param_labels else p
            row = {'Parameter': label, 'Metric': metric_name}
            pooled = all_pooled.get(p)
            if pooled and not np.isnan(pooled.get(metric_name, np.nan)):
                row['Pooled'] = f"{pooled[metric_name]:.2f}\n[{pooled[f'{metric_name}_CI_low']:.2f},{pooled[f'{metric_name}_CI_up']:.2f}]"
            else:
                row['Pooled'] = 'N/A'
            df = all_frac_metrics[p]
            for frac in range(1, max_fraction + 1):
                v = df[df['Fraction'] == frac]
                if len(v) > 0 and not np.isnan(v[metric_name].values[0]):
                    row[f'Frac{frac}'] = f"{v[metric_name].values[0]:.2f}\n[{v[f'{metric_name}_CI_low'].values[0]:.2f},{v[f'{metric_name}_CI_up'].values[0]:.2f}]"
                else:
                    row[f'Frac{frac}'] = 'N/A'
            frac_csv_data.append(row)
    if save_csv_dir:
        df_frac = pd.DataFrame(frac_csv_data)
        cols = ['Parameter', 'Metric', 'Pooled'] + [f'Frac{frac}' for frac in range(1, max_fraction + 1)]
        df_frac = df_frac[cols]
        csv_path = os.path.join(save_csv_dir, 'results_fraction.csv')
        df_frac.to_csv(csv_path, index=False)
        print(f"\n  [Fraction results saved to: {csv_path}]")

    print(f"\n{'=' * (8 + 14 * (len(all_dose_labels) + 1))}")
    print(f"  Analysis by Cumulative EQD2 Bin")
    print(f"{'=' * (8 + 14 * (len(all_dose_labels) + 1))}")
    dose_csv_data = []
    for metric_name in ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']:
        print(f"\n  --- {metric_name} ---")
        _print_metric_block_dose(metric_name, all_bin_metrics, all_pooled,
                                  param_names, param_labels, all_dose_labels)
        for p_idx, p in enumerate(param_names):
            label = param_labels[p_idx] if param_labels else p
            row = {'Parameter': label, 'Metric': metric_name}
            pooled = all_pooled.get(p)
            if pooled and not np.isnan(pooled.get(metric_name, np.nan)):
                row['Pooled'] = f"{pooled[metric_name]:.2f}\n[{pooled[f'{metric_name}_CI_low']:.2f},{pooled[f'{metric_name}_CI_up']:.2f}]"
            else:
                row['Pooled'] = 'N/A'
            bm = all_bin_metrics[p]
            for dlabel in all_dose_labels:
                safe_label = dlabel.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
                if dlabel in bm and bm[dlabel]:
                    m = bm[dlabel]
                    row[safe_label] = f"{m[metric_name]:.2f}\n[{m[f'{metric_name}_CI_low']:.2f},{m[f'{metric_name}_CI_up']:.2f}]"
                else:
                    row[safe_label] = 'N/A'
            dose_csv_data.append(row)
    if save_csv_dir:
        df_dose = pd.DataFrame(dose_csv_data)
        cols = ['Parameter', 'Metric', 'Pooled']
        for dlabel in all_dose_labels:
            safe_label = dlabel.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
            cols.append(safe_label)
        df_dose = df_dose[cols]
        csv_path = os.path.join(save_csv_dir, 'results_dose_bin.csv')
        df_dose.to_csv(csv_path, index=False)
        print(f"\n  [Dose results saved to: {csv_path}]")

def export_delong_table_csv(
    all_frac_predictions, all_frac_raw_values,
    all_bin_predictions, all_bin_raw_values,
    all_pooled, param_names, all_dose_labels, max_fraction,
    save_path
):
    rows = []

    for param in param_names:
        pooled = all_pooled[param]
        if pooled is None:
            continue
        split_line = pooled['split_line']
        good_is_upper = pooled.get('good_is_upper', True)

        f_static = {}
        f_strat = {}
        f_p = {}

        for frac in range(1, max_fraction + 1):
            if (frac not in all_frac_predictions[param] or
                frac not in all_frac_raw_values[param]):
                continue

            yt, yp_strat = all_frac_predictions[param][frac]
            raw = all_frac_raw_values[param][frac]
            gv = raw['good_vals']
            pv = raw['poor_vals']
            all_vals = np.concatenate([gv, pv])
            all_labels = np.array([1]*len(gv) + [0]*len(pv))

            if len(all_vals) < 3:
                continue

            if good_is_upper:
                yp_static = np.where(all_vals >= split_line, 1.0, 0.0)
            else:
                yp_static = np.where(all_vals <= split_line, 1.0, 0.0)

            auc_static, _, _ = auc_ci_delong(all_labels, yp_static)
            auc_strat, _, _ = auc_ci_delong(yt, yp_strat)

            _, _, p_val, _ = delong_test(all_labels, yp_static, yp_strat)

            f_static[frac] = auc_static
            f_strat[frac] = auc_strat
            f_p[frac] = p_val

        b_static = {}
        b_strat = {}
        b_p = {}

        for label in all_dose_labels:
            if (label not in all_bin_predictions[param] or
                all_bin_predictions[param][label] is None or
                label not in all_bin_raw_values[param]):
                continue

            yt, yp_strat = all_bin_predictions[param][label]
            raw = all_bin_raw_values[param][label]
            gv = raw['good_vals']
            pv = raw['poor_vals']
            all_vals = np.concatenate([gv, pv])
            all_labels = np.array([1]*len(gv) + [0]*len(pv))

            if len(all_vals) < 3:
                continue

            if good_is_upper:
                yp_static = np.where(all_vals >= split_line, 1.0, 0.0)
            else:
                yp_static = np.where(all_vals <= split_line, 1.0, 0.0)

            auc_static, _, _ = auc_ci_delong(all_labels, yp_static)
            auc_strat, _, _ = auc_ci_delong(yt, yp_strat)

            _, _, p_val, _ = delong_test(all_labels, yp_static, yp_strat)

            b_static[label] = auc_static
            b_strat[label] = auc_strat
            b_p[label] = p_val

        max_rows = max(max_fraction, len(all_dose_labels))

        for i in range(max_rows):
            frac = i + 1
            if frac <= max_fraction and frac in f_strat:
                f_frac = str(frac)
                f_s = f"{f_static[frac]:.2f}"
                f_t = f"{f_strat[frac]:.2f}"
                p = f_p[frac]
                if p < 0.001:     f_pv = '<0.001'
                elif p < 0.01:    f_pv = '<0.01'
                elif p < 0.05:    f_pv = '<0.05'
                else:             f_pv = f"{p:.2f}"
            else:
                f_frac = f_s = f_t = f_pv = ''

            if i < len(all_dose_labels):
                label = all_dose_labels[i]
                dl = label.replace("(", "").replace("]", "").replace("[", "").replace(")", "")
                dl = dl.replace(",", "~") if dl != "0" else "0"

                if label in b_strat:
                    b_s = f"{b_static[label]:.2f}"
                    b_t = f"{b_strat[label]:.2f}"
                    p = b_p[label]
                    if p < 0.001:     b_pv = '<0.001'
                    elif p < 0.01:    b_pv = '<0.01'
                    elif p < 0.05:    b_pv = '<0.05'
                    else:             b_pv = f"{p:.2f}"
                else:
                    b_s = b_t = b_pv = 'N/A'
            else:
                dl = b_s = b_t = b_pv = ''

            rows.append([f_frac, f_s, f_t, f_pv, dl, b_s, b_t, b_pv])

    with open(save_path, 'w', encoding='utf-8-sig') as f:
        f.write('Fraction-based monitoring,,,,,Cumulative dose-based monitoring,,,\n')
        f.write('Fraction,AUC(static),AUC(stratified),p value,,Cumulative Dose [Gy],AUC(static),AUC(stratified),p value\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},,{r[4]},{r[5]},{r[6]},{r[7]}\n")

    print(f"\n  [DeLong results saved to: {save_path}]")

if __name__ == "__main__":
    base_path = r'XXX'
    param_names = ['ADC', 'Dslow', 'Dfast', 'Ffast']
    param_labels = [r"$ADC$", r"$D$", r"$D^*$", r"$f$"]
    DOSE_EDGES = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]

    QA_CSV_PATH = r'XXX'
    save_csv_dir = r'XXX'

    (all_frac_metrics, all_frac_predictions, all_frac_raw_values,
     all_bin_metrics, all_bin_predictions, all_bin_raw_values,
     all_pooled, all_pooled_predictions,
     all_dose_labels) = batch_analyze(base_path, param_names, DOSE_EDGES,
                                       verbose=True, qa_csv=QA_CSV_PATH,
                                       alpha_beta=10.0)

    print_summary_table(all_frac_metrics, all_bin_metrics, all_pooled,
                         param_names, all_dose_labels, param_labels=param_labels,
                         save_csv_dir=save_csv_dir)

    print_decision_values(all_frac_metrics, all_bin_metrics, all_pooled,
                           param_names, all_dose_labels)

    export_delong_table_csv(
        all_frac_predictions, all_frac_raw_values,
        all_bin_predictions, all_bin_raw_values,
        all_pooled, param_names, all_dose_labels, max_fraction=8,
        save_path="XXX")

    plot_auc_comparison(all_frac_metrics, all_bin_metrics, all_pooled,
                         param_names, all_dose_labels, plot_name=param_labels,
                       save_path="XXX")

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)
