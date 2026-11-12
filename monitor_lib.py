# -*- coding: utf-8 -*-
"""
monitor_lib.py — Analysis library for MRL-guided liver radiotherapy
treatment monitoring with longitudinal IVIM parameters (ADC, D, D*, f).

Method summary:
  1) Dosimetry: the per-fraction dose metric is Dmean (arithmetic mean over
     GTV voxels); it is converted to EQD2 with the linear-quadratic model
     (alpha/beta = 10 Gy) and accumulated over previous fractions into
     cumulative EQD2. Dmean matches the GTV-mean parameter outcome and is
     the most robust per-fraction summary against segmentation / registration
     noise on adaptive fractions.
  2) Monitoring bands: separated response / non-response filled bands with
     split lines (IQR-based, stepwise separation of overlapping bands).
  3) Band direction protocol:
       - Global band orientation (good_is_upper) is obtained by majority vote
         over the local orientations of all fractions / dose bins; ties are
         resolved with the band direction of all time points pooled.
       - A time point may deviate from the global orientation (i.e. flipping
         is allowed) ONLY if any within-group comparison versus baseline
         (fraction 1 / 0 Gy bin) is significant (paired Wilcoxon, BH-FDR
         q < 0.05). All other time points are forced to the global direction.
       - LOOCV scoring: a fold score is inverted when the fold band direction
         differs from the effective direction of that time point; there is no
         whole-fraction flip at AUC < 0.5.
  4) Statistics: within-group paired Wilcoxon signed-rank versus baseline
     (n_pairs >= 3 for both groups); between-group Mann-Whitney U;
     Benjamini-Hochberg FDR (within family = response + non-response
     combined; between family = all fractions / bins).
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import mannwhitneyu, wilcoxon

# ============================================================
# Global figure style
# ============================================================
rcParams.update({"font.family": 'Times New Roman',
                 "mathtext.fontset": 'cm',
                 "font.serif": ['Times New Roman']})
plt.rc('font', family='Times New Roman')
plt.rcParams['text.usetex'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ALPHA_BETA = 10.0            # alpha/beta ratio [Gy]
# Per-fraction GTV dose summary = Dmean (arithmetic mean over GTV voxels)

COLOR_RESPONSE = '#2E86AB'
COLOR_NONRESPONSE = '#F18F01'
COLOR_RESPONSE_DARK = '#1A5C7A'
COLOR_NONRESPONSE_DARK = '#B8650A'
COLOR_OVERALL = '#B0B0B0'
COLOR_OVERALL_DARK = '#666666'
PALETTE = ['#e6f0de', '#a8d6bb', '#6ea7ad', '#415079']   # cubehelix(rot=-.5, gamma=.5, dark=.1)
MARKERS = ['o', 's', '^', 'D']
C_ALL, C_MET, C_PRI = '#E15759', '#6A3D9A', '#1B7837'    # Overall / Metastasis / Primary liver

def traversalDir_FirstDir(path):
    """First-level subdirectories of path."""
    return [os.path.join(path, file) for file in os.listdir(path)
            if os.path.exists(os.path.join(path, file))]


def load_qa_pass_dict(csv_path):
    """Load the QA record CSV -> {(patient, fraction_number): all_pass}."""
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: QA record not found; all fractions will be used.")
        return None
    df = pd.read_csv(csv_path)
    qa_dict = {}
    for _, row in df.iterrows():
        patient = str(row['patient'])
        frac_num = int(row['fraction_number'])
        all_pass = row['all_pass'] if isinstance(row['all_pass'], bool) \
            else (str(row['all_pass']).lower() == 'true')
        qa_dict[(patient, frac_num)] = all_pass
    return qa_dict


def extract_adc_dose_means_4class(base_path, parameter, func, multi=1,
                                  qa_dict=None, alpha_beta=ALPHA_BETA):
    """
    Extract per-fraction GTV mean of the parameter map and the GTV mean dose
    (Dmean), then accumulate EQD2.
    Returns [cls0..cls3][name] = {'adc':..., 'dose':..., 'cum_eqd2':...}.
    Mask binarization (>0.5); fractions flagged as failed in the QA ledger
    are skipped.
    """
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
                        mask_flat = sitk.GetArrayFromImage(mask_img > 0.5).reshape(-1,)

                        ADC = sitk.GetArrayFromImage(
                            sitk.ReadImage(adc_path, sitk.sitkFloat64)).reshape(-1,)
                        roi_adc = ADC[np.where(mask_flat)] * multi
                        if roi_adc.size > 0:
                            adc_out[fraction_N - 1] = func(roi_adc)

                        Dose = sitk.GetArrayFromImage(
                            sitk.ReadImage(dose_path, sitk.sitkFloat64)).reshape(-1,)
                        roi_dose = Dose[np.where(mask_flat)]
                        if roi_dose.size > 0:
                            dose_out[fraction_N - 1] = round(float(np.mean(roi_dose)), 2)
                    except Exception:
                        continue

                # forward-fill missing dose values
                for i in range(total_N):
                    if np.isnan(dose_out[i]) and i > 0:
                        dose_out[i] = dose_out[i - 1]

                # per-fraction EQD2 and cumulative EQD2 (dose of previous fractions)
                eqd2_per_frac = dose_out * (alpha_beta + dose_out) / (alpha_beta + 2)
                cum_eqd2 = np.cumsum(np.nan_to_num(eqd2_per_frac))
                cum_eqd2 = np.insert(cum_eqd2[:-1], 0, 0)
                for i in range(total_N):
                    if np.isnan(adc_out[i]):
                        cum_eqd2[i] = np.nan
                    if i > 0 and np.isnan(dose_out[i - 1]):
                        cum_eqd2[i] = np.nan

                out[cls][name] = {"adc": adc_out, "dose": dose_out,
                                  "cum_eqd2": cum_eqd2}
            except Exception:
                continue
    return out

def compute_filled_bands(good_vals, poor_vals, force_upper=None):
    """
    Separated response / non-response bands built from the IQR boxes:
    separated groups keep their geometric orientation; overlapping groups are
    shrunk stepwise until separated. force_upper (bool) overrides the
    orientation choice in the overlapping case only (separated bands cannot
    be re-oriented without contradicting their geometry).
    """
    if len(good_vals) < 3 or len(poor_vals) < 3:
        return None
    good_median = np.median(good_vals)
    poor_median = np.median(poor_vals)
    good_25 = np.percentile(good_vals, 25)
    good_75 = np.percentile(good_vals, 75)
    poor_25 = np.percentile(poor_vals, 25)
    poor_75 = np.percentile(poor_vals, 75)

    if good_75 < poor_25:          # already separated, good below (geometric)
        split_line = (good_75 + poor_25) / 2.0
        return {'good_band': (good_25, split_line), 'poor_band': (split_line, poor_75),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0, 'split_line': split_line,
                'expanded': True, 'good_is_upper': False}
    if poor_75 < good_25:          # already separated, good above (geometric)
        split_line = (poor_75 + good_25) / 2.0
        return {'good_band': (split_line, good_75), 'poor_band': (poor_25, split_line),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0, 'split_line': split_line,
                'expanded': True, 'good_is_upper': True}

    # overlapping: shrink stepwise; orientation from data or force_upper
    if force_upper is not None:
        good_is_upper = bool(force_upper)
    else:
        good_is_upper = bool(good_75 > poor_75)
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
                        'split_line': split_line, 'shrunk': True,
                        'good_is_upper': True}
        split_line = (good_median + poor_median) / 2.0
        return {'good_band': (split_line, good_75), 'poor_band': (poor_25, split_line),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0, 'split_line': split_line,
                'shrunk': True, 'good_is_upper': True}
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
                        'split_line': split_line, 'shrunk': True,
                        'good_is_upper': False}
        split_line = (good_median + poor_median) / 2.0
        return {'good_band': (good_25, split_line), 'poor_band': (split_line, poor_75),
                'good_median': good_median, 'poor_median': poor_median,
                'overlap': False, 'separation': 0.0, 'split_line': split_line,
                'shrunk': True, 'good_is_upper': False}


def classify_patient_raw(value, bands):
    """Map a parameter value to a response-likelihood score in [0, 1]."""
    if bands is None:
        return 0.5
    gl, gu = bands['good_band']
    pl, pu = bands['poor_band']
    split_line = bands.get('split_line', (gu + pl) / 2.0)
    if gl <= value <= gu:
        return 0.5 + 0.5 * abs(value - split_line) / (gu - gl) if gu - gl > 0 else 0.75
    if pl <= value <= pu:
        return 0.5 - 0.5 * abs(value - split_line) / (pu - pl) if pu - pl > 0 else 0.25
    if value < min(gl, pl):
        return 0.0 if gl > pl else 1.0
    return 1.0 if gl > pl else 0.0


def bh_adjust(ps):
    """Benjamini-Hochberg FDR adjustment."""
    ps = np.asarray(ps, dtype=float)
    n = ps.size
    if n == 0:
        return ps
    order = np.argsort(ps)
    ranked = ps[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def sig_stars(p, alpha=0.05):
    """Significance stars from an (adjusted) p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < alpha:
        return '*'
    return ''


def stars_from_fdr(fdr_dict, alpha=0.05):
    """FDR dict -> star dict (non-significant / missing entries dropped)."""
    return {k: sig_stars(v, alpha) for k, v in fdr_dict.items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))}


def fraction_statistical_analysis(good_group, poor_group, max_fraction=8,
                                  n_min_good=3, n_min_poor=3):
    """
    Between-group: Mann-Whitney U per fraction; FDR family = all fractions.
    Within-group vs fraction 1: paired Wilcoxon per patient;
    FDR family = response + non-response within-group p-values combined.
    good/poor_group: {name: per-fraction value array}.
    """
    between_pvals = {}
    for frac in range(1, max_fraction + 1):
        g = [arr[frac - 1] for arr in good_group.values()
             if frac <= len(arr) and not np.isnan(arr[frac - 1])]
        p = [arr[frac - 1] for arr in poor_group.values()
             if frac <= len(arr) and not np.isnan(arr[frac - 1])]
        between_pvals[frac] = mannwhitneyu(g, p, alternative='two-sided').pvalue \
            if len(g) >= 3 and len(p) >= 3 else np.nan
    fracs = sorted(between_pvals)
    valid = [f for f in fracs if not np.isnan(between_pvals[f])]
    fdr_between = dict(zip(valid, bh_adjust(np.array([between_pvals[f] for f in valid])))) \
        if valid else {}

    within_pvals = {'g': {}, 'p': {}}
    for frac in range(2, max_fraction + 1):
        for key, grp, n_min in (('g', good_group, n_min_good), ('p', poor_group, n_min_poor)):
            pairs = [(arr[0], arr[frac - 1]) for arr in grp.values()
                     if frac <= len(arr) and len(arr) >= 1
                     and not np.isnan(arr[0]) and not np.isnan(arr[frac - 1])]
            within_pvals[key][frac] = wilcoxon([a for a, _ in pairs], [b for _, b in pairs],
                                               alternative='two-sided').pvalue \
                if len(pairs) >= n_min else np.nan
    frac_list_w = sorted(within_pvals['g'])
    all_p = ([within_pvals['g'][f] for f in frac_list_w] +
             [within_pvals['p'][f] for f in frac_list_w])
    valid_p = np.array(all_p)[~np.isnan(all_p)]
    fdr_w = {'g': {}, 'p': {}}
    if len(valid_p):
        adj = bh_adjust(valid_p)
        idx = 0
        for key in ('g', 'p'):
            for f in frac_list_w:
                if not np.isnan(within_pvals[key][f]):
                    fdr_w[key][f] = adj[idx]
                    idx += 1
    overall_pvals = {}
    all_grp = {**good_group, **poor_group}
    for frac in range(2, max_fraction + 1):
        pairs = [(arr[0], arr[frac - 1]) for arr in all_grp.values()
                 if frac <= len(arr) and len(arr) >= 1
                 and not np.isnan(arr[0]) and not np.isnan(arr[frac - 1])]
        overall_pvals[frac] = wilcoxon(
            [a for a, _ in pairs], [b for _, b in pairs],
            alternative='two-sided').pvalue if len(pairs) >= n_min_good else np.nan
    valid_o = [f for f in sorted(overall_pvals) if not np.isnan(overall_pvals[f])]
    fdr_o = dict(zip(valid_o, bh_adjust(np.array([overall_pvals[f] for f in valid_o])))) \
        if len(valid_o) > 0 else {}

    return {'between': {'pvals': between_pvals, 'fdr': fdr_between},
            'within_good': {'pvals': within_pvals['g'], 'fdr': fdr_w['g']},
            'within_poor': {'pvals': within_pvals['p'], 'fdr': fdr_w['p']},
            'overall': {'pvals': overall_pvals, 'fdr': fdr_o}}


def bin_values_per_t(good_group, poor_group, dose_edges):
    """
    Per-patient mean parameter value within each cumulative-EQD2 bin.
    Returns ({label: (good_vals, poor_vals)}, labels) with labels including '0'.
    """
    labels = ["0", f"(0,{dose_edges[0]}]"] + \
             [f"({dose_edges[i]},{dose_edges[i+1]}]" for i in range(len(dose_edges) - 1)]
    conds = {"0": lambda d: d == 0,
             labels[1]: lambda d, hi=dose_edges[0]: 0 < d <= hi}
    for i in range(len(dose_edges) - 1):
        conds[labels[i + 2]] = lambda d, lo=dose_edges[i], hi=dose_edges[i + 1]: lo < d <= hi

    def _means(grp, cond):
        out = {}
        for name, d in grp.items():
            vals = [a for dd, a in zip(d['cum_eqd2'], d['adc'])
                    if not np.isnan(dd) and not np.isnan(a) and cond(dd)]
            if vals:
                out[name] = np.mean(vals)
        return out

    vals = {}
    for l in labels:
        gm = _means(good_group, conds[l])
        pm = _means(poor_group, conds[l])
        vals[l] = (np.array(list(gm.values())), np.array(list(pm.values())))
    return vals, labels


def dose_statistical_analysis(good_group, poor_group, dose_edges, min_samples=3,
                              n_min_good=3, n_min_poor=3):
    """Same protocol as fraction_statistical_analysis on cumulative-EQD2 bins
    (baseline = bin '0'; valid bins require >= min_samples per group)."""
    vals, labels = bin_values_per_t(good_group, poor_group, dose_edges)
    valid_labels = [l for l in labels
                    if len(vals[l][0]) >= min_samples and len(vals[l][1]) >= min_samples]
    between_pvals = {}
    for l in valid_labels:
        g, p = vals[l]
        between_pvals[l] = mannwhitneyu(g, p, alternative='two-sided').pvalue \
            if len(g) >= 3 and len(p) >= 3 else np.nan
    valid_b = [l for l in valid_labels if not np.isnan(between_pvals[l])]
    fdr_between = dict(zip(valid_b, bh_adjust(np.array([between_pvals[l] for l in valid_b])))) \
        if valid_b else {}

    # named bin means are needed for patient-paired within-group tests
    conds = {"0": lambda d: d == 0,
             labels[1]: lambda d, hi=dose_edges[0]: 0 < d <= hi}
    for i in range(len(dose_edges) - 1):
        conds[labels[i + 2]] = lambda d, lo=dose_edges[i], hi=dose_edges[i + 1]: lo < d <= hi

    def _named(grp, cond):
        out = {}
        for name, d in grp.items():
            v = [a for dd, a in zip(d['cum_eqd2'], d['adc'])
                 if not np.isnan(dd) and not np.isnan(a) and cond(dd)]
            if v:
                out[name] = np.mean(v)
        return out

    named = {l: (_named(good_group, conds[l]), _named(poor_group, conds[l]))
             for l in labels}
    within_pvals = {'g': {}, 'p': {}}
    within_labels = [l for l in valid_labels if l != "0"]
    for l in within_labels:
        for key, idx in (('g', 0), ('p', 1)):
            base = named["0"][idx]
            cur = named[l][idx]
            pairs = [(base[n], cur[n]) for n in base if n in cur]
            n_min = n_min_good if key == 'g' else n_min_poor
            within_pvals[key][l] = wilcoxon([a for a, _ in pairs], [b for _, b in pairs],
                                            alternative='two-sided').pvalue \
                if len(pairs) >= n_min else np.nan
    all_p = ([within_pvals['g'][l] for l in within_labels] +
             [within_pvals['p'][l] for l in within_labels])
    valid_p = np.array(all_p)[~np.isnan(all_p)]
    fdr_w = {'g': {}, 'p': {}}
    if len(valid_p):
        adj = bh_adjust(valid_p)
        idx = 0
        for key in ('g', 'p'):
            for l in within_labels:
                if not np.isnan(within_pvals[key][l]):
                    fdr_w[key][l] = adj[idx]
                    idx += 1

    overall_pvals = {}
    base_all = {**named["0"][0], **named["0"][1]}
    for l in within_labels:
        cur_all = {**named[l][0], **named[l][1]}
        pairs = [(base_all[n], cur_all[n]) for n in base_all if n in cur_all]
        overall_pvals[l] = wilcoxon(
            [a for a, _ in pairs], [b for _, b in pairs],
            alternative='two-sided').pvalue if len(pairs) >= n_min_good else np.nan
    valid_o = [l for l in overall_pvals if not np.isnan(overall_pvals[l])]
    fdr_o = dict(zip(valid_o, bh_adjust(np.array([overall_pvals[l] for l in valid_o])))) \
        if len(valid_o) > 0 else {}

    return {'between': {'pvals': between_pvals, 'fdr': fdr_between},
            'within_good': {'pvals': within_pvals['g'], 'fdr': fdr_w['g']},
            'within_poor': {'pvals': within_pvals['p'], 'fdr': fdr_w['p']},
            'overall': {'pvals': overall_pvals, 'fdr': fdr_o}}


def timepoint_directions(vals_per_t, stats, baseline, alpha=0.05):
    """
    vals_per_t: {t: (good_vals, poor_vals)}
    stats     : output of fraction/dose_statistical_analysis (within-group FDR)
    baseline  : baseline key (1 for fractions, '0' for bins)
    Returns (dirs, info): dirs[t] = effective good_is_upper at t.
    Rule: global = majority vote of local band orientations (tie -> pooled);
    a time point may keep its local orientation (flip allowed) only when a
    within-group vs-baseline comparison is significant (FDR q < alpha).
    """
    locals_ = {}
    for t, (gv, pv) in vals_per_t.items():
        b = compute_filled_bands(gv, pv)
        if b is not None:
            locals_[t] = bool(b['good_is_upper'])
    if not locals_:
        return {}, {'global': None, 'locals': {}, 'sig': {}}
    votes = sum(locals_.values())
    if votes * 2 != len(locals_):
        glob = votes * 2 > len(locals_)          # majority vote
    else:                                        # tie -> pooled band orientation
        gv = np.concatenate([v[0] for v in vals_per_t.values() if len(v[0])])
        pv = np.concatenate([v[1] for v in vals_per_t.values() if len(v[1])])
        pb = compute_filled_bands(gv, pv)
        glob = bool(pb['good_is_upper']) if pb is not None else True
    sig = {}
    for t in vals_per_t:
        if t == baseline:
            sig[t] = False
            continue
        qg = stats['within_good']['fdr'].get(t, np.nan)
        qp = stats['within_poor']['fdr'].get(t, np.nan)
        sig[t] = bool((not np.isnan(qg) and qg < alpha) or
                      (not np.isnan(qp) and qp < alpha))
    dirs = {}
    for t in vals_per_t:
        dirs[t] = locals_.get(t, glob) if sig[t] else glob
    return dirs, {'global': glob, 'locals': locals_, 'sig': sig}


def forced_band(gv, pv, want_dir):
    """Band with the effective direction: overlapping bands are re-built with
    force_upper; separated bands keep their geometric orientation."""
    b = compute_filled_bands(gv, pv)
    if b is None or want_dir is None:
        return b
    if b['good_is_upper'] != bool(want_dir):
        b2 = compute_filled_bands(gv, pv, force_upper=bool(want_dir))
        if b2 is not None:
            b = b2
    return b

def fraction_loocv_scores(good_group, poor_group, dirs, max_fraction=8):
    """good/poor_group: {name: array}; returns {frac: (names, yt, yp)}."""
    allp = {n: {'v': a, 'l': 1} for n, a in good_group.items()}
    allp.update({n: {'v': a, 'l': 0} for n, a in poor_group.items()})
    preds = {}
    for frac in range(1, max_fraction + 1):
        valid = [n for n in allp if frac <= len(allp[n]['v'])
                 and not np.isnan(allp[n]['v'][frac - 1])]
        ng = sum(allp[n]['l'] == 1 for n in valid)
        np_ = len(valid) - ng
        if len(valid) < 6 or ng < 3 or np_ < 3:
            continue
        want = dirs.get(frac)
        yn, yt, yp = [], [], []
        for t in valid:
            train = [n for n in valid if n != t]
            tg = np.array([allp[n]['v'][frac - 1] for n in train if allp[n]['l'] == 1])
            tp = np.array([allp[n]['v'][frac - 1] for n in train if allp[n]['l'] == 0])
            if len(tg) < 3 or len(tp) < 3:
                continue
            b = compute_filled_bands(tg, tp)
            if b is None:
                continue
            s = classify_patient_raw(allp[t]['v'][frac - 1], b)
            if want is not None and b['good_is_upper'] != bool(want):
                s = 1.0 - s
            yn.append(t)
            yt.append(allp[t]['l'])
            yp.append(s)
        if yt:
            preds[frac] = (np.array(yn), np.array(yt), np.array(yp))
    return preds


def bin_loocv_scores(good_group, poor_group, dose_edges, dirs, min_samples=3):
    """good/poor_group: {name: {'adc','cum_eqd2'}};
    returns ({label: (names, yt, yp)}, labels)."""
    vals, labels = bin_values_per_t(good_group, poor_group, dose_edges)
    conds = {"0": lambda d: d == 0,
             labels[1]: lambda d, hi=dose_edges[0]: 0 < d <= hi}
    for i in range(len(dose_edges) - 1):
        conds[labels[i + 2]] = lambda d, lo=dose_edges[i], hi=dose_edges[i + 1]: lo < d <= hi

    def _named(grp, cond):
        out = {}
        for name, d in grp.items():
            v = [a for dd, a in zip(d['cum_eqd2'], d['adc'])
                 if not np.isnan(dd) and not np.isnan(a) and cond(dd)]
            if v:
                out[name] = np.mean(v)
        return out

    preds = {}
    for l in labels:
        gm = _named(good_group, conds[l])
        pm = _named(poor_group, conds[l])
        if len(gm) < min_samples or len(pm) < min_samples:
            continue
        names = list(gm.keys()) + list(pm.keys())
        allv = np.array(list(gm.values()) + list(pm.values()))
        ally = np.array([1] * len(gm) + [0] * len(pm))
        n = len(allv)
        want = dirs.get(l)
        yn, yt, yp = [], [], []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            tg = allv[mask & (ally == 1)]
            tp = allv[mask & (ally == 0)]
            if len(tg) < 3 or len(tp) < 3:
                continue
            b = compute_filled_bands(tg, tp)
            if b is None:
                continue
            s = classify_patient_raw(allv[i], b)
            if want is not None and b['good_is_upper'] != bool(want):
                s = 1.0 - s
            yn.append(names[i])
            yt.append(ally[i])
            yp.append(s)
        if yt:
            preds[l] = (np.array(yn), np.array(yt), np.array(yp))
    return preds, labels

def auc_ci_delong(y_true, y_score):
    """DeLong AUC with 95% CI."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    gt = pos[:, None] > neg[None, :]
    eq = pos[:, None] == neg[None, :]
    auc = (gt.sum() + 0.5 * eq.sum()) / (m * n)
    v10 = (gt + 0.5 * eq).mean(axis=1)
    v01 = (gt + 0.5 * eq).mean(axis=0)
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    se = np.sqrt(s10 / m + s01 / n)
    return auc, auc - 1.96 * se, auc + 1.96 * se


def auc_of(y, s):
    """AUC + CI with the evaluation gate used by the subgroup figures
    (needs >= 6 samples and both classes)."""
    if len(np.unique(y)) < 2 or len(y) < 6:
        return None
    return auc_ci_delong(y, s)


def roc_xy(y, score):
    """ROC curve (FPR, TPR) by score ranking."""
    order = np.argsort(-np.asarray(score, dtype=float))
    yy = np.asarray(y)[order]
    tpr = np.cumsum(yy) / max(yy.sum(), 1)
    fpr = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])

def whisker_range(v):
    """Extent of the drawn box whiskers (matplotlib 1.5 x IQR rule)."""
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    if inside.size == 0:
        return np.nanmin(v), np.nanmax(v)
    return inside.min(), inside.max()


def apply_six_even_ticks(ax, lo, hi, n=6):
    """Six evenly spaced y ticks spanning [lo, hi] (inclusive)."""
    ticks = np.linspace(lo, hi, n)
    dec = 2 if (hi - lo) < 2 else 1
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{v:.{dec}f}' for v in ticks])
    for lab in ax.get_yticklabels():
        lab.set_fontfamily('Times New Roman')
        lab.set_fontsize(18)


def style_axes(ax, xlabel=None, ylabel=None, xticks=None, xticklabels=None,
               rotate=0, xtick_fs=18):
    """House axis style: Times New Roman, 18 pt labels/ticks, spine 0.8."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18, fontfamily='Times New Roman')
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18, fontfamily='Times New Roman')
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=xtick_fs, rotation=rotate)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily('Times New Roman')
        lab.set_fontsize(18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def draw_grouped_box(ax, pos, gv, pv):
    """Three side-by-side boxes at x=pos: Overall (gray) / Response / Non-response."""
    ov = np.concatenate([gv, pv])
    bp = ax.boxplot([ov, gv, pv], positions=[pos - 0.25, pos, pos + 0.25],
                    widths=0.22, patch_artist=True, showfliers=True, zorder=3)
    faces = [COLOR_OVERALL, COLOR_RESPONSE, COLOR_NONRESPONSE]
    edges = [COLOR_OVERALL_DARK, COLOR_RESPONSE_DARK, COLOR_NONRESPONSE_DARK]
    for k in range(3):
        bp['boxes'][k].set_facecolor(faces[k])
        bp['boxes'][k].set_alpha(0.3)
        bp['boxes'][k].set_edgecolor(edges[k])
        bp['boxes'][k].set_linewidth(1.5)
        for art in (bp['whiskers'][2 * k], bp['whiskers'][2 * k + 1],
                    bp['caps'][2 * k], bp['caps'][2 * k + 1]):
            art.set_color(edges[k])
            art.set_linewidth(1.5)
        bp['medians'][k].set_color(edges[k])
        bp['medians'][k].set_linewidth(2.5)
        fl = bp['fliers'][k]
        fl.set_marker('o')
        fl.set_markersize(4)
        fl.set_markerfacecolor(faces[k])
        fl.set_markeredgecolor('none')
        fl.set_alpha(0.3)
        fl.set_linestyle('none')


def annotate_significance(ax, xs, keys, vals_per_x, base_idx, span,
                          star_overall, star_good, star_poor, star_between):
    """
    Significance annotations on the grouped box plots:
      1) star above each Response / Non-response box: within-group difference
         versus baseline (fraction 1 / 0 Gy bin);
      2) bracket above the Response and Non-response boxes with the star of
         the between-group (Response vs Non-response) comparison.
    Returns the highest y position used (-inf if none).
    """
    needed = -np.inf
    for i, x in enumerate(xs):
        ov, gv, pv = vals_per_x[i]
        tops = {k: (whisker_range(v)[1] if len(v) else np.nan)
                for k, v in zip(('o', 'g', 'p'), (ov, gv, pv))}
        if all(np.isnan(t) for t in tops.values()):
            continue
        key = keys[i]
        if i != base_idx:
            for dx, kk, smap in zip((-0.25, 0.0, 0.25), ('o', 'g', 'p'),
                                    (star_overall, star_good, star_poor)):
                star = smap.get(key, '')
                if star:
                    y = tops[kk] + 0.04 * span
                    ax.text(x + dx, y, star, ha='center', va='bottom',
                            fontsize=18, fontfamily='Times New Roman')
                    needed = max(needed, y + 0.09 * span)
        star = star_between.get(key, '')
        if star and not np.isnan(tops['g']) and not np.isnan(tops['p']):
            h2 = np.nanmax([tops['g'], tops['p']]) + 0.18 * span
            ax.plot([x, x + 0.25], [h2, h2], color='black', lw=1.2, zorder=6)
            ax.plot([x, x], [tops['g'] + 0.12 * span, h2], color='black', lw=1.2, zorder=6)
            ax.plot([x + 0.25, x + 0.25], [tops['p'] + 0.12 * span, h2],
                    color='black', lw=1.2, zorder=6)
            ax.text(x + 0.125, h2 + 0.02 * span, star, ha='center', va='bottom',
                    fontsize=18, fontfamily='Times New Roman')
            needed = max(needed, h2 + 0.09 * span)
    return needed
