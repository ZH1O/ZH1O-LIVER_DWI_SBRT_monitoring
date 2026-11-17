# -*- coding: utf-8 -*-
"""
run_figures.py — Main figure generator.

Produces seven figures (dpi=1000) into figures/, named by analysis content:
  final_bands_fraction_and_EQD2_with_cases.png
      separated response / non-response final bands per fraction and per
      cumulative-EQD2 bin, with two representative case trajectories
  grouped_boxplots_fraction_and_EQD2_with_significance.png
      per-fraction / per-bin grouped box plots (Overall / Response /
      Non-response) with within- and between-group significance stars
  subgroup_parameter_AUC_by_fraction_and_EQD2.png
      tumor-type subgroup (Overall / Metastasis / Primary liver) parameter
      AUC from raw values, by fraction (left) and cumulative EQD2 (right)
  monitor_accuracy_by_fraction_and_EQD2.png
      overall-cohort LOOCV accuracy, two stacked panels (fraction | EQD2),
      four parameter lines
  bin_width_sensitivity_AUC.png
      EQD2 bin-width sensitivity of the direct parameter AUC (DeLong 95% CI; 6 / 7 / 8 Gy; 7 Gy main analysis)
  bin_width_sample_counts.png
      per-bin responder / non-responder counts for 6 / 7 / 8 Gy with the
      n=3 band-construction floor
  AUC_by_fraction_and_EQD2.png
      four-parameter direct parameter AUC versus fraction and cumulative
      EQD2 bins (bin '0' included as the baseline timepoint)
Analysis protocol: see monitor_lib.py (Dmean dosimetry; IQR bands with a
Youden-optimal split line; band orientation flipping allowed only at time
points with a significant between-group difference (BH-FDR q<0.05).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

import monitor_lib as ml

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, 'figures')
os.makedirs(FIGD, exist_ok=True)

# ============================================================
# Hyperparameters
# ============================================================
base_path = r'XXX'
param_names = ['ADC', 'Dslow', 'Dfast', 'Ffast']
param_labels = [r"$ADC \; [\times 10^{-3}\ \text{mm}^2/\text{s}]$",
                r"$D \; [\times 10^{-3}\ \text{mm}^2/\text{s}]$",
                r"$D* \; [\times 10^{-3}\ \text{mm}^2/\text{s}]$",
                r"$f \;[\%]$"]
PSYM = {'ADC': r'$ADC$', 'Dslow': r'$D$', 'Dfast': r'$D^*$', 'Ffast': r'$f$'}
multi_list = [1000, 1000, 1000, 100]

QA_CSV_PATH = r'xxx'
MAX_FRACTION = 8
STAT_MAX_FRACTION = None  # set per parameter from observed data
DOSE_EDGES = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
GOOD_CASE_IDX = 5
POOR_CASE_IDX = 7
INCLUDE_CASES = True
YLIM_MANUAL = {'ADC': (0.8, 1.9), 'Dslow': (0.9, 1.15),
               'Dfast': (46, 50), 'Ffast': (8, 25)}
WIDTHS_ALL = [5, 6, 7, 8, 9, 10]
WIDTHS_FIG = [6, 7, 8]

# ============================================================
# Subgroup (tumor type) mapping, read directly from the clinical Excel:
# primary site containing the CJK keywords for liver / HCC / bile duct
# -> Primary liver, otherwise Metastasis.
# ============================================================
XLSX_PATH = r'xxx'
_xls = pd.read_excel(XLSX_PATH)
_xls['pid'] = _xls['id'].astype(str).str.lower().str.lstrip('p').str.strip()
_site = {r['pid']: (str(r['原发部位']) if pd.notna(r['原发部位']) else '')
         for _, r in _xls.iterrows()}


def tumor_type_of_pid(pid):
    v = _site.get(pid, _site.get(pid[:8], ''))
    return 'Primary liver' if any(k in v for k in ['肝', 'HCC', '胆管']) else 'Metastasis'


def pid_of_name(name):
    """Patient folder name (e.g. P00006666-duan dehua) -> numeric pid."""
    m = re.match(r'[pP]?(\d+)', name)
    return m.group(1) if m else ''


def sort_bin_labels(labels):
    """Order dose-bin labels numerically ('0' first)."""
    def key(l):
        if l == '0':
            return (0, 0)
        clean = l.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        return (float(clean.split(',')[0]), 1)
    return sorted(labels, key=key)


def width_edges(w, dmax=70):
    """Bin-edge grid for a given width, including trailing-interval handling."""
    edges = list(range(w, dmax + 1, w))
    if edges[-1] < dmax:
        if dmax - edges[-1] >= w / 2:
            edges.append(dmax)
        else:
            edges.pop()
            edges.append(dmax)
    return sorted(set(edges))


# ============================================================
# Extraction (Dmean) + per-parameter bands / directions / LOOCV / stats /
# subgroup trajectories / bin-width sweep
# ============================================================
qa_dict = ml.load_qa_pass_dict(QA_CSV_PATH)
extracted = {}
for pn, multi in zip(param_names, multi_list):
    print(f"  Extracting (Dmean): {pn}")
    extracted[pn] = ml.extract_adc_dose_means_4class(
        base_path, pn, np.mean, multi=multi, qa_dict=qa_dict)

P = {}                       # per-parameter analysis quantities
traj_frac, traj_bin = [], []
bw_rows = []
cut_rows = []

# global tumor-type map built from all parameters (avoids scope leakage)
_all_names_global = set()
for _pn in param_names:
    _o = extracted[_pn]
    _all_names_global |= set(_o[0]) | set(_o[1]) | set(_o[2]) | set(_o[3])
name_type_global = {n: tumor_type_of_pid(pid_of_name(n))
                    for n in _all_names_global}


def _bin_dict(group, label):
    if label == '0':
        cond = lambda d: d == 0
    else:
        lo, hi = label.strip('()]').split(',')
        lo, hi = float(lo), float(hi)
        cond = lambda d: (lo < d) & (d <= hi)
    out = {}
    for n, d in group.items():
        v = [a for dd, a in zip(d['cum_eqd2'], d['adc'])
             if not np.isnan(dd) and not np.isnan(a) and cond(dd)]
        if v:
            out[n] = float(np.mean(v))
    return out


def _fold_cutoffs(gv, pv):
    out = []
    for held in list(gv) + list(pv):
        tg = [v for n, v in gv.items() if n != held]
        tp = [v for n, v in pv.items() if n != held]
        if len(tg) < 3 or len(tp) < 3:
            continue
        b = ml.compute_filled_bands(tg, tp)
        if b is not None:
            out.append(float(b['split_line']))
    return out

for pn in param_names:
    out = extracted[pn]
    gd = {**out[0], **out[1]}
    pd_ = {**out[2], **out[3]}
    ga = {n: d['adc'] for n, d in gd.items()}
    pa = {n: d['adc'] for n, d in pd_.items()}
    name_type = {n: name_type_global[n] for n in list(ga) + list(pa)}

    # ---- fractions ----
    vals_f = {}
    for f in range(1, MAX_FRACTION + 1):
        gv = np.array([a[f - 1] for a in ga.values() if len(a) >= f and not np.isnan(a[f - 1])])
        pv = np.array([a[f - 1] for a in pa.values() if len(a) >= f and not np.isnan(a[f - 1])])
        vals_f[f] = (gv, pv)
    stat_max = min(max([len(a) for a in list(ga.values()) + list(pa.values())]), MAX_FRACTION)
    stats_f = ml.fraction_statistical_analysis(ga, pa, stat_max)
    dirs_f, info_f = ml.timepoint_directions(vals_f, stats_f, baseline=1)
    frac_bands = {f: ml.forced_band(gv, pv, dirs_f.get(f)) for f, (gv, pv) in vals_f.items()}
    frac_raw = {f: {'good_vals': gv, 'poor_vals': pv} for f, (gv, pv) in vals_f.items()}
    frac_scores = ml.fraction_loocv_scores(ga, pa, dirs_f, MAX_FRACTION)
    for f, (names, yt, yp) in frac_scores.items():
        for nm, y, s in zip(names, yt, yp):
            traj_frac.append(dict(param=pn, pid=nm, Fraction=f, score=s,
                                  resp=int(y), type=name_type.get(nm, 'Metastasis')))
    for f in range(1, MAX_FRACTION + 1):
        gvf = {n: a[f - 1] for n, a in ga.items()
               if len(a) >= f and not np.isnan(a[f - 1])}
        pvf = {n: a[f - 1] for n, a in pa.items()
               if len(a) >= f and not np.isnan(a[f - 1])}
        for c in _fold_cutoffs(gvf, pvf):
            cut_rows.append(dict(param=pn, axis='fraction',
                                 timepoint=str(f), cutoff=c))

    # ---- dose bins ----
    vals_b, labels_b = ml.bin_values_per_t(gd, pd_, DOSE_EDGES)
    stats_d = ml.dose_statistical_analysis(gd, pd_, DOSE_EDGES)
    dirs_b, info_b = ml.timepoint_directions(vals_b, stats_d, baseline='0')
    dose_bands = {l: ml.forced_band(gv, pv, dirs_b.get(l)) for l, (gv, pv) in vals_b.items()}
    dose_raw = {l: {'good_vals': gv, 'poor_vals': pv} for l, (gv, pv) in vals_b.items()}
    bin_scores, _ = ml.bin_loocv_scores(gd, pd_, DOSE_EDGES, dirs_b)
    for l, (names, yt, yp) in bin_scores.items():
        for nm, y, s in zip(names, yt, yp):
            traj_bin.append(dict(param=pn, pid=nm, bin=l, score=s,
                                 resp=int(y), type=name_type.get(nm, 'Metastasis')))
    for l in labels_b:
        gvb = _bin_dict(gd, l)
        pvb = _bin_dict(pd_, l)
        for c in _fold_cutoffs(gvb, pvb):
            cut_rows.append(dict(param=pn, axis='bin',
                                 timepoint=str(l), cutoff=c))

    # ---- bin-width sweep (5-10 Gy): direct parameter AUC + bootstrap CI ----
    for w in WIDTHS_ALL:
        edges = width_edges(w)
        vals_w, labs_w = ml.bin_values_per_t(gd, pd_, edges)
        for l in labs_w:
            gv, pv = vals_w[l]
            n, nN = len(gv) + len(pv), len(pv)
            if len(gv) < 3 or len(pv) < 3:
                bw_rows.append(dict(param=pn, width=w, bin=l, n=n, nN=nN,
                                    auc=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                    usable=False))
                continue
            y = np.r_[np.ones(len(gv)), np.zeros(len(pv))]
            s = np.r_[gv, pv]
            a, lo, hi = ml.auc_ci_delong(y, s)
            bw_rows.append(dict(param=pn, width=w, bin=l, n=n, nN=nN,
                                auc=round(a, 4), ci_lo=round(lo, 4),
                                ci_hi=round(hi, 4), usable=True))

    P[pn] = dict(gd=gd, pd=pd_, ga=ga, pa=pa, vals_f=vals_f, stats_f=stats_f,
                 dirs_f=dirs_f, info_f=info_f, frac_bands=frac_bands, frac_raw=frac_raw,
                 vals_b=vals_b, labels_b=labels_b, stats_d=stats_d, dirs_b=dirs_b,
                 dose_bands=dose_bands, dose_raw=dose_raw,
                 frac_scores=frac_scores, bin_scores=bin_scores)
    print(f"  {pn}: global direction (fractions) = {info_f['global']}; "
          f"flip-allowed fractions = {[t for t, s in info_f['sig'].items() if s]}")

SS = pd.DataFrame(traj_frac)
BS = pd.DataFrame(traj_bin)


def _youden_rows(df, keycol, axis):
    """LOOCV Youden index J = Se + Sp - 1 at score >= 0.5."""
    rows = []
    for (pn, k), d in df.groupby(['param', keycol]):
        y = d.resp.values.astype(int)
        pred = (d.score.values >= 0.5).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        se = tp / (tp + fn) if (tp + fn) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(dict(param=pn, axis=axis, timepoint=str(k),
                         n=len(d), nR=int(y.sum()), nNR=int((y == 0).sum()),
                         SEN=se, SPE=sp, J=se + sp - 1))
    return rows


df_j = pd.DataFrame(_youden_rows(SS, 'Fraction', 'fraction') +
                    _youden_rows(BS, 'bin', 'bin'))
df_j.to_csv(os.path.join(HERE, 'loocv_youden_index.csv'),
            index=False, encoding='utf-8-sig')
print('  saved -> loocv_youden_index.csv')


def _seed_of(*parts):
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) % (2 ** 32 - 1)
    return h


def _class_metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    se = tp / (tp + fn) if (tp + fn) else np.nan
    sp = tn / (tn + fp) if (tn + fp) else np.nan
    acc = (tp + tn) / len(y) if len(y) else np.nan
    return se, sp, acc


def _class_ci(y, pred, n_boot=2000, seed=0):
    """Percentile bootstrap (patient resampling) for Se/Sp/Acc."""
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int)
    se, sp, acc = _class_metrics(y, pred)
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out[b] = _class_metrics(y[idx], pred[idx])
    lo = np.nanpercentile(out, 2.5, axis=0)
    hi = np.nanpercentile(out, 97.5, axis=0)
    return se, sp, acc, lo, hi


def _class_rows(df, keycol, axis):
    rows = []
    for (pn, k), d in df.groupby(['param', keycol]):
        y = d.resp.values.astype(int)
        pred = (d.score.values >= 0.5).astype(int)
        se, sp, acc, lo, hi = _class_ci(
            y, pred, seed=_seed_of(pn, axis, k))
        rows.append(dict(param=pn, axis=axis, timepoint=str(k),
                         n=len(d), nR=int(y.sum()), nNR=int((y == 0).sum()),
                         ACC=acc, ACC_low=lo[2], ACC_high=hi[2],
                         SEN=se, SEN_low=lo[0], SEN_high=hi[0],
                         SPE=sp, SPE_low=lo[1], SPE_high=hi[1]))
    return rows


df_c = pd.DataFrame(_class_rows(SS, 'Fraction', 'fraction') +
                    _class_rows(BS, 'bin', 'bin'))


def _fmt(v, lo, hi):
    return f'{v:.2f} [{lo:.2f}, {hi:.2f}]'


df_c['ACC_95CI'] = [_fmt(v, l, h) for v, l, h in
                    zip(df_c.ACC, df_c.ACC_low, df_c.ACC_high)]
df_c['SEN_95CI'] = [_fmt(v, l, h) for v, l, h in
                    zip(df_c.SEN, df_c.SEN_low, df_c.SEN_high)]
df_c['SPE_95CI'] = [_fmt(v, l, h) for v, l, h in
                    zip(df_c.SPE, df_c.SPE_low, df_c.SPE_high)]
df_c.to_csv(os.path.join(HERE, 'loocv_acc_sen_spe.csv'),
            index=False, encoding='utf-8-sig')
print('  saved -> loocv_acc_sen_spe.csv')

df_cut = pd.DataFrame(cut_rows)
df_cut.to_csv(os.path.join(HERE, 'loocv_cutoff_range.csv'),
              index=False, encoding='utf-8-sig')
print('  saved -> loocv_cutoff_range.csv')

# full-data band summary (fixed Q25/Q75 outer edges + Youden split)
band_rows = []
for pn in param_names:
    for f, b in P[pn]['frac_bands'].items():
        if b is None:
            continue
        gv = P[pn]['frac_raw'][f]['good_vals']
        pv = P[pn]['frac_raw'][f]['poor_vals']
        g25, g75 = np.percentile(gv, [25, 75])
        p25, p75 = np.percentile(pv, [25, 75])
        band_rows.append(dict(param=pn, axis='fraction', timepoint=str(f),
                              response_Q25=g25, response_Q75=g75,
                              nonresponse_Q25=p25, nonresponse_Q75=p75,
                              good_low=b['good_band'][0], good_high=b['good_band'][1],
                              poor_low=b['poor_band'][0], poor_high=b['poor_band'][1],
                              good_is_upper=int(b['good_is_upper']),
                              cutoff=b['split_line']))
    for l, b in P[pn]['dose_bands'].items():
        if b is None:
            continue
        gv = P[pn]['dose_raw'][l]['good_vals']
        pv = P[pn]['dose_raw'][l]['poor_vals']
        g25, g75 = np.percentile(gv, [25, 75])
        p25, p75 = np.percentile(pv, [25, 75])
        band_rows.append(dict(param=pn, axis='bin', timepoint=str(l),
                              response_Q25=g25, response_Q75=g75,
                              nonresponse_Q25=p25, nonresponse_Q75=p75,
                              good_low=b['good_band'][0], good_high=b['good_band'][1],
                              poor_low=b['poor_band'][0], poor_high=b['poor_band'][1],
                              good_is_upper=int(b['good_is_upper']),
                              cutoff=b['split_line']))
pd.DataFrame(band_rows).to_csv(os.path.join(HERE, 'band_summary.csv'),
                               index=False, encoding='utf-8-sig')
print('  saved -> band_summary.csv')

# bootstrap CI for the full-data Youden cutoff (fixed band direction)
def _cutoff_bootstrap(gv, pv, upper, n_boot=2000, seed=0):
    g = np.asarray(gv, dtype=float); p = np.asarray(pv, dtype=float)
    if len(g) < 3 or len(p) < 3:
        return np.nan, np.nan, np.nan
    point, _ = ml.youden_threshold(g, p, bool(upper))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        gg = g[rng.integers(0, len(g), len(g))]
        pp = p[rng.integers(0, len(p), len(p))]
        b = ml.compute_filled_bands(gg, pp, force_upper=bool(upper))
        if b is not None and np.isfinite(b['split_line']):
            vals.append(b['split_line'])
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


cut_summary = []
for pn in param_names:
    for f, b in P[pn]['frac_bands'].items():
        if b is None:
            continue
        gv = P[pn]['frac_raw'][f]['good_vals']
        pv = P[pn]['frac_raw'][f]['poor_vals']
        y, lo, hi = _cutoff_bootstrap(
            gv, pv, b['good_is_upper'], seed=_seed_of(pn, 'f', f))
        cut_summary.append(dict(param=pn, axis='fraction', timepoint=str(f),
                                nR=len(gv), nNR=len(pv), cutoff=y,
                                cutoff_low=lo, cutoff_high=hi))
    for l, b in P[pn]['dose_bands'].items():
        if b is None:
            continue
        gv = P[pn]['dose_raw'][l]['good_vals']
        pv = P[pn]['dose_raw'][l]['poor_vals']
        y, lo, hi = _cutoff_bootstrap(
            gv, pv, b['good_is_upper'], seed=_seed_of(pn, 'b', l))
        cut_summary.append(dict(param=pn, axis='bin', timepoint=str(l),
                                nR=len(gv), nNR=len(pv), cutoff=y,
                                cutoff_low=lo, cutoff_high=hi))

df_cs = pd.DataFrame(cut_summary)
df_cs['cutoff_95CI'] = [
    f'{v:.2f} [{l:.2f}, {h:.2f}]' if pd.notna(v) and pd.notna(l) and pd.notna(h)
    else (f'{v:.2f} [NA, NA]' if pd.notna(v) else 'NA')
    for v, l, h in zip(df_cs.cutoff, df_cs.cutoff_low, df_cs.cutoff_high)]
df_cs.to_csv(os.path.join(HERE, 'cutoff_summary.csv'),
             index=False, encoding='utf-8-sig')
print('  saved -> cutoff_summary.csv')

# ------------------------------------------------------------------
# Direct subgroup parameter AUC (raw values, not LOOCV scores)
# ------------------------------------------------------------------
def _direct_subgroup_auc():
    rows = []
    for pn in param_names:
        ga, pa = P[pn]['ga'], P[pn]['pa']
        gd, pd_ = P[pn]['gd'], P[pn]['pd']
        allnames = set(ga) | set(pa)
        pri = {n for n in allnames if name_type_global.get(n) == 'Primary liver'}
        met = allnames - pri
        scopes = [('Overall', allnames), ('Metastasis', met),
                  ('Primary liver', pri)]

        def _add(gg, pp, axis, tp):
            for scope, subset in scopes:
                g2 = {n: v for n, v in gg.items() if n in subset}
                p2 = {n: v for n, v in pp.items() if n in subset}
                if len(g2) < 2 or len(p2) < 2:
                    continue
                y = np.r_[np.ones(len(g2)), np.zeros(len(p2))]
                s = np.r_[list(g2.values()), list(p2.values())]
                a, lo, hi = ml.auc_ci_delong(y, s)
                rows.append(dict(param=pn, axis=axis, timepoint=tp,
                                 scope=scope, n=len(y), nR=len(g2),
                                 nNR=len(p2), AUC=a, DeLong_low=lo,
                                 DeLong_high=hi))

        for f in range(1, MAX_FRACTION + 1):
            gv = {n: a[f - 1] for n, a in ga.items()
                  if len(a) >= f and not np.isnan(a[f - 1])}
            pv = {n: a[f - 1] for n, a in pa.items()
                  if len(a) >= f and not np.isnan(a[f - 1])}
            if len(gv) < 2 or len(pv) < 2:
                continue
            _add(gv, pv, 'fraction', str(f))

        labels = sort_bin_labels(list(P[pn]['labels_b']))
        for lab in labels:
            gv = _bin_dict(gd, lab)
            pv = _bin_dict(pd_, lab)
            if len(gv) < 2 or len(pv) < 2:
                continue
            _add(gv, pv, 'bin', str(lab))
    return pd.DataFrame(rows)

BW = pd.DataFrame(bw_rows)


# ============================================================
# Final bands (fraction | cumulative EQD2) with individual case trajectories
# ============================================================
def _patient_bin_value(ad, label):
    """Mean parameter value of one patient within a dose bin."""
    if label == '0':
        vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                if not np.isnan(d) and not np.isnan(a) and d == 0]
    elif label.startswith('(0,'):
        hi = float(label.split(',')[1].rstrip(']'))
        vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                if not np.isnan(d) and not np.isnan(a) and 0 < d <= hi]
    else:
        lo, hi = [float(x) for x in label.replace('(', '').replace(']', '').split(',')]
        vals = [a for d, a in zip(ad['cum_eqd2'], ad['adc'])
                if not np.isnan(d) and not np.isnan(a) and lo < d <= hi]
    return np.mean(vals) if vals else None


def plot_final_bands(save_path):
    n = len(param_names)
    fig = plt.figure(figsize=(19, 4.5 * n))
    gs = fig.add_gridspec(n, 2, left=0.055, right=0.99, top=0.99, bottom=0.055,
                          wspace=0.14, hspace=0.34)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(n)])
    handles = {}
    for ri, pn in enumerate(param_names):
        ylim = YLIM_MANUAL[pn]
        frac_keys = [f for f in sorted(P[pn]['frac_bands'])
                     if P[pn]['frac_bands'][f] is not None]
        dose_keys = sort_bin_labels([l for l in P[pn]['dose_bands']
                                     if P[pn]['dose_bands'][l] is not None])
        for ci, (keys, bands) in enumerate((
                (frac_keys, P[pn]['frac_bands']),
                (dose_keys, P[pn]['dose_bands']))):
            ax = axes[ri, ci]
            if len(keys) < 2:
                continue
            xs = np.array(keys, dtype=float) if ci == 0 else np.arange(len(keys), dtype=float)
            gl = np.array([bands[k]['good_band'][0] for k in keys])
            gu = np.array([bands[k]['good_band'][1] for k in keys])
            pl = np.array([bands[k]['poor_band'][0] for k in keys])
            pu = np.array([bands[k]['poor_band'][1] for k in keys])
            g_upper = ((gl + gu) / 2)[0] > ((pl + pu) / 2)[0]
            uc, ul = (ml.COLOR_RESPONSE, 'Response band') if g_upper else \
                     (ml.COLOR_NONRESPONSE, 'Non-response band')
            lc, ll = (ml.COLOR_NONRESPONSE, 'Non-response band') if g_upper else \
                     (ml.COLOR_RESPONSE, 'Response band')
            up_l, up_u = (gl, gu) if g_upper else (pl, pu)
            lo_l, lo_u = (pl, pu) if g_upper else (gl, gu)
            ax.fill_between(xs, up_l, up_u, alpha=0.3, color=uc, edgecolor='none', label=ul)
            ax.fill_between(xs, lo_l, lo_u, alpha=0.3, color=lc, edgecolor='none', label=ll)
            if INCLUDE_CASES:
                gnames = list(P[pn]['gd'].keys())
                pnames_ = list(P[pn]['pd'].keys())
                gc = gnames[min(GOOD_CASE_IDX, len(gnames) - 1)]
                pc = pnames_[min(POOR_CASE_IDX, len(pnames_) - 1)]
                for case, grp, mk, cc, lab in (
                        (gc, P[pn]['gd'], 'o-', ml.COLOR_RESPONSE_DARK, 'Response case'),
                        (pc, P[pn]['pd'], 's-', ml.COLOR_NONRESPONSE_DARK, 'Non-response case')):
                    if ci == 0:
                        v = grp[case]['adc']
                        fx = np.arange(1, len(v) + 1)
                        m = ~np.isnan(v)
                        ax.plot(fx[m], v[m], mk, color=cc, lw=2.5, ms=10, label=lab, zorder=5)
                    else:
                        bx, by = [], []
                        for i, k in enumerate(keys):
                            v = _patient_bin_value(grp[case], k)
                            if v is not None:
                                bx.append(i)
                                by.append(v)
                        if bx:
                            ax.plot(bx, by, mk, color=cc, lw=2.5, ms=10, label=lab, zorder=5)
            ax.set_ylim(ylim)
            ml.apply_six_even_ticks(ax, ylim[0], ylim[1])
            if ci == 0:
                ml.style_axes(ax, xlabel='Fraction', ylabel=param_labels[ri],
                              xticks=keys, xticklabels=[str(k) for k in keys])
            else:
                ml.style_axes(ax, xlabel='Cumulative EQD$_2$ [Gy]', ylabel=param_labels[ri],
                              xticks=np.arange(len(keys)), xticklabels=keys,
                              rotate=30, xtick_fs=14)
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handles:
                    handles[l] = h
    # frame alignment: identical box height/position for both panels of a row
    for r in range(n):
        pl_, pr_ = axes[r, 0].get_position(), axes[r, 1].get_position()
        y0, y1 = min(pl_.y0, pr_.y0), max(pl_.y1, pr_.y1)
        axes[r, 0].set_position([pl_.x0, y0, pl_.width, y1 - y0])
        axes[r, 1].set_position([pr_.x0, y0, pr_.width, y1 - y0])
    want = ['Response band', 'Non-response band', 'Response case', 'Non-response case']
    hs = [handles[k] for k in want if k in handles]
    ls = [k for k in want if k in handles]
    fig.legend(hs, ls, loc='lower center', fontsize=16, ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.05))
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Grouped box plots (Overall / Response / Non-response) with significance annotations
# ============================================================
def plot_grouped_boxplots(save_path):
    n = len(param_names)
    fig = plt.figure(figsize=(19, 4.5 * n))
    gs = fig.add_gridspec(n, 2, left=0.055, right=0.99, top=0.99, bottom=0.055,
                          wspace=0.14, hspace=0.34)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(n)])
    for ri, pn in enumerate(param_names):
        lo_e, hi_e = YLIM_MANUAL[pn]
        mid = (lo_e + hi_e) / 2
        span = (hi_e - lo_e) * 1.44            # 20% widening applied twice
        lo_e, hi_e = mid - span / 2, mid + span / 2
        for d in list(P[pn]['frac_raw'].values()) + list(P[pn]['dose_raw'].values()):
            for v in (d['good_vals'], d['poor_vals']):
                if len(v):
                    w_lo, w_hi = ml.whisker_range(v)
                    lo_e, hi_e = min(lo_e, w_lo), max(hi_e, w_hi)
        needs = []
        for ci in (0, 1):
            ax = axes[ri, ci]
            raw = P[pn]['frac_raw'] if ci == 0 else P[pn]['dose_raw']
            cand = [k for k in raw
                    if len(raw[k]['good_vals']) and len(raw[k]['poor_vals'])]
            keys = sorted(cand) if ci == 0 else sort_bin_labels(cand)
            xs = np.array(keys, dtype=float) if ci == 0 else np.arange(len(keys), dtype=float)
            for x, k in zip(xs, keys):
                ml.draw_grouped_box(ax, x, raw[k]['good_vals'], raw[k]['poor_vals'])
            stats = P[pn]['stats_f'] if ci == 0 else P[pn]['stats_d']
            vpx = [(np.concatenate([raw[k]['good_vals'], raw[k]['poor_vals']]),
                    raw[k]['good_vals'], raw[k]['poor_vals']) for k in keys]
            nd = ml.annotate_significance(
                ax, xs, list(keys), vpx, 0, hi_e - lo_e,
                ml.stars_from_fdr(stats['overall']['fdr']),
                ml.stars_from_fdr(stats['within_good']['fdr']),
                ml.stars_from_fdr(stats['within_poor']['fdr']),
                ml.stars_from_fdr(stats['between']['fdr']))
            needs.append(nd)
            if ci == 0:
                ml.style_axes(ax, xlabel='Fraction', ylabel=param_labels[ri],
                              xticks=keys, xticklabels=[str(k) for k in keys])
            else:
                ml.style_axes(ax, xlabel='Cumulative EQD$_2$ [Gy]', ylabel=param_labels[ri],
                              xticks=np.arange(len(keys)), xticklabels=keys,
                              rotate=30, xtick_fs=14)
        top = hi_e
        for nd in needs:
            if np.isfinite(nd):
                top = max(top, nd)
        top += 0.05 * (top - lo_e)             # extra headroom for annotations
        for ci in (0, 1):
            axes[ri, ci].set_ylim(lo_e, top)
            ml.apply_six_even_ticks(axes[ri, ci], lo_e, top)
    for r in range(n):
        pl_, pr_ = axes[r, 0].get_position(), axes[r, 1].get_position()
        y0, y1 = min(pl_.y0, pr_.y0), max(pl_.y1, pr_.y1)
        axes[r, 0].set_position([pl_.x0, y0, pl_.width, y1 - y0])
        axes[r, 1].set_position([pr_.x0, y0, pr_.width, y1 - y0])
    hs = [Patch(facecolor=ml.COLOR_OVERALL, alpha=0.3, edgecolor=ml.COLOR_OVERALL_DARK,
                linewidth=1.5, label='Overall'),
          Patch(facecolor=ml.COLOR_RESPONSE, alpha=0.3, edgecolor=ml.COLOR_RESPONSE_DARK,
                linewidth=1.5, label='Response'),
          Patch(facecolor=ml.COLOR_NONRESPONSE, alpha=0.3,
                edgecolor=ml.COLOR_NONRESPONSE_DARK, linewidth=1.5, label='Non-response')]
    fig.legend(hs, [h.get_label() for h in hs], loc='lower center', fontsize=16,
               ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, -0.085,
             'Within-group vs baseline (Fraction 1 / 0 Gy): paired Wilcoxon signed-rank;  '
             'between-group: Mann-Whitney U;  p values FDR-corrected (Benjamini-Hochberg): '
             '* p<0.05, ** p<0.01, *** p<0.001;  '
             'stars above boxes: vs baseline (gray: whole cohort; blue/orange: response groups);  '
             'brackets: Response vs Non-response',
             ha='center', va='center', fontsize=14, fontfamily='Times New Roman')
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Subgroup direct parameter AUC (Overall / Metastasis / Primary liver)
# ============================================================
def plot_subgroup_parameter_auc(sub_df, save_path):
    """4 rows (parameters) x 2 cols; direct parameter AUC by subgroup.

    AUC is calculated from raw parameter values within each subgroup (not
    from LOOCV monitor scores)."""
    keys_f = list(range(1, MAX_FRACTION + 1))
    keys_b = sort_bin_labels(list(P[param_names[0]]['labels_b']))
    data = [('fraction', keys_f), ('bin', keys_b)]
    n = len(param_names)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.6 * n), squeeze=False)

    for ci, (axis, keys) in enumerate(data):
        xpos = {str(k): i for i, k in enumerate(keys)}
        for i, pn in enumerate(param_names):
            ax = axes[i, ci]
            for scope, col in [('Overall', ml.C_ALL),
                               ('Metastasis', ml.C_MET),
                               ('Primary liver', ml.C_PRI)]:
                sub = sub_df[(sub_df.param == pn) & (sub_df.axis == axis) &
                             (sub_df.scope == scope)].copy()
                if sub.empty:
                    continue
                pts = []
                for _, r in sub.iterrows():
                    pts.append((xpos[str(r['timepoint'])], r['AUC']))
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        'o-' if axis == 'fraction' else 's-',
                        color=col, lw=3, ms=8, label=scope)
            ax.axhline(0.5, color='grey', ls=':', lw=0.8)
            ax.set_ylim(0.2, 1.03)
            if ci == 0:
                ax.set_ylabel(PSYM[pn], fontsize=18)
            if axis == 'fraction':
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels([str(k) for k in keys], fontsize=16)
            else:
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels(keys, fontsize=14, rotation=30, ha='right')
                ax.set_xlim(-0.5, len(keys) - 0.5)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontfamily('Times New Roman')
                lab.set_fontsize(16)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            if i == n - 1:
                ax.set_xlabel('Fraction' if axis == 'fraction'
                              else 'Cumulative EQD$_2$ [Gy]', fontsize=18)
            if i == 0:
                ax.set_title('AUC by fraction' if axis == 'fraction'
                             else 'AUC by cumulative EQD$_2$',
                             fontsize=18)

    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=3, frameon=True,
               fontsize=16, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Overall Accuracy (LOOCV; fraction | cumulative EQD2)
# ============================================================
def plot_accuracy_overall(frac_df, bin_df, save_path):
    """Two stacked panels (fraction | cumulative EQD2), four parameter lines."""
    keys_f = list(range(1, MAX_FRACTION + 1))
    keys_b = sort_bin_labels(list(P[param_names[0]]['labels_b']))
    data = [('fraction', frac_df, 'Fraction', keys_f),
            ('bin', bin_df, 'bin', keys_b)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    for ax, (kind, df, keycol, keys) in zip(axes, data):
        for i, pn in enumerate(param_names):
            xs, ys = [], []
            for k in keys:
                d = df[(df.param == pn) & (df[keycol] == k)]
                if len(d) == 0:
                    continue
                pred = (d.score.values >= 0.5).astype(int)
                ys.append(float((pred == d.resp.values).mean()))
                xs.append(k)
            if not xs:
                continue
            xp = xs if kind == 'fraction' else [keys.index(k) for k in xs]
            ax.plot(xp, ys, 'o-' if kind == 'fraction' else 's-',
                    color=ml.PALETTE[i], lw=3, ms=8, label=PSYM[pn])
        ax.axhline(0.5, color='grey', ls=':', lw=0.8)
        ax.set_ylim(0.3, 1.0)
        if kind == 'fraction':
            ml.style_axes(ax, xlabel='Fraction', ylabel='Monitor accuracy',
                          xticks=keys, xticklabels=[str(k) for k in keys])
        else:
            ml.style_axes(ax, xlabel='Cumulative EQD$_2$ [Gy]', ylabel='Monitor accuracy',
                          xticks=range(len(keys)), xticklabels=keys,
                          rotate=30, xtick_fs=14)
            ax.set_xlim(-0.5, len(keys) - 0.5)
        ax.legend(fontsize=16, ncol=4, loc='upper center', frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Bin-width sensitivity: direct parameter AUC and per-bin counts
# ============================================================
def plot_bin_width_auc(save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    for ax, w in zip(axes, WIDTHS_FIG):
        sub = BW[BW.width == w]
        bins = sort_bin_labels(list(dict.fromkeys(sub['bin'])))
        xmap = {b: k for k, b in enumerate(bins)}
        for pi, pn in enumerate(param_names):
            g = sub[(sub.param == pn) & sub.usable & sub.auc.notna()].copy()
            g['x'] = g['bin'].map(xmap)
            g = g.sort_values('x')
            ax.plot(g.x.values, g.auc.values, marker=ml.MARKERS[pi], linestyle='-',
                    color=ml.PALETTE[pi], linewidth=3, markersize=8,
                    label=PSYM[pn], alpha=0.9)
        nu = sub[~sub.usable | sub.auc.isna()]
        if len(nu):   # bins that cannot be evaluated (sample-size gate)
            ax.plot(nu['bin'].map(xmap).values, np.full(len(nu), 0.5), 'x',
                    color='#7F7F7F', ms=10, mew=2.2, alpha=0.9)
        ax.axhline(y=0.5, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xticks([xmap[b] for b in bins])
        ax.set_xticklabels(bins, fontsize=18, rotation=30, ha='right')
        ml.style_axes(ax, xlabel='Cumulative EQD$_2$ [Gy]', ylabel='AUC')
        _lo = min(0.45, float(sub.auc.min()) - 0.02) if sub.auc.notna().any() else 0.45
        ax.set_ylim(_lo, 1.05)
        ax.legend(fontsize=16, loc='upper center', ncol=4, frameon=True)
        ax.set_title(f'{w} Gy bins', fontsize=18,
                     fontweight='bold' if w == 7 else 'normal')
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


def plot_bin_width_counts(save_path):
    R2 = BW[BW.param == 'ADC'].copy()      # counts are parameter-independent
    # all bins (evaluable and non-evaluable) are retained in BW
    _cnt = R2.copy()
    _cnt['n_responders'] = _cnt['n'] - _cnt['nN']
    _cnt['n_nonresponders'] = _cnt['nN']
    _cnt[['width', 'bin', 'n', 'n_responders', 'n_nonresponders', 'usable']].to_csv(
        os.path.join(HERE, 'bin_width_sample_counts.csv'),
        index=False, encoding='utf-8-sig')
    R2['mid'] = R2['bin'].str.extract(r'\((\d+),(\d+)').astype(float).mean(axis=1)
    R2.loc[R2['bin'] == '0', 'mid'] = 0.0
    wcol = {6: '#6A3D9A', 7: '#000000', 8: '#E15759'}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
    for ax, grp, ylab, title in [(axes[0], 'R', 'Number of responders', 'Responders'),
                                 (axes[1], 'N', 'Number of non-responders',
                                  'Non-responders')]:
        for w in WIDTHS_FIG:
            sub = R2[R2.width == w].sort_values('mid')
            y = (sub.n - sub.nN) if grp == 'R' else sub.nN
            ax.plot(sub.mid, y, '-', color=wcol[w], lw=2.4 if w == 7 else 1.3,
                    alpha=1.0 if w == 7 else 0.65, marker='o',
                    ms=7 if w == 7 else 5, mec='w' if w == 7 else 'none', mew=0.8,
                    label=f'{w} Gy')
        ax.axhline(3, color='#E15759', lw=1.0, ls=':')   # n=3 band-construction floor
        ax.set_ylim(0, None)
        ax.set_ylabel(ylab, fontsize=16)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, 72)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        for lab in ax.get_yticklabels():
            lab.set_fontsize(14)
            lab.set_fontfamily('Times New Roman')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    axes[0].legend(ncol=3, fontsize=12, frameon=True, loc='upper right', title='bin width')
    axes[1].set_xlabel('Cumulative EQD$_2$ [Gy]', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Direct parameter AUC: fraction | cumulative EQD2 bins (bin '0' included
#        as the baseline timepoint)
# ============================================================
def plot_parameter_auc(sub_auc, save_path):
    """Direct parameter AUC by fraction and cumulative EQD2 (overall cohort)."""
    overall = sub_auc[sub_auc.scope == 'Overall']
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    labels_all = sort_bin_labels(list(P[param_names[0]]['labels_b']))

    for ax, axis, keys in [(axes[0], 'fraction',
                            list(range(1, MAX_FRACTION + 1))),
                           (axes[1], 'bin', labels_all)]:
        xpos = {str(k): i for i, k in enumerate(keys)}
        for pi, pn in enumerate(param_names):
            d = overall[(overall.param == pn) & (overall.axis == axis)]
            if d.empty:
                continue
            pts = sorted((xpos[str(t)], v) for t, v in
                         zip(d.timepoint, d.AUC))
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker=ml.MARKERS[pi], linestyle='-', color=ml.PALETTE[pi],
                    linewidth=3, markersize=8, label=PSYM[pn], alpha=0.9)
        ax.axhline(0.5, color='grey', ls=':', lw=0.8)
        if axis == 'fraction':
            ml.style_axes(ax, xlabel='Fraction', ylabel='Parameter AUC',
                          xticks=range(len(keys)),
                          xticklabels=[str(k) for k in keys])
        else:
            ml.style_axes(ax, xlabel='Cumulative EQD$_2$ [Gy]',
                          ylabel='Parameter AUC',
                          xticks=range(len(keys)), xticklabels=keys,
                          rotate=30, xtick_fs=14)
            ax.set_xlim(-0.5, len(keys) - 0.5)
        ax.set_ylim(0.2, 1.03)
        ax.legend(fontsize=16, loc='upper center', ncol=4, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Generate all figures
# ============================================================
plot_final_bands(os.path.join(FIGD, 'final_bands_fraction_and_EQD2_with_cases.png'))
plot_grouped_boxplots(os.path.join(FIGD, 'grouped_boxplots_fraction_and_EQD2_with_significance.png'))
BS_all = BS.reset_index(drop=True)
sub_auc = _direct_subgroup_auc()
for _c in ['AUC', 'DeLong_low', 'DeLong_high']:
    sub_auc[_c] = sub_auc[_c].map(lambda v: np.nan if pd.isna(v)
                                  else np.floor(v * 100 + 0.5) / 100)
sub_auc['AUC_95CI'] = [
    f'{v:.2f} [{l:.2f}, {h:.2f}]' if pd.notna(v) and pd.notna(l) and pd.notna(h)
    else (f'{v:.2f} [NA, NA]' if pd.notna(v) else 'NA')
    for v, l, h in zip(sub_auc.AUC, sub_auc.DeLong_low, sub_auc.DeLong_high)]
sub_auc.to_csv(os.path.join(HERE, 'subgroup_parameter_AUC.csv'),
               index=False, encoding='utf-8-sig')
plot_subgroup_parameter_auc(
    sub_auc, os.path.join(FIGD,
                          'subgroup_parameter_AUC_by_fraction_and_EQD2.png'))
plot_accuracy_overall(
    SS, BS_all, os.path.join(FIGD, 'monitor_accuracy_by_fraction_and_EQD2.png'))
plot_bin_width_auc(os.path.join(FIGD, 'bin_width_sensitivity_AUC.png'))
plot_bin_width_counts(os.path.join(FIGD, 'bin_width_sample_counts.png'))
plot_parameter_auc(sub_auc, os.path.join(FIGD, 'AUC_by_fraction_and_EQD2.png'))
sub_auc[sub_auc.scope == 'Overall'][['param', 'axis', 'timepoint', 'n', 'nR', 'nNR', 'AUC', 'DeLong_low', 'DeLong_high', 'AUC_95CI']].to_csv(os.path.join(HERE, 'parameter_AUC_summary.csv'), index=False, encoding='utf-8-sig')
print("  All figures done.")
