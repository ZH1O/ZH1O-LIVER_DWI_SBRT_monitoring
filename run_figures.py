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
  subgroup_ROC_and_AUC_by_fraction.png
      tumor-type subgroup (Overall / Metastasis / Primary liver) patient-level
      ROC curves and LOOCV AUC per fraction
  subgroup_ROC_and_AUC_by_EQD2.png
      same subgroup comparison versus cumulative-EQD2 bins (bin '0' excluded)
  bin_width_sensitivity_AUC.png
      EQD2 bin-width sensitivity of the LOOCV AUC (6 / 7 / 8 Gy; 7 Gy primary)
  bin_width_sample_counts.png
      per-bin responder / non-responder counts for 6 / 7 / 8 Gy with the
      n=3 band-construction floor
  AUC_by_fraction_and_EQD2.png
      four-parameter LOOCV AUC curves versus fraction and versus cumulative
      EQD2 bins (bin '0' excluded)
Analysis protocol: see monitor_lib.py (Dmean dosimetry; global band
orientation with flipping allowed only at time points showing a significant
within-group difference versus baseline).
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
STAT_MAX_FRACTION = 9
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

for pn in param_names:
    out = extracted[pn]
    gd = {**out[0], **out[1]}
    pd_ = {**out[2], **out[3]}
    ga = {n: d['adc'] for n, d in gd.items()}
    pa = {n: d['adc'] for n, d in pd_.items()}
    name_type = {n: tumor_type_of_pid(pid_of_name(n)) for n in list(ga) + list(pa)}

    # ---- fractions ----
    vals_f = {}
    for f in range(1, MAX_FRACTION + 1):
        gv = np.array([a[f - 1] for a in ga.values() if len(a) >= f and not np.isnan(a[f - 1])])
        pv = np.array([a[f - 1] for a in pa.values() if len(a) >= f and not np.isnan(a[f - 1])])
        vals_f[f] = (gv, pv)
    stats_f = ml.fraction_statistical_analysis(ga, pa, STAT_MAX_FRACTION)
    dirs_f, info_f = ml.timepoint_directions(vals_f, stats_f, baseline=1)
    frac_bands = {f: ml.forced_band(gv, pv, dirs_f.get(f)) for f, (gv, pv) in vals_f.items()}
    frac_raw = {f: {'good_vals': gv, 'poor_vals': pv} for f, (gv, pv) in vals_f.items()}
    frac_scores = ml.fraction_loocv_scores(ga, pa, dirs_f, MAX_FRACTION)
    for f, (names, yt, yp) in frac_scores.items():
        for nm, y, s in zip(names, yt, yp):
            traj_frac.append(dict(param=pn, pid=nm, Fraction=f, score=s,
                                  resp=int(y), type=name_type.get(nm, 'Metastasis')))

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

    # ---- bin-width sweep (5-10 Gy, same protocol per width) ----
    for w in WIDTHS_ALL:
        edges = width_edges(w)
        vals_w, labs_w = ml.bin_values_per_t(gd, pd_, edges)
        stats_w = ml.dose_statistical_analysis(gd, pd_, edges)
        dirs_w, _ = ml.timepoint_directions(vals_w, stats_w, baseline=labs_w[0])
        scores_w, _ = ml.bin_loocv_scores(gd, pd_, edges, dirs_w)
        for l in labs_w:
            if l == '0':
                continue
            gv, pv = vals_w[l]
            n, nN = len(gv) + len(pv), len(pv)
            if l in scores_w:
                _, yt, yp = scores_w[l]
                a, lo, hi = ml.auc_ci_delong(yt, yp)
                bw_rows.append(dict(param=pn, width=w, bin=l, n=len(yt),
                                    nN=int((yt == 0).sum()), auc=round(a, 4),
                                    ci_lo=round(lo, 4), ci_hi=round(hi, 4), usable=True))
            else:
                bw_rows.append(dict(param=pn, width=w, bin=l, n=n, nN=nN,
                                    auc=np.nan, ci_lo=np.nan, ci_hi=np.nan, usable=False))

    P[pn] = dict(gd=gd, pd=pd_, ga=ga, pa=pa, vals_f=vals_f, stats_f=stats_f,
                 dirs_f=dirs_f, info_f=info_f, frac_bands=frac_bands, frac_raw=frac_raw,
                 vals_b=vals_b, labels_b=labels_b, stats_d=stats_d, dirs_b=dirs_b,
                 dose_bands=dose_bands, dose_raw=dose_raw,
                 frac_scores=frac_scores, bin_scores=bin_scores)
    print(f"  {pn}: global direction (fractions) = {info_f['global']}; "
          f"flip-allowed fractions = {[t for t, s in info_f['sig'].items() if s]}")

SS = pd.DataFrame(traj_frac)
BS = pd.DataFrame(traj_bin)
BW = pd.DataFrame(bw_rows)


# ============================================================
# Fig1 — final bands (fraction | dose) with individual case trajectories
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


def plot_fig1(save_path):
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
                ml.style_axes(ax, xlabel='Cumulative $EQD_2$ [Gy]', ylabel=param_labels[ri],
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
    fig.legend(hs, ls, loc='lower center', fontsize=18, ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.05))
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Fig2 — grouped box plots (Overall / Response / Non-response) + stars
# ============================================================
def plot_fig2(save_path):
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
                ml.style_axes(ax, xlabel='Cumulative $EQD_2$ [Gy]', ylabel=param_labels[ri],
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
    fig.legend(hs, [h.get_label() for h in hs], loc='lower center', fontsize=18,
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
# Fig3 / Fig4 — subgroup (Overall / Metastasis / Primary liver) ROC + AUC curves
# ============================================================
def plot_subgroup_auc(traj_df, kind, header_right, save_path):
    scopes = [('Overall', ml.C_ALL, None),
              ('Metastasis', ml.C_MET, set(traj_df[traj_df.type == 'Metastasis'].pid)),
              ('Primary liver', ml.C_PRI, set(traj_df[traj_df.type == 'Primary liver'].pid))]
    n = len(param_names)
    fig = plt.figure(figsize=(19, 4.5 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[0.95, 1.75], left=0.075, right=0.985,
                          top=0.93, bottom=0.06, wspace=0.10, hspace=0.42)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(n)])
    labels_all = sort_bin_labels([l for l in P[param_names[0]]['labels_b'] if l != '0'])
    keys = list(range(1, MAX_FRACTION + 1)) if kind == 'fraction' else labels_all
    keycol = 'Fraction' if kind == 'fraction' else 'bin'
    for i, pn in enumerate(param_names):
        # left: patient-level ROC (mean score over the whole course)
        ax = axes[i, 0]
        d0 = traj_df[traj_df.param == pn].groupby('pid').agg(
            score=('score', 'mean'), resp=('resp', 'first'))
        for label, col, subset in scopes:
            d = d0 if subset is None else d0[d0.index.isin(subset)]
            r = ml.auc_of(d.resp.values, d.score.values)
            if r is None:
                continue
            fpr, tpr = ml.roc_xy(d.resp.values, d.score.values)
            ax.plot(fpr, tpr, color=col, lw=2.0, label=label)
        ax.plot([0, 1], [0, 1], ':', color='grey', lw=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_aspect('equal', anchor='E')
        ml.style_axes(ax, xlabel='FPR' if i == n - 1 else None)
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        # right: AUC trajectory per fraction / per dose bin (bin '0' excluded)
        ax = axes[i, 1]
        for label, col, subset in scopes:
            xs, ys = [], []
            for k in keys:
                d = traj_df[(traj_df.param == pn) & (traj_df[keycol] == k)]
                if subset is not None:
                    d = d[d.pid.isin(subset)]
                r = ml.auc_of(d.resp.values, d.score.values)
                if r is not None:
                    xs.append(k)
                    ys.append(r[0])
            if not xs:
                continue
            xp = xs if kind == 'fraction' else [keys.index(k) for k in xs]
            ax.plot(xp, ys, 'o-' if kind == 'fraction' else 's-',
                    color=col, lw=2.5, ms=8, label=label)
        if kind == 'bin':
            ml.style_axes(ax, xlabel='Cumulative $EQD_2$ [Gy]' if i == n - 1 else None,
                          xticks=np.arange(len(keys)), xticklabels=keys,
                          rotate=30, xtick_fs=14)
            ax.set_xlim(-0.5, len(keys) - 0.5)
        else:
            ml.style_axes(ax, xlabel='Fraction' if i == n - 1 else None,
                          xticks=keys, xticklabels=[str(k) for k in keys])
        ax.axhline(0.5, color='k', lw=0.8, alpha=0.4)
        ax.set_ylim(0.2, 1.03)
    fig.canvas.draw()
    for i, pn in enumerate(param_names):
        pos = axes[i, 0].get_position()
        fig.text(pos.x0 - 0.045, (pos.y0 + pos.y1) / 2, PSYM[pn], rotation=90,
                 ha='center', va='center', fontsize=20)
    pl_ = axes[0, 0].get_position()
    pr_ = axes[0, 1].get_position()
    fig.text((pl_.x0 + pl_.x1) / 2, 0.965, 'Overall ROC', ha='center', fontsize=18)
    fig.text((pr_.x0 + pr_.x1) / 2, 0.965, header_right, ha='center', fontsize=18)
    hs = [mlines.Line2D([], [], color=c, lw=2.5, marker='o', ms=6) for _, c, _ in scopes]
    fig.legend(hs, [s for s, _, _ in scopes], loc='lower center', ncol=3, frameon=True,
               fontsize=18, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Fig5 / Fig6 — bin-width sensitivity (AUC curves; per-bin counts)
# ============================================================
def plot_fig5(save_path):
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
                    color=ml.PALETTE[pi], linewidth=5, markersize=12,
                    label=PSYM[pn], alpha=0.9)
        nu = sub[~sub.usable | sub.auc.isna()]
        if len(nu):   # bins that cannot be evaluated (sample-size gate)
            ax.plot(nu['bin'].map(xmap).values, np.full(len(nu), 0.5), 'x',
                    color='#7F7F7F', ms=10, mew=2.2, alpha=0.9)
        ax.axhline(y=0.5, color='grey', linestyle=':', linewidth=2.5)
        ax.set_xticks([xmap[b] for b in bins])
        ax.set_xticklabels(bins, fontsize=18, rotation=30, ha='right')
        ml.style_axes(ax, xlabel='Cumulative $EQD_2$ [Gy]', ylabel='AUC')
        _lo = min(0.45, float(sub.auc.min()) - 0.02) if sub.auc.notna().any() else 0.45
        ax.set_ylim(_lo, 1.05)
        ax.legend(fontsize=24, loc='upper center', ncol=4, frameon=True)
        tag = ' (primary)' if w == 7 else ''
        ax.set_title(f'{w} Gy bins{tag}', fontsize=18,
                     fontweight='bold' if w == 7 else 'normal')
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


def plot_fig6(save_path):
    R2 = BW[BW.param == 'ADC'].copy()      # counts are parameter-independent
    R2['mid'] = R2['bin'].str.extract(r'\((\d+),(\d+)').astype(float).mean(axis=1)
    wcol = {6: '#6A3D9A', 7: '#000000', 8: '#E15759'}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
    for ax, grp, ylab, title in [(axes[0], 'R', 'responders per bin (nR)', 'Response group'),
                                 (axes[1], 'N', 'non-responders per bin (nN)',
                                  'Non-response group')]:
        for w in WIDTHS_FIG:
            sub = R2[R2.width == w].sort_values('mid')
            y = (sub.n - sub.nN) if grp == 'R' else sub.nN
            ax.plot(sub.mid, y, '-', color=wcol[w], lw=2.4 if w == 7 else 1.3,
                    alpha=1.0 if w == 7 else 0.65, marker='o',
                    ms=7 if w == 7 else 5, mec='w' if w == 7 else 'none', mew=0.8,
                    label=f'{w} Gy')
        ax.axhline(3, color='#E15759', lw=1.0, ls=':')   # n=3 band-construction floor
        ax.set_ylim(0, None)
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 72)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        for lab in ax.get_yticklabels():
            lab.set_fontsize(11)
            lab.set_fontfamily('Times New Roman')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    axes[0].legend(ncol=3, fontsize=10, frameon=True, loc='upper right', title='bin width')
    axes[1].set_xlabel('Cumulative EQD2 bin midpoint [Gy]', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Fig7 — four-parameter AUC: fraction | cumulative EQD2 bins (bin '0' excluded)
# ============================================================
def plot_fig7(save_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax = axes[0]
    for pi, pn in enumerate(param_names):
        xs, ys = [], []
        for f, (names, yt, yp) in sorted(P[pn]['frac_scores'].items()):
            r = ml.auc_of(yt, yp)
            if r is not None:
                xs.append(f)
                ys.append(r[0])
        ax.plot(xs, ys, marker=ml.MARKERS[pi], linestyle='-', color=ml.PALETTE[pi],
                linewidth=5, markersize=12, label=PSYM[pn], alpha=0.9)
    ax.axhline(0.5, color='grey', ls=':', lw=2.5)
    ml.style_axes(ax, xlabel='Fraction', ylabel='AUC',
                  xticks=range(1, MAX_FRACTION + 1),
                  xticklabels=[str(f) for f in range(1, MAX_FRACTION + 1)])
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=24, loc='upper center', ncol=4, frameon=True)
    ax = axes[1]
    labels_nz = sort_bin_labels([l for l in P[param_names[0]]['labels_b'] if l != '0'])
    for pi, pn in enumerate(param_names):
        xs, ys = [], []
        for l in labels_nz:
            if l in P[pn]['bin_scores']:
                _, yt, yp = P[pn]['bin_scores'][l]
                r = ml.auc_of(yt, yp)
                if r is not None:
                    xs.append(labels_nz.index(l))
                    ys.append(r[0])
        ax.plot(xs, ys, marker=ml.MARKERS[pi], linestyle='-', color=ml.PALETTE[pi],
                linewidth=5, markersize=12, label=PSYM[pn], alpha=0.9)
    ax.axhline(0.5, color='grey', ls=':', lw=2.5)
    ml.style_axes(ax, xlabel='Cumulative $EQD_2$ [Gy]', ylabel='AUC',
                  xticks=np.arange(len(labels_nz)), xticklabels=labels_nz, rotate=30)
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=24, loc='upper center', ncol=4, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)
    print('  saved ->', save_path)


# ============================================================
# Generate all figures
# ============================================================
plot_fig1(os.path.join(FIGD, 'final_bands_fraction_and_EQD2_with_cases.png'))
plot_fig2(os.path.join(FIGD, 'grouped_boxplots_fraction_and_EQD2_with_significance.png'))
plot_subgroup_auc(SS, 'fraction', 'AUC by fraction',
                  os.path.join(FIGD, 'subgroup_ROC_and_AUC_by_fraction.png'))
BS_nz = BS[BS['bin'] != '0'].reset_index(drop=True)
plot_subgroup_auc(BS_nz, 'bin', 'AUC by cumulative EQD2',
                  os.path.join(FIGD, 'subgroup_ROC_and_AUC_by_EQD2.png'))
plot_fig5(os.path.join(FIGD, 'bin_width_sensitivity_AUC.png'))
plot_fig6(os.path.join(FIGD, 'bin_width_sample_counts.png'))
plot_fig7(os.path.join(FIGD, 'AUC_by_fraction_and_EQD2.png'))
print("  All figures done.")
