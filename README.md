
  Longitudinal IVIM-based Treatment Monitoring during MR-Linac Liver SBRT
=

<img width="5512" height="3228" alt="Figure 1" src="https://github.com/user-attachments/assets/9c3a13a4-c959-4550-b6fa-ab931553c8da" />


Overview
--------
This project implements a temporally stratified treatment monitoring framework
using longitudinal intravoxel incoherent motion (IVIM) parameters (ADC, D, D*,
f) acquired during MR-Linac-based liver SBRT. The framework evaluates two
complementary monitoring dimensions: fraction-based (time) and cumulative
EQD2-based (accumulated biological dose), and additionally provides
tumor-type subgroup analyses (primary liver vs metastasis) and an EQD2
bin-width sensitivity analysis supporting the 7 Gy bin choice.
The code is organized as a self-contained analysis library (monitor_lib.py)
plus a figure-generation program (run_figures.py). Only figures are produced;
no tables, CSVs or other intermediate files are written.

Scripts
-------
1. monitor_lib.py
   Self-contained analysis library:
   - Extraction: per-fraction GTV mean of each IVIM parameter and the GTV
     mean dose (Dmean); cumulative EQD2 bookkeeping (alpha/beta = 10 Gy);
     QA filtering through an external fraction-level QA ledger (qa_pass.csv).
   - Monitoring bands: separated response / non-response filled bands with
     split lines (IQR-based, stepwise separation of overlapping bands).
   - Band direction protocol: global band orientation = majority vote of the
     local orientations over all fractions / dose bins (ties resolved with
     all time points pooled); a time point may deviate from the global
     orientation (flip) ONLY if a within-group comparison versus baseline
     (fraction 1 / 0 Gy bin) is significant (paired Wilcoxon, BH-FDR
     q < 0.05); there is no whole-time-point flip at AUC < 0.5.
   - LOOCV scoring: leave-one-out bands per fraction and per EQD2 bin; a fold
     score is inverted when the fold band orientation differs from the
     effective direction of that time point.
   - Statistics: within-group paired Wilcoxon signed-rank versus baseline
     (unified threshold n_pairs >= 3 for both groups); between-group
     Mann-Whitney U; Benjamini-Hochberg FDR (families: within = response +
     non-response combined; between = all fractions / bins).
   - AUC / ROC: DeLong AUC with 95% CI; ROC curves; subgroup evaluation
     subsets (Overall / Metastasis / Primary liver; LOOCV training always
     uses the full cohort).
   - Plotting helpers implementing the project figure style (Times New
     Roman, 18 pt labels/ticks, spine linewidth 0.8, six evenly spaced y
     ticks, per-row frame alignment, framed bottom legends).

2. run_figures.py
   Generates all figures (dpi = 1000) into figures/:
   - final_bands_fraction_and_EQD2_with_cases.png: 4 parameters x
     (fraction final bands | cumulative-EQD2 final bands), overlaid with two
     representative case trajectories (response / non-response).
   - grouped_boxplots_fraction_and_EQD2_with_significance.png: per-fraction
     and per-bin grouped box plots (Overall gray / Response / Non-response)
     with significance stars above boxes (within-group vs baseline) and
     brackets between the Response and Non-response boxes (between-group).
   - subgroup_ROC_and_AUC_by_fraction.png and
     subgroup_ROC_and_AUC_by_EQD2.png: tumor-type subgroup patient-level ROC
     curves (whole-course mean score) and LOOCV AUC trajectories per fraction
     / per cumulative-EQD2 bin (bin 0 excluded).
   - bin_width_sensitivity_AUC.png: LOOCV AUC versus cumulative-EQD2 bins
     for 6 / 7 / 8 Gy bin widths (7 Gy marked primary); non-evaluable bins
     marked with gray crosses.
   - bin_width_sample_counts.png: per-bin responder / non-responder counts
     for 6 / 7 / 8 Gy with the n = 3 band-construction floor line.
   - AUC_by_fraction_and_EQD2.png: four-parameter LOOCV AUC by fraction and
     by cumulative-EQD2 bin (bin 0 excluded).
   Output: figures only.

Data structure (identical to the previous release)
--------------------------------------------------
```
base_path/
│
├── patient folder (name=N1=N2)
│   │
│   ├── fraction folder (name=N3)
│   │   │
│   │   ├── parameter_map/
│   │   │   ├── ADC.nii.gz
│   │   │   ├── Dslow.nii.gz
│   │   │   ├── Dfast.nii.gz
│   │   │   └── Ffast.nii.gz
│   │   │
│   │   ├── RTst_registration/
│   │   │   └── Tumor.nii.gz
│   │   │
│   │   ├── RT/
│   │   │   └── Dose.nii.gz
│   │   │
│   │   └── DWI_registration/
│   │       ├── 0.nii.gz   (b = 0 s/mm²)
│   │       ├── 1.nii.gz   (b = 30)
│   │       ├── 2.nii.gz   (b = 50)
│   │       ├── 3.nii.gz   (b = 100)
│   │       ├── 4.nii.gz   (b = 150)
│   │       ├── 5.nii.gz   (b = 300)
│   │       ├── 6.nii.gz   (b = 400)
│   │       └── 7.nii.gz   (b = 500)
│   │
│   ├── fraction folder (name=2)
│   │   └── ...
│   │
│   └── fraction folder (name=3)
│       └── ...
│
└── patient folder (name=...)
    └── ...
```

Naming conventions:
- Patient folder: `name=N1=N2` where N1 = class label, N2 = total fractions
  (class 0/1 = response, 2/3 = non-response; response group = 0+1,
  non-response group = 2+3)
- Fraction folder: `name=N` where N = fraction number (starting from 1)

Key Parameters
--------------
- α/β = 10 Gy (uniform across all tumor types)
- EQD2 formula: EQD2 = d × (α/β + d) / (α/β + 2)
- Per-fraction dose metric: Dmean (arithmetic mean over GTV voxels)
- Cumulative EQD2 bins: 7 Gy grid [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
  Gy; bin-width sensitivity evaluated over 5-10 Gy (figures: 6/7/8 Gy);
  bin "0" (pre-treatment) is excluded from all dose-axis figures
- QA: external fraction-level ledger qa_pass.csv (patient, fraction,
  all_pass); fractions with all_pass = False are excluded during extraction.
  The QA computation itself (DWI SNR in normal liver parenchyma across
  b-values) is not part of this package; the ledger is an input.
- Subgroups: primary site containing the CJK keywords for liver / HCC /
  bile duct -> Primary liver, otherwise Metastasis (82 lesions = 16 primary
  + 66 metastasis); read from the clinical Excel at run time (input, not
  shipped)
- LOOCV gates: >= 6 valid patients and >= 3 per group per fraction;
  >= 3 per group per bin
- Statistics: within-group paired Wilcoxon signed-rank vs baseline
  (fraction 1 / bin 0; unified n_pairs >= 3); between-group Mann-Whitney U;
  Benjamini-Hochberg FDR, α = 0.05
- Band direction protocol: global orientation by majority vote over all
  time points; flipping allowed only at time points with a significant
  within-group vs-baseline difference (FDR q < 0.05); no AUC < 0.5 whole flip
- Significance stars: * p < 0.05, ** p < 0.01, *** p < 0.001 (FDR-adjusted)
- AUC confidence intervals: DeLong

Output Files
------------
figures/final_bands_fraction_and_EQD2_with_cases.png
figures/grouped_boxplots_fraction_and_EQD2_with_significance.png
figures/subgroup_ROC_and_AUC_by_fraction.png
figures/subgroup_ROC_and_AUC_by_EQD2.png
figures/bin_width_sensitivity_AUC.png
figures/bin_width_sample_counts.png
figures/AUC_by_fraction_and_EQD2.png

Dependencies
------------
Python 3.12.7, NumPy 1.26.4, SimpleITK 2.4.1, Pandas 2.2.2,
Matplotlib 3.9.2, SciPy 1.13.1
(Seaborn and Statsmodels are not required; FDR is implemented internally.)


Dependencies
------------
Python 3.12.7, NumPy 1.26.4, SimpleITK 2.4.1, Pandas 2.2.2, Matplotlib 3.9.2, Seaborn 0.13.2, SciPy 1.13.1, Statsmodels 0.14.2
