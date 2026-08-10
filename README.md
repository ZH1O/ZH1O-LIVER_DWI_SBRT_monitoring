======================================
  Longitudinal IVIM-based Treatment Monitoring during MR-Linac Liver SBRT
======================================

<img width="5512" height="2244" alt="示意图" src="https://github.com/user-attachments/assets/a4057611-5203-4830-8866-f3f739bc23bb" />

Overview
--------
This project implements a temporally stratified treatment monitoring framework
using longitudinal intravoxel incoherent motion (IVIM) parameters (ADC, D, D*,
f) acquired during MR-Linac-based liver SBRT. The framework evaluates two
complementary monitoring dimensions: fraction-based (time) and cumulative
EQD₂-based (accumulated biological dose).


Scripts
-------
1. Image QA.py
   Quality assurance of DWI images. Computes SNR in normal liver parenchyma
   (excluding tumor) across b-values (0, 30, 50, 100, 150, 300, 400, 500 s/mm²)
   using QIBA thresholds (SNR ≥ 7 for b ≤ 10, SNR ≥ 3 for b > 10).
   Output: qa_pass.csv (fraction-level pass/fail records).

2. FDR.py
   Statistical analysis with FDR correction. Performs:
   - Between-group comparisons (Mann-Whitney U) at each fraction and dose bin
   - Within-group comparisons (Wilcoxon signed-rank) vs baseline (fraction 1 /
     bin 0)
   Output: console-logged p-values and significance levels.

3. Fraction Band.py
   Visualization of fraction-based monitoring. Generates a 4-row × 2-column
   figure showing 25th–75th percentile bands and filled monitoring bands with
   split lines, overlaid with individual patient trajectories.
   Output: fraction-based monitoring figure.

4. Dose Band.py
   Visualization of cumulative EQD₂-based monitoring. Same layout as Fraction
   Band.py but grouped by 7 Gy EQD₂ bins.
   Output: cumulative EQD₂-based monitoring figure.

5. LOOCV+Delong.py
   Core analysis pipeline. Performs:
   - Leave-one-out cross-validation (LOOCV) for fraction-based and dose-based
     temporally stratified monitors
   - Pooled (static) monitor as comparator
   - DeLong test comparing stratified vs static monitors
   - AUC plot generation
   Outputs: results_fraction.csv, results_dose_bin.csv, delong_comparison.csv,
            AUC figure.


Data Structure
--------------
Expected directory layout:

  base_path/
  |
  +-- patient folder (named "name=N1=N2")
  |     |
  |     +-- fraction folder (named "name=N3")
  |           |
  |           +-- parameter_map/
  |           |     +-- ADC.nii.gz
  |           |     +-- Dslow.nii.gz
  |           |     +-- Dfast.nii.gz
  |           |     +-- Ffast.nii.gz
  |           |
  |           +-- RTst_registration/
  |           |     +-- Tumor.nii.gz
  |           |
  |           +-- RT/
  |           |     +-- Dose.nii.gz
  |           |
  |           +-- DWI_registration/
  |                 +-- 0.nii.gz   (b = 0 s/mm2)
  |                 +-- 1.nii.gz   (b = 30)
  |                 +-- 2.nii.gz   (b = 50)
  |                 +-- 3.nii.gz   (b = 100)
  |                 +-- 4.nii.gz   (b = 150)
  |                 +-- 5.nii.gz   (b = 300)
  |                 +-- 6.nii.gz   (b = 400)
  |                 +-- 7.nii.gz   (b = 500)

Patient folder naming: "name=N1=N2"
  - name: patient identifier
  - N1:   class label (integer)
  - N2:   total number of fractions (integer)

Fraction folder naming: "name=N3"
  - name: same as parent folder
  - N3:   fraction number (integer, starting from 1)


Key Parameters
--------------
- α/β = 10 Gy (uniform across all tumor types)
- EQD₂ formula: EQD₂ = d × (α/β + d) / (α/β + 2)
- Per-fraction dose: D10 (90th percentile within GTV)
- Cumulative EQD₂ bins: [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70] Gy
- QA: SNR thresholds, liver parenchyma with 3-voxel erosion
- LOOCV: minimum 3 samples per group per bin/fraction
- FDR correction: Benjamini-Hochberg, α = 0.05


Output Files
------------
qa_pass.csv              - QA results (patient, fraction, pass/fail)
results_fraction.csv     - AUC/Accuracy/Sensitivity/Specificity by fraction
results_dose_bin.csv     - Same metrics by cumulative EQD₂ bin
delong_comparison.csv    - DeLong test: stratified vs static AUC with p-values


Dependencies
------------
Python 3.12.7, NumPy 1.26.4, SimpleITK 2.4.1, Pandas 2.2.2, Matplotlib 3.9.2, Seaborn 0.13.2, SciPy 1.13.1, Statsmodels 0.14.2
