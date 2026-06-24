# Exoplanet Detection Pipeline:- BAH 2026, Problem Statement 7
**AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves**

This folder contains a complete, *run-and-verified* ML pipeline trained on
the NASA Exoplanet Archive's Kepler KOI cumulative table (`koi_table.csv`,
9,564 Kepler Objects of Interest).

---

## 1. What the data actually is?

`koi_table.csv` is **not** a raw light curve - it has no time/flux
columns. It is NASA's catalog of *already-extracted* transit parameters
(period, depth, duration, SNR, stellar properties, disposition) for every
KOI ever flagged by the Kepler pipeline. 
<br>So:
- Pipeline **Steps 1–2** (flux cleaning, BLS period search, phase-folding)
  were already performed by NASA to *produce* this table - that's what
  `Section 12` of the code reproduces one real raw flux, optionally,
  on your own machine.
- Pipeline **Steps 3–6** (feature-based classification, vetting,
  output) is what we trained and evaluated here, end-to-end, on real data.

A
model trained on a handful of examples can't learn the real distribution
of transit shapes. This table gives you **9,564 labeled examples**
(2,747 CONFIRMED, 1,978 CANDIDATE, 4,839 FALSE POSITIVE) -three orders of
magnitude more signal for the model to learn from.

## 2. The data-leakage trap we avoided 
The table contains `koi_score` and `koi_fpflag_nt/ss/co/ec` - these are
**NASA's own vetting decision**, not independent physical measurements.
Training on them lets the model "cheat" by reading the answer key.

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Clean model (no leakage) | **0.853** | **0.817** |
| Same model + `koi_score`/`fpflag_*` included | 0.944 | 0.923 |

The jump is the leakage effect, not real detection skill - see
`outputs/leakage_ablation.png`.

## 3. Final model results (leakage-free, real, reproducible)

- **Algorithm:** XGBoost multiclass classifier (Random Forest trained too,
  as a comparison baseline - see `outputs/metrics_summary.json`)
- **Features:** 25 leakage-free, physically meaningful engineered features:
  transit shape (period, duration, depth, impact parameter, radius ratio),
  signal detection statistics (model SNR, single/multi-event statistics,
  odd-even depth significance), host-star properties, and **real
  centroid-shift measurements** (`koi_dicco_msky`, `koi_dikco_msky`,
  `koi_fwm_stat_sig`) - this is what actually powers the "reject
  blended binaries / centroid shift" vetting step in your diagram.
- **Train/test split:** 80/20, stratified, random_state=42

| Metric | Value |
|---|---|
| 3-class test accuracy | **85.3%** |
| 3-class macro F1 | **0.82** |
| CONFIRMED precision / recall | 0.89 / 0.91 |
| FALSE POSITIVE precision / recall | 0.88 / 0.92 |
| CANDIDATE precision / recall | 0.70 / 0.60 |
| Binary (planet-like vs false positive) ROC-AUC | **0.972** |
| Binary PR-AUC | **0.973** |

**Why CANDIDATE is weaker:** CANDIDATE means "not yet resolved by anyone" -
by definition these sit closest to the decision boundary between real
planets and false positives, so lower recall here is scientifically
expected, not a bug. 

**Top features driving the model** (see `outputs/feature_importance.png`):
1. `koi_count` — number of KOIs found around the same star (multi-planet
   systems are far more likely to be real planets than false alarms — this
   is the real "validation by multiplicity" principle used in actual
   exoplanet confirmation papers)
2. `koi_prad` — planet radius
3. `koi_dikco_msky` — centroid shift during transit (confirms the model
   is learning to reject blended eclipsing binaries, exactly as intended)
4. `koi_model_snr` — transit signal-to-noise

## 4. Files in this project

```
outputs/
  model_xgb.pkl                     - trained model bundle (load with joblib)
  metrics_summary.json              - every number above, machine-readable
  confusion_matrix.png
  feature_importance.png
  roc_pr_curves.png
  leakage_ablation.png
  simulated_transit_example.png
  feature_metadata.json
  real_lightcurve_KIC10797460.png
  class_distribution.png
```

## 5. How to run it yourself

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
then run the folder containing exoplanet_detection_pipeline.ipynb along with the koi_table.csv file
```

Last section of the code additionally needs `pip install lightkurve`
and internet access (it pulls real data from NASA's MAST archive) - run it
separately on your own laptop, not required for the core pipeline above.
