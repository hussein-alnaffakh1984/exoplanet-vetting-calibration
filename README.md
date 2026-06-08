# Calibrated, Uncertainty-Aware Vetting of Kepler Exoplanet Candidates

Code accompanying the manuscript *"Calibration, Measurement-Aware Uncertainty,
and Error-Bar Training: A Probabilistic Machine-Learning Framework for Vetting
Kepler Exoplanet Candidates."*

This repository contains everything needed to reproduce the analysis from fully
public archives, with no proprietary Data Validation products. It evaluates
three probabilistic classifiers (LightGBM, NGBoost, a Monte-Carlo-Dropout
Bayesian network) on two open input representations of Kepler Objects of
Interest, and measures not only accuracy but calibration and uncertainty, which
are the properties that actually govern follow-up prioritisation.

**Author:** Hussein Ali Hussein Al-Naffakh
**Affiliation:** University of Babylon and University of Alkafeel, Najaf, Iraq
**Contact / GitHub:** https://github.com/hussein-alnaffakh1984

## What the code produces

| Script | Output | Manuscript element |
| --- | --- | --- |
| `src/flux_calibration.py` | `results/flux_calibration.json` | Table 2, Figure 1 (left) |
| `src/physical_model.py` | `results/physical_model.json`, `phys_oof.npz` | Tables 3, 4; Figure 2; Appendix B |
| `src/measurement_error.py` | `results/measurement_error.json` | Table 5, Figure 3; fragility ablation |
| `src/error_aware_training.py` | `results/error_aware_training.json` | Table 10, Figure 7 |
| `src/cross_mission.py` | `results/cross_mission.json` | Table 8, Figure 6 |
| `src/stats_tests.py` | `results/stats_tests.json` | Table 9 |

The `results/` folder ships with the JSON outputs from the reported runs, and
`figures/` ships the seven figures, so the numbers can be inspected without
re-running anything.

## Data (download separately)

The input files are large and are **not** stored in the repository. Download
them and place them in `data/` (or set `EXO_DATA_DIR`). See `data/README.md`.

1. **Phase-folded light curves** (flux representation): Macedo & Zalewski (2024),
   Mendeley Data, DOI `10.17632/wctcv34962.3`. Save the global-view CSV as
   `data/all_global.csv` (2001 flux columns plus a `label` column).
2. **Kepler DR25 KOI table** (physical features): NASA Exoplanet Archive. Save as
   `data/koi_dr25_full.csv` (must include `koi_disposition`, the seven features
   in `src/config.py`, and their `_err1`/`_err2` columns).
3. **TESS TOI catalogue** (cross-mission): NASA Exoplanet Archive. Save as
   `data/TOI.csv` (must include `tfopwg_disp` and the `pl_*` columns).

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src
python physical_model.py        # 5-fold CV physical model (Tables 3, 4; Fig 2)
python measurement_error.py     # fragility analysis (Table 5; Fig 3)
python error_aware_training.py  # error-bar augmentation (Table 10; Fig 7)
python cross_mission.py         # Kepler -> TESS (Table 8; Fig 6)
python stats_tests.py           # McNemar, DeLong, ECE robustness (Table 9)
python flux_calibration.py      # flux calibration, 3 seeds (Table 2; Fig 1)
```

The flux scripts read a one-time compact cache (`data/flux_global.npz`) that is
built automatically from `all_global.csv` on first run.

## Notes on methodology

- The natural class balance is kept throughout; no resampling (e.g. SMOTE),
  because resampling would distort the very posterior probabilities being
  calibrated.
- Robovetter flags and `koi_score` are **excluded** from the physical feature
  set: they are vetting-pipeline outputs and would make the task circular.
- Measurement-error propagation assumes independent Gaussian errors of the
  catalogued magnitude. This is a first-order approximation (the DR25 errors are
  mildly correlated; Thompson et al. 2018).
- Exact numbers may vary at the third decimal with library versions and CPU vs
  GPU; the qualitative findings are stable.

## License

MIT (see `LICENSE`).
