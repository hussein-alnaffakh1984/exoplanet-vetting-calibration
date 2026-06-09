# Exoplanet Vetting: Calibration and Uncertainty (Kepler KOIs)

Code to reproduce a calibrated, uncertainty-aware machine-learning analysis of
Kepler Objects of Interest (KOIs) from fully public archives, with no
proprietary Data Validation products. It evaluates three probabilistic
classifiers (LightGBM, NGBoost, and a Monte-Carlo-Dropout Bayesian network) on
two open input representations of the same objects, and measures not only
accuracy but calibration and uncertainty, which are the properties that govern
follow-up prioritisation.

**Author:** Hussein Ali Hussein Al-Naffakh
**Affiliation:** University of Alkafeel, Najaf, Iraq
**GitHub:** https://github.com/hussein-alnaffakh1984

## What the code produces

| Script | Output |
| --- | --- |
| `src/flux_calibration.py` | `results/flux_calibration.json` |
| `src/physical_model.py` | `results/physical_model.json`, `results/phys_oof.npz` |
| `src/measurement_error.py` | `results/measurement_error.json` |
| `src/error_aware_training.py` | `results/error_aware_training.json` |
| `src/cross_mission.py` | `results/cross_mission.json` |
| `src/stats_tests.py` | `results/stats_tests.json` |
| `src/candidate_analysis.py` | `results/candidate_analysis.json` |
| `src/matched_comparison.py` | `results/matched_and_significance.json` |

The `results/` folder ships the JSON outputs from the reported runs (including
`xgboost_physical.json`, a deterministic XGBoost re-run of the physical model
under the same five-fold protocol, as a model-agnosticism check), and
`figures/` ships the generated plots, so the numbers can be inspected without
re-running anything.

## Data (download separately)

The input files are large and are **not** stored in the repository. Download
them and place them in `data/` (or set `EXO_DATA_DIR`). See `data/README.md`.

1. **Phase-folded light curves** (flux representation): Macedo and Zalewski
   (2024), Mendeley Data, DOI `10.17632/wctcv34962.3`. Save the global-view CSV
   as `data/all_global.csv` (2001 flux columns plus a `label` column).
2. **Kepler DR25 KOI table** (physical features): NASA Exoplanet Archive. Save
   as `data/koi_dr25_full.csv` (must include `koi_disposition`, the seven
   features in `src/config.py`, and their `_err1`/`_err2` columns).
3. **TESS TOI catalogue** (cross-mission): NASA Exoplanet Archive. Save as
   `data/TOI.csv` (must include `tfopwg_disp` and the `pl_*` columns).

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src
python physical_model.py        # 5-fold CV physical model
python measurement_error.py     # measurement-error fragility analysis
python error_aware_training.py  # error-bar augmentation
python cross_mission.py         # Kepler -> TESS transfer
python stats_tests.py           # McNemar, DeLong, ECE robustness
python flux_calibration.py      # flux calibration, three seeds
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
