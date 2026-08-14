# What is in this package, and what has to be regenerated

This archive contains the **code** for the round-2 revision. It does **not**
contain `results/` or `figures/`, because those are produced by running the
pipeline and must be regenerated rather than copied.

## New in this round

| File | Purpose |
| --- | --- |
| `src/k_sweep.py` | augmentation-count sweep with a unified Monte-Carlo seed, plus the per-object fragility flags |
| `src/nested_cv.py` | twenty-five outer folds, K selected on an inner validation split, confidence intervals |
| `src/unseen_noise.py` | both models under nine perturbation families, eight of them unseen in training |
| `src/feature_tiers.py` | three operational feature tiers |
| `src/recent_method.py` | head-to-head against TabM under the identical protocol |
| `src/recent_method_aug.py` | augmentation on top of TabM, with a clean unaugmented validation split |
| `src/calibration_significance.py` | the paired calibration test quoted in Section 5.1 |
| `verify_paper_numbers.py` | checks every regenerated value against the published tables |

## Corrected in this round

| File | Correction |
| --- | --- |
| `src/noise_sensitivity.py` | the split-normal sampler drew the sign and the scale independently, which makes the distribution symmetric; it now uses one draw for both and takes the two scales from the catalogued upper and lower error bars. Three correlation strengths, a bounded uniform family and an error-scale sweep were added |
| `src/measurement_error.py` | now saves the per-candidate flip rates and uncertainties |
| `src/error_aware_training.py` | Monte-Carlo seed unified from 1 to 0, so the fragile fraction agrees exactly with `measurement_error.py` instead of to within sampling noise |
| `src/physical_three_models.py` | now emits the Brier score and calibration error, without which two columns of Table S2 were not reproducible, and saves the Bayesian-network out-of-fold mean and standard deviation |
| `src/flux_calibration.py` | `_bnn` can return the predictive standard deviation |
| `reproduce.py` | four stages were missing from the driver; the retired fragility script was removed |
| `requirements.txt` | versions pinned exactly |

## Removed

`src/fragility_ablation.py` produced a curve whose meaning differed from the
table it was supposed to support. The threshold sweep now comes from
`measurement_error.py` alone, so the figure and the table agree by construction.

## Still to be regenerated on the machine that runs the pipeline

`results/*.json`, `results/*.npz`, `figures/*.png`, and
`results/environment.json`.

## Order

```
pip install -r requirements.txt
# place the input files in data/ as described in data/README.md
python reproduce.py
python verify_paper_numbers.py      # must print no mismatches
python reproduce.py --with-recent   # needs pytabkit and a GPU
python verify_paper_numbers.py
```

`verify_paper_numbers.py` exits non-zero on the first mismatch. Do not push a
run that does not pass it.
