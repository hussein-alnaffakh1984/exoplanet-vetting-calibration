#!/usr/bin/env python3
"""
reproduce.py  -  Master reproduction script.

Runs the full analysis pipeline and regenerates the result files that back the
tables, then regenerates the figures. All randomness is fixed by the seeds in
src/config.py and by torch.manual_seed(0) inside the Bayesian network, so the
pipeline is deterministic.

    python reproduce.py                 # analyses + figures
    python reproduce.py --no-figs       # analyses only
    python reproduce.py --with-recent   # also the TabM comparison (needs pytabkit)
"""
import subprocess, sys, os, time

STAGES = [
    "src/flux_calibration.py",
    "src/physical_model.py",
    "src/measurement_error.py",          # also writes per-candidate flip rates
    "src/error_aware_training.py",
    "src/k_sweep.py",                    # K in {0,2,4,8} + fragility McNemar
    "src/cross_mission.py",
    "src/matched_comparison.py",
    "src/candidate_analysis.py",
    "src/stats_tests.py",
    "src/calibration_significance.py",
    "src/feature_leakage_ablation.py",
    "src/physical_three_models.py",      # also writes the BNN out-of-fold std
    "src/xgboost_physical.py",
    "src/noise_sensitivity.py",
    "src/unseen_noise.py",               # nine perturbation families, paired
    "src/nested_cv.py",                  # 25 outer folds, K chosen on validation
    "src/feature_tiers.py",              # operational feature tiers
]

# Require pytabkit and are much faster on a GPU; run with --with-recent
OPTIONAL_STAGES = [
    "src/recent_method.py",
    "src/recent_method_aug.py",
]

def run(path):
    print(f"\n>>> {path}")
    t = time.time()
    # scripts write into ../results from inside src/
    r = subprocess.run([sys.executable, os.path.basename(path)], cwd="src")
    if r.returncode != 0:
        print(f"!! FAILED: {path}"); sys.exit(r.returncode)
    print(f"    done in {time.time()-t:.1f}s")

def main():
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    for s in STAGES:
        run(s)
    if "--with-recent" in sys.argv:
        for s in OPTIONAL_STAGES:
            run(s)
    if "--no-figs" not in sys.argv:
        print("\n>>> make_figures.py")
        subprocess.run([sys.executable, "make_figures.py"])
    print("\nAll results and figures regenerated.")

if __name__ == "__main__":
    main()
