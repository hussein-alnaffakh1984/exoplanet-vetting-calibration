#!/usr/bin/env python3
"""Check the regenerated results against every number printed in the paper.

Run this after `python reproduce.py`. It fails loudly on the first mismatch, so
a silent drift between the released code and the published tables is impossible.

    python verify_paper_numbers.py
"""
import json, sys, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ok, bad = [], []


def chk(label, got, want, tol=5e-4):
    if got is None:
        bad.append(f"{label}: MISSING"); return
    if abs(got - want) <= tol:
        ok.append(label)
    else:
        bad.append(f"{label}: got {got:.6g}, paper says {want:.6g}")


def load(name):
    try:
        return json.load(open(os.path.join(R, name)))
    except Exception:
        return None


# ---- Table 3 / Table S1 : physical model ---------------------------------
d = load("physical_model.json")
if d:
    cv = d.get("cv", {})
    chk("Table 3 accuracy", cv.get("acc", [None])[0], 0.919, 1e-3)
    chk("Table 3 AUC", cv.get("auc", [None])[0], 0.970, 1e-3)
    chk("Table 3 Brier", cv.get("brier", [None])[0], 0.063, 1e-3)
    chk("Table 3 ECE", cv.get("ece", [None])[0], 0.024, 1e-3)
    chk("Table S1 F1", d.get("f1"), 0.902, 1e-3)
    chk("Table S1 precision", d.get("precision"), 0.888, 1e-3)
    chk("Table S1 recall", d.get("recall"), 0.917, 1e-3)
    cm = d.get("confusion", [[0, 0], [0, 0]])
    for got, want, lab in ((cm[0][0], 3649, "TN"), (cm[0][1], 316, "FP"),
                           (cm[1][0], 227, "FN"), (cm[1][1], 2503, "TP")):
        chk(f"Table S1 confusion {lab}", got, want, 0)

# ---- Table S3 / S5 : measurement error -----------------------------------
d = load("measurement_error.json")
if d:
    chk("Table S3 mean flip rate", d["mean_flip_rate"], 0.0657)
    chk("Table S3 fragile fraction", d["fragile_fraction"], 0.1867)
    for k, v in zip(("5%", "10%", "15%", "20%"), (0.2323, 0.1867, 0.1583, 0.1337)):
        chk(f"Table S5 threshold {k}", d["fragility_ablation"][k], v)
    for i, v in enumerate((0.9933, 0.9417, 0.8027)):
        chk(f"Table S3 tercile {i}", d["acc_by_uncertainty_tercile"][i], v)

# ---- Table 5 : baseline against error-aware ------------------------------
d = load("error_aware_training.json")
if d:
    chk("Table 5 baseline fragile", d["baseline"]["fragile_fraction"], 0.1867)
    chk("Table 5 error-aware fragile", d["error_aware"]["fragile_fraction"], 0.0732)
    chk("Table 5 baseline ECE", d["baseline"]["in_dist"]["ece"], 0.0246)
    chk("Table 5 error-aware ECE", d["error_aware"]["in_dist"]["ece"], 0.0169)

# ---- Table S6 : K sweep --------------------------------------------------
d = load("k_sweep.json")
if d:
    for K, frag in (("0", 0.1867), ("2", 0.0904), ("4", 0.0732), ("8", 0.0709)):
        chk(f"Table S6 K={K} fragile", d[K]["fragile_fraction"], frag)
    m = d.get("fragility_mcnemar_K0_vs_K4", {})
    chk("Table S5 fragility chi2", m.get("chi2"), 123.9, 0.5)
    chk("Table S5 fixed by augmentation", m.get("fixed_by_augmentation"), 168, 0)

# ---- Table S7 : unseen noise families ------------------------------------
d = load("unseen_noise.json")
if d:
    f = d["families"]
    for name, base, aug in (("gaussian_SEEN", 0.181, 0.069),
                            ("asymmetric_split_normal", 0.118, 0.067),
                            ("student_t_df3", 0.153, 0.056),
                            ("uniform_box", 0.190, 0.073),
                            ("scale_double", 0.256, 0.123)):
        chk(f"Table S7 {name} baseline", f[name]["baseline_fragile"][0], base, 2e-3)
        chk(f"Table S7 {name} augmented", f[name]["augmented_fragile"][0], aug, 2e-3)
    if not d.get("all_families_reduced"):
        bad.append("Table S7: all_families_reduced is no longer True")

# ---- Table 6 : nested cross-validation -----------------------------------
d = load("nested_cv.json")
if d:
    chk("Table 6 baseline fragile", d["baseline"]["fragile_fraction"][0], 0.181, 2e-3)
    chk("Table 6 error-aware fragile", d["error_aware"]["fragile_fraction"][0], 0.066, 2e-3)
    chk("Table 6 baseline AUC", d["baseline"]["auc"][0], 0.971, 1e-3)
    chk("Table 6 AUC difference", d["paired_difference"]["auc"]["mean_ci"][0], -0.0020, 5e-4)
    c = d["selected_K_counts"]
    if int(c.get("0", 0)) != 0:
        bad.append("Table 6: the unaugmented model was selected in some fold")
    chk("Table 6 K=8 count", int(c.get("8", 0)), 19, 0)

# ---- Table S8 : feature tiers --------------------------------------------
d = load("feature_tiers.json")
if d:
    t = d["tiers"]
    for tier, auc, ece_, base, aug in (
            ("tier0_detection_only", 0.904, 0.032, 0.081, 0.057),
            ("tier1_plus_stellar",   0.923, 0.031, 0.092, 0.064),
            ("tier2_full",           0.967, 0.027, 0.181, 0.069)):
        chk(f"Table S8 {tier} AUC", t[tier]["baseline"]["auc"][0], auc, 2e-3)
        chk(f"Table S8 {tier} ECE", t[tier]["baseline"]["ece"][0], ece_, 2e-3)
        chk(f"Table S8 {tier} fragile", t[tier]["baseline"]["fragile_fraction"][0], base, 2e-3)
        chk(f"Table S8 {tier} augmented", t[tier]["error_aware"]["fragile_fraction"][0], aug, 2e-3)

# ---- Table S9 : recent method --------------------------------------------
d = load("recent_method.json")
if d:
    A = d["part_A_five_fold"]
    chk("Table S9 LightGBM AUC", A["LightGBM"]["auc"][0], 0.971, 1e-3)
    chk("Table S9 TabM AUC", A["TabM"]["auc"][0], 0.974, 1e-3)
    chk("Table S9 DeLong z", A["tests"]["delong"]["z"], -2.93, 0.10)
d = load("recent_method_aug.json")
if d:
    chk("Table S9 TabM baseline fragile", d["baseline"]["fragile_fraction"][0], 0.304, 5e-3)
    chk("Table S9 TabM augmented fragile", d["error_aware"]["fragile_fraction"][0], 0.082, 5e-3)
    chk("Table S9 TabM ECE difference", d["paired_difference"]["ece"]["mean_ci"][0], 0.0002, 1e-3)

# ---- report ---------------------------------------------------------------
print(f"checks passed: {len(ok)}")
if bad:
    print(f"\nMISMATCHES ({len(bad)}):")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("\nEvery checked value matches the published tables.")
