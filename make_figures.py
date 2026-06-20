#!/usr/bin/env python3
"""
make_figures.py  -  Regenerate every manuscript figure from saved result files.

Each figure is rebuilt purely from the artifacts in results/ (produced by
reproduce.py). Because the figures plot stored values, regeneration is exact and
independent of any random state. Each function documents the figure it produces
and the result file it reads, giving a figure -> data -> script trace.

    python make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = "results", "figures"
os.makedirs(OUT, exist_ok=True)
GREEN, AMBER, RED, BLUE, ORANGE = "#2ca02c", "#e1a700", "#d62728", "#1f77b4", "#ff7f0e"


def _reliability(p, y, nb=10):
    b = np.linspace(0, 1, nb + 1); xs, ys, e = [], [], 0.0
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1]) if i < nb - 1 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean())
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return np.array(xs), np.array(ys), e


def fig1_calibration():
    """Figure 1: flux reliability (left) + epistemic-uncertainty terciles (right).
       Reads flux_fig_preds.npz."""
    z = np.load(f"{RES}/flux_fig_preds.npz"); y = z["yte"].astype(int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.286, 5.087), dpi=300)
    a1.plot([0, 1], [0, 1], "--", color="gray", lw=1.4, label="Perfect calibration")
    for key, lab, col in [("lgb_raw", "LightGBM, raw", RED), ("lgb_iso", "LightGBM, isotonic", BLUE),
                          ("bnn_raw", "Bayesian network, raw", ORANGE), ("bnn_iso", "Bayesian network, isotonic", GREEN)]:
        xs, ys, e = _reliability(z[key].astype(float), y)
        a1.plot(xs, ys, "o-", color=col, ms=4, lw=1.6, label=f"{lab} (ECE {e:.2f})")
    a1.set_xlabel("Mean predicted probability of CONFIRMED", fontsize=11)
    a1.set_ylabel("Observed fraction CONFIRMED", fontsize=11)
    a1.set_title("Reliability of flux classifiers", fontsize=12, fontweight="bold")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.grid(alpha=0.25)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True)
    std, braw = z["bnn_std"], z["bnn_raw"]; correct = ((braw >= 0.5).astype(int) == y).astype(int)
    parts = np.array_split(np.argsort(std, kind="stable"), 3)
    accs = [correct[p].mean() * 100 for p in parts]; ns = [len(p) for p in parts]
    bars = a2.bar(range(3), accs, color=[GREEN, AMBER, RED], width=0.62, edgecolor="black", lw=0.6)
    for b, a, n in zip(bars, accs, ns):
        a2.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    a2.set_xticks(range(3)); a2.set_xticklabels(["Low", "Medium", "High"], fontsize=11)
    a2.set_xlabel("Epistemic-uncertainty tercile", fontsize=11); a2.set_ylabel("Accuracy (%)", fontsize=11)
    a2.set_title("Epistemic uncertainty flags unreliable cases", fontsize=12, fontweight="bold")
    a2.set_ylim(0, 105); a2.grid(axis="y", alpha=0.25)
    plt.tight_layout(); fig.savefig(f"{OUT}/fig1_calibration.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  fig1_calibration.png")


def fig2_physical_model():
    """Figure 2: physical-model reliability (left) + SHAP feature importance (right).
       Reads phys_oof.npz and physical_model.json."""
    z = np.load(f"{RES}/phys_oof.npz"); d = json.load(open(f"{RES}/physical_model.json"))
    y = z["y"].astype(int); fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.286, 5.087), dpi=300)
    a1.plot([0, 1], [0, 1], "--", color="gray", lw=1.4, label="Perfect calibration")
    xs, ys, e = _reliability(z["oof_p"].astype(float), y); a1.plot(xs, ys, "o-", color=RED, ms=4, lw=1.6, label=f"Raw (ECE {e:.2f})")
    xs, ys, e = _reliability(z["oof_p_iso"].astype(float), y); a1.plot(xs, ys, "s-", color=BLUE, ms=4, lw=1.6, label=f"Isotonic (ECE {e:.2f})")
    a1.set_xlabel("Mean predicted probability of CONFIRMED", fontsize=11); a1.set_ylabel("Observed fraction CONFIRMED", fontsize=11)
    a1.set_title("Reliability, physical-parameter model", fontsize=12, fontweight="bold")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.grid(alpha=0.25); a1.legend(fontsize=9, loc="upper left")
    sh = d["shap"]; feats = list(sh.keys())[::-1]; vals = [sh[f] for f in feats]
    a2.barh(range(len(feats)), vals, color=GREEN, edgecolor="black", lw=0.6)
    a2.set_yticks(range(len(feats))); a2.set_yticklabels([f.replace("koi_", "") for f in feats], fontsize=10)
    a2.set_xlabel("Mean absolute SHAP value", fontsize=11); a2.set_title("Feature importance (physical model)", fontsize=12, fontweight="bold")
    a2.grid(axis="x", alpha=0.25)
    plt.tight_layout(); fig.savefig(f"{OUT}/fig2_physical_model.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  fig2_physical_model.png")


def figS1_roc_confusion():
    """Figure S1: ROC (left) + out-of-fold confusion matrix (right). Reads phys_oof.npz, physical_model.json."""
    z = np.load(f"{RES}/phys_oof.npz"); d = json.load(open(f"{RES}/physical_model.json"))
    y = z["y"].astype(int); p = z["oof_p"].astype(float)
    thr = np.linspace(0, 1, 200); P, N = y.sum(), (1 - y).sum()
    tpr = [((p >= t) & (y == 1)).sum() / P for t in thr]; fpr = [((p >= t) & (y == 0)).sum() / N for t in thr]
    auc = float(d["cv"]["auc"][0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
    a1.plot(fpr, tpr, color=BLUE, lw=1.8, label=f"Physical model (AUC {auc:.3f})"); a1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    a1.set_xlabel("False-positive rate", fontsize=11); a1.set_ylabel("True-positive rate", fontsize=11)
    a1.set_title("ROC, physical-parameter model", fontsize=12, fontweight="bold"); a1.legend(fontsize=9); a1.grid(alpha=0.25)
    cm = np.array(d["confusion"])
    a2.imshow(cm, cmap="Blues"); a2.set_xticks([0, 1]); a2.set_yticks([0, 1])
    a2.set_xticklabels(["FALSE POSITIVE", "CONFIRMED"], fontsize=9); a2.set_yticklabels(["FALSE POSITIVE", "CONFIRMED"], fontsize=9)
    for i in range(2):
        for j in range(2):
            a2.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    a2.set_xlabel("Predicted", fontsize=11); a2.set_ylabel("True", fontsize=11); a2.set_title("Confusion matrix (out-of-fold)", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(f"{OUT}/figS1_roc_confusion.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  figS1_roc_confusion.png")


def figS2_three_classifiers():
    """Figure S2: accuracy and AUC for three classifiers on physical features. Reads physical_three_models.json."""
    d = json.load(open(f"{RES}/physical_three_models.json")); names = list(d.keys())
    acc = [d[n]["acc"][0] for n in names]; accs = [d[n]["acc"][1] for n in names]
    auc = [d[n]["auc"][0] for n in names]; aucs = [d[n]["auc"][1] for n in names]
    x = np.arange(len(names)); w = 0.36; fig, ax = plt.subplots(figsize=(7, 4.4), dpi=300)
    ax.bar(x - w / 2, acc, w, yerr=accs, capsize=3, color=BLUE, edgecolor="black", lw=0.6, label="Accuracy")
    ax.bar(x + w / 2, auc, w, yerr=aucs, capsize=3, color=GREEN, edgecolor="black", lw=0.6, label="AUC")
    for i in range(len(names)):
        ax.text(i - w / 2, acc[i] + 0.012, f"{acc[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, auc[i] + 0.012, f"{auc[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10); ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Score", fontsize=11); ax.set_title("Three classifiers on physical features", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); fig.savefig(f"{OUT}/figS2_three_classifiers.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  figS2_three_classifiers.png")


def figS4_measurement_terciles():
    """Figure S4: accuracy by measurement-uncertainty tercile. Reads measurement_error.json."""
    d = json.load(open(f"{RES}/measurement_error.json")); acc = d.get("acc_by_uncertainty_tercile")
    if not acc:
        print("  [skip] figS4: key missing"); return
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    bars = ax.bar(["Low", "Medium", "High"], [a * 100 for a in acc], color=[GREEN, AMBER, RED], edgecolor="black", lw=0.6, width=0.62)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, a * 100 + 1.5, f"{a*100:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Measurement-uncertainty tercile")
    ax.set_title("Accuracy falls with measurement uncertainty", fontweight="bold"); ax.set_ylim(0, 105)
    plt.tight_layout(); fig.savefig(f"{OUT}/figS4_measurement_terciles.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  figS4_measurement_terciles.png")


def figS5_cross_mission():
    """Figure S5: cross-mission transfer Kepler -> TESS. Reads cross_mission.json."""
    d = json.load(open(f"{RES}/cross_mission.json"))
    keys = [("kep_heldout", "Kepler\nhold-out"), ("tess_zero", "TESS\nzero-shot"), ("tess_recal", "TESS\nrecalibrated")]
    keys = [(k, l) for k, l in keys if k in d]
    if not keys:
        print("  [skip] figS5: keys missing"); return
    auc = [d[k].get("auc", np.nan) for k, _ in keys]; ece = [d[k].get("ece", np.nan) for k, _ in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), dpi=300)
    a1.bar([l for _, l in keys], auc, color=BLUE, edgecolor="black", lw=0.6)
    for i, v in enumerate(auc): a1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    a1.set_ylabel("AUC"); a1.set_ylim(0, 1.05); a1.set_title("Discrimination", fontweight="bold")
    a2.bar([l for _, l in keys], ece, color=RED, edgecolor="black", lw=0.6)
    for i, v in enumerate(ece): a2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold")
    a2.set_ylabel("ECE"); a2.set_title("Calibration", fontweight="bold")
    plt.tight_layout(); fig.savefig(f"{OUT}/figS5_cross_mission.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  figS5_cross_mission.png")


def figS6_feature_leakage():
    """Figure S6: feature-leakage ablation (accuracy / AUC as label-proximal features are removed).
       Reads feature_leakage_ablation.json."""
    d = json.load(open(f"{RES}/feature_leakage_ablation.json"))
    order = [("full_7feat", "All 7\nfeatures"), ("drop_prad_only", "Drop\nradius"), ("drop_prad_snr_impact", "Drop radius,\nSNR, impact")]
    order = [(k, l) for k, l in order if k in d]
    acc = [d[k]["acc"][0] for k, _ in order]; auc = [d[k]["auc"][0] for k, _ in order]
    x = np.arange(len(order)); w = 0.36; fig, ax = plt.subplots(figsize=(7, 4.4), dpi=300)
    ax.bar(x - w / 2, acc, w, color=BLUE, edgecolor="black", lw=0.6, label="Accuracy")
    ax.bar(x + w / 2, auc, w, color=GREEN, edgecolor="black", lw=0.6, label="AUC")
    for i in range(len(order)):
        ax.text(i - w / 2, acc[i] + 0.008, f"{acc[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, auc[i] + 0.008, f"{auc[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=10); ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Score", fontsize=11); ax.set_title("Performance under feature-leakage ablation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); fig.savefig(f"{OUT}/figS6_feature_leakage.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  figS6_feature_leakage.png")


def figS7_noise_robustness():
    """Figure S7: fragile fraction across noise models and noise levels.
       Reads noise_sensitivity.json and fragility_ablation.json."""
    ns = json.load(open(f"{RES}/noise_sensitivity.json")); fr = json.load(open(f"{RES}/fragility_ablation.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)
    labs = {"independent_gaussian": "Gaussian", "correlated_rho0.5": "Correlated", "asymmetric_split_normal": "Split-normal", "student_t_df3": "Student-t"}
    names = [labs.get(k, k) for k in ns]; frac = [ns[k]["fragile_fraction"] * 100 for k in ns]
    a1.bar(names, frac, color=ORANGE, edgecolor="black", lw=0.6)
    for i, v in enumerate(frac): a1.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    a1.set_ylabel("Fragile fraction (%)"); a1.set_title("Robustness across noise models", fontweight="bold"); a1.tick_params(axis="x", labelsize=9)
    levels = sorted(fr.keys(), key=float); xs = [float(k) for k in levels]; ys = [fr[k] * 100 for k in levels]
    a2.plot(xs, ys, "o-", color=RED, lw=1.8, ms=6)
    for xv, yv in zip(xs, ys): a2.text(xv, yv + 0.5, f"{yv:.1f}%", ha="center", fontsize=9)
    a2.set_xlabel("Relative measurement-noise level"); a2.set_ylabel("Fragile fraction (%)")
    a2.set_title("Fragility vs noise level", fontweight="bold"); a2.grid(alpha=0.25)
    plt.tight_layout(); fig.savefig(f"{OUT}/figS7_noise_robustness.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  figS7_noise_robustness.png")


if __name__ == "__main__":
    print("Regenerating figures into figures/ ...")
    for fn in (fig1_calibration, fig2_physical_model, figS1_roc_confusion, figS2_three_classifiers,
               figS4_measurement_terciles, figS5_cross_mission, figS6_feature_leakage, figS7_noise_robustness):
        try:
            fn()
        except Exception as ex:
            print(f"  [error] {fn.__name__}: {ex}")
    print("Done.")
