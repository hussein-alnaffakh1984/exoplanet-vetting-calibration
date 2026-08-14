"""Augmentation on top of TabM, with a clean object-level validation split.

TabM carves an internal validation set from its training data. If the augmented
pool is passed directly, perturbed copies of the same object land on both sides
of that split, which corrupts early stopping and calibration. Here the inner
validation objects are held out FIRST, are never augmented, and are passed
explicitly, so baseline and augmented models see identical validation data and
differ only in their training set.
"""
import json, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import pytabkit, config
from metrics import ece
from measurement_error import _sigma
from error_aware_training import _augment

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, N_REPEATS, N_MC = 4, 8, 200


def ci(a):
    a = np.asarray(a, float)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return [float(a.mean()), float(a.mean() - h), float(a.mean() + h)]


def flips(m, X, sig, seed):
    base = (m.predict_proba(X)[:, 1] >= 0.5).astype(int)
    f = np.zeros(len(X)); r = np.random.default_rng(seed)
    for _ in range(N_MC):
        f += ((m.predict_proba(X + r.normal(0, 1, X.shape) * np.nan_to_num(sig))[:, 1] >= 0.5)
              .astype(int) != base)
    return f / N_MC


def run():
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    F = config.PHYS_FEATURES
    X = df[F].fillna(df[F].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in F]).T

    b_all, a_all = [], []
    for rep in range(N_REPEATS):
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=rep)
        fit_i, val_i = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=rep)
        Xv, yv = X[val_i], y[val_i]                       # clean, never augmented
        Xa, ya = _augment(X[fit_i], y[fit_i], sig[fit_i], K=K, seed=rep)
        models = {
            "base": pytabkit.TabM_D_Classifier(device=DEV, random_state=rep, verbosity=0)
                    .fit(X[fit_i], y[fit_i], X_val=Xv, y_val=yv),
            "aug":  pytabkit.TabM_D_Classifier(device=DEV, random_state=rep, verbosity=0)
                    .fit(Xa, ya, X_val=Xv, y_val=yv),
        }
        for tag, m in models.items():
            p = m.predict_proba(X[te])[:, 1]
            fr = flips(m, X[te], sig[te], 1000 + rep)
            rec = dict(acc=float(((p >= .5).astype(int) == y[te]).mean()),
                       auc=float(roc_auc_score(y[te], p)), ece=float(ece(p, y[te])),
                       brier=float(brier_score_loss(y[te], p)),
                       fragile_fraction=float((fr > 0.10).mean()),
                       mean_flip_rate=float(fr.mean()))
            (b_all if tag == "base" else a_all).append(rec)
        print(f"  repeat {rep+1}/{N_REPEATS}  ece {b_all[-1]['ece']:.4f} -> {a_all[-1]['ece']:.4f}"
              f"   fragile {b_all[-1]['fragile_fraction']:.3f} -> "
              f"{a_all[-1]['fragile_fraction']:.3f}", flush=True)

    M = ("acc", "auc", "ece", "brier", "fragile_fraction", "mean_flip_rate")
    out = {"method": "TabM (ICLR 2025) with a clean unaugmented inner validation split",
           "K": K, "n_repeats": N_REPEATS, "n_mc": N_MC,
           "baseline": {q: ci([r[q] for r in b_all]) for q in M},
           "error_aware": {q: ci([r[q] for r in a_all]) for q in M},
           "paired_difference": {}}
    for q in M:
        d = np.array([a[q] - b[q] for a, b in zip(a_all, b_all)])
        out["paired_difference"][q] = dict(mean_ci=ci(d),
                                           wilcoxon_p=float(stats.wilcoxon(d).pvalue))
    json.dump(out, open("../results/recent_method_aug.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    r = run()
    print()
    for q in ("acc", "auc", "ece", "brier", "fragile_fraction", "mean_flip_rate"):
        b, a, p = r["baseline"][q], r["error_aware"][q], r["paired_difference"][q]
        print(f"  {q:18s} {b[0]:.4f} -> {a[0]:.4f}   diff {p['mean_ci'][0]:+.4f} "
              f"[{p['mean_ci'][1]:+.4f}, {p['mean_ci'][2]:+.4f}]  p {p['wilcoxon_p']:.4f}")
