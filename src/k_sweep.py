"""Augmentation-count sweep with a unified Monte-Carlo seed.

Runs K in {0, 2, 4, 8} on the single held-out split used throughout the paper
and stores, in addition to the summary metrics, the per-object fragility flags
so that the K = 0 against K = 4 comparison has a released artifact. The
Monte-Carlo seed is the same one used by measurement_error.py, so the K = 0 row
reproduces the fragile fraction reported there exactly rather than to within
sampling noise.

Usage:  python k_sweep.py
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy import stats

import config
from metrics import ece, mcnemar
from measurement_error import _sigma
from error_aware_training import _augment

K_GRID = (0, 2, 4, 8)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)


def _flip_rate(model, X, sig, n_mc=200, seed=0):
    base = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    flips = np.zeros(len(X))
    rng = np.random.default_rng(seed)
    for _ in range(n_mc):
        flips += ((model.predict_proba(X + rng.normal(0, 1, X.shape) * np.nan_to_num(sig))[:, 1]
                   >= 0.5).astype(int) != base)
    return flips / n_mc


def run(n_mc=200, thresh=0.10):
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    feats = config.PHYS_FEATURES
    X = df[feats].fillna(df[feats].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in feats]).T

    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)

    out, flags = {}, {}
    for K in K_GRID:
        if K == 0:
            Xa, ya = X[tr], y[tr]
        else:
            Xa, ya = _augment(X[tr], y[tr], sig[tr], K=K, seed=0)
        m = lgb.LGBMClassifier(**PARAMS).fit(Xa, ya)
        p = m.predict_proba(X[te])[:, 1]
        fr = _flip_rate(m, X[te], sig[te], n_mc=n_mc, seed=0)
        flags[f"K{K}"] = (fr > thresh)
        out[str(K)] = dict(
            acc=float(((p >= 0.5).astype(int) == y[te]).mean()),
            auc=float(roc_auc_score(y[te], p)),
            ece=float(ece(p, y[te])),
            fragile_fraction=float((fr > thresh).mean()),
            mean_flip_rate=float(fr.mean()),
        )
        print(f"  K={K}: fragile {out[str(K)]['fragile_fraction']:.4f}", flush=True)

    # does the augmentation fix more objects than it breaks?
    a, b = flags["K0"].astype(int), flags["K4"].astype(int)
    fixed = int(np.sum((a == 1) & (b == 0)))
    broke = int(np.sum((a == 0) & (b == 1)))
    chi2 = (abs(fixed - broke) - 1) ** 2 / (fixed + broke) if (fixed + broke) else 0.0
    out["fragility_mcnemar_K0_vs_K4"] = dict(
        fixed_by_augmentation=fixed, newly_fragile=broke,
        chi2=float(chi2), p=float(stats.chi2.sf(chi2, df=1)), n=int(len(te)))

    np.savez_compressed("../results/k_sweep_fragile_flags.npz",
                        test_index=te, **{k: v for k, v in flags.items()})
    json.dump(out, open("../results/k_sweep.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
