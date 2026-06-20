"""Fragility ablation: fraction of held-out candidates whose verdict-flip rate
exceeds a range of thresholds (0.05 to 0.20). Uses the same Gaussian
error-propagation as measurement_error.py.

Usage:  python fragility_ablation.py
"""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import config
from measurement_error import _sigma


def run(n_mc=200):
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    feats = config.PHYS_FEATURES
    X = df[feats].fillna(df[feats].median()); y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in feats]).T
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
    m.fit(X.values[tr], y[tr])
    Xte, site = X.values[te], sig[te]
    base = (m.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    flips = np.zeros(len(te)); rng = np.random.default_rng(0)
    for _ in range(n_mc):
        Xp = Xte + rng.normal(0, 1, Xte.shape) * np.nan_to_num(site)
        flips += ((m.predict_proba(Xp)[:, 1] >= 0.5).astype(int) != base)
    fr = flips / n_mc
    out = {f"{th:.2f}": float((fr > th).mean()) for th in (0.05, 0.10, 0.15, 0.20)}
    json.dump(out, open("../results/fragility_ablation.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
