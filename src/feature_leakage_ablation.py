"""Feature-leakage ablation: physical model on progressively reduced feature
sets, to show performance is not driven by label-proximal features.

Reuses the exact five-fold pipeline of physical_model.py (LightGBM, random_state
0, inner calibration split per fold) on three feature subsets.

Usage:  python feature_leakage_ablation.py
"""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pandas as pd
import config

SUBSETS = {
    "full_7feat": config.PHYS_FEATURES,
    "drop_prad_only": [f for f in config.PHYS_FEATURES if f != "koi_prad"],
    "drop_prad_snr_impact": [f for f in config.PHYS_FEATURES
                             if f not in ("koi_prad", "koi_model_snr", "koi_impact")],
}


def _cv(X, y):
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        itr, _ical = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=0)
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
        m.fit(X[itr], y[itr])
        p = m.predict_proba(X[te])[:, 1]
        accs.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
        aucs.append(float(roc_auc_score(y[te], p)))
    return accs, aucs


def run():
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    out = {}
    for name, feats in SUBSETS.items():
        X = df[feats].fillna(df[feats].median()).values
        accs, aucs = _cv(X, y)
        out[name] = {"features": feats,
                     "acc": [float(np.mean(accs)), float(np.std(accs))],
                     "auc": [float(np.mean(aucs)), float(np.std(aucs))]}
    json.dump(out, open("../results/feature_leakage_ablation.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
