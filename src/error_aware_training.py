"""Measurement-error-aware training (error-bar augmentation).

Each training example is augmented with K
perturbed copies, every feature displaced by Gaussian noise equal to its
catalogued error bar, teaching the classifier that points within a candidate's
measurement uncertainty share its label. Compares a baseline classifier with
the augmented one on in-distribution metrics, verdict fragility, and zero-shot
TESS transfer.

Usage:  python error_aware_training.py
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import config
import data as D
from metrics import all_metrics
from measurement_error import _sigma


def _augment(X, y, sig, K=4, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(K):
        Xs.append(X + rng.normal(0, 1, X.shape) * np.nan_to_num(sig))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def _fragile(model, X, sig, n_mc=200, thresh=0.10):
    base = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    flips = np.zeros(len(X))
    rng = np.random.default_rng(1)
    for _ in range(n_mc):
        Xp = X + rng.normal(0, 1, X.shape) * np.nan_to_num(sig)
        flips += ((model.predict_proba(Xp)[:, 1] >= 0.5).astype(int) != base)
    fr = flips / n_mc
    return float((fr > thresh).mean()), float(fr.mean())


def run():
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    feats = config.PHYS_FEATURES
    X = df[feats].fillna(df[feats].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in feats]).T

    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)

    base = lgb.LGBMClassifier(**params).fit(X[tr], y[tr])
    Xa, ya = _augment(X[tr], y[tr], sig[tr], K=4)
    aug = lgb.LGBMClassifier(**params).fit(Xa, ya)

    res = {}
    for name, mdl in [("baseline", base), ("error_aware", aug)]:
        md = all_metrics(mdl.predict_proba(X[te])[:, 1], y[te])
        fr, mean_fr = _fragile(mdl, X[te], sig[te])
        res[name] = dict(in_dist=md, fragile_fraction=fr, mean_flip_rate=mean_fr)

    # zero-shot TESS transfer on the shared feature schema
    try:
        Xk, yk = D.load_kepler_shared()
        Xt, yt = D.load_tess_shared()
        for name in ("baseline", "error_aware"):
            mk = lgb.LGBMClassifier(**params)
            if name == "baseline":
                mk.fit(Xk.values, yk)
            else:
                sk = np.full(Xk.shape, np.nan)  # error bars unavailable on shared subset -> 5%
                sk = 0.05 * np.abs(Xk.values)
                Xak, yak = _augment(Xk.values, yk, sk, K=4)
                mk.fit(Xak, yak)
            res[name]["tess"] = all_metrics(mk.predict_proba(Xt.values)[:, 1], yt)
    except Exception as e:  # TESS file optional
        res["tess_note"] = f"skipped ({e})"

    json.dump(res, open("../results/error_aware_training.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    run()
