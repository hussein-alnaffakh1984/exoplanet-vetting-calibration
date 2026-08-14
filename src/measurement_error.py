"""Measurement-error-aware uncertainty: propagate catalogued error bars.

For each
held-out candidate, the catalogued per-feature errors are sampled (Gaussian,
independent) and pushed through the trained classifier; the verdict-flip rate
and a measurement-driven uncertainty follow. A candidate is "fragile" if its
verdict flips in more than 10 percent of draws.

Note: the independent-Gaussian assumption is a first-order approximation; the
DR25 errors are mildly correlated (Thompson et al. 2018).

Usage:  python measurement_error.py
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import config

ERR_SUFFIX = ("_err1", "_err2")  # NASA Exoplanet Archive upper/lower error columns


def _sigma(df, feat):
    """Mean of |err1| and |err2| as a per-row Gaussian sigma; fallback to 5%."""
    e1, e2 = feat + ERR_SUFFIX[0], feat + ERR_SUFFIX[1]
    if e1 in df and e2 in df:
        s = (df[e1].abs().fillna(0) + df[e2].abs().fillna(0)) / 2.0
        s = s.replace(0, np.nan)
        s = s.fillna(0.05 * df[feat].abs())
        return s.values
    return (0.05 * df[feat].abs()).values


def run(n_mc=200, frag_thresh=0.10):
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    feats = config.PHYS_FEATURES
    X = df[feats].fillna(df[feats].median())
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in feats]).T  # (n, n_feat)

    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
    m.fit(X.values[tr], y[tr])

    Xte, site, yte = X.values[te], sig[te], y[te]
    p0 = m.predict_proba(Xte)[:, 1]
    base = (p0 >= 0.5).astype(int)
    flips = np.zeros(len(te))
    draws = np.zeros((len(te), n_mc))
    rng = np.random.default_rng(0)
    for k in range(n_mc):
        Xp = Xte + rng.normal(0, 1, Xte.shape) * np.nan_to_num(site)
        pk = m.predict_proba(Xp)[:, 1]
        draws[:, k] = pk
        flips += ((pk >= 0.5).astype(int) != base)
    flip_rate = flips / n_mc
    unc = draws.std(1)

    # accuracy by tercile of measurement-driven uncertainty
    order = np.argsort(unc)
    terc = np.array_split(order, 3)
    acc_terc = [float((base[t] == yte[t]).mean()) for t in terc]
    ablation = {f"{int(th*100)}%": float((flip_rate > th).mean())
                for th in (0.05, 0.10, 0.15, 0.20)}

    summary = dict(
        mean_flip_rate=float(flip_rate.mean()),
        fragile_fraction=float((flip_rate > frag_thresh).mean()),
        acc_by_uncertainty_tercile=acc_terc,
        fragility_ablation=ablation,
    )
    np.savez_compressed("../results/measurement_error_percandidate.npz",
                        test_index=te, flip_rate=flip_rate, uncertainty=unc,
                        p0=p0, base_pred=base, y_true=yte)
    json.dump(summary, open("../results/measurement_error.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    run()
