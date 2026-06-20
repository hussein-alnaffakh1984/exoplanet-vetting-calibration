"""Noise-model sensitivity: verdict-flip rate and fragile fraction under four
measurement-noise models. The independent-Gaussian case reproduces
measurement_error.py exactly; the other three test robustness of the fragility
conclusion to the noise assumption.

Usage:  python noise_sensitivity.py
"""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import config
from measurement_error import _sigma


def _flip(m, Xte, sig, sampler, n_mc=200, thresh=0.10):
    base = (m.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    flips = np.zeros(len(Xte)); rng = np.random.default_rng(0)
    for _ in range(n_mc):
        Xp = Xte + sampler(rng, Xte.shape) * np.nan_to_num(sig)
        flips += ((m.predict_proba(Xp)[:, 1] >= 0.5).astype(int) != base)
    fr = flips / n_mc
    return float(fr.mean()), float((fr > thresh).mean())


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
    samplers = {
        "independent_gaussian": lambda r, sh: r.normal(0, 1, sh),
        "correlated_rho0.5": lambda r, sh: 0.707 * r.normal(0, 1, (sh[0], 1)) + 0.707 * r.normal(0, 1, sh),
        "asymmetric_split_normal": lambda r, sh: np.where(r.normal(0, 1, sh) > 0, 1.0, 0.7) * r.normal(0, 1, sh),
        "student_t_df3": lambda r, sh: r.standard_t(3, sh) / np.sqrt(3),
    }
    out = {}
    for name, samp in samplers.items():
        mfr, ff = _flip(m, Xte, site, samp, n_mc)
        out[name] = {"mean_flip_rate": round(mfr, 4), "fragile_fraction": round(ff, 4)}
    json.dump(out, open("../results/noise_sensitivity.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
