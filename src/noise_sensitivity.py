"""Noise-model sensitivity of the fragility measurement.

The independent-Gaussian case reproduces measurement_error.py exactly, by
construction: the same split, the same model, the same seed and the same number
of Monte-Carlo draws. The remaining families test whether the fragility
conclusion depends on that assumption.

Two corrections relative to the first version of this script:

  * the asymmetric case previously drew two independent normal variates, one to
    pick the scale and one to supply the value, which makes the result symmetric
    and therefore not a split-normal at all. It now uses a single draw to set
    both the sign and the scale, and takes the two scales from the catalogued
    upper and lower error bars, so the asymmetry is the catalogue's own;
  * samplers now return the full displacement rather than a multiplier, which is
    what the split-normal family requires.

Usage:  python noise_sensitivity.py
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import config
from measurement_error import _sigma, ERR_SUFFIX


def _sigma_pm(df, feat):
    """Separate upper and lower one-sigma error bars; fallback to 5 percent."""
    e1, e2 = feat + ERR_SUFFIX[0], feat + ERR_SUFFIX[1]
    fb = 0.05 * df[feat].abs()
    if e1 in df and e2 in df:
        up = df[e1].abs().replace(0, np.nan).fillna(fb)
        lo = df[e2].abs().replace(0, np.nan).fillna(fb)
        return up.values, lo.values
    return fb.values, fb.values


def _families(shape, s, up, lo):
    """Every entry returns a full displacement array of the given shape."""
    def corr(rho):
        a, b = np.sqrt(rho), np.sqrt(1.0 - rho)
        return lambda r: (a * r.normal(0, 1, (shape[0], 1)) + b * r.normal(0, 1, shape)) * s

    def split_normal(r):
        z = r.normal(0, 1, shape)
        return np.where(z > 0, up, lo) * z

    def scaled(c):
        return lambda r: r.normal(0, 1, shape) * s * c

    return {
        "independent_gaussian":     lambda r: r.normal(0, 1, shape) * s,
        "correlated_rho0.3":        corr(0.3),
        "correlated_rho0.5":        corr(0.5),
        "correlated_rho0.7":        corr(0.7),
        "asymmetric_split_normal":  split_normal,
        "student_t_df3":            lambda r: r.standard_t(3, shape) / np.sqrt(3) * s,
        "uniform_box":              lambda r: r.uniform(-np.sqrt(3), np.sqrt(3), shape) * s,
    }, {f"{c:.1f}": scaled(c) for c in (0.5, 1.0, 1.5, 2.0)}


def _flip(m, Xte, gen, n_mc=200, thresh=0.10, seed=0):
    base = (m.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    flips = np.zeros(len(Xte))
    rng = np.random.default_rng(seed)
    for _ in range(n_mc):
        flips += ((m.predict_proba(Xte + gen(rng))[:, 1] >= 0.5).astype(int) != base)
    fr = flips / n_mc
    return float(fr.mean()), float((fr > thresh).mean())


def run(n_mc=200):
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    feats = config.PHYS_FEATURES
    X = df[feats].fillna(df[feats].median())
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in feats]).T
    up = np.nan_to_num(np.vstack([_sigma_pm(df, f)[0] for f in feats]).T)
    lo = np.nan_to_num(np.vstack([_sigma_pm(df, f)[1] for f in feats]).T)

    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
    m.fit(X.values[tr], y[tr])

    Xte = X.values[te]
    fams, sweep = _families(Xte.shape, np.nan_to_num(sig[te]), up[te], lo[te])

    out = {"families": {}, "scale_sweep": {}}
    for name, gen in fams.items():
        mfr, ff = _flip(m, Xte, gen, n_mc)
        out["families"][name] = {"mean_flip_rate": round(mfr, 4), "fragile_fraction": round(ff, 4)}
    for name, gen in sweep.items():
        mfr, ff = _flip(m, Xte, gen, n_mc)
        out["scale_sweep"][name] = {"mean_flip_rate": round(mfr, 4), "fragile_fraction": round(ff, 4)}

    json.dump(out, open("../results/noise_sensitivity.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
