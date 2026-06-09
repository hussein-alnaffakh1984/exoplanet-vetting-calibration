"""Matched-protocol representation comparison and fragile-fraction significance test.

Re-evaluates the physical model under the SAME 3-repeat 70/15/15 holdout
protocol as the flux models and on a random subsample matched to the flux
sample size (n = 5,302), so the two representations can be compared under
matched resampling and sample size (objects still cannot be matched one-to-one
because the public flux matrix carries no identifier).

Tests whether the fragile-fraction reduction from error-bar augmentation
is statistically significant, using a paired McNemar test on per-candidate
fragility flags plus a Fisher exact test on the two-by-two counts.

Usage:  python matched_comparison.py
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from scipy import stats
import lightgbm as lgb

import config
from metrics import all_metrics
from measurement_error import _sigma
from error_aware_training import _augment, _fragile


def matched_physical(X, y, params, n_match=5302, seeds=(0, 1, 2)):
    rng = np.random.default_rng(42)
    sub = rng.choice(len(X), n_match, replace=False)
    Xs, ys = X[sub], y[sub]
    acc, auc, ece = [], [], []
    for s in seeds:
        itr, itmp = train_test_split(np.arange(len(Xs)), test_size=0.30, stratify=ys, random_state=s)
        _, ite = train_test_split(itmp, test_size=0.50, stratify=ys[itmp], random_state=s)
        m = lgb.LGBMClassifier(**params).fit(Xs[itr], ys[itr])
        md = all_metrics(m.predict_proba(Xs[ite])[:, 1], ys[ite])
        acc.append(md["acc"]); auc.append(md["auc"]); ece.append(md["ece"])
    return dict(n=n_match, protocol="3-repeat 70/15/15 holdout (matched to flux)",
                acc=[float(np.mean(acc)), float(np.std(acc))],
                auc=[float(np.mean(auc)), float(np.std(auc))],
                ece=[float(np.mean(ece)), float(np.std(ece))])


def fragile_significance(X, y, sig, params):
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)

    def flags(model):
        base = (model.predict_proba(X[te])[:, 1] >= 0.5).astype(int)
        flips = np.zeros(len(te)); rng = np.random.default_rng(1)
        for _ in range(200):
            Xp = X[te] + rng.normal(0, 1, X[te].shape) * np.nan_to_num(sig[te])
            flips += ((model.predict_proba(Xp)[:, 1] >= 0.5).astype(int) != base)
        return ((flips / 200) > 0.10).astype(int)

    base = lgb.LGBMClassifier(**params).fit(X[tr], y[tr])
    Xa, ya = _augment(X[tr], y[tr], sig[tr], K=4)
    aug = lgb.LGBMClassifier(**params).fit(Xa, ya)
    fb, fa = flags(base), flags(aug)
    n10 = int(np.sum((fb == 1) & (fa == 0)))  # fixed by augmentation
    n01 = int(np.sum((fb == 0) & (fa == 1)))  # newly fragile
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) > 0 else 0.0
    ct = [[int(fb.sum()), int(len(fb) - fb.sum())], [int(fa.sum()), int(len(fa) - fa.sum())]]
    _, p_fisher = stats.fisher_exact(ct)
    return dict(frac_baseline=float(fb.mean()), frac_erraware=float(fa.mean()),
                fixed_by_aug=n10, newly_fragile=n01,
                mcnemar_chi2=float(chi2), mcnemar_p=float(stats.chi2.sf(chi2, 1)),
                fisher_p=float(p_fisher))


def run():
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    feats = config.PHYS_FEATURES
    X = df[feats].fillna(df[feats].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in feats]).T
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)

    out = dict(matched_physical=matched_physical(X, y, params),
               fragile_significance=fragile_significance(X, y, sig, params))
    json.dump(out, open("../results/matched_and_significance.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run()
