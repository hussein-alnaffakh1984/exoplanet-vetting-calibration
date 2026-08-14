"""Operational feature tiers.

Tiers are defined by WHEN a quantity becomes available in the transit pipeline,
not by how predictive it is:

  Tier 0  period, duration, depth                available from the detection itself
  Tier 1  + equilibrium temperature              requires stellar parameters
  Tier 2  + radius, impact parameter, model SNR  transit-fit outputs, label-proximal

For each tier the baseline and the error-bar-augmented model are both retrained
over five independent repeats, so the question is not only whether accuracy
survives the restriction but whether calibration and the augmentation benefit do.
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import lightgbm as lgb
import config
from metrics import ece
from measurement_error import _sigma
from error_aware_training import _augment

TIERS = {
    "tier0_detection_only": ["koi_period", "koi_duration", "koi_depth"],
    "tier1_plus_stellar":   ["koi_period", "koi_duration", "koi_depth", "koi_teq"],
    "tier2_full":           config.PHYS_FEATURES,
}
K, N_REPEATS, N_MC = 4, 5, 200
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1,
           random_state=0, deterministic=True, force_col_wise=True, n_jobs=1)


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
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    out = {"base_rate": float(y.mean()), "K": K, "n_repeats": N_REPEATS, "tiers": {}}

    for tier, feats in TIERS.items():
        Xall = df[feats].fillna(df[feats].median()).values
        sig = np.vstack([_sigma(df, f) for f in feats]).T
        b_all, a_all = [], []
        for rep in range(N_REPEATS):
            tr, te = train_test_split(np.arange(len(Xall)), test_size=0.2,
                                      stratify=y, random_state=rep)
            mb = lgb.LGBMClassifier(**{**LGB, "random_state": rep}).fit(Xall[tr], y[tr])
            Xa, ya = _augment(Xall[tr], y[tr], sig[tr], K=K, seed=rep)
            ma = lgb.LGBMClassifier(**{**LGB, "random_state": rep}).fit(Xa, ya)
            for m, store in ((mb, b_all), (ma, a_all)):
                p = m.predict_proba(Xall[te])[:, 1]
                fr = flips(m, Xall[te], sig[te], 1000 + rep)
                store.append(dict(acc=float(((p >= .5).astype(int) == y[te]).mean()),
                                  auc=float(roc_auc_score(y[te], p)),
                                  ece=float(ece(p, y[te])),
                                  brier=float(brier_score_loss(y[te], p)),
                                  fragile_fraction=float((fr > 0.10).mean()),
                                  mean_flip_rate=float(fr.mean())))
        M = ("acc", "auc", "ece", "brier", "fragile_fraction", "mean_flip_rate")
        d = {"n_features": len(feats), "features": feats,
             "baseline": {q: ci([r[q] for r in b_all]) for q in M},
             "error_aware": {q: ci([r[q] for r in a_all]) for q in M},
             "paired_difference": {}}
        for q in M:
            diff = np.array([a[q] - b[q] for a, b in zip(a_all, b_all)])
            d["paired_difference"][q] = dict(mean_ci=ci(diff),
                                             wilcoxon_p=float(stats.wilcoxon(diff).pvalue))
        out["tiers"][tier] = d
        bf, af = d["baseline"]["fragile_fraction"][0], d["error_aware"]["fragile_fraction"][0]
        print(f"  {tier:22s} n={len(feats)}  acc {d['baseline']['acc'][0]:.3f}"
              f"  auc {d['baseline']['auc'][0]:.3f}  ece {d['baseline']['ece'][0]:.3f}"
              f"  fragile {bf:.3f} -> {af:.3f}  ({100*(bf-af)/bf:.1f}% reduction)", flush=True)

    json.dump(out, open("../results/feature_tiers.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    r = run()
    print(f"\n  base rate (majority-class floor): {1-r['base_rate']:.3f}")
