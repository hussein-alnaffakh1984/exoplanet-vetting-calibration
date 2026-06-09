"""Exploratory analysis of the unresolved CANDIDATE population.

Addresses the operational-relevance gap: the main study trains and evaluates on
resolved CONFIRMED / FALSE POSITIVE objects, but follow-up prioritisation
applies to the unresolved CANDIDATE objects. Here the physical-parameter model
is trained on the resolved set and applied to the CANDIDATE objects, reporting
where they fall on the probability axis and how many are observationally
fragile under measurement-error propagation.

Usage:  python candidate_analysis.py
"""
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb

import config
from measurement_error import _sigma


def run(n_mc=200, frag_thresh=0.10):
    df = pd.read_csv(config.KOI_DR25_CSV)
    feats = config.PHYS_FEATURES
    med = df[feats].median()

    res = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    Xr = res[feats].fillna(med).values
    yr = (res["koi_disposition"] == config.POS_LABEL).astype(int).values

    cand = df[df["koi_disposition"] == "CANDIDATE"].reset_index(drop=True)
    Xc = cand[feats].fillna(med).values
    sigc = np.vstack([_sigma(cand, f) for f in feats]).T

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
    m.fit(Xr, yr)
    pc = m.predict_proba(Xc)[:, 1]

    # measurement-error fragility on candidates
    base = (pc >= 0.5).astype(int)
    flips = np.zeros(len(Xc))
    rng = np.random.default_rng(0)
    for _ in range(n_mc):
        Xp = Xc + rng.normal(0, 1, Xc.shape) * np.nan_to_num(sigc)
        flips += ((m.predict_proba(Xp)[:, 1] >= 0.5).astype(int) != base)
    flip_rate = flips / n_mc

    summary = dict(
        n_candidates=int(len(Xc)),
        prob_mean=float(pc.mean()),
        prob_median=float(np.median(pc)),
        frac_planet_like=float((pc >= 0.5).mean()),
        frac_high_conf_planet=float((pc >= 0.9).mean()),
        frac_high_conf_fp=float((pc <= 0.1).mean()),
        frac_uncertain_band=float(((pc > 0.4) & (pc < 0.6)).mean()),
        fragile_fraction=float((flip_rate > frag_thresh).mean()),
        mean_flip_rate=float(flip_rate.mean()),
    )
    json.dump(summary, open("../results/candidate_analysis.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    run()
