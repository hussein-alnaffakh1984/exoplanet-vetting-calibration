"""Cross-mission validation: Kepler -> TESS.

Reproduces Table 8 / Figure 6. Trains on the six shared physical features using
Kepler KOIs only, evaluates in-distribution (Kepler hold-out) and zero-shot on
TESS, then recalibrates on a small TESS subset to separate the (repairable)
calibration failure from the (non-repairable) discrimination loss under the
distribution shift.

Usage:  python cross_mission.py
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

import data as D
from metrics import all_metrics


def run():
    Xk, yk = D.load_kepler_shared()
    Xt, yt = D.load_tess_shared()
    ktr, kte = train_test_split(np.arange(len(Xk)), test_size=0.2, stratify=yk, random_state=0)

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
    m.fit(Xk.values[ktr], yk[ktr])

    kepler_holdout = all_metrics(m.predict_proba(Xk.values[kte])[:, 1], yk[kte])
    pt = m.predict_proba(Xt.values)[:, 1]
    tess_zero = all_metrics(pt, yt)

    # recalibrate on a small TESS subset
    cal, ev = train_test_split(np.arange(len(Xt)), test_size=0.7, stratify=yt, random_state=0)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pt[cal], yt[cal])
    tess_recal = all_metrics(np.clip(iso.predict(pt[ev]), 0, 1), yt[ev])

    summary = dict(kepler_holdout=kepler_holdout, tess_zero_shot=tess_zero,
                   tess_recalibrated=tess_recal)
    json.dump(summary, open("../results/cross_mission.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    run()
