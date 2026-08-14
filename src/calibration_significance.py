"""Paired significance of the single isotonic calibration step, flux features.

Section 5.1 quotes a McNemar test comparing raw against isotonically calibrated
hard predictions. That test is computed on ONE representative split, seed 0 of
the three flux repeats, and not on the pooled three-repeat evaluation behind the
Table 2 means; the two are different quantities and the paper now says so. This
script exists so that the quoted numbers have a released artifact.

Usage:  python calibration_significance.py
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

import config
import data as D
from metrics import mcnemar
from flux_calibration import _split, _bnn


def run(seed=0):
    X, Xds, y = D.load_flux()
    tr, cal, te = _split(len(X), y, seed)

    res = {"protocol": dict(representation="flux", seed=seed, n_test=int(len(te)),
                            note="one representative split, not the pooled three-repeat "
                                 "evaluation behind the Table 2 means")}

    # LightGBM, raw against isotonic
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
    m.fit(X[tr], y[tr])
    pte, pcal = m.predict_proba(X[te])[:, 1], m.predict_proba(X[cal])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(pcal, y[cal])
    pte_iso = np.clip(iso.predict(pte), 0, 1)
    chi2, p, n01, n10 = mcnemar(y[te], (pte >= 0.5).astype(int), (pte_iso >= 0.5).astype(int))
    res["LightGBM"] = dict(acc_raw=float(((pte >= 0.5).astype(int) == y[te]).mean()),
                           acc_isotonic=float(((pte_iso >= 0.5).astype(int) == y[te]).mean()),
                           n01=n01, n10=n10, chi2=chi2, p=p)

    # Bayesian network, raw against isotonic
    pb, pb_cal = _bnn(X[tr], y[tr], X[cal], X[te])
    isob = IsotonicRegression(out_of_bounds="clip").fit(pb_cal, y[cal])
    pb_iso = np.clip(isob.predict(pb), 0, 1)
    chi2, p, n01, n10 = mcnemar(y[te], (pb >= 0.5).astype(int), (pb_iso >= 0.5).astype(int))
    res["BayesianNetwork"] = dict(acc_raw=float(((pb >= 0.5).astype(int) == y[te]).mean()),
                                  acc_isotonic=float(((pb_iso >= 0.5).astype(int) == y[te]).mean()),
                                  n01=n01, n10=n10, chi2=chi2, p=p)

    json.dump(res, open("../results/calibration_significance.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
