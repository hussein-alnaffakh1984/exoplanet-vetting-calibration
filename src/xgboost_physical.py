"""XGBoost baseline on the physical features (five-fold CV with isotonic
calibration), reported as an alternative gradient-boosted-trees learner.

Mirrors physical_model.py exactly, swapping LightGBM for XGBoost.

Usage:  python xgboost_physical.py
"""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import config
import data as D
from metrics import ece


def run():
    X, y = D.load_physical(); Xv = X.values
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    accs, aucs, eces = [], [], []
    oof_p, oof_p_iso = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in skf.split(Xv, y):
        itr, ical = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=0)
        m = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                          subsample=1.0, colsample_bytree=1.0, eval_metric="logloss",
                          random_state=0, verbosity=0)
        m.fit(Xv[itr], y[itr])
        p = m.predict_proba(Xv[te])[:, 1]
        accs.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
        aucs.append(float(roc_auc_score(y[te], p)))
        eces.append(ece(p, y[te]))
        oof_p[te] = p
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xv[ical])[:, 1], y[ical])
        oof_p_iso[te] = np.clip(iso.predict(p), 0, 1)
    summary = dict(model="XGBoost", n=int(len(y)),
                   acc=[float(np.mean(accs)), float(np.std(accs))],
                   auc=[float(np.mean(aucs)), float(np.std(aucs))],
                   ece_fold=[float(np.mean(eces)), float(np.std(eces))],
                   ece_oof_raw=ece(oof_p, y), ece_oof_iso=ece(oof_p_iso, y))
    json.dump(summary, open("../results/xgboost_physical.json", "w"), indent=1)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
