"""Three probabilistic classifiers on the physical features: LightGBM, NGBoost,
and the Monte-Carlo-dropout Bayesian network. Five-fold cross-validation,
random_state 0. The Bayesian network reuses the architecture in flux_calibration.

Usage:  python physical_three_models.py
"""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
import config, data as D
from flux_calibration import _bnn


def _score(p, y):
    return float(((p >= 0.5).astype(int) == y).mean()), float(roc_auc_score(y, p))


def run():
    X, y = D.load_physical(); Xv = X.values.astype(np.float32)
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    res = {k: {"acc": [], "auc": []} for k in ("LightGBM", "NGBoost", "BNN")}
    for tr, te in skf.split(Xv, y):
        # LightGBM
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
        m.fit(Xv[tr], y[tr]); a, u = _score(m.predict_proba(Xv[te])[:, 1], y[te])
        res["LightGBM"]["acc"].append(a); res["LightGBM"]["auc"].append(u)
        # NGBoost
        ng = NGBClassifier(Dist=Bernoulli, n_estimators=400, learning_rate=0.03, verbose=False, random_state=0)
        ng.fit(Xv[tr], y[tr]); pn = ng.predict_proba(Xv[te])[:, 1]; a, u = _score(pn, y[te])
        res["NGBoost"]["acc"].append(a); res["NGBoost"]["auc"].append(u)
        # BNN (MC-dropout), reuse flux architecture
        pb = _bnn(Xv[tr], y[tr], Xv[tr], Xv[te])[0]
        a, u = _score(pb, y[te]); res["BNN"]["acc"].append(a); res["BNN"]["auc"].append(u)
    out = {k: {"acc": [float(np.mean(v["acc"])), float(np.std(v["acc"]))],
               "auc": [float(np.mean(v["auc"])), float(np.std(v["auc"]))]} for k, v in res.items()}
    json.dump(out, open("../results/physical_three_models.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
