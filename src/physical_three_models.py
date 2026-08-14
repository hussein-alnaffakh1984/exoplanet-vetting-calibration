"""Three probabilistic classifiers on the physical features: LightGBM, NGBoost,
and the Monte-Carlo-dropout Bayesian network. Five-fold cross-validation,
random_state 0. The Bayesian network reuses the architecture in flux_calibration.

Usage:  python physical_three_models.py
"""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
import config, data as D
from metrics import ece
from flux_calibration import _bnn


MET = ("acc", "auc", "brier", "ece")


def _score(p, y):
    """Accuracy, AUC, Brier and ECE, so Table S2 is fully reproducible."""
    return (float(((p >= 0.5).astype(int) == y).mean()), float(roc_auc_score(y, p)),
            float(brier_score_loss(y, p)), float(ece(p, y)))


def run():
    X, y = D.load_physical(); Xv = X.values.astype(np.float32)
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    res = {k: {m: [] for m in MET} for k in ("LightGBM", "NGBoost", "BNN")}
    bnn_p = np.zeros(len(y)); bnn_s = np.zeros(len(y))
    for tr, te in skf.split(Xv, y):
        # LightGBM
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
        m.fit(Xv[tr], y[tr])
        for k, v in zip(MET, _score(m.predict_proba(Xv[te])[:, 1], y[te])):
            res["LightGBM"][k].append(v)
        # NGBoost
        ng = NGBClassifier(Dist=Bernoulli, n_estimators=400, learning_rate=0.03, verbose=False, random_state=0)
        ng.fit(Xv[tr], y[tr]); pn = ng.predict_proba(Xv[te])[:, 1]
        for k, v in zip(MET, _score(pn, y[te])):
            res["NGBoost"][k].append(v)
        # BNN (MC-dropout), reuse flux architecture
        pb, sb, _ = _bnn(Xv[tr], y[tr], Xv[tr], Xv[te], with_std=True)
        bnn_p[te] = pb; bnn_s[te] = sb
        for k, v in zip(MET, _score(pb, y[te])):
            res["BNN"][k].append(v)
    out = {k: {m: [float(np.mean(v[m])), float(np.std(v[m]))] for m in MET}
           for k, v in res.items()}
    np.savez_compressed("../results/phys_bnn_oof.npz", p=bnn_p, std=bnn_s, y=y)
    json.dump(out, open("../results/physical_three_models.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
