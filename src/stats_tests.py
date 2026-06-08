"""Statistical significance and robustness tests.

Reproduces Table 9: McNemar (LightGBM vs NGBoost classifications), DeLong (AUC
difference), and ECE robustness across equal-width and adaptive bin counts.
Operates on a common physical-feature test set.

Usage:  python stats_tests.py
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import config
import data as D
from metrics import ece, ece_adaptive, mcnemar, delong_auc_test


def run():
    X, y = D.load_physical()
    Xv = X.values
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=0)

    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
    lgbm.fit(Xv[tr], y[tr])
    p_lgb = lgbm.predict_proba(Xv[te])[:, 1]

    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    ng = NGBClassifier(Dist=Bernoulli, n_estimators=400, learning_rate=0.03, verbose=False)
    ng.fit(Xv[tr], y[tr])
    p_ng = ng.predict_proba(Xv[te])[:, 1]

    chi2, p_mc, n01, n10 = mcnemar(y[te], (p_lgb >= 0.5).astype(int), (p_ng >= 0.5).astype(int))
    auc_a, auc_b, z, p_dl = delong_auc_test(y[te], p_lgb, p_ng)

    summary = dict(
        mcnemar=dict(chi2=chi2, p=p_mc, n01=n01, n10=n10),
        delong=dict(auc_lgb=auc_a, auc_ng=auc_b, z=z, p=p_dl),
        ece_equal_width={str(b): ece(p_lgb, y[te], b) for b in (10, 15, 20)},
        ece_adaptive={str(b): ece_adaptive(p_lgb, y[te], b) for b in (10, 15, 20)},
    )
    json.dump(summary, open("../results/stats_tests.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    run()
