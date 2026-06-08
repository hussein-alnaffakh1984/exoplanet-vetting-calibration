"""Physical-parameter model: 5-fold cross-validated calibration + SHAP.

Reproduces Table 3, Table 4 (confusion), Figure 2 and the SHAP ranking in
Appendix B. Trains a gradient-boosted classifier on the seven physical transit
features, with an inner calibration split per fold, and saves out-of-fold
predictions for downstream analyses.

Usage:  python physical_model.py
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import lightgbm as lgb

import config
import data as D
from metrics import all_metrics, ece


def run():
    X, y = D.load_physical()
    Xv = X.values
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    fold_metrics, oof_p, oof_p_iso, oof_idx = [], np.zeros(len(y)), np.zeros(len(y)), []
    shap_abs = np.zeros(X.shape[1])

    for tr, te in skf.split(Xv, y):
        itr, ical = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=0)
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1)
        m.fit(Xv[itr], y[itr])
        pte = m.predict_proba(Xv[te])[:, 1]
        fold_metrics.append(all_metrics(pte, y[te]))
        oof_p[te] = pte
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xv[ical])[:, 1], y[ical])
        oof_p_iso[te] = np.clip(iso.predict(pte), 0, 1)
        oof_idx.extend(te.tolist())
        # SHAP via LightGBM native predict_contrib (booster API)
        contrib = m.booster_.predict(Xv[te], pred_contrib=True)[:, :-1]
        shap_abs += np.abs(contrib).mean(0)

    shap_abs /= config.N_FOLDS
    pred = (oof_p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    summary = dict(
        cv={k: [float(np.mean([f[k] for f in fold_metrics])),
                float(np.std([f[k] for f in fold_metrics]))] for k in fold_metrics[0]},
        f1=float(f1_score(y, pred)),
        precision=float(precision_score(y, pred)),
        recall=float(recall_score(y, pred)),
        ece_oof_raw=ece(oof_p, y),
        ece_oof_iso=ece(oof_p_iso, y),
        confusion=cm.tolist(),
        shap={f: float(v) for f, v in sorted(zip(config.PHYS_FEATURES, shap_abs),
                                             key=lambda t: -t[1])},
    )
    np.savez_compressed("../results/phys_oof.npz", oof_p=oof_p, oof_p_iso=oof_p_iso, y=y)
    json.dump(summary, open("../results/physical_model.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    run()
