"""Repeated nested cross-validation for error-bar-augmented training.

Outer loop: 5-fold stratified CV repeated 5 times (25 independent outer folds),
with a complete retrain in every fold. Inner loop: the augmentation count K is
selected on a validation split carved out of the OUTER TRAINING DATA ONLY, so
the outer fold is touched exactly once, for evaluation.

Selection rule, fixed in advance: choose the K with the lowest inner-validation
fragile fraction among those whose inner AUC is within 0.005 of the K = 0 value.

Confidence intervals: folds within a repeat are dependent, so the interval is
built from the five repeat-level means using Student t with four degrees of
freedom rather than from the 25 folds directly.
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import lightgbm as lgb
import config
from metrics import ece
from measurement_error import _sigma
from error_aware_training import _augment

K_GRID = (0, 2, 4, 8)
AUC_TOL = 0.005
METRICS = ("acc", "auc", "ece", "brier", "fragile_fraction", "mean_flip_rate")


def _params(seed):
    return dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1,
                random_state=seed, deterministic=True, force_col_wise=True, n_jobs=1)


def _flip_rate(m, X, sig, n_mc, seed):
    base = (m.predict_proba(X)[:, 1] >= 0.5).astype(int)
    f = np.zeros(len(X)); rng = np.random.default_rng(seed)
    for _ in range(n_mc):
        f += ((m.predict_proba(X + rng.normal(0, 1, X.shape) * np.nan_to_num(sig))[:, 1] >= 0.5)
              .astype(int) != base)
    return f / n_mc


def _fit(X, y, sig, K, seed):
    Xa, ya = (X, y) if K == 0 else _augment(X, y, sig, K=K, seed=seed)
    return lgb.LGBMClassifier(**_params(seed)).fit(Xa, ya)


def _evaluate(m, X, y, sig, n_mc, seed):
    p = m.predict_proba(X)[:, 1]
    fr = _flip_rate(m, X, sig, n_mc, seed)
    return dict(acc=float(((p >= 0.5).astype(int) == y).mean()),
                auc=float(roc_auc_score(y, p)),
                ece=float(ece(p, y)),
                brier=float(brier_score_loss(y, p)),
                fragile_fraction=float((fr > 0.10).mean()),
                mean_flip_rate=float(fr.mean()))


def _ci(per_repeat):
    """95% interval from the repeat-level means, Student t with n-1 df."""
    a = np.asarray(per_repeat, float); n = len(a)
    if n < 2:
        return [float(a.mean()), float("nan"), float("nan")]
    h = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return [float(a.mean()), float(a.mean() - h), float(a.mean() + h)]


def run(n_repeats=5, n_splits=5, n_mc_inner=100, n_mc_outer=200):
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    F = config.PHYS_FEATURES
    X = df[F].fillna(df[F].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in F]).T

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    base_folds, aug_folds, chosen_K = [], [], []

    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        itr, ival = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=fold)

        inner = {}
        for K in K_GRID:
            m = _fit(X[itr], y[itr], sig[itr], K, seed=fold)
            inner[K] = _evaluate(m, X[ival], y[ival], sig[ival], n_mc_inner, seed=fold)
        ok = [K for K in K_GRID if inner[K]["auc"] >= inner[0]["auc"] - AUC_TOL]
        K_star = min(ok, key=lambda K: inner[K]["fragile_fraction"])
        chosen_K.append(int(K_star))

        base_folds.append(_evaluate(_fit(X[tr], y[tr], sig[tr], 0, seed=fold),
                                    X[te], y[te], sig[te], n_mc_outer, seed=1000 + fold))
        aug_folds.append(_evaluate(_fit(X[tr], y[tr], sig[tr], K_star, seed=fold),
                                   X[te], y[te], sig[te], n_mc_outer, seed=1000 + fold))
        print(f"  fold {fold+1:2d}/{n_repeats*n_splits}  K*={K_star}  "
              f"fragile {base_folds[-1]['fragile_fraction']:.3f} -> "
              f"{aug_folds[-1]['fragile_fraction']:.3f}", flush=True)

    def repeat_means(folds, key):
        v = np.array([f[key] for f in folds]).reshape(n_repeats, n_splits)
        return v.mean(1)

    out = {"protocol": dict(n_splits=n_splits, n_repeats=n_repeats,
                            n_outer_folds=n_repeats * n_splits,
                            K_grid=list(K_GRID), auc_tolerance=AUC_TOL,
                            n_mc_inner=n_mc_inner, n_mc_outer=n_mc_outer,
                            ci="95% Student-t on repeat-level means, 4 df"),
           "selected_K_counts": {str(k): int(chosen_K.count(k)) for k in K_GRID},
           "baseline": {}, "error_aware": {}, "paired_difference": {}}

    for k in METRICS:
        out["baseline"][k] = _ci(repeat_means(base_folds, k))
        out["error_aware"][k] = _ci(repeat_means(aug_folds, k))
        d_fold = np.array([a[k] - b[k] for a, b in zip(aug_folds, base_folds)])
        d_rep = d_fold.reshape(n_repeats, n_splits).mean(1)
        w = stats.wilcoxon(d_fold, alternative="two-sided")
        out["paired_difference"][k] = dict(mean_ci=_ci(d_rep),
                                           wilcoxon_stat=float(w.statistic),
                                           wilcoxon_p=float(w.pvalue))

    json.dump(out, open("../results/nested_cv.json", "w"), indent=1)
    np.savez_compressed("../results/nested_cv_folds.npz",
                        **{f"base_{k}": np.array([f[k] for f in base_folds]) for k in METRICS},
                        **{f"aug_{k}": np.array([f[k] for f in aug_folds]) for k in METRICS},
                        chosen_K=np.array(chosen_K))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
