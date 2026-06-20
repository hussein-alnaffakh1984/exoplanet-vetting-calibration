"""Probability-quality and significance metrics used throughout the study.

All functions operate on 1-D numpy arrays of predicted positive-class
probabilities (p) and binary ground-truth labels (y in {0, 1}).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from scipy import stats


def ece(p, y, n_bins=10):
    """Expected Calibration Error (equal-width binning)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        hi = edges[i + 1] if i < n_bins - 1 else 1.01
        m = (p >= edges[i]) & (p < hi)
        if m.sum() > 0:
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def ece_adaptive(p, y, n_bins=10):
    """Adaptive (equal-count) ECE."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    order = np.argsort(p)
    p, y = p[order], y[order]
    e = 0.0
    n = len(p)
    for chunk in np.array_split(np.arange(n), n_bins):
        if len(chunk) == 0:
            continue
        e += (len(chunk) / n) * abs(y[chunk].mean() - p[chunk].mean())
    return float(e)


def reliability_curve(p, y, n_bins=10):
    """Return (mean_confidence, observed_accuracy, count) per bin for plotting."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        hi = edges[i + 1] if i < n_bins - 1 else 1.01
        m = (p >= edges[i]) & (p < hi)
        if m.sum() > 0:
            conf.append(p[m].mean())
            acc.append(y[m].mean())
            cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def all_metrics(p, y, thresh=0.5):
    """Standard bundle: accuracy, AUC, Brier, ECE."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    return dict(
        acc=float(accuracy_score(y, (p >= thresh).astype(int))),
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=ece(p, y),
    )


def mcnemar(y_true, pred_a, pred_b):
    """McNemar test comparing two classifiers' hard predictions.

    Returns (chi2, p_value, n01, n10) where n01 = A wrong & B right, etc.
    """
    y_true = np.asarray(y_true, int)
    a = (np.asarray(pred_a) == y_true)
    b = (np.asarray(pred_b) == y_true)
    n01 = int(np.sum(~a & b))   # A wrong, B right
    n10 = int(np.sum(a & ~b))   # A right, B wrong
    # continuity-corrected McNemar
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    pval = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), pval, n01, n10


def delong_auc_test(y, p_a, p_b):
    """DeLong test for two correlated ROC AUCs on the same sample.

    Returns (auc_a, auc_b, z, p_value). Implementation of the fast
    DeLong (Sun & Xu 2014) covariance estimator for two predictors.
    """
    y = np.asarray(y, int)
    order = np.argsort(-np.vstack([p_a, p_b]), axis=1)  # not used; kept explicit below

    def _midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N)
        T2[J] = T
        return T2

    def _fast_delong(preds, label_1_count):
        m = label_1_count
        n = preds.shape[1] - m
        k = preds.shape[0]
        pos = preds[:, :m]
        neg = preds[:, m:]
        tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
        for r in range(k):
            tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
        aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
        v01 = (tz[:, :m] - tx) / n
        v10 = 1.0 - (tz[:, m:] - ty) / m
        sx = np.cov(v01); sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    # order samples so positives come first
    idx = np.argsort(-y)  # 1s first
    ys = y[idx]
    preds = np.vstack([np.asarray(p_a)[idx], np.asarray(p_b)[idx]])
    m = int(ys.sum())
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), pval
