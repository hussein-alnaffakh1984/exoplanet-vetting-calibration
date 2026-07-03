"""Paired McNemar test: does isotonic calibration change accuracy significantly?

Reads results/flux_fig_preds.npz (released held-out predictions) and, for each
flux classifier, compares the raw and isotonic 0.5-threshold verdicts on the
SAME test cases. Reports discordant pairs (b: raw-only correct, c: iso-only
correct) and the exact binomial McNemar p-value. Writes
results/calibration_significance.json. Supports the manuscript statement that
the small accuracy change from calibration is not statistically significant.
"""
import json, numpy as np
from scipy.stats import binomtest

def mcnemar_raw_vs_iso(y, p_raw, p_iso):
    raw = (p_raw >= 0.5).astype(int); iso = (p_iso >= 0.5).astype(int)
    b = int(((raw == y) & (iso != y)).sum())   # raw correct, iso wrong
    c = int(((raw != y) & (iso == y)).sum())   # iso correct, raw wrong
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return dict(acc_raw=float((raw == y).mean()), acc_iso=float((iso == y).mean()),
                b_raw_only=b, c_iso_only=c, n_discordant=n, mcnemar_exact_p=float(p))

def run():
    z = np.load("../results/flux_fig_preds.npz"); y = z["yte"].astype(int)
    out = {}
    for name, raw, iso in [("LightGBM", "lgb_raw", "lgb_iso"),
                           ("Bayesian_network", "bnn_raw", "bnn_iso")]:
        if raw in z and iso in z:
            out[name] = mcnemar_raw_vs_iso(y, z[raw].astype(float), z[iso].astype(float))
    json.dump(out, open("../results/calibration_significance.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    run()
