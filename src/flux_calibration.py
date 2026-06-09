"""Flux representation: calibration of three probabilistic classifiers.

For each of three random splits it trains
LightGBM (raw + isotonic), NGBoost (on the downsampled flux) and a Monte-Carlo
Dropout Bayesian network (raw + isotonic), and reports accuracy, AUC, Brier and
ECE as mean +/- std across the splits.

Usage:  python flux_calibration.py
"""
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

import config
import data as D
from metrics import all_metrics


def _split(n, y, seed):
    tr, tmp = train_test_split(np.arange(n), test_size=0.30, stratify=y, random_state=seed)
    cal, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, cal, te


def _bnn(Xtr, ytr, Xcal, Xte, T=200, epochs=45):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xcal, Xte = (Xtr - mu) / sd, (Xcal - mu) / sd, (Xte - mu) / sd

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):
            return self.f(x)

    net = Net(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr.astype(np.float32))
    yt = torch.tensor(ytr.astype(np.float32)).view(-1, 1)
    n, bs = len(xt), 128
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(xt[j]), yt[j]).backward()
            opt.step()

    def mc(Xnp):
        net.train()  # keep dropout active
        x = torch.tensor(Xnp.astype(np.float32))
        ps = []
        with torch.no_grad():
            for _ in range(T):
                ps.append(torch.sigmoid(net(x)).numpy().ravel())
        P = np.stack(ps)
        return P.mean(0)

    return mc(Xte), mc(Xcal)


def run():
    X, Xds, y = D.load_flux()
    out = {k: [] for k in ["lgb_raw", "lgb_iso", "ngboost", "ngboost_iso", "bnn_raw", "bnn_iso"]}
    for s in config.SEEDS:
        tr, cal, te = _split(len(X), y, s)
        # LightGBM (full features)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
        m.fit(X[tr], y[tr])
        pte, pcal = m.predict_proba(X[te])[:, 1], m.predict_proba(X[cal])[:, 1]
        out["lgb_raw"].append(all_metrics(pte, y[te]))
        iso = IsotonicRegression(out_of_bounds="clip").fit(pcal, y[cal])
        out["lgb_iso"].append(all_metrics(np.clip(iso.predict(pte), 0, 1), y[te]))
        # NGBoost (downsampled flux), raw + isotonic
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        ng = NGBClassifier(Dist=Bernoulli, n_estimators=250, learning_rate=0.04,
                           minibatch_frac=0.5, verbose=False)
        ng.fit(Xds[tr], y[tr])
        png_te, png_cal = ng.predict_proba(Xds[te])[:, 1], ng.predict_proba(Xds[cal])[:, 1]
        out["ngboost"].append(all_metrics(png_te, y[te]))
        ng_iso = IsotonicRegression(out_of_bounds="clip").fit(png_cal, y[cal])
        out["ngboost_iso"].append(all_metrics(np.clip(ng_iso.predict(png_te), 0, 1), y[te]))
        # Bayesian network (MC Dropout)
        pte_b, pcal_b = _bnn(X[tr], y[tr], X[cal], X[te])
        out["bnn_raw"].append(all_metrics(pte_b, y[te]))
        isob = IsotonicRegression(out_of_bounds="clip").fit(pcal_b, y[cal])
        out["bnn_iso"].append(all_metrics(np.clip(isob.predict(pte_b), 0, 1), y[te]))
        print(f"seed {s} done")

    summary = {k: {m: [float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))]
                   for m in v[0]} for k, v in out.items()}
    json.dump(summary, open("../results/flux_calibration.json", "w"), indent=1)
    for k, v in summary.items():
        print(k, {m: f"{a:.3f}+/-{b:.3f}" for m, (a, b) in v.items()})


if __name__ == "__main__":
    run()
