"""Controlled head-to-head comparison with a recent tabular method (TabM, 2025).

TabM is chosen over TabPFN because it is installable and runnable with a single
pip command under an open licence, with no registration or API token, so a
referee can reproduce this comparison without creating an account.

Part A: identical five-fold split, identical features, identical metric
definitions; LightGBM and TabM are compared, with DeLong and McNemar on the
pooled out-of-fold predictions.

Part B: error-bar augmentation is applied ON TOP of TabM over five independent
repeats, testing whether the method generalises beyond gradient-boosted trees.
"""
import json, warnings, time
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
import lightgbm as lgb
import pytabkit
import config
from metrics import ece, delong_auc_test, mcnemar
from measurement_error import _sigma
from error_aware_training import _augment

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1,
           random_state=0, deterministic=True, force_col_wise=True, n_jobs=1)


def tabm(seed):
    return pytabkit.TabM_D_Classifier(device=DEV, random_state=seed, verbosity=0)


def scores(p, y):
    return dict(acc=float(((p >= 0.5).astype(int) == y).mean()),
                auc=float(roc_auc_score(y, p)), ece=float(ece(p, y)),
                brier=float(brier_score_loss(y, p)))


def flips(m, X, sig, n_mc, seed):
    base = (m.predict_proba(X)[:, 1] >= 0.5).astype(int)
    f = np.zeros(len(X)); r = np.random.default_rng(seed)
    for _ in range(n_mc):
        f += ((m.predict_proba(X + r.normal(0, 1, X.shape) * np.nan_to_num(sig))[:, 1] >= 0.5)
              .astype(int) != base)
    return f / n_mc


def ci(a):
    a = np.asarray(a, float)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return [float(a.mean()), float(a.mean() - h), float(a.mean() + h)]


def run(n_mc=200, n_repeats=5, K=4):
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    F = config.PHYS_FEATURES
    X = df[F].fillna(df[F].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in F]).T

    # ---------- Part A: same five folds, same features, same metrics ----------
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    oof = {"LightGBM": np.zeros(len(y)), "TabM": np.zeros(len(y))}
    fold = {"LightGBM": [], "TabM": []}
    for k, (tr, te) in enumerate(skf.split(X, y)):
        a = lgb.LGBMClassifier(**LGB).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        b = tabm(0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof["LightGBM"][te] = a; oof["TabM"][te] = b
        fold["LightGBM"].append(scores(a, y[te])); fold["TabM"].append(scores(b, y[te]))
        print(f"  fold {k+1}/5  LightGBM AUC {fold['LightGBM'][-1]['auc']:.4f} "
              f"| TabM AUC {fold['TabM'][-1]['auc']:.4f}", flush=True)

    partA = {m: {q: [float(np.mean([f[q] for f in fold[m]])),
                     float(np.std([f[q] for f in fold[m]]))]
                 for q in ("acc", "auc", "ece", "brier")} for m in fold}
    aa, ab, z, pz = delong_auc_test(y, oof["LightGBM"], oof["TabM"])
    chi2, pm, n01, n10 = mcnemar(y, (oof["LightGBM"] >= .5).astype(int),
                                    (oof["TabM"] >= .5).astype(int))
    partA["tests"] = dict(delong=dict(auc_lightgbm=aa, auc_tabm=ab, z=z, p=pz),
                          mcnemar=dict(chi2=chi2, p=pm, n01=n01, n10=n10, n=int(len(y))))

    # ---------- Part B: augmentation applied on top of TabM ----------
    base_r, aug_r = [], []
    for rep in range(n_repeats):
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=rep)
        mb = tabm(rep).fit(X[tr], y[tr])
        Xa, ya = _augment(X[tr], y[tr], sig[tr], K=K, seed=rep)
        ma = tabm(rep).fit(Xa, ya)
        for tag, m, store in (("base", mb, base_r), ("aug", ma, aug_r)):
            p = m.predict_proba(X[te])[:, 1]
            fr = flips(m, X[te], sig[te], n_mc, 1000 + rep)
            d = scores(p, y[te]); d["fragile_fraction"] = float((fr > 0.10).mean())
            d["mean_flip_rate"] = float(fr.mean()); store.append(d)
        print(f"  repeat {rep+1}/{n_repeats}  fragile "
              f"{base_r[-1]['fragile_fraction']:.3f} -> "
              f"{aug_r[-1]['fragile_fraction']:.3f}", flush=True)

    M = ("acc", "auc", "ece", "brier", "fragile_fraction", "mean_flip_rate")
    partB = {"baseline": {q: ci([r[q] for r in base_r]) for q in M},
             "error_aware": {q: ci([r[q] for r in aug_r]) for q in M},
             "paired_difference": {}}
    for q in M:
        d = np.array([a[q] - b[q] for a, b in zip(aug_r, base_r)])
        partB["paired_difference"][q] = dict(
            mean_ci=ci(d), wilcoxon_p=float(stats.wilcoxon(d).pvalue))

    out = dict(method="TabM (Gorishniy et al., ICLR 2025), pytabkit TabM_D_Classifier",
               device=DEV, K=K, n_repeats=n_repeats, n_mc=n_mc,
               licence_note="open licence, pip-installable, no registration or API token",
               part_A_five_fold=partA, part_B_augmentation_on_tabm=partB)
    json.dump(out, open("../results/recent_method.json", "w"), indent=1)
    np.savez_compressed("../results/recent_method_oof.npz",
                        lightgbm=oof["LightGBM"], tabm=oof["TabM"], y=y)
    return out


if __name__ == "__main__":
    t0 = time.time(); r = run()
    A = r["part_A_five_fold"]
    print("\n== Part A: identical five-fold protocol ==")
    for m in ("LightGBM", "TabM"):
        s = A[m]
        print(f"  {m:9s} acc {s['acc'][0]:.3f}+-{s['acc'][1]:.3f}  "
              f"auc {s['auc'][0]:.3f}+-{s['auc'][1]:.3f}  "
              f"ece {s['ece'][0]:.3f}+-{s['ece'][1]:.3f}  "
              f"brier {s['brier'][0]:.3f}+-{s['brier'][1]:.3f}")
    d = A["tests"]["delong"]; mm = A["tests"]["mcnemar"]
    print(f"  DeLong  AUC {d['auc_lightgbm']:.4f} vs {d['auc_tabm']:.4f}  "
          f"z {d['z']:.2f}  p {d['p']:.2e}")
    print(f"  McNemar chi2 {mm['chi2']:.2f}  p {mm['p']:.2e}  "
          f"({mm['n01']} vs {mm['n10']}, n={mm['n']})")
    B = r["part_B_augmentation_on_tabm"]
    print("\n== Part B: error-bar augmentation applied to TabM ==")
    for q in ("auc", "ece", "fragile_fraction", "mean_flip_rate"):
        b, a, p = B["baseline"][q], B["error_aware"][q], B["paired_difference"][q]
        print(f"  {q:18s} {b[0]:.4f} -> {a[0]:.4f}   diff {p['mean_ci'][0]:+.4f} "
              f"[{p['mean_ci'][1]:+.4f}, {p['mean_ci'][2]:+.4f}]  p {p['wilcoxon_p']:.4f}")
    print(f"\nelapsed {(time.time()-t0)/60:.1f} min")
