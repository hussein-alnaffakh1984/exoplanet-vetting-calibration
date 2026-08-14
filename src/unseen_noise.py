"""Baseline versus error-bar-augmented training under UNSEEN perturbation families.

Training uses Gaussian perturbations scaled by the catalogued error bars, and
nothing else. Evaluation then applies families the model was never trained
against, so a reduction in fragility under those families cannot be explained by
invariance to the training perturbation alone. Five independent repeats, each
with its own split, model seed and augmentation draw.
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from scipy import stats
import lightgbm as lgb
import config
from measurement_error import _sigma
from error_aware_training import _augment
from noise_sensitivity import _sigma_pm

K = 4
N_REPEATS = 5
N_MC = 200
THRESH = 0.10


def _params(seed):
    return dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1,
                random_state=seed, deterministic=True, force_col_wise=True, n_jobs=1)


def _families(shape, s, U, L):
    def corr(rho):
        a, b = np.sqrt(rho), np.sqrt(1 - rho)
        return lambda r: (a * r.normal(0, 1, (shape[0], 1)) + b * r.normal(0, 1, shape)) * s

    def split_normal(r):
        z = r.normal(0, 1, shape)
        return np.where(z > 0, U, L) * z

    return {
        "gaussian_SEEN":            lambda r: r.normal(0, 1, shape) * s,
        "correlated_rho0.3":        corr(0.3),
        "correlated_rho0.5":        corr(0.5),
        "correlated_rho0.7":        corr(0.7),
        "asymmetric_split_normal":  split_normal,
        "student_t_df3":            lambda r: r.standard_t(3, shape) / np.sqrt(3) * s,
        "uniform_box":              lambda r: r.uniform(-np.sqrt(3), np.sqrt(3), shape) * s,
        "scale_half":               lambda r: r.normal(0, 1, shape) * s * 0.5,
        "scale_double":             lambda r: r.normal(0, 1, shape) * s * 2.0,
    }


def _fragile(m, X, gen, seed):
    base = (m.predict_proba(X)[:, 1] >= 0.5).astype(int)
    f = np.zeros(len(X)); r = np.random.default_rng(seed)
    for _ in range(N_MC):
        f += ((m.predict_proba(X + gen(r))[:, 1] >= 0.5).astype(int) != base)
    f /= N_MC
    return float((f > THRESH).mean()), float(f.mean())


def run():
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].reset_index(drop=True)
    F = config.PHYS_FEATURES
    X = df[F].fillna(df[F].median()).values
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    sig = np.vstack([_sigma(df, f) for f in F]).T
    up = np.nan_to_num(np.vstack([_sigma_pm(df, f)[0] for f in F]).T)
    lo = np.nan_to_num(np.vstack([_sigma_pm(df, f)[1] for f in F]).T)

    per_rep = {}
    for rep in range(N_REPEATS):
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2,
                                  stratify=y, random_state=rep)
        base_m = lgb.LGBMClassifier(**_params(rep)).fit(X[tr], y[tr])
        Xa, ya = _augment(X[tr], y[tr], sig[tr], K=K, seed=rep)   # Gaussian only
        aug_m = lgb.LGBMClassifier(**_params(rep)).fit(Xa, ya)

        fams = _families(X[te].shape, sig[te], up[te], lo[te])
        for name, gen in fams.items():
            fb, mb = _fragile(base_m, X[te], gen, 1000 + rep)
            fa, ma = _fragile(aug_m,  X[te], gen, 1000 + rep)
            d = per_rep.setdefault(name, {"base": [], "aug": [], "flip_base": [], "flip_aug": []})
            d["base"].append(fb); d["aug"].append(fa)
            d["flip_base"].append(mb); d["flip_aug"].append(ma)
        print(f"  repeat {rep+1}/{N_REPEATS} done", flush=True)

    def ci(a):
        a = np.asarray(a, float)
        h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        return [float(a.mean()), float(a.mean() - h), float(a.mean() + h)]

    out = {"protocol": dict(K=K, n_repeats=N_REPEATS, n_mc=N_MC, threshold=THRESH,
                            training_perturbation="Gaussian scaled by catalogued error bars only",
                            note="every family except gaussian_SEEN is unseen during training"),
           "families": {}}
    all_reduced = True
    for name, d in per_rep.items():
        b, a = np.array(d["base"]), np.array(d["aug"])
        red = 100 * (b.mean() - a.mean()) / b.mean()
        w = stats.wilcoxon(a - b, alternative="less")
        diff = ci(a - b)
        out["families"][name] = dict(
            baseline_fragile=ci(b), augmented_fragile=ci(a),
            paired_difference=diff, reduction_percent=float(red),
            wilcoxon_p_one_sided=float(w.pvalue),
            baseline_mean_flip=ci(d["flip_base"]), augmented_mean_flip=ci(d["flip_aug"]))
        if diff[2] >= 0:
            all_reduced = False
    out["all_families_reduced"] = bool(all_reduced)
    json.dump(out, open("../results/unseen_noise.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    r = run()
    print()
    print(f"{'family':26s} {'baseline':>9s} {'augmented':>10s} {'reduction':>10s} "
          f"{'95% CI of difference':>26s}")
    for k, v in r["families"].items():
        d = v["paired_difference"]
        print(f"{k:26s} {v['baseline_fragile'][0]:9.3f} {v['augmented_fragile'][0]:10.3f} "
              f"{v['reduction_percent']:9.1f}% [{d[1]:+.3f}, {d[2]:+.3f}]")
    print()
    print("ALL UNSEEN FAMILIES REDUCED:", r["all_families_reduced"])
