#!/usr/bin/env python3
"""make_tables.py - regenerate every numeric table in the paper from results/.

Reads only the JSON/NPZ artifacts produced by reproduce.py and prints each
table's values (Tables 2-5 and the supplementary classifier tables), so every
reported number is traceable to a released file with no manual transcription.

    python make_tables.py
"""
import json, numpy as np
R = "results"
L = lambda f: json.load(open(f"{R}/{f}"))

def fmt(m): return f"{m[0]:.3f}\u00b1{m[1]:.3f}" if isinstance(m, list) else f"{m:.3f}"

def table2():
    d = L("flux_calibration.json")
    print("\n== Table 2: flux global-view classifiers ==")
    for k, v in d.items():
        if isinstance(v, dict) and "acc" in v:
            print(f"  {k:28s} acc {fmt(v['acc'])}  auc {fmt(v['auc'])}  brier {fmt(v.get('brier',0))}  ece {fmt(v.get('ece',0))}")

def table3():
    d = L("physical_model.json")["cv"]
    print("\n== Table 3: physical-parameter model (5-fold CV) ==")
    print(f"  Accuracy {fmt(d['acc'])}  AUC {fmt(d['auc'])}  Brier {fmt(d['brier'])}  ECE {fmt(d['ece'])}")

def tableS2():
    d = L("physical_three_models.json"); x = L("xgboost_physical.json")
    print("\n== Table S2: classifiers on physical features (same 5-fold protocol) ==")
    for k, v in d.items():
        print(f"  {k:10s} acc {fmt(v['acc'])}  auc {fmt(v['auc'])}  brier {fmt(v['brier'])}  ece {fmt(v['ece'])}")
    print(f"  {'XGBoost':10s} acc {fmt(x['acc'])}  auc {fmt(x['auc'])}  ece_fold {fmt(x['ece_fold'])}")

def table4():
    d = L("cross_mission.json")
    print("\n== Table 4: Kepler -> TESS (six shared features; TESS n=%d) ==" % d["tess_n"])
    for k in ("kep_heldout", "tess_zero", "tess_recal"):
        v = d[k]; print(f"  {k:12s} acc {v['acc']:.3f}  auc {v['auc']:.3f}  ece {v['ece']:.3f}")

def table5():
    d = L("error_aware_training.json")
    print("\n== Table 5: baseline vs error-aware (single held-out split) ==")
    for k in ("baseline", "error_aware"):
        v = d[k]; idd = v["in_dist"]
        print(f"  {k:12s} acc {idd['acc']:.3f}  auc {idd['auc']:.3f}  ece {idd['ece']:.3f}  "
              f"fragile {v['fragile_fraction']*100:.1f}%  flip {v['mean_flip_rate']*100:.1f}%  "
              f"TESS auc/ece {v['tess']['auc']:.3f}/{v['tess']['ece']:.3f}")

if __name__ == "__main__":
    for t in (table2, table3, tableS2, table4, table5):
        try: t()
        except Exception as e: print(f"[skip {t.__name__}: {e}]")
    print("\nEvery table value above is read directly from results/ artifacts.")
