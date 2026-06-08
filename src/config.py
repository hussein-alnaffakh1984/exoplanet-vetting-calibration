"""Central configuration: data paths and feature definitions.

Edit DATA_DIR (or set the EXO_DATA_DIR environment variable) to point at the
folder where the public input files have been downloaded. See data/README.md
for the download links and expected file names.
"""
import os

DATA_DIR = os.environ.get("EXO_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# Phase-folded light-curve benchmark (Macedo & Zalewski 2024, Mendeley Data
# DOI 10.17632/wctcv34962.3): 2001 global-view flux bins (columns "0".."2000")
# plus a "label" column with values CONFIRMED / FALSE POSITIVE.
FLUX_GLOBAL_CSV = os.path.join(DATA_DIR, "all_global.csv")
FLUX_CACHE_NPZ = os.path.join(DATA_DIR, "flux_global.npz")  # built by data.cache_flux()

# Kepler DR25 KOI table (NASA Exoplanet Archive). Binary subset
# (CONFIRMED / FALSE POSITIVE) with the physical features below.
KOI_DR25_CSV = os.path.join(DATA_DIR, "koi_dr25_full.csv")

# TESS TOI catalogue (NASA Exoplanet Archive) for cross-mission validation.
TOI_CSV = os.path.join(DATA_DIR, "TOI.csv")

# Seven physical transit / derived features. Robovetter flags and koi_score are
# deliberately EXCLUDED: they are outputs of the official vetting pipeline and
# would make the classification circular (answer leakage).
PHYS_FEATURES = [
    "koi_period", "koi_duration", "koi_depth",
    "koi_prad", "koi_impact", "koi_model_snr", "koi_teq",
]

# Six features shared between Kepler (koi_*) and TESS (pl_*) catalogues,
# used for the zero-shot Kepler -> TESS transfer experiment.
SHARED_KEPLER = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_insol"]
SHARED_TESS = ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_eqt", "pl_insol"]

POS_LABEL = "CONFIRMED"
NEG_LABEL = "FALSE POSITIVE"
SEEDS = [0, 1, 2]  # flux repeats
N_FOLDS = 5        # physical-model cross-validation
