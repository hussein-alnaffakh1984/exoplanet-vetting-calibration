"""Data loading and caching helpers."""
import os
import numpy as np
import pandas as pd

import config


def cache_flux():
    """Read the (large) flux CSV once and cache it as a compact npz.

    Produces arrays X (full 2001-bin flux, float32), Xds (every 4th bin, for
    NGBoost) and y (1 = CONFIRMED, 0 = FALSE POSITIVE).
    """
    if os.path.exists(config.FLUX_CACHE_NPZ):
        return
    df = pd.read_csv(config.FLUX_GLOBAL_CSV)
    y = (df["label"] == config.POS_LABEL).astype(np.int8).values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(config.FLUX_CACHE_NPZ, X=X, Xds=X[:, ::4], y=y)


def load_flux():
    """Return X (full), Xds (downsampled), y for the flux representation."""
    cache_flux()
    d = np.load(config.FLUX_CACHE_NPZ)
    return d["X"], d["Xds"], d["y"]


def load_physical():
    """Return X (DataFrame of PHYS_FEATURES) and y for the binary physical task."""
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].copy()
    X = df[config.PHYS_FEATURES].copy()
    X = X.fillna(X.median())
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    return X.reset_index(drop=True), y


def load_kepler_shared():
    """Kepler physical features shared with TESS (for cross-mission transfer)."""
    df = pd.read_csv(config.KOI_DR25_CSV)
    df = df[df["koi_disposition"].isin([config.POS_LABEL, config.NEG_LABEL])].copy()
    X = df[config.SHARED_KEPLER].copy().fillna(df[config.SHARED_KEPLER].median())
    y = (df["koi_disposition"] == config.POS_LABEL).astype(int).values
    return X.reset_index(drop=True), y


def load_tess_shared():
    """TESS TOI features mapped onto the shared schema, with a binary label.

    Positives: CP (confirmed) and KP (known planet). Negatives: FP (false
    positive) and FA (false alarm). PC (planet candidate) and APC (ambiguous)
    are dropped as having no ground-truth disposition.
    """
    df = pd.read_csv(config.TOI_CSV, comment="#")
    disp = df["tfopwg_disp"].astype(str)
    pos = disp.isin(["CP", "KP"])
    neg = disp.isin(["FP", "FA"])
    keep = pos | neg
    df = df[keep].copy()
    y = pos[keep].astype(int).values
    X = df[config.SHARED_TESS].copy()
    X.columns = config.SHARED_KEPLER  # align column names to the Kepler schema
    X = X.fillna(X.median())
    return X.reset_index(drop=True), y
