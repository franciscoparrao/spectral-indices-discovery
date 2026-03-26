#!/usr/bin/env python3
"""
Full evaluation: PySR indices vs classical indices vs RF vs Embeddings.
Includes intra-site CV (III Región) and cross-site validation (III→IV).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from pathlib import Path
import json

GT_DIR = Path("data/ground_truth")
EMB_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")

CLASS_NAMES = {
    1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
    4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn",
}

# ===== CLASSICAL INDICES =====
# Adapted from ASTER/Landsat literature to Sentinel-2

def clay_ratio(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """B11/B12 — Ninomiya 2005, standard OH-bearing mineral index"""
    return np.where(B12 > 0, B11 / B12, 0)

def iron_oxide_ratio(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """B04/B02 — Sabins 1999, Fe³⁺ detection"""
    return np.where(B02 > 0, B04 / B02, 0)

def ferrous_ratio(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """B12/B8A — Fe²⁺ minerals (chlorite, epidote)"""
    return np.where(B8A > 0, B12 / B8A, 0)

def alunite_index(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """(B11-B12)/(B11+B12) — Normalized SWIR difference for Al-OH"""
    denom = B11 + B12
    return np.where(denom > 0, (B11 - B12) / denom, 0)

def gossan_index(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """B04/B02 — Gossan detection (similar to iron oxide)"""
    return np.where(B02 > 0, B04 / B02, 0)

def ndvi(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """(B8A-B04)/(B8A+B04) — Vegetation baseline"""
    denom = B8A + B04
    return np.where(denom > 0, (B8A - B04) / denom, 0)

def oh_minerals(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """B02/B11 — OH mineral absorption proxy"""
    return np.where(B11 > 0, B02 / B11, 0)

def silica_index(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """B12/B11 — Inverse clay ratio, silicification indicator"""
    return np.where(B11 > 0, B12 / B11, 0)

# ===== PySR DISCOVERED INDICES =====
def sr_silicic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return B04 - 0.13496

def sr_adv_argillic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return 0.83176 - (B02 / np.maximum(B05, 1e-6))

def sr_argillic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return 0.09014 / np.maximum(B05, 1e-6)

def sr_propylitic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return B03 - (B11 * 0.48311)

def sr_iron_oxide(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return (np.sqrt(B12) - B11) ** 2

def sr_potassic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return (B03 * (B12 / np.maximum(B07 ** 2, 1e-6))) - 0.45114


def evaluate_index(scores, y, cls_id):
    """Evaluate a single index for one-vs-rest classification."""
    y_bin = (y == cls_id).astype(int)
    if y_bin.sum() < 5 or (1 - y_bin).sum() < 5:
        return {"auc": np.nan, "f1": np.nan}

    scores_clean = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
    auc = roc_auc_score(y_bin, scores_clean)

    # Also try inverted (some indices are inversely related)
    auc_inv = roc_auc_score(y_bin, -scores_clean)
    if auc_inv > auc:
        auc = auc_inv
        scores_clean = -scores_clean

    best_f1 = 0
    for t in np.arange(0.01, 0.99, 0.02):
        f1 = f1_score(y_bin, (scores_clean > np.percentile(scores_clean, t * 100)).astype(int),
                      zero_division=0)
        best_f1 = max(best_f1, f1)

    return {"auc": float(auc), "f1": float(best_f1)}


def main():
    # ===== LOAD DATA =====
    print("Loading data...")

    # III Región (training site)
    train_data = np.load(GT_DIR / "maricunga_training_s2_gee.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    band_names = [str(b) for b in sorted(train_data["band_names"])]
    print(f"III Región: {len(y_train)} samples, bands: {band_names}")

    # IV Región (validation site)
    val_csv = EMB_DIR / "iv_region_validation.csv"
    df_val = pd.read_csv(val_csv)
    band_cols = sorted([c for c in df_val.columns if c.startswith("B") and len(c) <= 3])
    df_val = df_val.dropna(subset=band_cols)
    X_val = df_val[band_cols].values
    y_val = df_val["class_id"].values
    print(f"IV Región: {len(y_val)} samples, bands: {band_cols}")

    # Embeddings
    emb_train_df = pd.read_csv(EMB_DIR / "atlas_alteration_embeddings_pixels.csv")
    emb_val_df = df_val.copy()
    emb_cols = sorted([c for c in emb_train_df.columns if c.startswith("A") and c[1:].isdigit()])[:64]

    # Extract band arrays for formula application
    def get_bands(X, names):
        d = {names[i]: X[:, i] for i in range(len(names))}
        return d

    bands_train = get_bands(X_train, band_names)
    bands_val = get_bands(X_val, band_cols)

    # ===== DEFINE ALL INDICES =====
    classical_indices = {
        "Clay Ratio (B11/B12)": clay_ratio,
        "Iron Oxide (B04/B02)": iron_oxide_ratio,
        "Ferrous (B12/B8A)": ferrous_ratio,
        "Alunite Idx (B11-B12)/(B11+B12)": alunite_index,
        "OH Minerals (B02/B11)": oh_minerals,
        "Silica (B12/B11)": silica_index,
        "NDVI": ndvi,
    }

    sr_indices = {
        "SR: B04 - 0.135": sr_silicic,
        "SR: 0.83 - B02/B05": sr_adv_argillic,
        "SR: 0.09/B05": sr_argillic,
        "SR: B03 - B11*0.48": sr_propylitic,
        "SR: (sqrt(B12)-B11)²": sr_iron_oxide,
        "SR: B03*B12/B07² - 0.45": sr_potassic,
    }

    # Target classes for SR
    sr_target = {
        "SR: B04 - 0.135": 1,
        "SR: 0.83 - B02/B05": 2,
        "SR: 0.09/B05": 3,
        "SR: B03 - B11*0.48": 4,
        "SR: (sqrt(B12)-B11)²": 5,
        "SR: B03*B12/B07² - 0.45": 6,
    }

    # ===== 1. INTRA-SITE CV (III REGIÓN) =====
    print("\n" + "=" * 70)
    print("1. INTRA-SITE 5-FOLD CV — III Región (Maricunga)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    present_classes = sorted(set(y_train))

    # RF S2 baseline
    rf = RandomForestClassifier(200, max_depth=15, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    y_proba_rf = cross_val_predict(rf, X_train, y_train, cv=cv, method="predict_proba")
    rf.fit(X_train, y_train)  # Fit for later use

    # RF Embeddings
    emb_train_filtered = emb_train_df[emb_train_df["class_id"].isin(present_classes)]
    X_emb = emb_train_filtered[emb_cols].values
    y_emb = emb_train_filtered["class_id"].values

    rf_emb = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight="balanced")
    y_proba_emb = cross_val_predict(rf_emb, X_emb, y_emb, cv=cv, method="predict_proba")
    rf_emb.fit(X_emb, y_emb)

    results_intra = {}

    print(f"\n{'Index':<35} ", end="")
    for cls_id in present_classes:
        print(f"{CLASS_NAMES.get(cls_id, '?')[:12]:>12}", end="")
    print(f"{'  Mean':>8}")
    print("-" * (35 + 12 * len(present_classes) + 8))

    # Classical indices
    for idx_name, idx_fn in classical_indices.items():
        scores = idx_fn(**bands_train)
        aucs = []
        for cls_id in present_classes:
            res = evaluate_index(scores, y_train, cls_id)
            aucs.append(res["auc"])
        mean_auc = np.nanmean(aucs)
        print(f"{idx_name:<35} ", end="")
        for a in aucs:
            print(f"{a:>12.3f}", end="")
        print(f"{mean_auc:>8.3f}")
        results_intra[idx_name] = {str(c): a for c, a in zip(present_classes, aucs)}
        results_intra[idx_name]["mean"] = mean_auc

    print("-" * (35 + 12 * len(present_classes) + 8))

    # SR indices
    for idx_name, idx_fn in sr_indices.items():
        target_cls = sr_target[idx_name]
        if target_cls not in present_classes:
            continue
        scores = idx_fn(**bands_train)
        aucs = []
        for cls_id in present_classes:
            res = evaluate_index(scores, y_train, cls_id)
            aucs.append(res["auc"])
        mean_auc = np.nanmean(aucs)
        print(f"{idx_name:<35} ", end="")
        for a in aucs:
            print(f"{a:>12.3f}", end="")
        print(f"{mean_auc:>8.3f}")
        results_intra[idx_name] = {str(c): a for c, a in zip(present_classes, aucs)}
        results_intra[idx_name]["mean"] = mean_auc

    print("-" * (35 + 12 * len(present_classes) + 8))

    # RF S2
    aucs_rf = []
    for cls_id in present_classes:
        y_bin = (y_train == cls_id).astype(int)
        idx = list(rf.classes_).index(cls_id)
        auc = roc_auc_score(y_bin, y_proba_rf[:, idx])
        aucs_rf.append(auc)
    print(f"{'RF S2 (10 bands, 5-fold CV)':<35} ", end="")
    for a in aucs_rf:
        print(f"{a:>12.3f}", end="")
    print(f"{np.mean(aucs_rf):>8.3f}")

    # RF Embeddings
    aucs_emb = []
    for cls_id in present_classes:
        y_bin = (y_emb == cls_id).astype(int)
        if cls_id in rf_emb.classes_:
            idx = list(rf_emb.classes_).index(cls_id)
            auc = roc_auc_score(y_bin, y_proba_emb[:, idx])
        else:
            auc = 0.5
        aucs_emb.append(auc)
    print(f"{'RF Embeddings (64-dim, 5-fold CV)':<35} ", end="")
    for a in aucs_emb:
        print(f"{a:>12.3f}", end="")
    print(f"{np.mean(aucs_emb):>8.3f}")

    # ===== 2. CROSS-SITE VALIDATION (III → IV) =====
    print("\n" + "=" * 70)
    print("2. CROSS-SITE VALIDATION — III Región → IV Región")
    print("=" * 70)

    val_classes = sorted(set(y_val))

    print(f"\n{'Index':<35} ", end="")
    for cls_id in val_classes:
        print(f"{CLASS_NAMES.get(cls_id, '?')[:12]:>12}", end="")
    print(f"{'  Mean':>8}")
    print("-" * (35 + 12 * len(val_classes) + 8))

    results_cross = {}

    # Classical indices
    for idx_name, idx_fn in classical_indices.items():
        scores = idx_fn(**bands_val)
        aucs = []
        for cls_id in val_classes:
            res = evaluate_index(scores, y_val, cls_id)
            aucs.append(res["auc"])
        mean_auc = np.nanmean(aucs)
        print(f"{idx_name:<35} ", end="")
        for a in aucs:
            print(f"{a:>12.3f}", end="")
        print(f"{mean_auc:>8.3f}")
        results_cross[idx_name] = {str(c): a for c, a in zip(val_classes, aucs)}

    print("-" * (35 + 12 * len(val_classes) + 8))

    # SR indices
    for idx_name, idx_fn in sr_indices.items():
        target_cls = sr_target[idx_name]
        scores = idx_fn(**bands_val)
        aucs = []
        for cls_id in val_classes:
            res = evaluate_index(scores, y_val, cls_id)
            aucs.append(res["auc"])
        mean_auc = np.nanmean(aucs)
        print(f"{idx_name:<35} ", end="")
        for a in aucs:
            print(f"{a:>12.3f}", end="")
        print(f"{mean_auc:>8.3f}")
        results_cross[idx_name] = {str(c): a for c, a in zip(val_classes, aucs)}

    print("-" * (35 + 12 * len(val_classes) + 8))

    # RF S2 cross-site
    y_pred_rf_val = rf.predict_proba(X_val)
    aucs_rf_cross = []
    for cls_id in val_classes:
        y_bin = (y_val == cls_id).astype(int)
        if cls_id in rf.classes_:
            idx = list(rf.classes_).index(cls_id)
            auc = roc_auc_score(y_bin, y_pred_rf_val[:, idx])
        else:
            auc = 0.5
        aucs_rf_cross.append(auc)
    print(f"{'RF S2 (trained III, tested IV)':<35} ", end="")
    for a in aucs_rf_cross:
        print(f"{a:>12.3f}", end="")
    print(f"{np.mean(aucs_rf_cross):>8.3f}")

    # RF Embeddings cross-site
    if emb_cols[0] in df_val.columns:
        X_emb_val = df_val[emb_cols].dropna().values
        y_emb_val = y_val[:len(X_emb_val)]
        y_pred_emb_val = rf_emb.predict_proba(X_emb_val)
        aucs_emb_cross = []
        for cls_id in val_classes:
            y_bin = (y_emb_val == cls_id).astype(int)
            if cls_id in rf_emb.classes_:
                idx = list(rf_emb.classes_).index(cls_id)
                auc = roc_auc_score(y_bin, y_pred_emb_val[:, idx])
            else:
                auc = 0.5
            aucs_emb_cross.append(auc)
        print(f"{'RF Embeddings (trained III, IV)':<35} ", end="")
        for a in aucs_emb_cross:
            print(f"{a:>12.3f}", end="")
        print(f"{np.mean(aucs_emb_cross):>8.3f}")

    # Save all results
    all_results = {"intra_site": results_intra, "cross_site": results_cross}
    with open(RESULTS_DIR / "full_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'full_evaluation.json'}")


if __name__ == "__main__":
    main()
