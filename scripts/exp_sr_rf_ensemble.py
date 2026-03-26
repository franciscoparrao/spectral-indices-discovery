#!/usr/bin/env python3
"""
Experiment: SR-informed RF ensemble.
Compare:
  A) RF(10 raw bands) — upper bound
  B) RF(6 SR indices) — pure SR features
  C) RF(6 SR + low-correlation raw bands) — hybrid
  D) RF(7 classical indices) — classical features
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from itertools import combinations

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")

# Load data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X_raw = data['X']
y = data['y']
band_names = list(data['band_names'])
B = {name: i for i, name in enumerate(band_names)}

print(f"Data: {X_raw.shape}, bands: {band_names}")
print(f"Classes: {np.unique(y, return_counts=True)}")

eps = 1e-6

# ===== Compute feature sets =====

# SR indices
sr_features = np.column_stack([
    X_raw[:, B['B04']] - 0.135,                                              # Silicic
    0.83 - X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B05']], eps),          # Adv Argillic
    0.09 / np.maximum(X_raw[:, B['B05']], eps),                               # Argillic
    X_raw[:, B['B03']] - 0.48 * X_raw[:, B['B11']],                          # Propylitic
    (np.sqrt(X_raw[:, B['B12']]) - X_raw[:, B['B11']]) ** 2,                 # Iron Oxide
    X_raw[:, B['B03']] * X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B07']]**2, eps) - 0.45,  # Potassic
])
sr_names = ['SR_Silicic', 'SR_AdvArg', 'SR_Argillic', 'SR_Propylitic', 'SR_IronOx', 'SR_Potassic']

# Classical indices
classical_features = np.column_stack([
    X_raw[:, B['B11']] / np.maximum(X_raw[:, B['B12']], eps),                 # Clay Ratio
    X_raw[:, B['B04']] / np.maximum(X_raw[:, B['B02']], eps),                 # Iron Oxide
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B8A']], eps),                 # Ferrous
    (X_raw[:, B['B11']] - X_raw[:, B['B12']]) / np.maximum(X_raw[:, B['B11']] + X_raw[:, B['B12']], eps),  # Alunite
    X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B11']], eps),                 # OH Minerals
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B11']], eps),                 # Silica
    (X_raw[:, B['B8A']] - X_raw[:, B['B04']]) / np.maximum(X_raw[:, B['B8A']] + X_raw[:, B['B04']], eps),  # NDVI
])
classical_names = ['Clay_Ratio', 'Iron_Oxide', 'Ferrous', 'Alunite', 'OH_Minerals', 'Silica', 'NDVI']

# ===== Correlation analysis: SR indices vs raw bands =====
print("\n" + "=" * 70)
print("CORRELATION: SR indices vs raw bands (|Pearson r|)")
print("=" * 70)

corr_matrix = np.zeros((len(sr_names), len(band_names)))
for i, sr_name in enumerate(sr_names):
    for j, band_name in enumerate(band_names):
        r = np.corrcoef(sr_features[:, i], X_raw[:, j])[0, 1]
        corr_matrix[i, j] = r

# Print correlation table
header = f"{'SR Index':<16}" + "".join(f"{b:>6}" for b in band_names)
print(header)
print("-" * len(header))
for i, sr_name in enumerate(sr_names):
    row = f"{sr_name:<16}" + "".join(f"{corr_matrix[i,j]:>6.2f}" for j in range(len(band_names)))
    print(row)

# Max absolute correlation per band across all SR indices
max_corr_per_band = np.max(np.abs(corr_matrix), axis=0)
print(f"\n{'Max |r| per band:':<16}" + "".join(f"{max_corr_per_band[j]:>6.2f}" for j in range(len(band_names))))

# Select bands with max |r| < 0.7 (low correlation with any SR index)
CORR_THRESHOLD = 0.70
low_corr_bands = [i for i, mc in enumerate(max_corr_per_band) if mc < CORR_THRESHOLD]
low_corr_names = [band_names[i] for i in low_corr_bands]
print(f"\nBands with max |r| < {CORR_THRESHOLD}: {low_corr_names}")

# Also try threshold 0.5
low_corr_05 = [i for i, mc in enumerate(max_corr_per_band) if mc < 0.50]
low_corr_05_names = [band_names[i] for i in low_corr_05]
print(f"Bands with max |r| < 0.50: {low_corr_05_names}")

# ===== Build feature sets =====
feature_sets = {
    'A: RF(10 raw bands)': (X_raw, band_names),
    'B: RF(6 SR indices)': (sr_features, sr_names),
    f'C: RF(6 SR + {len(low_corr_names)} bands |r|<0.7)': (
        np.column_stack([sr_features, X_raw[:, low_corr_bands]]),
        sr_names + low_corr_names
    ),
    'D: RF(7 classical)': (classical_features, classical_names),
    'E: RF(6 SR + 7 classical)': (
        np.column_stack([sr_features, classical_features]),
        sr_names + classical_names
    ),
}

# If there are low-corr bands at 0.5 threshold and it differs from 0.7
if low_corr_05 != low_corr_bands:
    feature_sets[f'C2: RF(6 SR + {len(low_corr_05_names)} bands |r|<0.5)'] = (
        np.column_stack([sr_features, X_raw[:, low_corr_05]]),
        sr_names + low_corr_05_names
    )

# ===== Evaluate all feature sets with 5-fold CV =====
print("\n" + "=" * 70)
print("EVALUATION: 5-fold stratified CV")
print("=" * 70)

class_ids = sorted(np.unique(y))
CLASS_NAMES = {1: "Silicic", 2: "Adv_Arg", 3: "Arg_Phyl", 4: "Propyl", 5: "Iron_Ox", 6: "Pot_Sk"}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_results = {}

for set_name, (X_set, feat_names) in feature_sets.items():
    print(f"\n--- {set_name} ({len(feat_names)} features: {feat_names}) ---")

    rf = RandomForestClassifier(200, max_depth=15, random_state=42,
                                 n_jobs=-1, class_weight='balanced')

    # Per-fold, per-class AUC
    fold_aucs = {c: [] for c in class_ids}
    fold_ba = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_set, y)):
        X_tr, X_te = X_set[train_idx], X_set[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        rf.fit(X_tr, y_tr)
        y_proba = rf.predict_proba(X_te)
        y_pred = rf.predict(X_te)

        ba = balanced_accuracy_score(y_te, y_pred)
        fold_ba.append(ba)

        for c in class_ids:
            y_bin = (y_te == c).astype(int)
            if c in rf.classes_:
                idx = list(rf.classes_).index(c)
                auc = roc_auc_score(y_bin, y_proba[:, idx])
                fold_aucs[c].append(auc)

    # Summary
    result = {
        'n_features': len(feat_names),
        'feature_names': feat_names,
        'balanced_accuracy': {
            'mean': float(np.mean(fold_ba)),
            'std': float(np.std(fold_ba)),
        },
    }

    print(f"  BA: {np.mean(fold_ba):.3f} ± {np.std(fold_ba):.3f}")
    print(f"  {'Class':<12} {'AUC mean':>10} {'± std':>8}")

    class_results = {}
    for c in class_ids:
        aucs = fold_aucs[c]
        m, s = np.mean(aucs), np.std(aucs)
        class_results[str(c)] = {'mean': float(m), 'std': float(s)}
        print(f"  {CLASS_NAMES[c]:<12} {m:>10.3f} {s:>8.3f}")

    result['per_class_auc'] = class_results
    result['mean_auc'] = float(np.mean([class_results[str(c)]['mean'] for c in class_ids]))
    all_results[set_name] = result

# ===== Summary comparison =====
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)

header = f"{'Feature Set':<45} {'BA':>6} {'Mean AUC':>9}"
for c in class_ids:
    header += f" {CLASS_NAMES[c]:>8}"
print(header)
print("-" * len(header))

for set_name, result in all_results.items():
    row = f"{set_name:<45} {result['balanced_accuracy']['mean']:>6.3f} {result['mean_auc']:>9.3f}"
    for c in class_ids:
        row += f" {result['per_class_auc'][str(c)]['mean']:>8.3f}"
    print(row)

# Save
output = RESULTS_DIR / "sr_rf_ensemble.json"
with open(output, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {output}")
