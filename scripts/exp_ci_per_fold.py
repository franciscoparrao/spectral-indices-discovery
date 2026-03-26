#!/usr/bin/env python3
"""E2: Compute SR index AUC per fold for confidence intervals."""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")

# Load training data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X = data['X']
y = data['y']
band_names = list(data['band_names'])

# Band index lookup
B = {name: i for i, name in enumerate(band_names)}

# SR formulas as functions
def sr_silicic(x):
    return x[:, B['B04']] - 0.135

def sr_adv_argillic(x):
    return 0.83 - x[:, B['B02']] / np.maximum(x[:, B['B05']], 1e-6)

def sr_argillic(x):
    return 0.09 / np.maximum(x[:, B['B05']], 1e-6)

def sr_propylitic(x):
    return x[:, B['B03']] - 0.48 * x[:, B['B11']]

def sr_iron_oxide(x):
    return (np.sqrt(x[:, B['B12']]) - x[:, B['B11']]) ** 2

def sr_potassic(x):
    return x[:, B['B03']] * x[:, B['B12']] / np.maximum(x[:, B['B07']] ** 2, 1e-6) - 0.45

# Classical indices
def clay_ratio(x):
    return x[:, B['B11']] / np.maximum(x[:, B['B12']], 1e-6)

def iron_oxide_idx(x):
    return x[:, B['B04']] / np.maximum(x[:, B['B02']], 1e-6)

SR_INDICES = {
    'SR_Silicic': (sr_silicic, 1),
    'SR_Adv_Argillic': (sr_adv_argillic, 2),
    'SR_Argillic_Phyllic': (sr_argillic, 3),
    'SR_Propylitic': (sr_propylitic, 4),
    'SR_Iron_Oxide': (sr_iron_oxide, 5),
    'SR_Potassic_Skarn': (sr_potassic, 6),
}

CLASSICAL_INDICES = {
    'Clay_Ratio': clay_ratio,
    'Iron_Oxide': iron_oxide_idx,
}

class_ids = sorted(np.unique(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# SR indices: compute AUC per fold for target class
for name, (func, target_class) in SR_INDICES.items():
    fold_aucs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_te = X[test_idx]
        y_te = y[test_idx]
        scores = func(X_te)
        y_bin = (y_te == target_class).astype(int)

        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            auc = roc_auc_score(y_bin, scores)
            auc_neg = roc_auc_score(y_bin, -scores)
            fold_aucs.append(max(auc, auc_neg))

    results[name] = {
        'target_class': target_class,
        'per_fold': fold_aucs,
        'mean': float(np.mean(fold_aucs)),
        'std': float(np.std(fold_aucs)),
    }
    print(f"{name}: AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

# Classical indices: compute AUC per fold per class
for idx_name, func in CLASSICAL_INDICES.items():
    for target_class in class_ids:
        fold_aucs = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_te = X[test_idx]
            y_te = y[test_idx]
            scores = func(X_te)
            y_bin = (y_te == target_class).astype(int)

            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                auc = roc_auc_score(y_bin, scores)
                auc_neg = roc_auc_score(y_bin, -scores)
                fold_aucs.append(max(auc, auc_neg))

        key = f"{idx_name}_class{target_class}"
        results[key] = {
            'target_class': int(target_class),
            'per_fold': fold_aucs,
            'mean': float(np.mean(fold_aucs)),
            'std': float(np.std(fold_aucs)),
        }
        print(f"{key}: AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

output = RESULTS_DIR / "ci_per_fold.json"
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output}")
