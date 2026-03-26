#!/usr/bin/env python3
"""E1: Compute RF one-vs-rest AUC per class for fair comparison with SR indices."""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")

# Load training data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X = data['X']
y = data['y']
band_names = list(data['band_names'])

print(f"Data: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

# RF OvR: train RF with predict_proba, compute AUC per class
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class_ids = sorted(np.unique(y))
class_aucs = {str(c): [] for c in class_ids}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    rf.fit(X_tr, y_tr)
    y_proba = rf.predict_proba(X_te)

    for i, c in enumerate(rf.classes_):
        y_bin = (y_te == c).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            auc = roc_auc_score(y_bin, y_proba[:, i])
            class_aucs[str(c)].append(auc)

    print(f"  Fold {fold+1}/5 done")

# Compute mean and std per class
rf_results = {}
for c in class_ids:
    aucs = class_aucs[str(c)]
    rf_results[str(c)] = {
        'mean': float(np.mean(aucs)),
        'std': float(np.std(aucs)),
        'per_fold': [float(a) for a in aucs],
    }
    print(f"  Class {c}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

rf_results['overall_mean'] = float(np.mean([rf_results[str(c)]['mean'] for c in class_ids]))

# Save
output = RESULTS_DIR / "rf_ovr_auc.json"
with open(output, 'w') as f:
    json.dump(rf_results, f, indent=2)
print(f"\nSaved: {output}")
