#!/usr/bin/env python3
"""
E-M1: TOST (Two One-Sided Tests) equivalence test.
Formally tests: SR features ≡ raw bands within margin ε.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")

# Load data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X_raw = data['X']
y = data['y']
band_names = list(data['band_names'])
B = {name: i for i, name in enumerate(band_names)}

eps = 1e-6

# SR features
sr_features = np.column_stack([
    X_raw[:, B['B04']] - 0.135,
    0.83 - X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B05']], eps),
    0.09 / np.maximum(X_raw[:, B['B05']], eps),
    X_raw[:, B['B03']] - 0.48 * X_raw[:, B['B11']],
    (np.sqrt(X_raw[:, B['B12']]) - X_raw[:, B['B11']]) ** 2,
    X_raw[:, B['B03']] * X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B07']]**2, eps) - 0.45,
])

class_ids = sorted(np.unique(y))
CLASS_NAMES = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
               4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn"}

# 10-fold CV for more power
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Collect per-fold, per-class AUC for both feature sets
auc_raw_folds = []
auc_sr_folds = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
    # RF on raw bands
    rf_raw = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
    rf_raw.fit(X_raw[train_idx], y[train_idx])
    proba_raw = rf_raw.predict_proba(X_raw[test_idx])

    # RF on SR features
    rf_sr = RandomForestClassifier(200, max_depth=15, random_state=42,
                                    n_jobs=-1, class_weight='balanced')
    rf_sr.fit(sr_features[train_idx], y[train_idx])
    proba_sr = rf_sr.predict_proba(sr_features[test_idx])

    # Mean AUC across classes for this fold
    aucs_raw = []
    aucs_sr = []
    for c in class_ids:
        y_bin = (y[test_idx] == c).astype(int)
        if c in rf_raw.classes_ and y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            idx_raw = list(rf_raw.classes_).index(c)
            idx_sr = list(rf_sr.classes_).index(c)
            aucs_raw.append(roc_auc_score(y_bin, proba_raw[:, idx_raw]))
            aucs_sr.append(roc_auc_score(y_bin, proba_sr[:, idx_sr]))

    auc_raw_folds.append(np.mean(aucs_raw))
    auc_sr_folds.append(np.mean(aucs_sr))

auc_raw_folds = np.array(auc_raw_folds)
auc_sr_folds = np.array(auc_sr_folds)
diffs = auc_sr_folds - auc_raw_folds

print("=" * 60)
print("TOST EQUIVALENCE TEST: RF(6 SR) vs RF(10 raw bands)")
print("=" * 60)
print(f"\n10-fold CV results:")
print(f"  RF(raw):  mean AUC = {auc_raw_folds.mean():.4f} ± {auc_raw_folds.std():.4f}")
print(f"  RF(SR):   mean AUC = {auc_sr_folds.mean():.4f} ± {auc_sr_folds.std():.4f}")
print(f"  Diff:     mean = {diffs.mean():+.4f} ± {diffs.std():.4f}")

# Paired t-test (standard)
t_stat, p_paired = stats.ttest_rel(auc_sr_folds, auc_raw_folds)
print(f"\nPaired t-test: t={t_stat:.3f}, p={p_paired:.4f}")

# TOST equivalence test with different margins
results = {}
for epsilon in [0.01, 0.02, 0.03, 0.05]:
    n = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)

    # Upper bound test: H0: mean_diff >= epsilon
    t_upper = (mean_diff - epsilon) / se
    p_upper = stats.t.cdf(t_upper, df=n-1)  # one-sided, want small

    # Lower bound test: H0: mean_diff <= -epsilon
    t_lower = (mean_diff + epsilon) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=n-1)  # one-sided, want small

    p_tost = max(p_upper, p_lower)
    reject = p_tost < 0.05

    results[str(epsilon)] = {
        'epsilon': epsilon,
        'mean_diff': float(mean_diff),
        'se': float(se),
        't_upper': float(t_upper),
        'p_upper': float(p_upper),
        't_lower': float(t_lower),
        'p_lower': float(p_lower),
        'p_tost': float(p_tost),
        'reject_h0': bool(reject),
        'equivalent': bool(reject),
    }

    sig = "EQUIVALENT" if reject else "not equivalent"
    print(f"\n  TOST ε={epsilon:.2f}: p={p_tost:.4f} → {sig} (α=0.05)")
    print(f"    H0: |ΔAUC| ≥ {epsilon}")
    print(f"    Upper: t={t_upper:.3f}, p={p_upper:.4f}")
    print(f"    Lower: t={t_lower:.3f}, p={p_lower:.4f}")

# Also do Cuprite
print("\n" + "=" * 60)
print("CUPRITE VALIDATION")
print("=" * 60)

cuprite_data = np.load(GT_DIR / "cuprite_training_s2.npz", allow_pickle=True)
X_cup = cuprite_data['X']
y_cup = cuprite_data['y']
cup_bands = list(cuprite_data['band_names'])
B_cup = {name: i for i, name in enumerate(cup_bands)}

sr_cup = np.column_stack([
    X_cup[:, B_cup['B04']] - 0.135,
    0.83 - X_cup[:, B_cup['B02']] / np.maximum(X_cup[:, B_cup['B05']], eps),
    0.09 / np.maximum(X_cup[:, B_cup['B05']], eps),
    X_cup[:, B_cup['B03']] - 0.48 * X_cup[:, B_cup['B11']],
    (np.sqrt(X_cup[:, B_cup['B12']]) - X_cup[:, B_cup['B11']]) ** 2,
    X_cup[:, B_cup['B03']] * X_cup[:, B_cup['B12']] / np.maximum(X_cup[:, B_cup['B07']]**2, eps) - 0.45,
])

cup_classes = sorted([c for c in np.unique(y_cup) if c > 0 and (y_cup == c).sum() >= 10])

skf_cup = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cup_raw_folds = []
cup_sr_folds = []

for fold, (train_idx, test_idx) in enumerate(skf_cup.split(X_cup, y_cup)):
    rf_raw = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
    rf_raw.fit(X_cup[train_idx], y_cup[train_idx])
    proba_raw = rf_raw.predict_proba(X_cup[test_idx])

    rf_sr = RandomForestClassifier(200, max_depth=15, random_state=42,
                                    n_jobs=-1, class_weight='balanced')
    rf_sr.fit(sr_cup[train_idx], y_cup[train_idx])
    proba_sr = rf_sr.predict_proba(sr_cup[test_idx])

    aucs_raw = []
    aucs_sr = []
    for c in cup_classes:
        y_bin = (y_cup[test_idx] == c).astype(int)
        if c in rf_raw.classes_ and c in rf_sr.classes_:
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                aucs_raw.append(roc_auc_score(y_bin, proba_raw[:, list(rf_raw.classes_).index(c)]))
                aucs_sr.append(roc_auc_score(y_bin, proba_sr[:, list(rf_sr.classes_).index(c)]))

    cup_raw_folds.append(np.mean(aucs_raw))
    cup_sr_folds.append(np.mean(aucs_sr))

cup_raw_folds = np.array(cup_raw_folds)
cup_sr_folds = np.array(cup_sr_folds)
cup_diffs = cup_sr_folds - cup_raw_folds

print(f"  RF(raw): {cup_raw_folds.mean():.4f} ± {cup_raw_folds.std():.4f}")
print(f"  RF(SR):  {cup_sr_folds.mean():.4f} ± {cup_sr_folds.std():.4f}")
print(f"  Diff:    {cup_diffs.mean():+.4f} ± {cup_diffs.std():.4f}")

# TOST for Cuprite
n = len(cup_diffs)
mean_diff = cup_diffs.mean()
se = cup_diffs.std(ddof=1) / np.sqrt(n)
epsilon = 0.02
t_upper = (mean_diff - epsilon) / se
p_upper = stats.t.cdf(t_upper, df=n-1)
t_lower = (mean_diff + epsilon) / se
p_lower = 1 - stats.t.cdf(t_lower, df=n-1)
p_tost = max(p_upper, p_lower)

results['cuprite_0.02'] = {
    'site': 'Cuprite',
    'epsilon': 0.02,
    'mean_diff': float(mean_diff),
    'p_tost': float(p_tost),
    'equivalent': bool(p_tost < 0.05),
}
print(f"  TOST ε=0.02: p={p_tost:.4f} → {'EQUIVALENT' if p_tost < 0.05 else 'not equivalent'}")

# Save
results['chile_summary'] = {
    'raw_mean': float(auc_raw_folds.mean()),
    'raw_std': float(auc_raw_folds.std()),
    'sr_mean': float(auc_sr_folds.mean()),
    'sr_std': float(auc_sr_folds.std()),
    'diff_mean': float(diffs.mean()),
    'diff_std': float(diffs.std()),
    'paired_t_p': float(p_paired),
    'per_fold_raw': auc_raw_folds.tolist(),
    'per_fold_sr': auc_sr_folds.tolist(),
}

output = RESULTS_DIR / "tost_equivalence.json"
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output}")
