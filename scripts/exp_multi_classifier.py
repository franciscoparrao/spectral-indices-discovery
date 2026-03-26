#!/usr/bin/env python3
"""
Robustness check: SR features vs raw bands across multiple classifiers.
RF, XGBoost, SVM-RBF, SVM-Linear.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")

# Load data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X_raw = data['X']
y = data['y']
band_names = list(data['band_names'])
B = {name: i for i, name in enumerate(band_names)}

eps = 1e-6

# Compute SR features
sr_features = np.column_stack([
    X_raw[:, B['B04']] - 0.135,
    0.83 - X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B05']], eps),
    0.09 / np.maximum(X_raw[:, B['B05']], eps),
    X_raw[:, B['B03']] - 0.48 * X_raw[:, B['B11']],
    (np.sqrt(X_raw[:, B['B12']]) - X_raw[:, B['B11']]) ** 2,
    X_raw[:, B['B03']] * X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B07']]**2, eps) - 0.45,
])
sr_names = ['SR_Sil', 'SR_AdvArg', 'SR_Arg', 'SR_Prop', 'SR_IronOx', 'SR_Pot']

# SR + classical
classical_features = np.column_stack([
    X_raw[:, B['B11']] / np.maximum(X_raw[:, B['B12']], eps),
    X_raw[:, B['B04']] / np.maximum(X_raw[:, B['B02']], eps),
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B8A']], eps),
    (X_raw[:, B['B11']] - X_raw[:, B['B12']]) / np.maximum(X_raw[:, B['B11']] + X_raw[:, B['B12']], eps),
    X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B11']], eps),
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B11']], eps),
    (X_raw[:, B['B8A']] - X_raw[:, B['B04']]) / np.maximum(X_raw[:, B['B8A']] + X_raw[:, B['B04']], eps),
])

sr_classical = np.column_stack([sr_features, classical_features])

feature_sets = {
    '10 raw bands': X_raw,
    '6 SR indices': sr_features,
    '6 SR + 7 classical': sr_classical,
}

# Classifiers with reasonable defaults (no exhaustive tuning)
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("XGBoost not available, skipping")

classifiers = {
    'RF': RandomForestClassifier(200, max_depth=15, random_state=42,
                                  n_jobs=-1, class_weight='balanced'),
    'SVM-RBF': Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                     class_weight='balanced', random_state=42)),
    ]),
    'SVM-Linear': Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1, probability=True,
                     class_weight='balanced', random_state=42)),
    ]),
}

if has_xgb:
    classifiers['XGBoost'] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric='mlogloss',
        use_label_encoder=False,
    )

class_ids = sorted(np.unique(y))
CLASS_NAMES = {1: "Silic", 2: "AdvArg", 3: "ArgPh", 4: "Prop", 5: "IrOx", 6: "PotSk"}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_results = {}

for clf_name, clf_template in classifiers.items():
    print(f"\n{'='*70}")
    print(f"CLASSIFIER: {clf_name}")
    print(f"{'='*70}")

    for feat_name, X_set in feature_sets.items():
        from sklearn.base import clone
        fold_aucs = {c: [] for c in class_ids}
        fold_ba = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_set, y)):
            X_tr, X_te = X_set[train_idx], X_set[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # For XGBoost, remap labels to 0-based
            if clf_name == 'XGBoost':
                label_map = {c: i for i, c in enumerate(class_ids)}
                inv_map = {i: c for c, i in label_map.items()}
                y_tr_mapped = np.array([label_map[v] for v in y_tr])
                y_te_mapped = np.array([label_map[v] for v in y_te])

                clf = clone(clf_template)
                clf.fit(X_tr, y_tr_mapped)
                y_proba = clf.predict_proba(X_te)
                y_pred = clf.predict(X_te)

                ba = balanced_accuracy_score(y_te_mapped, y_pred)
                fold_ba.append(ba)

                for i, c in enumerate(class_ids):
                    y_bin = (y_te_mapped == label_map[c]).astype(int)
                    if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                        auc = roc_auc_score(y_bin, y_proba[:, label_map[c]])
                        fold_aucs[c].append(auc)
            else:
                clf = clone(clf_template)
                clf.fit(X_tr, y_tr)
                y_proba = clf.predict_proba(X_te)
                y_pred = clf.predict(X_te)

                ba = balanced_accuracy_score(y_te, y_pred)
                fold_ba.append(ba)

                # Get class order from classifier
                if hasattr(clf, 'classes_'):
                    classes = clf.classes_
                elif hasattr(clf, 'named_steps'):
                    classes = clf.named_steps['svm'].classes_
                else:
                    classes = class_ids

                for c in class_ids:
                    y_bin = (y_te == c).astype(int)
                    if c in classes:
                        idx = list(classes).index(c)
                        auc = roc_auc_score(y_bin, y_proba[:, idx])
                        fold_aucs[c].append(auc)

        key = f"{clf_name} | {feat_name}"
        mean_auc = np.mean([np.mean(fold_aucs[c]) for c in class_ids])
        result = {
            'classifier': clf_name,
            'features': feat_name,
            'n_features': X_set.shape[1],
            'balanced_accuracy': float(np.mean(fold_ba)),
            'ba_std': float(np.std(fold_ba)),
            'mean_auc': float(mean_auc),
            'per_class_auc': {
                str(c): float(np.mean(fold_aucs[c])) for c in class_ids
            },
        }
        all_results[key] = result

        per_class_str = "  ".join(f"{CLASS_NAMES[c]}:{np.mean(fold_aucs[c]):.3f}" for c in class_ids)
        print(f"  {feat_name:25s} BA={np.mean(fold_ba):.3f}  mAUC={mean_auc:.3f}  {per_class_str}")

# ===== Summary table =====
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")

header = f"{'Classifier':<12} {'Features':<25} {'BA':>5} {'mAUC':>6}"
for c in class_ids:
    header += f" {CLASS_NAMES[c]:>6}"
print(header)
print("-" * len(header))

for key, r in all_results.items():
    row = f"{r['classifier']:<12} {r['features']:<25} {r['balanced_accuracy']:>5.3f} {r['mean_auc']:>6.3f}"
    for c in class_ids:
        row += f" {r['per_class_auc'][str(c)]:>6.3f}"
    print(row)

# Save
output = RESULTS_DIR / "multi_classifier.json"
with open(output, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {output}")
