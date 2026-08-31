#!/usr/bin/env python3
"""
GR3 — Spatial block CV to quantify autocorrelation inflation.
Compares pixel-level StratifiedKFold(5) vs spatial GroupKFold with
block sizes of 1 km and 5 km, for the 4 feature sets of the paper:
raw, SR, classical, SR + classical.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("data/results/spatial_block_cv.json")

df = pd.read_csv(CSV)
BANDS = ['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']
X = df[BANDS].values
y = df['class_id'].values
lon = df['lon'].values
lat = df['lat'].values

CLASS_NAMES = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
               4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn"}

eps = 1e-6
b = {n: i for i, n in enumerate(BANDS)}

# SR features
sr_feat = np.column_stack([
    X[:, b['B04']] - 0.135,
    0.83 - X[:, b['B02']] / np.maximum(X[:, b['B05']], eps),
    0.09 / np.maximum(X[:, b['B05']], eps),
    X[:, b['B03']] - 0.48 * X[:, b['B11']],
    (np.sqrt(np.maximum(X[:, b['B12']], 0)) - X[:, b['B11']]) ** 2,
    X[:, b['B03']] * X[:, b['B12']] / np.maximum(X[:, b['B07']]**2, eps) - 0.45,
])

# 7 classical indices (paper methods §3.5)
classical = np.column_stack([
    X[:, b['B11']] / np.maximum(X[:, b['B12']], eps),                              # Clay Ratio
    X[:, b['B04']] / np.maximum(X[:, b['B02']], eps),                              # Iron Oxide
    X[:, b['B12']] / np.maximum(X[:, b['B8A']], eps),                              # Ferrous
    (X[:, b['B11']] - X[:, b['B12']]) / np.maximum(X[:, b['B11']] + X[:, b['B12']], eps),  # Alunite
    X[:, b['B02']] / np.maximum(X[:, b['B11']], eps),                              # OH Minerals
    X[:, b['B12']] / np.maximum(X[:, b['B11']], eps),                              # Silica
    (X[:, b['B8A']] - X[:, b['B04']]) / np.maximum(X[:, b['B8A']] + X[:, b['B04']], eps),  # NDVI
])

sr_plus_classical = np.hstack([sr_feat, classical])

FEATURE_SETS = {
    'raw_10': X,
    'sr_6': sr_feat,
    'classical_7': classical,
    'sr_plus_classical_13': sr_plus_classical,
}


def eval_cv(Xf, y, splits):
    """Returns per-fold mean-AUC, per-fold per-class AUCs, per-fold BA."""
    fold_mauc = []
    fold_pc = []
    fold_ba = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        rf = RandomForestClassifier(200, max_depth=15, random_state=42,
                                    n_jobs=-1, class_weight='balanced')
        rf.fit(Xf[tr], y[tr])
        proba = rf.predict_proba(Xf[te])
        pc = {}
        for c in np.unique(y):
            yb = (y[te] == c).astype(int)
            if yb.sum() > 0 and yb.sum() < len(yb) and c in rf.classes_:
                idx = list(rf.classes_).index(c)
                pc[int(c)] = float(roc_auc_score(yb, proba[:, idx]))
        if pc:
            fold_mauc.append(float(np.mean(list(pc.values()))))
            fold_pc.append(pc)
            fold_ba.append(float(balanced_accuracy_score(y[te], rf.predict(Xf[te]))))
    return fold_mauc, fold_pc, fold_ba


def summarize(fold_mauc, fold_pc, fold_ba):
    pc_means = {}
    if fold_pc:
        for c in fold_pc[0].keys():
            pc_means[CLASS_NAMES.get(c, str(c))] = float(np.mean([f.get(c, np.nan) for f in fold_pc]))
    return {
        'mauc_mean': float(np.mean(fold_mauc)),
        'mauc_std':  float(np.std(fold_mauc)),
        'ba_mean':   float(np.mean(fold_ba)),
        'ba_std':    float(np.std(fold_ba)),
        'per_class_auc': pc_means,
        'n_folds': len(fold_mauc),
    }


# Build splits
SPLITS = {}

# Pixel-level stratified 5-fold CV (paper baseline)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SPLITS['pixel_stratified_5fold'] = {
    'splits': list(skf.split(X, y)),
    'description': 'Pixel-level StratifiedKFold(5) — paper baseline, pixels shuffled across folds',
    'n_groups': len(X),
}

# Spatial block GroupKFold
DEG_KM_LAT = 1 / 111.0
DEG_KM_LON = 1 / 99.0
for block_km in [1, 2, 5, 10]:
    bl = np.floor((lon - lon.min()) / (block_km * DEG_KM_LON)).astype(int)
    bla = np.floor((lat - lat.min()) / (block_km * DEG_KM_LAT)).astype(int)
    groups = bl * 100000 + bla
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        continue
    gkf = GroupKFold(n_splits=n_splits)
    SPLITS[f'spatial_block_{block_km}km_groupkfold'] = {
        'splits': list(gkf.split(X, y, groups=groups)),
        'description': f'Spatial GroupKFold({n_splits}) with {block_km} km blocks; train/test blocks are disjoint',
        'n_groups': int(n_groups),
    }


# Run evaluation
RESULTS = {}
for split_name, spec in SPLITS.items():
    print(f"\n{'='*70}\n{split_name}\n  {spec['description']}\n  {spec['n_groups']} groups\n{'='*70}")
    RESULTS[split_name] = {'description': spec['description'], 'n_groups': spec['n_groups'], 'feature_sets': {}}
    for feat_name, Xf in FEATURE_SETS.items():
        fold_mauc, fold_pc, fold_ba = eval_cv(Xf, y, spec['splits'])
        summary = summarize(fold_mauc, fold_pc, fold_ba)
        RESULTS[split_name]['feature_sets'][feat_name] = summary
        print(f"  {feat_name:25s}: mAUC = {summary['mauc_mean']:.4f} ± {summary['mauc_std']:.4f}  BA = {summary['ba_mean']:.4f}")

# Summary table
print(f"\n{'='*70}\nSUMMARY TABLE — mAUC ± std across validation schemes\n{'='*70}")
print(f"{'Feature set':<25} " + " ".join(f"{n:>20s}" for n in SPLITS.keys()))
for feat_name in FEATURE_SETS.keys():
    row = f"{feat_name:<25} "
    for split_name in SPLITS.keys():
        s = RESULTS[split_name]['feature_sets'][feat_name]
        row += f"{s['mauc_mean']:.4f}±{s['mauc_std']:.4f}      "
    print(row)

# Delta vs pixel baseline
print(f"\n{'='*70}\nΔ mAUC vs pixel-level baseline (autocorrelation inflation)\n{'='*70}")
baseline = RESULTS['pixel_stratified_5fold']['feature_sets']
print(f"{'Feature set':<25} " + " ".join(f"{k:>18s}" for k in SPLITS if k != 'pixel_stratified_5fold'))
for feat_name in FEATURE_SETS.keys():
    base_auc = baseline[feat_name]['mauc_mean']
    row = f"{feat_name:<25} "
    for split_name in SPLITS:
        if split_name == 'pixel_stratified_5fold':
            continue
        d = RESULTS[split_name]['feature_sets'][feat_name]['mauc_mean'] - base_auc
        row += f"{d:>+18.4f}    "
    print(row)

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nSaved → {OUT}")
