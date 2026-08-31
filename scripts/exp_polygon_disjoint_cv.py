#!/usr/bin/env python3
"""
GR3 extension — Polygon-disjoint CV via DBSCAN reconstruction of polygon IDs.
The CSV has no polygon_id, but pixels were sampled from compact Atlas polygons.
We reconstruct approximate polygon IDs by spatially clustering pixels within
each class using DBSCAN at ~100 m (below typical inter-polygon distance but
above S2's 20 m pixel size), then run GroupKFold on those clusters.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("data/results/polygon_disjoint_cv.json")

df = pd.read_csv(CSV)
BANDS = ['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']
X = df[BANDS].values
y = df['class_id'].values
lon = df['lon'].values
lat = df['lat'].values

CLASS_NAMES = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
               4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn"}

# ---- Reconstruct polygon IDs via DBSCAN within each class ----
# eps in degrees, haversine scale: 100 m at -27° lat ≈ 0.0009° in lon, 0.0009° in lat
# Use Euclidean on (lon, lat) with approx scaling; 0.001° is roughly 100 m.
EPS_DEG = 0.001  # ~110 m, safely above 20 m S2 pixel and below most inter-polygon gaps
poly_id = np.full(len(df), -1, dtype=int)
next_id = 0
for c in sorted(np.unique(y)):
    mask = y == c
    coords = np.column_stack([lon[mask], lat[mask]])
    if len(coords) < 2:
        poly_id[mask] = next_id
        next_id += 1
        continue
    # Scale lon by cos(lat) to make Euclidean distance approximately spatial
    lat_mean = coords[:, 1].mean()
    coords_scaled = coords.copy()
    coords_scaled[:, 0] *= np.cos(np.deg2rad(lat_mean))
    db = DBSCAN(eps=EPS_DEG, min_samples=2, metric='euclidean').fit(coords_scaled)
    labels = db.labels_
    # Assign unique poly_id per (class, cluster); singletons (-1) become unique polys
    for lbl in np.unique(labels):
        if lbl == -1:
            # singletons — each one its own polygon
            sing_idx = np.where(mask)[0][labels == -1]
            for i in sing_idx:
                poly_id[i] = next_id
                next_id += 1
        else:
            idxs = np.where(mask)[0][labels == lbl]
            poly_id[idxs] = next_id
            next_id += 1

n_polys = len(np.unique(poly_id))
print(f"Reconstructed {n_polys} polygons via DBSCAN (eps={EPS_DEG}° ≈ 110 m)")
# Distribution of pixels per polygon
sizes = pd.Series(poly_id).value_counts()
print(f"  Pixels per polygon: min={sizes.min()}, median={sizes.median():.0f}, mean={sizes.mean():.1f}, max={sizes.max()}")
print(f"  Polygons with only 1 pixel (singletons): {(sizes == 1).sum()}")

# Polygons per class
print("\nPolygons per class:")
for c in sorted(np.unique(y)):
    class_mask = y == c
    class_polys = len(np.unique(poly_id[class_mask]))
    print(f"  {CLASS_NAMES.get(c, str(c)):25s}: {class_polys} polygons, {class_mask.sum()} pixels")


# ---- Feature sets ----
eps = 1e-6
b = {n: i for i, n in enumerate(BANDS)}
sr_feat = np.column_stack([
    X[:, b['B04']] - 0.135,
    0.83 - X[:, b['B02']] / np.maximum(X[:, b['B05']], eps),
    0.09 / np.maximum(X[:, b['B05']], eps),
    X[:, b['B03']] - 0.48 * X[:, b['B11']],
    (np.sqrt(np.maximum(X[:, b['B12']], 0)) - X[:, b['B11']]) ** 2,
    X[:, b['B03']] * X[:, b['B12']] / np.maximum(X[:, b['B07']]**2, eps) - 0.45,
])
classical = np.column_stack([
    X[:, b['B11']] / np.maximum(X[:, b['B12']], eps),
    X[:, b['B04']] / np.maximum(X[:, b['B02']], eps),
    X[:, b['B12']] / np.maximum(X[:, b['B8A']], eps),
    (X[:, b['B11']] - X[:, b['B12']]) / np.maximum(X[:, b['B11']] + X[:, b['B12']], eps),
    X[:, b['B02']] / np.maximum(X[:, b['B11']], eps),
    X[:, b['B12']] / np.maximum(X[:, b['B11']], eps),
    (X[:, b['B8A']] - X[:, b['B04']]) / np.maximum(X[:, b['B8A']] + X[:, b['B04']], eps),
])
sr_plus = np.hstack([sr_feat, classical])

FEATURE_SETS = {
    'raw_10': X,
    'sr_6': sr_feat,
    'classical_7': classical,
    'sr_plus_classical_13': sr_plus,
}


def eval_cv(Xf, y, splits):
    fm, fp, fb = [], [], []
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
            fm.append(float(np.mean(list(pc.values()))))
            fp.append(pc)
            fb.append(float(balanced_accuracy_score(y[te], rf.predict(Xf[te]))))
    return fm, fp, fb


def summarize(fm, fp, fb):
    pc_means = {}
    if fp:
        for c in fp[0].keys():
            pc_means[CLASS_NAMES.get(c, str(c))] = float(np.mean([f.get(c, np.nan) for f in fp]))
    return {
        'mauc_mean': float(np.mean(fm)), 'mauc_std': float(np.std(fm)),
        'ba_mean': float(np.mean(fb)), 'ba_std': float(np.std(fb)),
        'per_class_auc': pc_means, 'n_folds': len(fm),
    }


# Polygon-disjoint 5-fold GroupKFold
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups=poly_id))

print(f"\n{'='*70}\nPolygon-disjoint GroupKFold(5) on {n_polys} reconstructed polygons\n{'='*70}")
RESULTS = {'n_polygons': int(n_polys), 'eps_deg': EPS_DEG, 'eps_meters_approx': 110,
           'feature_sets': {}}
for fname, Xf in FEATURE_SETS.items():
    fm, fp, fb = eval_cv(Xf, y, splits)
    s = summarize(fm, fp, fb)
    RESULTS['feature_sets'][fname] = s
    print(f"  {fname:25s}: mAUC = {s['mauc_mean']:.4f} ± {s['mauc_std']:.4f}  BA = {s['ba_mean']:.4f}")

# Per-class for the key feature sets
print("\nPer-class AUC (polygon-disjoint CV):")
for fname in ['raw_10', 'sr_6', 'sr_plus_classical_13']:
    print(f"\n  {fname}:")
    for cname, auc in RESULTS['feature_sets'][fname]['per_class_auc'].items():
        print(f"    {cname:25s}: {auc:.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nSaved → {OUT}")
