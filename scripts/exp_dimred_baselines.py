#!/usr/bin/env python3
"""
GR2 — PCA(6) and mutual-information top-6 baselines as
dimensionality-reduction references against SR(6).
Evaluated under pixel CV + polygon-disjoint CV + spatial block CV (5 km).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("data/results/dimred_baselines.json")

df = pd.read_csv(CSV)
BANDS = ['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']
X = df[BANDS].values
y = df['class_id'].values
lon = df['lon'].values
lat = df['lat'].values
CLASS_NAMES = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
               4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn"}


# --- Reconstruct polygon IDs (same as GR3) ---
EPS_DEG = 0.001
poly_id = np.full(len(df), -1, dtype=int)
next_id = 0
for c in sorted(np.unique(y)):
    mask = y == c
    coords = np.column_stack([lon[mask], lat[mask]])
    if len(coords) < 2:
        poly_id[mask] = next_id; next_id += 1; continue
    lat_mean = coords[:, 1].mean()
    coords_scaled = coords.copy()
    coords_scaled[:, 0] *= np.cos(np.deg2rad(lat_mean))
    db = DBSCAN(eps=EPS_DEG, min_samples=2).fit(coords_scaled)
    labels = db.labels_
    for lbl in np.unique(labels):
        if lbl == -1:
            for i in np.where(mask)[0][labels == -1]:
                poly_id[i] = next_id; next_id += 1
        else:
            poly_id[np.where(mask)[0][labels == lbl]] = next_id; next_id += 1

# --- SR features (same as paper) ---
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

# --- Classical 7 ---
classical = np.column_stack([
    X[:, b['B11']] / np.maximum(X[:, b['B12']], eps),
    X[:, b['B04']] / np.maximum(X[:, b['B02']], eps),
    X[:, b['B12']] / np.maximum(X[:, b['B8A']], eps),
    (X[:, b['B11']] - X[:, b['B12']]) / np.maximum(X[:, b['B11']] + X[:, b['B12']], eps),
    X[:, b['B02']] / np.maximum(X[:, b['B11']], eps),
    X[:, b['B12']] / np.maximum(X[:, b['B11']], eps),
    (X[:, b['B8A']] - X[:, b['B04']]) / np.maximum(X[:, b['B8A']] + X[:, b['B04']], eps),
])


def eval_cv(Xf, y, splits, fit_transform_fn=None):
    """fit_transform_fn is called with (X_train, y_train) → returns (X_train_t, transform_fn)
       used for fold-level PCA / MI (avoiding leakage)."""
    fm, fp, fb = [], [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        if fit_transform_fn is not None:
            Xtr, fn = fit_transform_fn(Xf[tr], y[tr])
            Xte = fn(Xf[te])
        else:
            Xtr, Xte = Xf[tr], Xf[te]
        rf = RandomForestClassifier(200, max_depth=15, random_state=42,
                                    n_jobs=-1, class_weight='balanced')
        rf.fit(Xtr, y[tr])
        proba = rf.predict_proba(Xte)
        pc = {}
        for c in np.unique(y):
            yb = (y[te] == c).astype(int)
            if yb.sum() > 0 and yb.sum() < len(yb) and c in rf.classes_:
                idx = list(rf.classes_).index(c)
                pc[int(c)] = float(roc_auc_score(yb, proba[:, idx]))
        if pc:
            fm.append(float(np.mean(list(pc.values()))))
            fp.append(pc)
            fb.append(float(balanced_accuracy_score(y[te], rf.predict(Xte))))
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


def pca6_fit_transform(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    pca = PCA(n_components=6, random_state=42).fit(X_train_s)
    return pca.transform(X_train_s), lambda X: pca.transform(scaler.transform(X))


def mi_top6_fit_transform(X_train, y_train):
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    top6 = np.argsort(mi)[-6:]
    return X_train[:, top6], lambda X: X[:, top6]


# --- Splits ---
SPLITS = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SPLITS['pixel_5fold'] = list(skf.split(X, y))

gkf = GroupKFold(n_splits=5)
SPLITS['polygon_disjoint'] = list(gkf.split(X, y, groups=poly_id))

DEG_KM_LAT, DEG_KM_LON = 1/111.0, 1/99.0
block_km = 5
bl = np.floor((lon - lon.min()) / (block_km * DEG_KM_LON)).astype(int)
bla = np.floor((lat - lat.min()) / (block_km * DEG_KM_LAT)).astype(int)
groups = bl * 100000 + bla
SPLITS[f'block_{block_km}km'] = list(GroupKFold(n_splits=5).split(X, y, groups=groups))


# --- Feature sets. For PCA/MI we pass the 10 bands + a fit_transform fn ---
FSETS = {
    'raw_10':         (X,        None),
    'sr_6':           (sr_feat,  None),
    'classical_7':    (classical, None),
    'sr_plus_cl_13':  (np.hstack([sr_feat, classical]), None),
    'pca_6':          (X,        pca6_fit_transform),
    'mi_top6':        (X,        mi_top6_fit_transform),
}


RESULTS = {}
for split_name, splits in SPLITS.items():
    print(f"\n{'='*70}\n{split_name}\n{'='*70}")
    RESULTS[split_name] = {}
    for fname, (Xf, fn) in FSETS.items():
        fm, fp, fb = eval_cv(Xf, y, splits, fit_transform_fn=fn)
        s = summarize(fm, fp, fb)
        RESULTS[split_name][fname] = s
        print(f"  {fname:18s}: mAUC = {s['mauc_mean']:.4f} ± {s['mauc_std']:.4f}")


print(f"\n{'='*80}\nSUMMARY — mAUC by feature set × validation scheme\n{'='*80}")
print(f"{'Feature set':<18} " + " ".join(f"{k:>18s}" for k in SPLITS))
for fname in FSETS:
    row = f"{fname:<18} "
    for sn in SPLITS:
        s = RESULTS[sn][fname]
        row += f"{s['mauc_mean']:.4f} ± {s['mauc_std']:.4f}   "
    print(row)

print(f"\n{'='*80}\nΔ(SR_6 vs PCA_6) and Δ(SR_6 vs MI_top6)\n{'='*80}")
for sn in SPLITS:
    sr = RESULTS[sn]['sr_6']['mauc_mean']
    pca = RESULTS[sn]['pca_6']['mauc_mean']
    mi = RESULTS[sn]['mi_top6']['mauc_mean']
    raw = RESULTS[sn]['raw_10']['mauc_mean']
    print(f"  {sn:18s}: SR {sr:.4f}  PCA {pca:.4f} (Δ {sr-pca:+.4f})  "
          f"MI {mi:.4f} (Δ {sr-mi:+.4f})  raw10 {raw:.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nSaved → {OUT}")
