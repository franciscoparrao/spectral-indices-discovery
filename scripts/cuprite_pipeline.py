#!/usr/bin/env python3
"""
Cuprite, Nevada — Third validation site pipeline.
1. Download alteration layers from USGS WMS
2. Download Sentinel-2 via GEE
3. Apply SR indices
4. Evaluate against USGS ground truth
"""

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import json
import requests
from PIL import Image
from io import BytesIO

ee.Initialize()

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")
S2_DIR = Path("data/sentinel2/cuprite")
S2_DIR.mkdir(parents=True, exist_ok=True)

# Cuprite area (Esmeralda County, Nevada)
# Bounding box in EPSG:4326
CUPRITE_BBOX = {
    'west': -117.28,
    'south': 37.47,
    'east': -117.10,
    'north': 37.60,
}

# WMS configuration
WMS_BASE = "https://mrdata.usgs.gov/services/alteration"
WMS_LAYERS = {
    'argillic': 2,       # Maps to our Adv_Argillic / Argillic
    'phyllic': 3,        # Maps to our Argillic_Phyllic
    'carbonate': 4,      # Propylitic (calcite)
    'epi_chlor': 4,      # Propylitic (epidote, chlorite) — merge with carbonate
    'hydro_silica': 1,   # Silicic
}

# Our class mapping
CLASS_NAMES = {
    0: "Unaltered",
    1: "Silicic",
    2: "Adv_Argillic",
    3: "Argillic_Phyllic",
    4: "Propylitic",
}

# Resolution for rasterization (match S2 at 20m → ~0.0002° at this latitude)
PIXEL_SIZE = 0.0002  # ~20m at 37.5°N
WIDTH = int((CUPRITE_BBOX['east'] - CUPRITE_BBOX['west']) / PIXEL_SIZE)
HEIGHT = int((CUPRITE_BBOX['north'] - CUPRITE_BBOX['south']) / PIXEL_SIZE)

print(f"Cuprite grid: {WIDTH} x {HEIGHT} pixels at ~20m")
print(f"Bounding box: {CUPRITE_BBOX}")


# ===== STEP 1: Download alteration layers from USGS WMS =====
print("\n" + "=" * 70)
print("STEP 1: Download USGS alteration layers via WMS")
print("=" * 70)

# Download each layer as a high-res PNG and convert to binary mask
alteration_masks = {}

for layer_name, class_id in WMS_LAYERS.items():
    print(f"  Downloading {layer_name} (class {class_id})...")

    # WMS GetMap request — use full resolution
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',
        'REQUEST': 'GetMap',
        'LAYERS': layer_name,
        'CRS': 'EPSG:4326',
        'BBOX': f"{CUPRITE_BBOX['south']},{CUPRITE_BBOX['west']},{CUPRITE_BBOX['north']},{CUPRITE_BBOX['east']}",
        'WIDTH': min(WIDTH, 2048),   # WMS may limit size
        'HEIGHT': min(HEIGHT, 2048),
        'FORMAT': 'image/png',
        'TRANSPARENT': 'TRUE',
    }

    resp = requests.get(WMS_BASE, params=params, timeout=60)
    if resp.status_code != 200:
        print(f"    ERROR: HTTP {resp.status_code}")
        continue

    img = Image.open(BytesIO(resp.content)).convert('RGBA')
    arr = np.array(img)

    # The alteration pixels are non-transparent (alpha > 0)
    mask = arr[:, :, 3] > 128  # alpha channel
    n_pixels = mask.sum()
    print(f"    {layer_name}: {n_pixels} altered pixels ({100*n_pixels/mask.size:.1f}%)")

    if layer_name in alteration_masks and class_id in [c for c in alteration_masks.values()]:
        # Merge with existing mask for same class (e.g., carbonate + epi_chlor → propylitic)
        for existing_layer, existing_class in list(alteration_masks.items()):
            if isinstance(existing_class, np.ndarray):
                continue
        # Store raw mask
        alteration_masks[layer_name] = mask
    else:
        alteration_masks[layer_name] = mask

# Build combined class raster (priority: silicic > argillic > phyllic > propylitic)
print("\n  Building class raster...")
gt_raster = np.zeros((min(HEIGHT, 2048), min(WIDTH, 2048)), dtype=np.uint8)

# Apply in order of priority (lower priority first, higher overwrites)
for layer_name in ['epi_chlor', 'carbonate', 'phyllic', 'argillic', 'hydro_silica']:
    if layer_name in alteration_masks:
        mask = alteration_masks[layer_name]
        class_id = WMS_LAYERS[layer_name]
        gt_raster[mask] = class_id

# Count pixels per class
for cid, cname in CLASS_NAMES.items():
    n = (gt_raster == cid).sum()
    print(f"    Class {cid} ({cname}): {n} pixels")

# Save ground truth raster
gt_transform = from_bounds(
    CUPRITE_BBOX['west'], CUPRITE_BBOX['south'],
    CUPRITE_BBOX['east'], CUPRITE_BBOX['north'],
    gt_raster.shape[1], gt_raster.shape[0]
)

gt_path = GT_DIR / "cuprite_ground_truth.tif"
with rasterio.open(gt_path, 'w', driver='GTiff',
                   height=gt_raster.shape[0], width=gt_raster.shape[1],
                   count=1, dtype='uint8', crs='EPSG:4326',
                   transform=gt_transform) as dst:
    dst.write(gt_raster, 1)
print(f"  Saved: {gt_path}")


# ===== STEP 2: Download Sentinel-2 via GEE =====
print("\n" + "=" * 70)
print("STEP 2: Download Sentinel-2 via GEE")
print("=" * 70)

aoi = ee.Geometry.Rectangle([
    CUPRITE_BBOX['west'], CUPRITE_BBOX['south'],
    CUPRITE_BBOX['east'], CUPRITE_BBOX['north']
])

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_RENAME = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
               "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
               "B11": "B11", "B12": "B12"}

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(aoi)
      .filterDate("2024-06-01", "2024-09-30")  # Summer for Nevada
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
      .select(S2_BANDS))

n_scenes = s2.size().getInfo()
print(f"  S2 scenes: {n_scenes}")

composite = s2.median().divide(10000).rename([BAND_RENAME[b] for b in S2_BANDS])

# Sample pixels at ground truth locations
# For each alteration class, sample N pixels
print("  Sampling pixels within alteration zones...")

# Create a grid of sample points across the AOI
sample_points = composite.sample(
    region=aoi,
    scale=20,
    numPixels=20000,
    seed=42,
    geometries=True,
)

# Download sample data
print("  Downloading sampled pixels...")
sample_size = sample_points.size().getInfo()
print(f"    Total samples available: {sample_size}")

all_rows = []
batch = 1000
for start in range(0, min(sample_size, 20000), batch):
    feats = sample_points.toList(min(batch, sample_size - start), start).getInfo()
    for feat in feats:
        props = feat.get('properties', {})
        geom = feat.get('geometry', {})
        if geom and 'coordinates' in geom:
            props['lon'] = geom['coordinates'][0]
            props['lat'] = geom['coordinates'][1]
        all_rows.append(props)
    print(f"    Downloaded {len(all_rows)} / {min(sample_size, 20000)}")

df_s2 = pd.DataFrame(all_rows)
band_cols = sorted([c for c in df_s2.columns if c.startswith("B") and c not in ('B08',) and len(c) <= 3])
# Include B08
band_cols = sorted([c for c in df_s2.columns if c.startswith("B")])
# Keep only our standard band names
keep_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
band_cols = [c for c in keep_bands if c in df_s2.columns]
df_s2 = df_s2.dropna(subset=band_cols)
print(f"  S2 pixels with valid data: {len(df_s2)}")


# ===== STEP 3: Assign ground truth labels to S2 pixels =====
print("\n" + "=" * 70)
print("STEP 3: Assign ground truth labels")
print("=" * 70)

# For each S2 pixel, look up the class in the ground truth raster
with rasterio.open(gt_path) as src:
    gt_data = src.read(1)
    gt_tf = src.transform

labels = []
for _, row in df_s2.iterrows():
    # Convert lat/lon to pixel coordinates in ground truth raster
    col, row_idx = ~gt_tf * (row['lon'], row['lat'])
    col, row_idx = int(col), int(row_idx)
    if 0 <= row_idx < gt_data.shape[0] and 0 <= col < gt_data.shape[1]:
        labels.append(gt_data[row_idx, col])
    else:
        labels.append(0)

df_s2['class_id'] = labels

print("  Class distribution:")
for cid, cname in CLASS_NAMES.items():
    n = (df_s2['class_id'] == cid).sum()
    print(f"    {cname}: {n}")


# ===== STEP 4: Apply SR indices and evaluate =====
print("\n" + "=" * 70)
print("STEP 4: Evaluate SR indices on Cuprite")
print("=" * 70)

eps = 1e-6
X_cuprite = df_s2[band_cols].values
y_cuprite = df_s2['class_id'].values

# Filter to pixels with alteration labels (class > 0) + unaltered (class == 0)
# Need at least some altered pixels
altered_mask = y_cuprite > 0
n_altered = altered_mask.sum()
n_unaltered = (~altered_mask).sum()
print(f"  Altered: {n_altered}, Unaltered: {n_unaltered}")

if n_altered < 50:
    print("  WARNING: Very few altered pixels. Results may be unreliable.")

# Band lookup
B = {name: i for i, name in enumerate(band_cols)}

# SR formulas
def sr_silicic(x): return x[:, B['B04']] - 0.135
def sr_adv_argillic(x): return 0.83 - x[:, B['B02']] / np.maximum(x[:, B['B05']], eps)
def sr_argillic(x): return 0.09 / np.maximum(x[:, B['B05']], eps)
def sr_propylitic(x): return x[:, B['B03']] - 0.48 * x[:, B['B11']]
def sr_iron_oxide(x): return (np.sqrt(x[:, B['B12']]) - x[:, B['B11']]) ** 2
def sr_potassic(x): return x[:, B['B03']] * x[:, B['B12']] / np.maximum(x[:, B['B07']]**2, eps) - 0.45

# Classical indices
def clay_ratio(x): return x[:, B['B11']] / np.maximum(x[:, B['B12']], eps)
def iron_oxide_idx(x): return x[:, B['B04']] / np.maximum(x[:, B['B02']], eps)

SR_INDICES = {
    'SR_Silicic': (sr_silicic, 1),
    'SR_Adv_Argillic': (sr_adv_argillic, 2),
    'SR_Argillic': (sr_argillic, 3),
    'SR_Propylitic': (sr_propylitic, 4),
}

CLASSICAL = {
    'Clay_Ratio': clay_ratio,
    'Iron_Oxide': iron_oxide_idx,
}

# Evaluate standalone indices
print("\n  Standalone SR indices (one-vs-rest AUC):")
standalone_results = {}

for name, (func, target_class) in SR_INDICES.items():
    if target_class not in np.unique(y_cuprite) or (y_cuprite == target_class).sum() < 10:
        print(f"    {name}: SKIPPED (insufficient samples for class {target_class})")
        continue

    scores = func(X_cuprite)
    y_bin = (y_cuprite == target_class).astype(int)
    scores_clean = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)

    auc = roc_auc_score(y_bin, scores_clean)
    auc_neg = roc_auc_score(y_bin, -scores_clean)
    auc = max(auc, auc_neg)

    standalone_results[name] = {'auc': float(auc), 'target_class': target_class}
    print(f"    {name}: AUC = {auc:.3f} (n_pos={y_bin.sum()})")

# Classical indices per class
print("\n  Classical indices:")
for idx_name, func in CLASSICAL.items():
    scores = func(X_cuprite)
    for target_class in [1, 2, 3, 4]:
        if (y_cuprite == target_class).sum() < 10:
            continue
        y_bin = (y_cuprite == target_class).astype(int)
        scores_clean = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
        auc = max(roc_auc_score(y_bin, scores_clean),
                  roc_auc_score(y_bin, -scores_clean))
        key = f"{idx_name}_class{target_class}"
        standalone_results[key] = {'auc': float(auc), 'target_class': target_class}
        print(f"    {key}: AUC = {auc:.3f}")


# ===== STEP 5: SR feature engineering evaluation =====
print("\n" + "=" * 70)
print("STEP 5: SR as feature engineering (Cuprite)")
print("=" * 70)

# Compute SR features
sr_features = np.column_stack([
    sr_silicic(X_cuprite),
    sr_adv_argillic(X_cuprite),
    sr_argillic(X_cuprite),
    sr_propylitic(X_cuprite),
    sr_iron_oxide(X_cuprite),
    sr_potassic(X_cuprite),
])

# Classical features
classical_features = np.column_stack([
    clay_ratio(X_cuprite),
    iron_oxide_idx(X_cuprite),
    X_cuprite[:, B['B12']] / np.maximum(X_cuprite[:, B['B8A']], eps),
    (X_cuprite[:, B['B11']] - X_cuprite[:, B['B12']]) / np.maximum(X_cuprite[:, B['B11']] + X_cuprite[:, B['B12']], eps),
    X_cuprite[:, B['B02']] / np.maximum(X_cuprite[:, B['B11']], eps),
    X_cuprite[:, B['B12']] / np.maximum(X_cuprite[:, B['B11']], eps),
    (X_cuprite[:, B['B8A']] - X_cuprite[:, B['B04']]) / np.maximum(X_cuprite[:, B['B8A']] + X_cuprite[:, B['B04']], eps),
])

sr_classical = np.column_stack([sr_features, classical_features])

# Only evaluate on pixels with class > 0 if we want multi-class,
# or include class 0 for binary
feature_sets = {
    'RF(10 raw)': X_cuprite,
    'RF(6 SR)': sr_features,
    'RF(6SR+7class)': sr_classical,
}

# Use all classes present
present_classes = sorted([c for c in np.unique(y_cuprite) if c > 0 and (y_cuprite == c).sum() >= 10])
print(f"  Classes with ≥10 samples: {[CLASS_NAMES.get(c, c) for c in present_classes]}")

from sklearn.model_selection import StratifiedKFold

# Only evaluate if we have enough samples per class
if len(present_classes) >= 2:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    ensemble_results = {}
    for feat_name, X_feat in feature_sets.items():
        fold_aucs = {c: [] for c in present_classes}
        fold_ba = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y_cuprite)):
            X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
            y_tr, y_te = y_cuprite[train_idx], y_cuprite[test_idx]

            rf = RandomForestClassifier(200, max_depth=15, random_state=42,
                                         n_jobs=-1, class_weight='balanced')
            rf.fit(X_tr, y_tr)
            y_proba = rf.predict_proba(X_te)
            y_pred = rf.predict(X_te)
            fold_ba.append(balanced_accuracy_score(y_te, y_pred))

            for c in present_classes:
                if c in rf.classes_:
                    y_bin = (y_te == c).astype(int)
                    idx = list(rf.classes_).index(c)
                    if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                        auc = roc_auc_score(y_bin, y_proba[:, idx])
                        fold_aucs[c].append(auc)

        mean_auc = np.mean([np.mean(fold_aucs[c]) for c in present_classes if fold_aucs[c]])
        mean_ba = np.mean(fold_ba)

        ensemble_results[feat_name] = {
            'ba': float(mean_ba),
            'mean_auc': float(mean_auc),
            'per_class': {CLASS_NAMES.get(c, str(c)): float(np.mean(fold_aucs[c]))
                          for c in present_classes if fold_aucs[c]},
        }

        per_class_str = "  ".join(f"{CLASS_NAMES.get(c, str(c))}:{np.mean(fold_aucs[c]):.3f}"
                                   for c in present_classes if fold_aucs[c])
        print(f"  {feat_name:20s} BA={mean_ba:.3f}  mAUC={mean_auc:.3f}  {per_class_str}")

else:
    print("  Insufficient classes for multi-class evaluation")
    ensemble_results = {}


# ===== STEP 6: Cross-site transfer (Chile → Cuprite) =====
print("\n" + "=" * 70)
print("STEP 6: Cross-site transfer (Chile III → Cuprite, Nevada)")
print("=" * 70)

# Load Chile training data
chile_data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X_chile = chile_data['X']
y_chile = chile_data['y']
chile_bands = list(chile_data['band_names'])

# Train RF on Chile, predict Cuprite
# Need to match band ordering
chile_B = {name: i for i, name in enumerate(chile_bands)}
cuprite_band_order = [chile_bands.index(b) if b in chile_bands else -1 for b in band_cols]

# Verify band alignment
print(f"  Chile bands: {chile_bands}")
print(f"  Cuprite bands: {band_cols}")

# Reorder Cuprite to match Chile band order
X_cuprite_aligned = np.zeros((len(X_cuprite), len(chile_bands)))
for i, bname in enumerate(chile_bands):
    if bname in band_cols:
        j = band_cols.index(bname)
        X_cuprite_aligned[:, i] = X_cuprite[:, j]

# Train on Chile (all classes), predict on Cuprite
rf_transfer = RandomForestClassifier(200, max_depth=15, random_state=42,
                                      n_jobs=-1, class_weight='balanced')
rf_transfer.fit(X_chile, y_chile)
y_proba_transfer = rf_transfer.predict_proba(X_cuprite_aligned)

print("\n  Cross-site RF transfer (trained Chile, tested Cuprite):")
transfer_results = {}
for c in present_classes:
    if c in rf_transfer.classes_:
        y_bin = (y_cuprite == c).astype(int)
        idx = list(rf_transfer.classes_).index(c)
        if y_bin.sum() >= 10:
            auc = roc_auc_score(y_bin, y_proba_transfer[:, idx])
            transfer_results[CLASS_NAMES.get(c, str(c))] = float(auc)
            print(f"    {CLASS_NAMES.get(c, str(c))}: AUC = {auc:.3f}")


# ===== SAVE ALL RESULTS =====
all_results = {
    'site': 'Cuprite, Nevada, USA',
    'bbox': CUPRITE_BBOX,
    'n_s2_pixels': len(df_s2),
    'class_distribution': {CLASS_NAMES.get(c, str(c)): int((y_cuprite == c).sum())
                           for c in sorted(np.unique(y_cuprite))},
    'ground_truth_source': 'USGS WMS — Hydrothermal alteration in Basin and Range (ASTER-derived)',
    's2_source': 'GEE COPERNICUS/S2_SR_HARMONIZED median Jun-Sep 2024',
    'standalone_indices': standalone_results,
    'ensemble_rf': ensemble_results,
    'cross_site_transfer_chile_to_cuprite': transfer_results,
}

output = RESULTS_DIR / "cuprite_validation.json"
with open(output, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {output}")

# Save training data for potential reuse
np.savez(GT_DIR / "cuprite_training_s2.npz",
         X=X_cuprite, y=y_cuprite, band_names=np.array(band_cols))
print(f"Saved: {GT_DIR / 'cuprite_training_s2.npz'}")
