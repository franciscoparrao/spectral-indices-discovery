#!/usr/bin/env python3
"""
Complete analysis pipeline:
1. Binary: altered vs unaltered
2. Multi-class: alteration types (92 classified polygons)
3. Sub-classify the 544 undifferentiated polygons
"""

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.cluster import KMeans
from pathlib import Path
import json

ee.Initialize()

ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
GT_DIR = Path("data/ground_truth")
EMB_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")
FIG_DIR = Path("figures")

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_RENAME = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
               "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
               "B11": "B11", "B12": "B12"}

ALTERATION_MAP = {
    "Alteración Silicea": 1, "vuggy silica": 1,
    "Alteracion Argilica y Argilica avanzada": 2, "Alteracion Solfatárica": 2,
    "Alteracion Argilica": 3, "Alteracion Sericitica": 3,
    "Alteración Cuarzo-Sericitica(Fílica)": 3,
    "Alteracion Propilitica": 4,
    "Oxidos e Hidróxidos de Hierro": 5,
    "Alteracion Potasica": 6, "skarn": 6,
}

CLASS_NAMES = {
    1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
    4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn",
}


# ===== SR FORMULAS =====
def sr_binary_altered(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    """Composite alteration score: combines the best SR indices."""
    # Use the best cross-site performer: (sqrt(B12)-B11)²
    return (np.sqrt(B12) - B11) ** 2

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


def polygon_to_ee(row):
    geom = row.geometry
    if geom.geom_type == "Polygon":
        coords = [list(c) for c in geom.exterior.coords]
        return ee.Feature(ee.Geometry.Polygon([coords]), {
            "class_id": int(row.get("class_id", 0)),
            "is_altered": int(row.get("is_altered", 1)),
        })
    elif geom.geom_type == "MultiPolygon":
        polys = [[list(c) for c in p.exterior.coords] for p in geom.geoms]
        return ee.Feature(ee.Geometry.MultiPolygon(polys), {
            "class_id": int(row.get("class_id", 0)),
            "is_altered": int(row.get("is_altered", 1)),
        })
    return None


def download_fc(fc, max_features=10000):
    size = fc.size().getInfo()
    if size == 0:
        return pd.DataFrame()
    all_rows = []
    batch = 1000
    for start in range(0, min(size, max_features), batch):
        feats = fc.toList(min(batch, size - start), start).getInfo()
        for feat in feats:
            props = feat.get("properties", {})
            all_rows.append(props)
    return pd.DataFrame(all_rows)


def evaluate_binary(scores, y_bin):
    """AUC + best F1 for binary classification."""
    scores = np.nan_to_num(scores, nan=0)
    auc = roc_auc_score(y_bin, scores)
    auc_inv = roc_auc_score(y_bin, -scores)
    if auc_inv > auc:
        auc = auc_inv
        scores = -scores
    best_f1 = 0
    for t in np.linspace(scores.min(), scores.max(), 50):
        f1 = f1_score(y_bin, (scores > t).astype(int), zero_division=0)
        best_f1 = max(best_f1, f1)
    return auc, best_f1


def main():
    # ===== LOAD POLYGONS =====
    print("Loading Atlas polygons...")
    gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
    attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
    gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
    gdf = gdf.set_crs("EPSG:32719")
    gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)

    # Separate groups
    classified = gdf[gdf["class_id"].isin([1, 2, 3, 4, 5, 6])].copy()
    undifferentiated = gdf[gdf["ALTERACION"] == "Alteracion hidrotermal indiferenciada (arg, lim y sil)"].copy()
    no_info = gdf[gdf["ALTERACION"] == "Sin Informacion"].copy()

    print(f"  Classified: {len(classified)}")
    print(f"  Undifferentiated: {len(undifferentiated)}")
    print(f"  No info: {len(no_info)}")

    # ===== BUILD S2 COMPOSITE =====
    print("\nBuilding S2 composite...")
    chile_aoi = ee.Geometry.Rectangle([-69.6, -28.1, -68.5, -25.9])
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(chile_aoi)
          .filterDate("2024-01-01", "2024-04-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .select(S2_BANDS))
    composite = s2.median().divide(10000).rename([BAND_RENAME[b] for b in S2_BANDS])
    print(f"  S2 scenes: {s2.size().getInfo()}")

    # ===== PHASE 1: BINARY (altered vs unaltered) =====
    print("\n" + "=" * 70)
    print("PHASE 1: BINARY — Altered vs Unaltered")
    print("=" * 70)

    # All alteration polygons (classified + undifferentiated) = altered
    all_altered = pd.concat([classified, undifferentiated], ignore_index=True)
    all_altered["is_altered"] = 1
    all_altered = all_altered.to_crs("EPSG:4326")

    # Sample "unaltered" from OUTSIDE all polygons
    # Use the bounding box but exclude polygon areas
    bounds = all_altered.total_bounds
    bbox = ee.Geometry.Rectangle([bounds[0]-0.1, bounds[1]-0.1, bounds[2]+0.1, bounds[3]+0.1])

    # Create altered FC
    altered_features = [polygon_to_ee(row) for _, row in all_altered.iterrows()
                       if polygon_to_ee(row) is not None]
    altered_fc = ee.FeatureCollection(altered_features)
    altered_region = altered_fc.geometry()

    # Sample altered pixels
    print("  Sampling altered pixels...")
    altered_sampled = composite.sample(
        region=altered_region, scale=20, numPixels=8000, seed=42, geometries=False
    )
    df_altered = download_fc(altered_sampled, max_features=8000)
    df_altered["is_altered"] = 1
    print(f"    Altered: {len(df_altered)} pixels")

    # Sample unaltered pixels from areas away from any alteration polygon
    # Use small boxes offset from the alteration zone (same geological province, no alteration)
    print("  Sampling unaltered pixels (offset regions)...")
    # 4 boxes around the alteration area, ~20km away
    b = all_altered.total_bounds  # [w, s, e, n] in EPSG:4326
    offset = 0.25  # ~25 km
    unaltered_boxes = [
        ee.Geometry.Rectangle([b[0]-offset-0.15, b[1], b[0]-offset, b[1]+0.15]),  # West
        ee.Geometry.Rectangle([b[2]+offset, b[1], b[2]+offset+0.15, b[1]+0.15]),  # East
        ee.Geometry.Rectangle([b[0], b[3]+offset, b[0]+0.15, b[3]+offset+0.15]),  # North
        ee.Geometry.Rectangle([b[0], b[1]-offset-0.15, b[0]+0.15, b[1]-offset]),  # South
    ]
    unaltered_region = ee.Geometry.MultiPolygon([box.coordinates() for box in unaltered_boxes])

    unaltered_sampled = composite.sample(
        region=unaltered_region, scale=20, numPixels=8000, seed=42, geometries=False
    )
    df_unaltered = download_fc(unaltered_sampled, max_features=8000)
    df_unaltered["is_altered"] = 0
    print(f"    Unaltered: {len(df_unaltered)} pixels")

    # Combine
    df_binary = pd.concat([df_altered, df_unaltered], ignore_index=True)
    band_cols = sorted([c for c in df_binary.columns if c.startswith("B") and len(c) <= 3])
    df_binary = df_binary.dropna(subset=band_cols)
    print(f"  Total binary: {len(df_binary)} ({(df_binary['is_altered']==1).sum()} altered, {(df_binary['is_altered']==0).sum()} unaltered)")

    # Evaluate binary detection
    bands = {col: df_binary[col].values for col in band_cols}
    y_binary = df_binary["is_altered"].values

    # PySR binary index
    score_sr = sr_binary_altered(**bands)
    auc_sr, f1_sr = evaluate_binary(score_sr, y_binary)

    # Classical indices
    score_clay = bands["B11"] / np.maximum(bands["B12"], 1e-6)
    auc_clay, f1_clay = evaluate_binary(score_clay, y_binary)

    score_iron = bands["B04"] / np.maximum(bands["B02"], 1e-6)
    auc_iron, f1_iron = evaluate_binary(score_iron, y_binary)

    # RF
    X_bin = df_binary[band_cols].values
    rf_bin = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    y_proba_bin = cross_val_predict(rf_bin, X_bin, y_binary, cv=cv, method="predict_proba")
    auc_rf, _ = evaluate_binary(y_proba_bin[:, 1], y_binary)
    ba_rf = balanced_accuracy_score(y_binary, (y_proba_bin[:, 1] > 0.5).astype(int))

    print(f"\n  Binary Detection Results:")
    print(f"    SR (sqrt(B12)-B11)²:    AUC={auc_sr:.4f}  F1={f1_sr:.4f}")
    print(f"    Clay Ratio B11/B12:     AUC={auc_clay:.4f}  F1={f1_clay:.4f}")
    print(f"    Iron Oxide B04/B02:     AUC={auc_iron:.4f}  F1={f1_iron:.4f}")
    print(f"    RF S2 (10 bands, CV):   AUC={auc_rf:.4f}  BA={ba_rf:.4f}")

    # ===== PHASE 2: Already done in full_evaluation.py =====
    # Skip, reference existing results

    # ===== PHASE 3: SUB-CLASSIFY UNDIFFERENTIATED =====
    print("\n" + "=" * 70)
    print("PHASE 3: SUB-CLASSIFICATION OF UNDIFFERENTIATED ZONES")
    print("=" * 70)

    # Sample S2 within undifferentiated polygons
    undiff_ll = undifferentiated.to_crs("EPSG:4326")
    undiff_features = [polygon_to_ee(row) for _, row in undiff_ll.iterrows()
                      if polygon_to_ee(row) is not None]

    if not undiff_features:
        print("  No undifferentiated features!")
        return

    undiff_fc = ee.FeatureCollection(undiff_features)
    undiff_region = undiff_fc.geometry()

    print("  Sampling undifferentiated pixels...")
    undiff_sampled = composite.sample(
        region=undiff_region, scale=20, numPixels=10000, seed=42, geometries=True
    )
    df_undiff = download_fc(undiff_sampled, max_features=10000)
    df_undiff = df_undiff.dropna(subset=band_cols)
    print(f"  Undifferentiated pixels: {len(df_undiff)}")

    if len(df_undiff) == 0:
        print("  No pixels extracted!")
        return

    # Apply SR formulas
    bands_undiff = {col: df_undiff[col].values for col in band_cols}

    sr_scores = {
        "Silicic": sr_silicic(**bands_undiff),
        "Adv_Argillic": sr_adv_argillic(**bands_undiff),
        "Argillic_Phyllic": sr_argillic(**bands_undiff),
        "Propylitic": sr_propylitic(**bands_undiff),
        "Iron_Oxide": sr_iron_oxide(**bands_undiff),
        "Potassic_Skarn": sr_potassic(**bands_undiff),
    }

    # Assign class based on highest SR score (normalized)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scores_matrix = np.column_stack([sr_scores[c] for c in sr_scores])
    scores_norm = scaler.fit_transform(scores_matrix)

    predicted_classes = np.array(list(sr_scores.keys()))[scores_norm.argmax(axis=1)]

    print(f"\n  SR-predicted sub-classification of undifferentiated zones:")
    for cls_name, count in pd.Series(predicted_classes).value_counts().items():
        pct = 100 * count / len(predicted_classes)
        print(f"    {cls_name:25s}: {count:5d} ({pct:.1f}%)")

    # Also train RF on classified polygons and predict undifferentiated
    print("\n  RF-predicted sub-classification:")
    train_data = np.load(GT_DIR / "maricunga_training_s2_gee.npz")
    X_train_cls = train_data["X"]
    y_train_cls = train_data["y"]

    rf_cls = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight="balanced")
    rf_cls.fit(X_train_cls, y_train_cls)

    X_undiff = df_undiff[sorted(band_cols)].values
    y_pred_rf = rf_cls.predict(X_undiff)
    y_proba_rf_undiff = rf_cls.predict_proba(X_undiff)

    for cls_id in sorted(np.unique(y_pred_rf)):
        count = (y_pred_rf == cls_id).sum()
        pct = 100 * count / len(y_pred_rf)
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        print(f"    {name:25s}: {count:5d} ({pct:.1f}%)")

    # K-means clustering
    print("\n  K-means clustering (k=4):")
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(X_undiff)

    for c in range(4):
        count = (clusters == c).sum()
        pct = 100 * count / len(clusters)
        # Characterize cluster by mean band values
        mask = clusters == c
        b11_mean = X_undiff[mask, sorted(band_cols).index("B11")].mean()
        b12_mean = X_undiff[mask, sorted(band_cols).index("B12")].mean()
        b04_mean = X_undiff[mask, sorted(band_cols).index("B04")].mean()
        ratio = b11_mean / max(b12_mean, 1e-6)
        print(f"    Cluster {c}: {count:5d} ({pct:.1f}%) — B11/B12={ratio:.2f}, B04={b04_mean:.3f}")

    # Agreement between methods
    print("\n  Agreement SR vs RF:")
    # Map SR predicted class names to IDs
    sr_to_id = {"Silicic": 1, "Adv_Argillic": 2, "Argillic_Phyllic": 3,
                "Propylitic": 4, "Iron_Oxide": 5, "Potassic_Skarn": 6}
    sr_pred_ids = np.array([sr_to_id[c] for c in predicted_classes])
    agreement = (sr_pred_ids == y_pred_rf).mean()
    print(f"    Overall agreement: {agreement:.1%}")

    # Save results
    results = {
        "phase1_binary": {
            "sr_auc": float(auc_sr), "sr_f1": float(f1_sr),
            "clay_ratio_auc": float(auc_clay),
            "iron_oxide_auc": float(auc_iron),
            "rf_auc": float(auc_rf), "rf_ba": float(ba_rf),
            "n_altered": int((y_binary == 1).sum()),
            "n_unaltered": int((y_binary == 0).sum()),
        },
        "phase3_subclassification": {
            "n_undifferentiated_pixels": len(df_undiff),
            "sr_distribution": dict(pd.Series(predicted_classes).value_counts()),
            "rf_distribution": {CLASS_NAMES.get(k, f"C{k}"): int(v)
                               for k, v in zip(*np.unique(y_pred_rf, return_counts=True))},
            "sr_rf_agreement": float(agreement),
        },
    }

    with open(RESULTS_DIR / "full_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'full_pipeline_results.json'}")


if __name__ == "__main__":
    main()
