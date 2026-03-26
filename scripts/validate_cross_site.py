#!/usr/bin/env python3
"""
Cross-site validation: Apply PySR formulas trained on III Región (Maricunga)
to IV Región (Vicuña-Pichasca, Andacollo, Ovalle) alteration polygons.
"""

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import json

ee.Initialize()

SHAPEFILES_DIR = Path("data/external/shapefiles_alteracion")
EMB_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_RENAME = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
               "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
               "B11": "B11", "B12": "B12"}

# Map IV Región alteration names to our class IDs
ALT_MAP_IV = {
    "Silícica": 1,
    "Argílica": 2,  # Map to Adv_Argillic (closest match)
    "Argílica, Cuarzo Sericítica": 3,
    "Argílica, Silícica": 2,
    "Cuarzo Sericítica, Silícica": 3,
    "Fílica (Cuarzo Sericítica; Sericitica)": 3,
    "Sericítica": 3,
    "Propilítica": 4,
}

CLASS_NAMES = {
    1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic", 4: "Propylitic",
}

# PySR formulas from Maricunga training (from pysr_results_gee.json)
def apply_sr_silicic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return B04 - 0.13496

def apply_sr_adv_argillic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return 0.83176 - (B02 / B05)

def apply_sr_argillic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return 0.09014 / B05

def apply_sr_propylitic(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12):
    return B03 - (B11 * 0.48311)


def polygon_to_ee(row):
    geom = row.geometry
    if geom.geom_type == "Polygon":
        coords = [list(c) for c in geom.exterior.coords]
        return ee.Feature(ee.Geometry.Polygon([coords]), {
            "class_id": int(row["class_id"]),
        })
    elif geom.geom_type == "MultiPolygon":
        polys = [[list(c) for c in p.exterior.coords] for p in geom.geoms]
        return ee.Feature(ee.Geometry.MultiPolygon(polys), {
            "class_id": int(row["class_id"]),
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
            if feat.get("geometry"):
                coords = feat["geometry"].get("coordinates", [None, None])
                props["lon"] = coords[0]
                props["lat"] = coords[1]
            all_rows.append(props)
    return pd.DataFrame(all_rows)


def main():
    # === 1. Load all IV Región shapefiles ===
    print("Loading IV Región alteration polygons...")
    all_gdfs = []

    for site_dir in sorted(SHAPEFILES_DIR.iterdir()):
        shp = site_dir / "GB_ALTERACION_P.shp"
        if not shp.exists():
            continue

        gdf = gpd.read_file(shp)

        # Find alteration column
        alt_col = None
        for col in ["d_ALTERACI", "ALTERACION"]:
            if col in gdf.columns:
                alt_col = col
                break

        if not alt_col:
            continue

        gdf["class_id"] = gdf[alt_col].map(ALT_MAP_IV).fillna(0).astype(int)
        gdf["source"] = site_dir.name
        gdf = gdf[gdf["class_id"] > 0]

        if len(gdf) > 0:
            all_gdfs.append(gdf)
            print(f"  {site_dir.name}: {len(gdf)} classified polygons")
            print(f"    {dict(gdf[alt_col].value_counts())}")

    if not all_gdfs:
        print("No classified polygons found!")
        return

    gdf_all = pd.concat(all_gdfs, ignore_index=True)
    gdf_all = gdf_all.to_crs("EPSG:4326")

    print(f"\nTotal classified: {len(gdf_all)}")
    for cls_id in sorted(gdf_all["class_id"].unique()):
        n = (gdf_all["class_id"] == cls_id).sum()
        print(f"  {cls_id} {CLASS_NAMES.get(cls_id, '?'):25s}: {n}")

    # === 2. Build S2 composite + embeddings via GEE ===
    print("\nBuilding S2 composite via GEE...")
    bounds = gdf_all.total_bounds
    aoi = ee.Geometry.Rectangle([bounds[0]-0.1, bounds[1]-0.1, bounds[2]+0.1, bounds[3]+0.1])

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate("2024-01-01", "2024-04-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .select(S2_BANDS))
    print(f"  S2 scenes: {s2.size().getInfo()}")

    composite = s2.median().divide(10000).select(S2_BANDS)
    composite = composite.rename([BAND_RENAME[b] for b in S2_BANDS])

    # Embeddings
    print("Loading embeddings...")
    emb_col = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
               .filter(ee.Filter.calendarRange(2024, 2024, "year"))
               .filterBounds(aoi))
    emb_image = emb_col.mosaic().setDefaultProjection("EPSG:32719", None, 10)
    print(f"  Embedding tiles: {emb_col.size().getInfo()}")

    # Combine S2 + embeddings
    combined = composite.addBands(emb_image)

    # === 3. Sample pixels within polygons ===
    print("\nSampling pixels...")
    all_dfs = []

    for cls_id in sorted(gdf_all["class_id"].unique()):
        cls_gdf = gdf_all[gdf_all["class_id"] == cls_id]
        cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

        features = []
        for _, row in cls_gdf.iterrows():
            feat = polygon_to_ee(row)
            if feat:
                features.append(feat)

        if not features:
            continue

        fc = ee.FeatureCollection(features)
        region = fc.geometry()

        max_px = max(300, min(3000, len(cls_gdf) * 150))

        sampled = combined.sample(
            region=region, scale=20, numPixels=max_px, seed=42, geometries=True
        )

        df = download_fc(sampled, max_features=max_px)
        if len(df) > 0:
            df["class_id"] = cls_id
            df["class_name"] = cls_name
            all_dfs.append(df)
            print(f"  {cls_id} {cls_name}: {len(df)} pixels")

    if not all_dfs:
        print("No pixels sampled!")
        return

    df_val = pd.concat(all_dfs, ignore_index=True)

    # === 4. Apply PySR formulas ===
    print(f"\nTotal validation samples: {len(df_val)}")

    band_cols = [c for c in df_val.columns if c.startswith("B") and len(c) <= 3]
    emb_cols = [c for c in df_val.columns if c.startswith("A") and c[1:].isdigit()]

    # Drop NaN
    df_val = df_val.dropna(subset=band_cols)
    print(f"After NaN removal: {len(df_val)}")

    print("\nClass distribution (validation):")
    for cls_id in sorted(df_val["class_id"].unique()):
        n = (df_val["class_id"] == cls_id).sum()
        print(f"  {cls_id} {CLASS_NAMES.get(cls_id, '?'):25s}: {n}")

    # Extract band arrays
    B02 = df_val["B02"].values
    B03 = df_val["B03"].values
    B04 = df_val["B04"].values
    B05 = df_val["B05"].values
    B06 = df_val["B06"].values
    B07 = df_val["B07"].values
    B08 = df_val["B08"].values
    B8A = df_val["B8A"].values
    B11 = df_val["B11"].values
    B12 = df_val["B12"].values
    y = df_val["class_id"].values

    # Apply SR formulas
    sr_formulas = {
        1: ("B04 - 0.135", apply_sr_silicic),
        2: ("0.83 - B02/B05", apply_sr_adv_argillic),
        3: ("0.09 / B05", apply_sr_argillic),
        4: ("B03 - B11*0.48", apply_sr_propylitic),
    }

    print("\n" + "=" * 70)
    print("CROSS-SITE VALIDATION: III Región (train) → IV Región (test)")
    print("=" * 70)

    results = {}

    for cls_id, (formula_str, formula_fn) in sr_formulas.items():
        cls_name = CLASS_NAMES[cls_id]
        y_bin = (y == cls_id).astype(int)
        n_pos = y_bin.sum()

        if n_pos < 5:
            print(f"\n  SKIP {cls_name}: only {n_pos} positive samples")
            continue

        score = formula_fn(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
        score = np.nan_to_num(score, nan=0)

        auc = roc_auc_score(y_bin, score) if len(np.unique(y_bin)) > 1 else 0

        best_f1, best_t = 0, 0.5
        for t in np.arange(0.01, 0.99, 0.02):
            f1 = f1_score(y_bin, (score > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        print(f"\n  {cls_name} (n={n_pos}):")
        print(f"    Formula: {formula_str}")
        print(f"    AUC: {auc:.4f}  F1: {best_f1:.4f} (thresh={best_t:.2f})")

        results[cls_name] = {
            "formula": formula_str, "auc": float(auc),
            "f1": float(best_f1), "n_positive": int(n_pos),
        }

    # === 5. RF on embeddings (cross-site upper bound) ===
    if emb_cols and len(emb_cols) >= 64:
        print("\n" + "=" * 70)
        print("EMBEDDING RF (upper bound)")
        print("=" * 70)

        # Load Maricunga training embeddings
        emb_train = pd.read_csv(EMB_DIR / "atlas_alteration_embeddings_pixels.csv")
        emb_train = emb_train[emb_train["class_id"].isin([1, 2, 3, 4])]

        if len(emb_train) > 0:
            X_train_emb = emb_train[emb_cols[:64]].values
            y_train_emb = emb_train["class_id"].values

            X_val_emb = df_val[emb_cols[:64]].dropna().values
            y_val_emb = y[:len(X_val_emb)]

            rf = RandomForestClassifier(200, max_depth=15, random_state=42,
                                       n_jobs=-1, class_weight="balanced")
            rf.fit(X_train_emb, y_train_emb)
            y_pred = rf.predict(X_val_emb)
            ba = balanced_accuracy_score(y_val_emb, y_pred)
            print(f"  Balanced Accuracy: {ba:.4f}")

            y_proba = rf.predict_proba(X_val_emb)
            for cls_id in sorted(np.unique(y_val_emb)):
                y_bin = (y_val_emb == cls_id).astype(int)
                if y_bin.sum() >= 5 and cls_id in rf.classes_:
                    idx = list(rf.classes_).index(cls_id)
                    auc = roc_auc_score(y_bin, y_proba[:, idx])
                    name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
                    print(f"    {name}: AUC={auc:.4f}")
                    results[f"emb_{name}"] = {"auc": float(auc)}

            results["emb_balanced_accuracy"] = float(ba)

    # === 6. RF on S2 bands (cross-site) ===
    print("\n" + "=" * 70)
    print("RF S2 BANDS (cross-site)")
    print("=" * 70)

    train_data = np.load("data/ground_truth/maricunga_training_s2_gee.npz")
    X_train_s2 = train_data["X"]
    y_train_s2 = train_data["y"]
    # Keep only classes present in validation
    valid_classes = set(df_val["class_id"].unique())
    mask = np.isin(y_train_s2, list(valid_classes))
    X_train_s2, y_train_s2 = X_train_s2[mask], y_train_s2[mask]

    X_val_s2 = df_val[sorted(band_cols)].values
    y_val_s2 = y

    rf_s2 = RandomForestClassifier(200, max_depth=15, random_state=42,
                                    n_jobs=-1, class_weight="balanced")
    rf_s2.fit(X_train_s2, y_train_s2)
    y_pred_s2 = rf_s2.predict(X_val_s2)
    ba_s2 = balanced_accuracy_score(y_val_s2, y_pred_s2)
    print(f"  Balanced Accuracy: {ba_s2:.4f}")

    y_proba_s2 = rf_s2.predict_proba(X_val_s2)
    for cls_id in sorted(np.unique(y_val_s2)):
        y_bin = (y_val_s2 == cls_id).astype(int)
        if y_bin.sum() >= 5 and cls_id in rf_s2.classes_:
            idx = list(rf_s2.classes_).index(cls_id)
            auc = roc_auc_score(y_bin, y_proba_s2[:, idx])
            name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            print(f"    {name}: AUC={auc:.4f}")
            results[f"rf_s2_{name}"] = {"auc": float(auc)}

    results["rf_s2_balanced_accuracy"] = float(ba_s2)

    # Save
    with open(RESULTS_DIR / "cross_site_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'cross_site_validation.json'}")

    # Save validation data
    val_path = EMB_DIR / "iv_region_validation.csv"
    df_val.to_csv(val_path, index=False)
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
