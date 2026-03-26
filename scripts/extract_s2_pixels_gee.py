#!/usr/bin/env python3
"""
Extract Sentinel-2 band values per-pixel within Atlas alteration polygons via GEE.
Same approach as embeddings extraction — ensures identical spatial sampling.
"""

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

ee.Initialize()

ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
OUT_DIR = Path("data/ground_truth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# S2 bands to extract (all useful bands at 10-20m)
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# Rename to match our convention
BAND_RENAME = {
    "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
    "B11": "B11", "B12": "B12",
}


def polygon_to_ee(row):
    """Convert GeoDataFrame row to EE Feature."""
    geom = row.geometry
    if geom.geom_type == "Polygon":
        coords = [list(c) for c in geom.exterior.coords]
        ee_geom = ee.Geometry.Polygon([coords])
    elif geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            coords = [list(c) for c in poly.exterior.coords]
            polys.append(coords)
        ee_geom = ee.Geometry.MultiPolygon(polys)
    else:
        return None
    return ee.Feature(ee_geom, {
        "class_id": int(row["class_id"]),
    })


def download_fc(fc, max_features=10000):
    """Download EE FeatureCollection in batches."""
    size = fc.size().getInfo()
    print(f"    Features to download: {size}")
    if size == 0:
        return pd.DataFrame()

    all_rows = []
    batch = 1000
    for start in range(0, min(size, max_features), batch):
        end = min(start + batch, size, max_features)
        feats = fc.toList(end - start, start).getInfo()
        for feat in feats:
            props = feat.get("properties", {})
            if feat.get("geometry"):
                coords = feat["geometry"].get("coordinates", [None, None])
                props["lon"] = coords[0]
                props["lat"] = coords[1]
            all_rows.append(props)
        print(f"      {len(all_rows)}/{min(size, max_features)}")

    return pd.DataFrame(all_rows)


def main():
    # === Build S2 median composite via GEE ===
    print("Building S2 L2A composite via GEE...")
    chile_aoi = ee.Geometry.Rectangle([-69.6, -28.1, -68.5, -25.9])

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(chile_aoi)
          .filterDate("2024-01-01", "2024-04-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .select(S2_BANDS))

    n_scenes = s2.size().getInfo()
    print(f"  Scenes found: {n_scenes}")

    # Cloud masking via SCL
    def mask_clouds(img):
        scl = img.select("SCL") if "SCL" in img.bandNames().getInfo() else None
        # Simple approach: use QA60 band
        qa = img.select("QA60") if "QA60" in S2_BANDS else None
        return img.divide(10000)  # Scale to reflectance

    # Actually, S2_SR_HARMONIZED already has surface reflectance
    # Just take median and scale
    composite = s2.median().divide(10000).select(S2_BANDS)

    # Rename bands to our convention
    composite = composite.rename([BAND_RENAME[b] for b in S2_BANDS])

    print(f"  Composite bands: {composite.bandNames().getInfo()}")

    # === Load alteration polygons ===
    print("\nLoading Atlas alteration polygons...")
    gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
    attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
    gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
    gdf = gdf.set_crs("EPSG:32719")
    gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)
    gdf_ll = gdf.to_crs("EPSG:4326")

    classified = gdf_ll[gdf_ll["class_id"].isin([1, 2, 3, 4, 5, 6])].copy()
    print(f"  Classified polygons: {len(classified)}")

    # === Sample per class ===
    all_dfs = []
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_gdf = classified[classified["class_id"] == cls_id]
        if len(cls_gdf) == 0:
            continue

        print(f"\n  Class {cls_id}: {cls_name} ({len(cls_gdf)} polygons)")

        # Convert to EE
        features = []
        for _, row in cls_gdf.iterrows():
            feat = polygon_to_ee(row)
            if feat:
                features.append(feat)

        if not features:
            continue

        fc = ee.FeatureCollection(features)
        region = fc.geometry()

        max_px = max(500, min(5000, len(cls_gdf) * 200))

        sampled = composite.sample(
            region=region,
            scale=20,  # 20m to match SWIR resolution
            numPixels=max_px,
            seed=42,
            geometries=True,
        )

        df = download_fc(sampled, max_features=max_px)
        if len(df) > 0:
            df["class_id"] = cls_id
            df["class_name"] = cls_name
            all_dfs.append(df)

    # === Combine and save ===
    if not all_dfs:
        print("ERROR: No data extracted!")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Check which bands we got
    band_cols = [c for c in df_all.columns if c.startswith("B")]
    print(f"\n=== TOTAL ===")
    print(f"Samples: {len(df_all)}")
    print(f"Bands: {sorted(band_cols)}")

    # Filter rows with NaN bands
    df_clean = df_all.dropna(subset=band_cols)
    print(f"After NaN removal: {len(df_clean)}")

    print(f"\nClass distribution:")
    for cls_id in sorted(df_clean["class_id"].unique()):
        n = (df_clean["class_id"] == cls_id).sum()
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        print(f"  {cls_id} {name:25s}: {n}")

    # Save
    path = OUT_DIR / "maricunga_training_s2_gee.csv"
    df_clean.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Also save as npz for PySR
    X = df_clean[sorted(band_cols)].values.astype(np.float32)
    y = df_clean["class_id"].values.astype(np.int32)

    npz_path = OUT_DIR / "maricunga_training_s2_gee.npz"
    np.savez_compressed(npz_path, X=X, y=y, band_names=sorted(band_cols))
    print(f"Saved: {npz_path} (X={X.shape}, y={y.shape})")

    # Stats
    print(f"\nBand statistics:")
    for b in sorted(band_cols):
        vals = df_clean[b]
        print(f"  {b:4s}: min={vals.min():.4f}  mean={vals.mean():.4f}  max={vals.max():.4f}")


if __name__ == "__main__":
    main()
