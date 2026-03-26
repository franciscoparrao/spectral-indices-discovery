#!/usr/bin/env python3
"""
Extract Google Satellite Embeddings (AlphaEarth v2.1) for study areas.
64-dimensional embeddings at 10m from Sentinel-1/2 + Landsat + LiDAR.

Extracts at:
1. Atlas Metalífero alteration polygon centroids (Maricunga)
2. Atlas yacimiento points (781 mineral deposits)
3. Grid sample over El Tatio area
"""

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time

ee.Initialize()

GT_DIR = Path("data/ground_truth")
ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEAR = 2024  # Most recent complete year
BAND_NAMES = [f"A{i:02d}" for i in range(64)]

# Alteration class mapping (same as build_training_atlas.py)
ALTERATION_MAP = {
    "Alteración Silicea": 1,
    "vuggy silica": 1,
    "Alteracion Argilica y Argilica avanzada": 2,
    "Alteracion Solfatárica": 2,
    "Alteracion Argilica": 3,
    "Alteracion Sericitica": 3,
    "Alteración Cuarzo-Sericitica(Fílica)": 3,
    "Alteracion Propilitica": 4,
    "Oxidos e Hidróxidos de Hierro": 5,
    "Alteracion Potasica": 6,
    "skarn": 6,
    "Alteracion hidrotermal indiferenciada (arg, lim y sil)": 7,
}


def extract_at_points(image, points_fc, scale=10):
    """Extract embedding values at point features."""
    sampled = image.sampleRegions(
        collection=points_fc,
        scale=scale,
        geometries=True,
    )
    return sampled


def fc_to_dataframe(fc, max_features=5000):
    """Convert EE FeatureCollection to pandas DataFrame in batches."""
    # Get size
    size = fc.size().getInfo()
    print(f"  Total features to download: {size}")

    if size == 0:
        return pd.DataFrame()

    all_rows = []
    batch_size = 1000

    for start in range(0, min(size, max_features), batch_size):
        end = min(start + batch_size, size, max_features)
        batch = fc.toList(end - start, start)
        features = batch.getInfo()
        for feat in features:
            props = feat.get("properties", {})
            if feat.get("geometry"):
                coords = feat["geometry"].get("coordinates", [None, None])
                props["_lon"] = coords[0] if coords else None
                props["_lat"] = coords[1] if coords else None
            all_rows.append(props)
        print(f"    Downloaded {len(all_rows)}/{min(size, max_features)}")

    return pd.DataFrame(all_rows)


def main():
    # Load embedding mosaic for 2024, filtered to Chile
    print(f"Loading embeddings for {YEAR}...")
    chile_aoi = ee.Geometry.Rectangle([-71, -28, -67, -22])
    collection = (ee.ImageCollection(EMBEDDING_COLLECTION)
                  .filter(ee.Filter.calendarRange(YEAR, YEAR, "year"))
                  .filterBounds(chile_aoi))
    n_images = collection.size().getInfo()
    print(f"  Found {n_images} tiles covering study area")
    image = collection.mosaic().setDefaultProjection("EPSG:32719", None, 10)

    # === 1. ATLAS ALTERATION POLYGONS (centroids) ===
    print("\n=== Atlas Alteration Polygons ===")

    # Load and prepare alteration data
    gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
    attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
    gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
    gdf = gdf.set_crs("EPSG:32719")
    gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)

    # Use centroids (compute in projected CRS, then convert to WGS84)
    centroids = gdf.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs("EPSG:4326")

    # Create EE features
    features = []
    for _, row in centroids.iterrows():
        geom = ee.Geometry.Point([row.geometry.x, row.geometry.y])
        props = {
            "class_id": int(row["class_id"]),
            "alteration": str(row["ALTERACION"])[:50],
            "int_orig": str(row["INT_ORIG"]),
        }
        features.append(ee.Feature(geom, props))

    fc_alt = ee.FeatureCollection(features)
    print(f"  Created {len(features)} polygon centroids")

    # Extract embeddings
    print("  Extracting embeddings at centroids...")
    sampled_alt = extract_at_points(image, fc_alt)
    df_alt = fc_to_dataframe(sampled_alt)

    if len(df_alt) > 0:
        path = EMB_DIR / "atlas_alteration_embeddings.csv"
        df_alt.to_csv(path, index=False)
        print(f"  Saved: {path} ({len(df_alt)} rows)")

        # Stats
        emb_cols = [c for c in df_alt.columns if c.startswith("A")]
        print(f"  Embedding dimensions: {len(emb_cols)}")
        print(f"  Classes: {df_alt['class_id'].value_counts().to_dict()}")
    else:
        print("  WARNING: No embeddings extracted!")

    # === 2. ATLAS YACIMIENTOS (mineral deposits) ===
    print("\n=== Atlas Yacimientos ===")
    yac = pd.read_csv(ATLAS_DIR / "yacimientos.csv")

    # Parse coordinates (UTM 19S WGS84)
    yac_valid = yac.dropna(subset=["UTM_ESTE", "UTM_NORTE"])
    yac_valid = yac_valid[(yac_valid["UTM_ESTE"] > 0) & (yac_valid["UTM_NORTE"] > 0)]

    # Convert UTM to lat/lon
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)

    features_yac = []
    for _, row in yac_valid.iterrows():
        try:
            lon, lat = transformer.transform(row["UTM_ESTE"], row["UTM_NORTE"])
            if -75 < lon < -60 and -35 < lat < -20:  # Sanity check
                props = {
                    "name": str(row.get("NOMBRE", ""))[:50],
                    "mena": str(row.get("MENA", ""))[:30],
                    "minerales_alt": str(row.get("MINERALES_ALTERACION", ""))[:100],
                    "genesis": str(row.get("GENESIS", ""))[:30],
                    "ambiente": str(row.get("AMBIENTE", ""))[:30],
                }
                features_yac.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))
        except:
            continue

    print(f"  Valid deposit points: {len(features_yac)}")

    if features_yac:
        fc_yac = ee.FeatureCollection(features_yac)
        print("  Extracting embeddings at deposits...")
        sampled_yac = extract_at_points(image, fc_yac)
        df_yac = fc_to_dataframe(sampled_yac)

        if len(df_yac) > 0:
            path = EMB_DIR / "atlas_yacimientos_embeddings.csv"
            df_yac.to_csv(path, index=False)
            print(f"  Saved: {path} ({len(df_yac)} rows)")

    # === 3. EL TATIO GRID ===
    print("\n=== El Tatio Grid Sample ===")
    # Sample a grid over El Tatio
    el_tatio_bbox = ee.Geometry.Rectangle([-68.10, -22.42, -67.92, -22.25])

    # Sample at regular grid
    grid_points = image.sample(
        region=el_tatio_bbox,
        scale=100,  # 100m grid to keep size manageable
        numPixels=5000,
        seed=42,
        geometries=True,
    )
    print("  Extracting El Tatio grid...")
    df_tatio = fc_to_dataframe(grid_points, max_features=5000)

    if len(df_tatio) > 0:
        path = EMB_DIR / "el_tatio_embeddings.csv"
        df_tatio.to_csv(path, index=False)
        print(f"  Saved: {path} ({len(df_tatio)} rows)")

    # === SUMMARY ===
    print("\n=== Summary ===")
    for f in EMB_DIR.glob("*.csv"):
        df = pd.read_csv(f)
        emb_cols = [c for c in df.columns if c.startswith("A")]
        print(f"  {f.name}: {len(df)} rows × {len(emb_cols)} embedding dims")

    print("\nDone! Embeddings extracted.")


if __name__ == "__main__":
    main()
