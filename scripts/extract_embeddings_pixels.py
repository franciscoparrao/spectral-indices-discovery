#!/usr/bin/env python3
"""
Extract Google Satellite Embeddings per-pixel within Atlas alteration polygons.
This gives thousands of samples instead of 89 centroids.
"""

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time

ee.Initialize()

ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEAR = 2024

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


def polygon_to_ee(row):
    """Convert a GeoDataFrame row to an EE Feature."""
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
        "alteration": str(row["ALTERACION"])[:50],
    })


def sample_class_polygons(image, gdf_class, cls_id, cls_name, max_pixels=5000):
    """Sample pixels within polygons of one class."""
    print(f"\n  Class {cls_id}: {cls_name} ({len(gdf_class)} polygons)")

    # Convert polygons to EE
    features = []
    for _, row in gdf_class.iterrows():
        feat = polygon_to_ee(row)
        if feat:
            features.append(feat)

    if not features:
        print(f"    No valid geometries")
        return pd.DataFrame()

    fc = ee.FeatureCollection(features)
    region = fc.geometry()

    # Sample pixels within polygons
    sampled = image.sample(
        region=region,
        scale=10,
        numPixels=max_pixels,
        seed=42,
        geometries=True,
    )

    # Download
    size = sampled.size().getInfo()
    print(f"    Sampled {size} pixels")

    if size == 0:
        return pd.DataFrame()

    all_rows = []
    batch_size = 1000
    for start in range(0, size, batch_size):
        batch = sampled.toList(min(batch_size, size - start), start)
        feats = batch.getInfo()
        for feat in feats:
            props = feat.get("properties", {})
            props["class_id"] = cls_id
            props["class_name"] = cls_name
            if feat.get("geometry"):
                coords = feat["geometry"].get("coordinates", [None, None])
                props["lon"] = coords[0]
                props["lat"] = coords[1]
            all_rows.append(props)
        print(f"      Downloaded {len(all_rows)}/{size}")

    return pd.DataFrame(all_rows)


def main():
    # Load embedding mosaic
    print(f"Loading embeddings for {YEAR}...")
    chile_aoi = ee.Geometry.Rectangle([-71, -28, -67, -22])
    collection = (ee.ImageCollection(EMBEDDING_COLLECTION)
                  .filter(ee.Filter.calendarRange(YEAR, YEAR, "year"))
                  .filterBounds(chile_aoi))
    image = collection.mosaic().setDefaultProjection("EPSG:32719", None, 10)
    print(f"  Tiles: {collection.size().getInfo()}")

    # Load alteration polygons
    print("Loading Atlas alteration polygons...")
    gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
    attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
    gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
    gdf = gdf.set_crs("EPSG:32719")
    gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)

    # Convert to WGS84 for GEE
    gdf_ll = gdf.to_crs("EPSG:4326")

    # Only classified polygons (exclude class 0 and "indiferenciada")
    classified = gdf_ll[gdf_ll["class_id"].isin([1, 2, 3, 4, 5, 6])].copy()
    print(f"Classified polygons: {len(classified)}")

    # Sample per class
    all_dfs = []
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_gdf = classified[classified["class_id"] == cls_id]
        if len(cls_gdf) == 0:
            continue

        # More pixels for larger classes, min 500 per class
        n_polys = len(cls_gdf)
        max_px = max(500, min(5000, n_polys * 200))

        df = sample_class_polygons(image, cls_gdf, cls_id, cls_name, max_pixels=max_px)
        if len(df) > 0:
            all_dfs.append(df)

    # Combine all
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        emb_cols = [c for c in df_all.columns if c.startswith("A") and c[1:].isdigit()]

        path = EMB_DIR / "atlas_alteration_embeddings_pixels.csv"
        df_all.to_csv(path, index=False)
        print(f"\n=== TOTAL ===")
        print(f"Saved: {path}")
        print(f"Total samples: {len(df_all)}")
        print(f"Embedding dims: {len(emb_cols)}")
        print(f"\nClass distribution:")
        for cls_id in sorted(df_all["class_id"].unique()):
            n = (df_all["class_id"] == cls_id).sum()
            name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            print(f"  {cls_id} {name:25s}: {n}")
    else:
        print("ERROR: No data extracted!")


if __name__ == "__main__":
    main()
