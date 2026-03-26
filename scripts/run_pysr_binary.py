#!/usr/bin/env python3
"""
PySR for binary altered/unaltered classification.
Uses the 16K samples from Phase 1 (8K altered + 8K unaltered).
"""

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
import time
import json

RESULTS_DIR = Path("data/results")

# Load Phase 1 results to rebuild the dataset
# We need to re-extract or load from the pipeline output
# Use the saved CSV if available, otherwise reconstruct
GT_DIR = Path("data/ground_truth")


def main():
    # Load binary data from full_pipeline results
    pipeline_results = json.load(open(RESULTS_DIR / "full_pipeline_results.json"))
    n_alt = pipeline_results["phase1_binary"]["n_altered"]
    n_unalt = pipeline_results["phase1_binary"]["n_unaltered"]
    print(f"Phase 1 had {n_alt} altered + {n_unalt} unaltered = {n_alt + n_unalt}")

    # We need to re-extract the data. Check if saved.
    binary_path = GT_DIR / "binary_training_s2.csv"

    if not binary_path.exists():
        print("Re-extracting binary data via GEE...")
        import ee
        ee.Initialize()

        import geopandas as gpd

        ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
        S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        BAND_RENAME = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
                       "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
                       "B11": "B11", "B12": "B12"}

        chile_aoi = ee.Geometry.Rectangle([-69.6, -28.1, -68.5, -25.9])
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(chile_aoi)
              .filterDate("2024-01-01", "2024-04-01")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
              .select(S2_BANDS))
        composite = s2.median().divide(10000).rename([BAND_RENAME[b] for b in S2_BANDS])

        gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
        attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
        gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
        gdf = gdf.set_crs("EPSG:32719").to_crs("EPSG:4326")

        # All alteration polygons
        def polygon_to_ee(row):
            geom = row.geometry
            if geom.geom_type == "Polygon":
                coords = [list(c) for c in geom.exterior.coords]
                return ee.Feature(ee.Geometry.Polygon([coords]))
            elif geom.geom_type == "MultiPolygon":
                polys = [[list(c) for c in p.exterior.coords] for p in geom.geoms]
                return ee.Feature(ee.Geometry.MultiPolygon(polys))
            return None

        features = [polygon_to_ee(row) for _, row in gdf.iterrows() if polygon_to_ee(row) is not None]
        fc = ee.FeatureCollection(features)
        region = fc.geometry()

        # Altered
        print("  Sampling altered...")
        alt_sampled = composite.sample(region=region, scale=20, numPixels=8000, seed=42)
        alt_list = alt_sampled.toList(8000).getInfo()
        df_alt = pd.DataFrame([f["properties"] for f in alt_list])
        df_alt["is_altered"] = 1
        print(f"    {len(df_alt)} pixels")

        # Unaltered (offset boxes)
        b = gdf.total_bounds
        offset = 0.25
        boxes = [
            ee.Geometry.Rectangle([b[0]-offset-0.15, b[1], b[0]-offset, b[1]+0.15]),
            ee.Geometry.Rectangle([b[2]+offset, b[1], b[2]+offset+0.15, b[1]+0.15]),
            ee.Geometry.Rectangle([b[0], b[3]+offset, b[0]+0.15, b[3]+offset+0.15]),
            ee.Geometry.Rectangle([b[0], b[1]-offset-0.15, b[0]+0.15, b[1]-offset]),
        ]
        unalt_region = ee.Geometry.MultiPolygon([box.coordinates() for box in boxes])

        print("  Sampling unaltered...")
        unalt_sampled = composite.sample(region=unalt_region, scale=20, numPixels=8000, seed=42)
        unalt_list = unalt_sampled.toList(8000).getInfo()
        df_unalt = pd.DataFrame([f["properties"] for f in unalt_list])
        df_unalt["is_altered"] = 0
        print(f"    {len(df_unalt)} pixels")

        df_binary = pd.concat([df_alt, df_unalt], ignore_index=True)
        df_binary.to_csv(binary_path, index=False)
        print(f"  Saved: {binary_path}")
    else:
        df_binary = pd.read_csv(binary_path)
        print(f"Loaded: {binary_path} ({len(df_binary)} rows)")

    band_cols = sorted([c for c in df_binary.columns if c.startswith("B") and len(c) <= 3])
    df_binary = df_binary.dropna(subset=band_cols)

    X = df_binary[band_cols].values.astype(np.float64)
    y = df_binary["is_altered"].values.astype(np.float64)

    print(f"Samples: {len(y)} ({(y==1).sum()} altered, {(y==0).sum()} unaltered)")
    print(f"Bands: {band_cols}")

    # Subsample
    MAX = 10000
    if len(y) > MAX:
        X, _, y, _ = train_test_split(X, y, train_size=MAX, stratify=y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # PySR
    print("\nRunning PySR for binary altered/unaltered...")
    model = PySRRegressor(
        niterations=120,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "square", "tanh"],
        maxsize=15,
        populations=30,
        population_size=50,
        parsimony=0.004,
        weight_optimize=0.001,
        constraints={"sqrt": 5, "log": 5, "square": 5, "tanh": 5},
        nested_constraints={
            "sqrt": {"sqrt": 0, "log": 0},
            "log": {"log": 0, "sqrt": 0},
            "square": {"square": 0},
        },
        select_k_features=5,
        progress=True,
        temp_equation_file=True,
        tempdir=str(RESULTS_DIR / "pysr_temp_binary"),
        turbo=True,
        bumper=True,
    )

    t0 = time.time()
    model.fit(X_train, y_train, variable_names=band_cols)
    elapsed = time.time() - t0

    best = model.get_best()
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.02):
        f1 = f1_score(y_test, (y_pred > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest equation: {best['equation']}")
    print(f"Complexity: {best['complexity']}")
    print(f"AUC: {auc:.4f}, F1: {best_f1:.4f} (thresh={best_t:.2f})")
    print(f"Time: {elapsed:.0f}s")

    # Save
    equations_df = model.equations_
    if equations_df is not None:
        equations_df.to_csv(RESULTS_DIR / "equations_binary.csv", index=False)

    results = {
        "best_equation": str(best["equation"]),
        "complexity": int(best["complexity"]),
        "auc": float(auc),
        "f1": float(best_f1),
        "threshold": float(best_t),
        "time_s": float(elapsed),
    }

    if equations_df is not None:
        results["pareto_front"] = []
        for _, row in equations_df.iterrows():
            results["pareto_front"].append({
                "equation": str(row.get("equation", "")),
                "complexity": int(row.get("complexity", 0)),
                "loss": float(row.get("loss", 0)),
            })

    with open(RESULTS_DIR / "pysr_binary_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'pysr_binary_results.json'}")

    # Print Pareto front
    print("\nPareto Front:")
    if equations_df is not None:
        for _, row in equations_df.iterrows():
            print(f"  C={row['complexity']:2d}  Loss={row['loss']:.6f}  {row['equation']}")


if __name__ == "__main__":
    main()
