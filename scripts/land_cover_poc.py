#!/usr/bin/env python3
"""
Proof-of-concept: SR feature engineering for land cover classification.
Demonstrates that the SR framework generalizes beyond hydrothermal alteration.

Uses ESA WorldCover 2021 labels + Sentinel-2 imagery.
Data acquisition via SurtGis (streaming, memory-safe).
Sampling via rasterio windowed reads (no full raster in RAM).

Memory budget: <4 GB total.
"""

import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.base import clone
from pathlib import Path
import json
import time
import gc
import sys

# ===== CONFIG =====
SURTGIS = Path.home() / "proyectos/surtgis/target/release/surtgis"
DATA_DIR = Path("data/land_cover")
RESULTS_DIR = Path("data/results")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Small AOI: Thuringia/Bavaria border (diverse land cover, flat+hills)
# 0.3° x 0.3° ≈ 900 km² — ~1650x1650 px at 20m ≈ 11 MB/band
AOI_BBOX = "11.0,50.2,11.3,50.5"
AOI_NAME = "thuringia"

# S2 bands to download (20m resolution for consistency)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Land cover classes from WorldCover
LC_CLASSES = {
    10: "Forest",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    80: "Water",
}

# Samples per class
SAMPLES_PER_CLASS = 1500

# Memory limit for SurtGis
MAX_MEMORY = "4G"


def run_surtgis(args, timeout=1200):
    """Run a SurtGis command with timeout (default 20 min)."""
    cmd = [str(SURTGIS)] + args
    print(f"  $ surtgis {' '.join(args[:4])}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:500]}", flush=True)
        return False
    if result.stdout.strip():
        lines = result.stdout.strip().split("\n")
        for line in lines[-3:]:
            print(f"    {line}", flush=True)
    return True


def step1_download_s2():
    """Download S2 composite bands via SurtGis STAC (streaming)."""
    print("=" * 70)
    print("STEP 1: Download Sentinel-2 composite via SurtGis")
    print(f"  AOI: {AOI_BBOX} ({AOI_NAME})")
    print(f"  Memory limit: {MAX_MEMORY}")
    print("=" * 70)

    # Check if already downloaded
    existing = [f for f in DATA_DIR.glob(f"{AOI_NAME}_*.tif") if f.stem.split("_")[-1] in S2_BANDS]
    if len(existing) >= len(S2_BANDS):
        print(f"  Already have {len(existing)} bands, skipping download.")
        return True

    # Download each band as a median composite (summer 2021)
    for band in S2_BANDS:
        outfile = DATA_DIR / f"{AOI_NAME}_{band}.tif"
        if outfile.exists():
            print(f"  {band}: already exists, skipping.")
            continue

        # Map band names to S2 STAC asset names
        asset_name = band.lower()  # B02 -> b02, etc.
        if band == "B8A":
            asset_name = "b8a"

        ok = run_surtgis([
            "stac", "composite",
            "--catalog", "pc",
            "--collection", "sentinel-2-l2a",
            "--asset", asset_name,
            "--bbox", AOI_BBOX,
            "--datetime", "2021-06-01/2021-09-01",
            "--max-scenes", "4",
            "--max-memory", MAX_MEMORY,
            "--compress",
            str(outfile),
        ])

        if not ok:
            print(f"  FAILED to download {band}")
            return False

        print(f"  {band}: done")

    return True


def step2_download_worldcover():
    """Download WorldCover 2021 for the AOI."""
    print("\n" + "=" * 70)
    print("STEP 2: Download ESA WorldCover 2021")
    print("=" * 70)

    outfile = DATA_DIR / f"{AOI_NAME}_worldcover.tif"
    if outfile.exists():
        print(f"  Already exists: {outfile}")
        return True

    # WorldCover is available on Earth Search as "esa-worldcover"
    # Try via SurtGis STAC fetch
    ok = run_surtgis([
        "stac", "fetch",
        "--catalog", "es",
        "--collections", "esa-worldcover",
        "--bbox", AOI_BBOX,
        "--datetime", "2021-01-01/2021-12-31",
        "--asset", "map",
        "--max-memory", MAX_MEMORY,
        str(outfile),
    ])

    if not ok:
        # Fallback: download via HTTPS from ESA WorldCover COG
        print("  SurtGis STAC failed, trying direct COG download...")
        ok = run_surtgis([
            "cog", "read",
            "--bbox", AOI_BBOX,
            "--max-memory", MAX_MEMORY,
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif",
            str(outfile),
        ])
        if not ok:
            print("  Also failed. Will try rasterio STAC approach.")
            return download_worldcover_rasterio()

    return True


def download_worldcover_rasterio():
    """Fallback: download WorldCover via planetary_computer + rasterio."""
    try:
        import planetary_computer as pc
        import pystac_client
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.crs import CRS

        print("  Using planetary_computer + rasterio fallback...")
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        bbox = [float(x) for x in AOI_BBOX.split(",")]
        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
        )
        items = list(search.items())
        if not items:
            print("  No WorldCover items found!")
            return False

        item = items[0]
        href = pc.sign(item.assets["map"].href)

        with rasterio.open(href) as src:
            window = from_bounds(*bbox, transform=src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1], height=data.shape[0],
                transform=transform, compress="deflate",
            )

        outfile = DATA_DIR / f"{AOI_NAME}_worldcover.tif"
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(data, 1)

        print(f"  Saved: {outfile} ({data.shape})")
        return True

    except Exception as e:
        print(f"  Fallback also failed: {e}")
        return False


def step3_sample_pixels():
    """Sample S2 band values at random locations, labeled by WorldCover class.

    Uses rasterio windowed reads — never loads full rasters into RAM.
    Memory usage: ~50 MB (just the sample arrays).
    """
    print("\n" + "=" * 70)
    print("STEP 3: Sample labeled pixels (memory-safe)")
    print("=" * 70)

    csv_path = RESULTS_DIR / "land_cover_s2_samples.csv"
    if csv_path.exists():
        print(f"  Already exists: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  {len(df)} samples, {df['class_name'].nunique()} classes")
        return df

    import rasterio
    from rasterio.transform import rowcol

    # Load WorldCover (small: ~11 MB for 0.3° x 0.3° at 10m)
    wc_path = DATA_DIR / f"{AOI_NAME}_worldcover.tif"
    if not wc_path.exists():
        print(f"  ERROR: WorldCover not found at {wc_path}")
        return None

    with rasterio.open(wc_path) as wc_src:
        wc_data = wc_src.read(1)
        wc_transform = wc_src.transform
        wc_crs = wc_src.crs
        print(f"  WorldCover: {wc_data.shape}, CRS: {wc_crs}")
        print(f"  Memory: {wc_data.nbytes / 1e6:.1f} MB")

    # Find pixel locations per class
    all_samples = []
    for lc_code, lc_name in LC_CLASSES.items():
        mask = wc_data == lc_code
        n_available = mask.sum()
        print(f"  {lc_name} (code {lc_code}): {n_available:,} pixels available")

        if n_available < 100:
            print(f"    Skipping: too few pixels")
            continue

        # Random sample of pixel indices
        rows, cols = np.where(mask)
        n_sample = min(SAMPLES_PER_CLASS, len(rows))
        rng = np.random.RandomState(42 + lc_code)
        idx = rng.choice(len(rows), size=n_sample, replace=False)
        sample_rows = rows[idx]
        sample_cols = cols[idx]

        # Convert pixel coords to geographic coords (WorldCover grid)
        xs, ys = rasterio.transform.xy(wc_transform, sample_rows, sample_cols)

        for i in range(n_sample):
            all_samples.append({
                "x": xs[i], "y": ys[i],
                "wc_row": int(sample_rows[i]),
                "wc_col": int(sample_cols[i]),
                "class_id": lc_code,
                "class_name": lc_name,
            })

    del wc_data, mask  # Free WorldCover from RAM
    gc.collect()

    df = pd.DataFrame(all_samples)
    print(f"\n  Total sample points: {len(df)}")

    # Extract S2 band values at sample locations
    # Read one band at a time to minimize RAM
    for band in S2_BANDS:
        band_path = DATA_DIR / f"{AOI_NAME}_{band}.tif"
        if not band_path.exists():
            print(f"  WARNING: {band_path} not found, skipping band {band}")
            df[band] = np.nan
            continue

        with rasterio.open(band_path) as src:
            # Convert geographic coords to this raster's pixel coords
            band_values = []
            for _, row in df.iterrows():
                try:
                    r, c = rowcol(src.transform, row["x"], row["y"])
                    # Read single pixel (1x1 window)
                    if 0 <= r < src.height and 0 <= c < src.width:
                        val = src.read(1, window=rasterio.windows.Window(c, r, 1, 1))
                        band_values.append(float(val[0, 0]))
                    else:
                        band_values.append(np.nan)
                except Exception:
                    band_values.append(np.nan)

            df[band] = band_values

        # Scale to reflectance (S2 L2A values are typically 0-10000)
        median_val = df[band].median()
        if median_val > 100:  # Needs scaling
            df[band] = df[band] / 10000.0

        print(f"  {band}: median={df[band].median():.4f}, range=[{df[band].min():.4f}, {df[band].max():.4f}]")

    # Drop rows with NaN bands
    df = df.dropna(subset=S2_BANDS)
    print(f"\n  Valid samples after dropping NaN: {len(df)}")

    # Save
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return df


def step3_sample_pixels_fast():
    """Faster version: read S2 bands as arrays (still memory-safe for small AOI).

    For a 0.3° x 0.3° AOI at 20m, each band is ~1650x1650 px = 11 MB.
    Loading one band at a time: peak ~22 MB (current + values array).
    """
    print("\n" + "=" * 70)
    print("STEP 3: Sample labeled pixels (fast, memory-safe)")
    print("=" * 70)

    csv_path = RESULTS_DIR / "land_cover_s2_samples.csv"
    if csv_path.exists():
        print(f"  Already exists: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  {len(df)} samples, {df['class_name'].nunique()} classes")
        return df

    import rasterio
    from rasterio.transform import rowcol
    from pyproj import Transformer

    # Load WorldCover
    wc_path = DATA_DIR / f"{AOI_NAME}_worldcover.tif"
    if not wc_path.exists():
        print(f"  ERROR: WorldCover not found at {wc_path}")
        return None

    with rasterio.open(wc_path) as wc_src:
        wc_data = wc_src.read(1)
        wc_transform = wc_src.transform
        wc_crs = wc_src.crs
        print(f"  WorldCover: {wc_data.shape}, CRS: {wc_crs}")

    # Get S2 CRS from any band file
    s2_band_path = DATA_DIR / f"{AOI_NAME}_{S2_BANDS[0]}.tif"
    with rasterio.open(s2_band_path) as s2_src:
        s2_crs = s2_src.crs
    print(f"  S2 CRS: {s2_crs}")

    # Prepare CRS transformer: WorldCover (EPSG:4326) -> S2 (EPSG:32632)
    transformer = Transformer.from_crs(wc_crs, s2_crs, always_xy=True)

    # Sample locations per class from WorldCover
    sample_points = {}  # class_id -> list of (x_utm, y_utm)
    for lc_code, lc_name in LC_CLASSES.items():
        mask = wc_data == lc_code
        n_available = int(mask.sum())
        print(f"  {lc_name} (code {lc_code}): {n_available:,} pixels")

        if n_available < 100:
            print(f"    Skipping: too few pixels")
            continue

        rows, cols = np.where(mask)
        n_sample = min(SAMPLES_PER_CLASS, len(rows))
        rng = np.random.RandomState(42 + lc_code)
        idx = rng.choice(len(rows), size=n_sample, replace=False)

        # Get geographic coords (lon, lat) from WorldCover grid
        lons, lats = rasterio.transform.xy(wc_transform, rows[idx], cols[idx])

        # Reproject to S2 CRS (UTM)
        xs_utm, ys_utm = transformer.transform(lons, lats)
        sample_points[lc_code] = list(zip(xs_utm, ys_utm))

    del wc_data
    gc.collect()

    # Build sample dataframe (coordinates in S2 CRS)
    records = []
    for lc_code, points in sample_points.items():
        for x, y in points:
            records.append({"x": x, "y": y, "class_id": lc_code,
                            "class_name": LC_CLASSES[lc_code]})
    df = pd.DataFrame(records)
    print(f"\n  Total sample points: {len(df)}")

    # Extract band values: load one S2 band at a time
    xs = df["x"].values
    ys = df["y"].values

    for band in S2_BANDS:
        band_path = DATA_DIR / f"{AOI_NAME}_{band}.tif"
        if not band_path.exists():
            print(f"  WARNING: {band_path} not found")
            df[band] = np.nan
            continue

        with rasterio.open(band_path) as src:
            band_data = src.read(1)
            band_transform = src.transform

            # Coords already in S2 CRS (UTM), direct rowcol lookup
            vals = np.full(len(df), np.nan)
            for i in range(len(df)):
                try:
                    r, c = rowcol(band_transform, xs[i], ys[i])
                    if 0 <= r < band_data.shape[0] and 0 <= c < band_data.shape[1]:
                        vals[i] = float(band_data[r, c])
                except Exception:
                    pass

            df[band] = vals
            del band_data
            gc.collect()

        # Scale if needed
        if df[band].median() > 100:
            df[band] = df[band] / 10000.0

        print(f"  {band}: median={df[band].median():.4f}")

    df = df.dropna(subset=S2_BANDS)
    print(f"\n  Valid samples: {len(df)}")

    # Class distribution
    print("\n  Class distribution:")
    for name, count in df["class_name"].value_counts().items():
        print(f"    {name:15s}: {count:5d}")

    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return df


def step4_run_pysr(df):
    """Run PySR one-vs-rest for each land cover class.

    Memory-safe: runs one class at a time, collects garbage between runs.
    PySR params reduced vs original (populations=15, pop_size=30).
    """
    print("\n" + "=" * 70)
    print("STEP 4: PySR symbolic regression (memory-safe)")
    print("=" * 70)

    sr_path = RESULTS_DIR / "pysr_land_cover_results.json"
    if sr_path.exists():
        print(f"  Loading existing results from {sr_path}")
        with open(sr_path) as f:
            results = json.load(f)
        for name, res in results.items():
            print(f"  {name}: {res['best_equation']} (AUC={res['auc']:.3f})")
        return results

    from pysr import PySRRegressor

    band_cols = sorted([c for c in df.columns if c.startswith("B") and len(c) <= 3])
    X = df[band_cols].values
    class_ids = df["class_id"].values

    print(f"  Bands: {band_cols}")
    print(f"  Samples: {len(X)}")
    print(f"  PySR config: niterations=80, populations=15, pop_size=30")

    results = {}
    for lc_code, lc_name in LC_CLASSES.items():
        if lc_code not in class_ids:
            continue

        y_bin = (class_ids == lc_code).astype(float)
        n_pos = int(y_bin.sum())
        if n_pos < 50:
            print(f"\n  Skipping {lc_name}: only {n_pos} positive samples")
            continue

        print(f"\n  === {lc_name} (n_pos={n_pos}) ===")
        t0 = time.time()

        model = PySRRegressor(
            niterations=80,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "square"],  # Fewer ops = less RAM
            maxsize=15,
            populations=15,           # Reduced from 20
            population_size=30,       # Reduced from 40
            parsimony=0.005,
            select_k_features=4,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            random_state=42,
            variable_names=band_cols,
            procs=0,                  # Single process (less RAM)
            multithreading=True,      # Use threads instead of processes
        )

        model.fit(X, y_bin)
        elapsed = time.time() - t0

        # Get best equation with complexity <= 8
        eqs = model.equations_
        eqs_simple = eqs[eqs["complexity"] <= 8]
        if len(eqs_simple) > 0:
            best_idx = eqs_simple["loss"].idxmin()
            best = eqs_simple.loc[best_idx]
        else:
            best = eqs.iloc[-1]

        # Evaluate AUC
        y_pred = model.predict(X)
        auc = roc_auc_score(y_bin, y_pred)
        auc_inv = roc_auc_score(y_bin, -y_pred)
        if auc_inv > auc:
            auc = auc_inv

        results[lc_name] = {
            "class_id": lc_code,
            "n_positive": n_pos,
            "best_equation": str(best["equation"]),
            "complexity": int(best["complexity"]),
            "loss": float(best["loss"]),
            "auc": float(auc),
            "time_s": elapsed,
        }

        # Pareto front
        pareto = []
        for _, row in eqs.iterrows():
            pareto.append({
                "equation": str(row["equation"]),
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
            })
        results[lc_name]["pareto_front"] = pareto

        print(f"    Formula: {best['equation']}")
        print(f"    Complexity: {best['complexity']}, AUC: {auc:.3f}")
        print(f"    Time: {elapsed:.0f}s")

        # Save equations CSV
        eqs.to_csv(RESULTS_DIR / f"equations_lc_{lc_name}.csv", index=False)

        # Cleanup between classes
        del model, eqs, y_pred
        gc.collect()

    with open(sr_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {sr_path}")

    return results


def step5_evaluate(df, sr_results):
    """Compare RF(SR features) vs RF(raw bands) — the core equivalence test."""
    print("\n" + "=" * 70)
    print("STEP 5: Feature Engineering Equivalence Test")
    print("=" * 70)

    band_cols = sorted([c for c in df.columns if c.startswith("B") and len(c) <= 3])
    X_raw = df[band_cols].values
    y = df["class_id"].values

    # Build SR features by evaluating discovered formulas
    # PySR with select_k_features uses x0,x1,...,x9 mapped to sorted band_cols
    sr_features = {}
    for lc_name, res in sr_results.items():
        eq = res["best_equation"]

        # Build namespace with both xN and band name mappings
        local_ns = {}
        for i, col in enumerate(band_cols):
            arr = df[col].values
            local_ns[col] = arr
            local_ns[f"x{i}"] = arr  # PySR's xN convention
        local_ns["sqrt"] = np.sqrt
        local_ns["log"] = lambda x: np.log(np.maximum(np.abs(x), 1e-10))
        local_ns["square"] = np.square
        local_ns["np"] = np

        try:
            eq_safe = eq.replace("^", "**")
            vals = eval(eq_safe, {"__builtins__": {}}, local_ns)
            vals = np.nan_to_num(vals, nan=0, posinf=0, neginf=0)
            sr_features[lc_name] = vals
            print(f"  {lc_name}: {eq} -> OK")
        except Exception as e:
            print(f"  WARNING: Could not evaluate '{eq}' for {lc_name}: {e}")

    if len(sr_features) == 0:
        print("  ERROR: No SR features could be evaluated!")
        return {}

    n_sr = len(sr_features)
    X_sr = np.column_stack(list(sr_features.values()))

    # Classical indices
    B02, B03, B04 = df["B02"].values, df["B03"].values, df["B04"].values
    B8A, B11, B12 = df["B8A"].values, df["B11"].values, df["B12"].values
    eps = 1e-6

    classical = np.column_stack([
        B11 / np.maximum(B12, eps),                    # Clay Ratio
        B04 / np.maximum(B02, eps),                    # Iron Oxide
        B12 / np.maximum(B8A, eps),                    # Ferrous
        (B11 - B12) / np.maximum(B11 + B12, eps),     # Alunite
        B02 / np.maximum(B11, eps),                    # OH Minerals
        B12 / np.maximum(B11, eps),                    # Silica
        (B8A - B04) / np.maximum(B8A + B04, eps),     # NDVI
    ])
    X_sr_classical = np.column_stack([X_sr, classical])

    # Feature sets
    feature_sets = {
        "10 raw bands": X_raw,
        f"{n_sr} SR indices": X_sr,
        f"{n_sr} SR + 7 classical": X_sr_classical,
    }

    # Classifiers
    classifiers = {
        "RF": RandomForestClassifier(200, max_depth=15, random_state=42,
                                      n_jobs=-1, class_weight="balanced"),
    }
    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            eval_metric="mlogloss",
        )
    except ImportError:
        pass

    from sklearn.svm import SVC
    classifiers["SVM-RBF"] = SVC(kernel="rbf", probability=True,
                                   class_weight="balanced", random_state=42)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    present_classes = sorted(set(y))

    results = {}
    print(f"\n  {'Classifier':<12} {'Features':<22} {'mAUC':>6} {'BA':>6}")
    print("  " + "-" * 50)

    for clf_name, clf_template in classifiers.items():
        for fs_name, X_fs in feature_sets.items():
            clf = clone(clf_template)

            try:
                y_proba = cross_val_predict(clf, X_fs, y, cv=cv, method="predict_proba")
                y_pred = cross_val_predict(clf, X_fs, y, cv=cv)

                aucs = []
                for cls_id in present_classes:
                    y_bin = (y == cls_id).astype(int)
                    if y_bin.sum() < 5:
                        continue
                    idx = list(sorted(set(y))).index(cls_id)
                    auc = roc_auc_score(y_bin, y_proba[:, idx])
                    aucs.append(auc)

                mean_auc = np.mean(aucs)
                ba = balanced_accuracy_score(y, y_pred)

                key = f"{clf_name}|{fs_name}"
                results[key] = {
                    "classifier": clf_name,
                    "features": fs_name,
                    "n_features": X_fs.shape[1],
                    "mean_auc": float(mean_auc),
                    "balanced_accuracy": float(ba),
                    "per_class_auc": [float(a) for a in aucs],
                }

                print(f"  {clf_name:<12} {fs_name:<22} {mean_auc:.3f}  {ba:.3f}")

            except Exception as e:
                print(f"  {clf_name:<12} {fs_name:<22} ERROR: {e}")

    # TOST equivalence
    rf_raw_key = "RF|10 raw bands"
    rf_sr_key = f"RF|{n_sr} SR indices"
    if rf_raw_key in results and rf_sr_key in results:
        delta = results[rf_sr_key]["mean_auc"] - results[rf_raw_key]["mean_auc"]
        results["tost_delta"] = float(delta)
        print(f"\n  Delta (SR - raw): {delta:+.4f}")
        if abs(delta) < 0.01:
            print(f"  ✓ Within equivalence margin (±0.01)")
        else:
            print(f"  ✗ Outside equivalence margin")

    with open(RESULTS_DIR / "land_cover_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'land_cover_evaluation.json'}")

    return results


def main():
    print(f"Memory available: {subprocess.getoutput('free -h | grep Mem')}")
    print()

    # Step 1-2: Download data via SurtGis
    if not step1_download_s2():
        print("\nFAILED at step 1 (S2 download). Check SurtGis.")
        sys.exit(1)

    if not step2_download_worldcover():
        print("\nFAILED at step 2 (WorldCover download). Check connectivity.")
        sys.exit(1)

    # Step 3: Sample pixels
    df = step3_sample_pixels_fast()
    if df is None or len(df) == 0:
        print("\nFAILED at step 3 (sampling).")
        sys.exit(1)

    # Step 4: PySR
    sr_results = step4_run_pysr(df)

    # Step 5: Evaluation
    eval_results = step5_evaluate(df, sr_results)

    print("\n" + "=" * 70)
    print("DONE — Land Cover POC complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
