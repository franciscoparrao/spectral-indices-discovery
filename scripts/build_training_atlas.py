#!/usr/bin/env python3
"""
Build training dataset from Atlas Metalífero III Región + Sentinel-2.
Uses FIELD-MAPPED alteration polygons as ground truth (no circular argument).

Pipeline:
1. Load alteration polygons from Atlas
2. Map alteration types to classes
3. Download S2 bands for the alteration area via SurtGis
4. Extract band values at polygon locations
5. Export training dataset for PySR
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pathlib import Path
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

DATA_DIR = Path("data/sentinel2/maricunga")
ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
GT_DIR = Path("data/ground_truth")
FIG_DIR = Path("figures")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SURTGIS = Path.home() / "proyectos/surtgis/target/release/surtgis"

# Map Atlas alteration types to our classes
# Grouping by dominant mineral assemblage
ALTERATION_MAP = {
    "Alteración Silicea": 1,             # Silicic
    "vuggy silica": 1,                   # Silicic (advanced)
    "Alteracion Argilica y Argilica avanzada": 2,  # Adv. Argillic
    "Alteracion Solfatárica": 2,         # Sulfuric acid → adv. argillic
    "Alteracion Argilica": 3,            # Argillic
    "Alteracion Sericitica": 3,          # Sericitic ≈ argillic
    "Alteración Cuarzo-Sericitica(Fílica)": 3,    # Phyllic ≈ argillic
    "Alteracion Propilitica": 4,         # Propylitic
    "Oxidos e Hidróxidos de Hierro": 5,  # Iron oxide
    "Alteracion Potasica": 6,            # Potassic (deep, less detectable by S2)
    "skarn": 6,                          # Skarn (contact metasomatic)
    # Excluded:
    # "Alteracion hidrotermal indiferenciada" → too generic
    # "Sin Informacion" → unknown
}

CLASS_NAMES = {
    0: ("No data", "#000000"),
    1: ("Silicic", "#FFFFFF"),
    2: ("Adv. Argillic", "#FF6600"),
    3: ("Argillic/Phyllic", "#FFCC00"),
    4: ("Propylitic", "#00AA00"),
    5: ("Iron Oxide", "#CC0000"),
    6: ("Other (Potassic/Skarn)", "#9966CC"),
}

SCALE = 1.0 / 10000.0
VALID_MAX = 10000


def download_s2_band(bbox_str, date_range, asset, output):
    """Download a S2 band mosaic via SurtGis."""
    if output.exists():
        print(f"  {output.name} already exists, skipping")
        return True
    cmd = [
        str(SURTGIS), "stac", "fetch-mosaic",
        "--catalog", "es",
        "--collection", "sentinel-2-l2a",
        f"--bbox={bbox_str}",
        "--datetime", date_range,
        "--compress",
        "--asset", asset,
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:]}")
        return False
    return True


def main():
    # === 1. LOAD ALTERATION POLYGONS ===
    print("Loading Atlas alteration polygons...")
    gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
    attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
    gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
    gdf = gdf.set_crs("EPSG:32719")

    # Map to classes
    gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)
    classified = gdf[gdf["class_id"] > 0].copy()

    print(f"\nClassified polygons: {len(classified)} / {len(gdf)}")
    print("\nClass distribution:")
    for cls_id, (cls_name, _) in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        n = (classified["class_id"] == cls_id).sum()
        if n > 0:
            print(f"  {cls_id} {cls_name:25s}: {n:4d} polygons")

    # === 2. FIND BEST AREA FOR S2 DOWNLOAD ===
    # Focus on area with most classified polygons (non-indiferenciada)
    # Get bounding box of classified polygons
    bounds = classified.total_bounds  # [minx, miny, maxx, maxy]
    print(f"\nClassified polygons bounds (UTM): {bounds}")

    # Convert to lat/lon for STAC search
    classified_ll = classified.to_crs("EPSG:4326")
    ll_bounds = classified_ll.total_bounds
    print(f"Classified polygons bounds (lat/lon): {ll_bounds}")

    # Use the full extent but limit to reasonable S2 download size
    # Split into tiles if too large
    west, south, east, north = ll_bounds
    bbox_str = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
    print(f"Download bbox: {bbox_str}")

    # The area is ~100km x 200km — too large for single fetch
    # Focus on Maricunga area where most epithermal deposits are
    # Maricunga belt: roughly -27.2 to -26.2 S, -69.5 to -68.5 W
    bbox_str = "-69.50,-27.50,-68.40,-26.20"
    print(f"Focused bbox (Maricunga belt): {bbox_str}")

    # === 3. FIND CLEAR S2 SCENE ===
    print("\nSearching for clear S2 scenes...")
    search_cmd = [
        str(SURTGIS), "stac", "search",
        "--catalog", "es",
        "--collections", "sentinel-2-l2a",
        f"--bbox={bbox_str}",
        "--datetime", "2024-01-01T00:00:00Z/2024-03-31T23:59:59Z",
        "--limit", "20",
    ]
    result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=60)
    print(result.stdout[:2000] if result.stdout else result.stderr[:500])

    # Use January 2024 (summer, less clouds)
    date_range = "2024-01-15T00:00:00Z/2024-01-15T23:59:59Z"
    print(f"\nUsing date: 2024-01-15")

    # === 4. DOWNLOAD S2 BANDS ===
    band_assets = {
        "B02": "blue", "B03": "green", "B04": "red",
        "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
        "B08": "nir", "B8A": "nir08",
        "B11": "swir16", "B12": "swir22",
    }

    print(f"\nDownloading S2 bands to {DATA_DIR}...")
    for band_name, asset_name in band_assets.items():
        output = DATA_DIR / f"{band_name}.tif"
        print(f"  Downloading {band_name} ({asset_name})...")
        ok = download_s2_band(bbox_str, date_range, asset_name, output)
        if not ok:
            # Try alternative date
            for alt_date in ["2024-01-20T00:00:00Z/2024-01-20T23:59:59Z",
                             "2024-02-01T00:00:00Z/2024-02-01T23:59:59Z",
                             "2024-01-10T00:00:00Z/2024-01-10T23:59:59Z"]:
                print(f"    Retrying with {alt_date}...")
                ok = download_s2_band(bbox_str, alt_date, asset_name, output)
                if ok:
                    break

    # Verify downloads
    downloaded = sorted(DATA_DIR.glob("B*.tif"))
    print(f"\nDownloaded {len(downloaded)} bands")
    if len(downloaded) < 6:
        print("ERROR: Need at least SWIR bands. Aborting.")
        return

    # === 5. RASTERIZE ALTERATION POLYGONS ===
    print("\nRasterizing alteration polygons...")
    # Use B11 as reference grid (20m)
    with rasterio.open(DATA_DIR / "B11.tif") as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = ref.shape
        print(f"  Reference grid: {ref_shape[1]}x{ref_shape[0]} at 20m, CRS={ref_crs}")

    # Reproject polygons to match raster CRS
    classified_proj = classified.to_crs(ref_crs)

    # Rasterize
    shapes = [(geom, cls_id) for geom, cls_id in
              zip(classified_proj.geometry, classified_proj.class_id)]

    gt_raster = rasterize(
        shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    )
    print(f"  Rasterized: {(gt_raster > 0).sum()} pixels classified")

    # Save GT raster
    gt_meta = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": ref_shape[0], "width": ref_shape[1],
        "crs": ref_crs, "transform": ref_transform,
        "compress": "deflate", "nodata": 0,
    }
    gt_path = GT_DIR / "maricunga_ground_truth_atlas.tif"
    with rasterio.open(gt_path, "w", **gt_meta) as dst:
        dst.write(gt_raster, 1)
    print(f"  Saved: {gt_path}")

    # Print class pixel counts
    print("\n  Class distribution (pixels):")
    for cls_id, (cls_name, _) in CLASS_NAMES.items():
        n = (gt_raster == cls_id).sum()
        if n > 0:
            print(f"    {cls_id} {cls_name:25s}: {n:8d}")

    # === 6. EXTRACT TRAINING DATA ===
    print("\nExtracting training data...")
    # Load all 20m bands (or resample 10m to 20m)
    bands_20m_names = ["B05", "B06", "B07", "B8A", "B11", "B12"]
    bands_all_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    def load_band(name):
        path = DATA_DIR / f"{name}.tif"
        if not path.exists():
            return None
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
        # Resample 10m to 20m if needed
        if data.shape != ref_shape:
            h, w = data.shape
            h2, w2 = h // 2, w // 2
            data = data[:h2*2, :w2*2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))
            # Trim/pad to match
            result = np.zeros(ref_shape, dtype=np.float32)
            mh, mw = min(data.shape[0], ref_shape[0]), min(data.shape[1], ref_shape[1])
            result[:mh, :mw] = data[:mh, :mw]
            data = result
        # Mask invalid
        mask = (data > 0) & (data <= VALID_MAX)
        return np.where(mask, data * SCALE, np.nan)

    # 20m bands set
    bands_20m = {}
    for name in bands_20m_names:
        b = load_band(name)
        if b is not None:
            bands_20m[name] = b
            valid = np.count_nonzero(~np.isnan(b))
            print(f"  {name}: {valid} valid ({100*valid/b.size:.1f}%)")

    if len(bands_20m) < 4:
        print("ERROR: Not enough bands loaded")
        return

    # Stack and extract
    stack_20m = np.stack([bands_20m[b] for b in bands_20m_names if b in bands_20m], axis=-1)
    available_20m = [b for b in bands_20m_names if b in bands_20m]

    all_valid = np.all(~np.isnan(stack_20m), axis=-1) & (gt_raster > 0)
    rows, cols = np.where(all_valid)
    X_20m = stack_20m[rows, cols, :]
    y_20m = gt_raster[rows, cols]

    print(f"\n=== 20m Training Set ===")
    print(f"Samples: {len(y_20m)}, Bands: {available_20m}")
    for cls_id in range(1, 7):
        n = (y_20m == cls_id).sum()
        if n > 0:
            print(f"  {cls_id} {CLASS_NAMES[cls_id][0]:25s}: {n:6d}")

    np.savez_compressed(
        GT_DIR / "maricunga_training_atlas_20m.npz",
        X=X_20m, y=y_20m,
        band_names=available_20m,
        class_names=[CLASS_NAMES.get(i, ("?",""))[0] for i in range(7)],
        rows=rows, cols=cols,
    )
    print(f"Saved: {GT_DIR / 'maricunga_training_atlas_20m.npz'}")

    # All bands set
    bands_all = {}
    for name in bands_all_names:
        b = load_band(name)
        if b is not None:
            bands_all[name] = b

    if len(bands_all) >= 8:
        stack_all = np.stack([bands_all[b] for b in bands_all_names if b in bands_all], axis=-1)
        available_all = [b for b in bands_all_names if b in bands_all]
        all_valid_full = np.all(~np.isnan(stack_all), axis=-1) & (gt_raster > 0)
        rows_a, cols_a = np.where(all_valid_full)
        X_all = stack_all[rows_a, cols_a, :]
        y_all = gt_raster[rows_a, cols_a]

        print(f"\n=== All-bands Training Set ===")
        print(f"Samples: {len(y_all)}, Bands: {available_all}")
        for cls_id in range(1, 7):
            n = (y_all == cls_id).sum()
            if n > 0:
                print(f"  {cls_id} {CLASS_NAMES[cls_id][0]:25s}: {n:6d}")

        np.savez_compressed(
            GT_DIR / "maricunga_training_atlas_all.npz",
            X=X_all, y=y_all,
            band_names=available_all,
            class_names=[CLASS_NAMES.get(i, ("?",""))[0] for i in range(7)],
            rows=rows_a, cols=cols_a,
        )
        print(f"Saved: {GT_DIR / 'maricunga_training_atlas_all.npz'}")

    # === 7. FIGURE ===
    print("\nGenerating figure...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    cmap_colors = [CLASS_NAMES[i][1] for i in range(7)]
    cmap = ListedColormap(cmap_colors)
    im = ax.imshow(gt_raster, cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    ax.set_title("Maricunga — Ground Truth from Atlas Metalífero III Región\n(Field-mapped alteration zones)", fontsize=13)
    patches = [mpatches.Patch(color=CLASS_NAMES[i][1], label=f"{i}: {CLASS_NAMES[i][0]}")
               for i in range(7) if (gt_raster == i).sum() > 100]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    ax.set_axis_off()
    fig.savefig(FIG_DIR / "maricunga_ground_truth_atlas.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'maricunga_ground_truth_atlas.png'}")

    print("\nDone! Ready for PySR with field-validated ground truth.")


if __name__ == "__main__":
    main()
