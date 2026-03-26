#!/usr/bin/env python3
"""
Create initial ground truth for El Tatio hydrothermal alteration mapping.
Uses spectral index thresholds + known locations to delineate training zones.

Classes:
  1 = Silicic (sinter, opal-A, high silica)
  2 = Advanced argillic (alunite, kaolinite)
  3 = Argillic (illite, smectite, montmorillonite)
  4 = Propylitic (chlorite, epidote)
  5 = Iron oxide (goethite, hematite, jarosite)
  6 = Unaltered (fresh volcanic rock, ignimbrite)
  7 = Vegetation
  0 = No data / unclassified
"""

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box, Point, mapping
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = Path("data/sentinel2/el_tatio")
GT_DIR = Path("data/ground_truth")
FIG_DIR = Path("figures")
GT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 1.0 / 10000.0
VALID_MAX = 10000

# Class definitions
CLASSES = {
    0: ("No data", "#000000"),
    1: ("Silicic", "#FFFFFF"),
    2: ("Adv. Argillic", "#FF6600"),
    3: ("Argillic", "#FFCC00"),
    4: ("Propylitic", "#00AA00"),
    5: ("Iron Oxide", "#CC0000"),
    6: ("Unaltered", "#888888"),
    7: ("Vegetation", "#00FF00"),
}


def load_band_20m(name):
    """Load band at 20m, block-average 10m bands."""
    bands_10m = {"B02", "B03", "B04", "B08"}
    with rasterio.open(DATA_DIR / f"{name}.tif") as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    if name in bands_10m:
        h, w = data.shape
        h2, w2 = h // 2, w // 2
        data = data[:h2*2, :w2*2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))

    # Mask invalid
    mask = (data > 0) & (data <= VALID_MAX)
    return np.where(mask, data * SCALE, np.nan)


def safe_ratio(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b != 0, a / b, np.nan)


def norm_diff(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((a + b) != 0, (a - b) / (a + b), np.nan)


def main():
    print("Loading bands...")
    bands = {}
    for name in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]:
        bands[name] = load_band_20m(name)

    # Get reference metadata from B11 (native 20m)
    with rasterio.open(DATA_DIR / "B11.tif") as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    shape = bands["B11"].shape
    # Trim/pad 10m bands to match 20m shape
    for name in ["B02", "B03", "B04", "B08"]:
        b = bands[name]
        result = np.full(shape, np.nan, dtype=np.float32)
        mh, mw = min(b.shape[0], shape[0]), min(b.shape[1], shape[1])
        result[:mh, :mw] = b[:mh, :mw]
        bands[name] = result

    print(f"Grid: {shape[1]}x{shape[0]} at 20m, CRS={crs}")

    # === COMPUTE INDICES ===
    print("Computing indices...")

    # Clay ratio: B11/B12 — high = OH-bearing clay minerals
    clay_ratio = safe_ratio(bands["B11"], bands["B12"])

    # Iron oxide: B04/B02 — high = Fe³⁺
    iron_oxide = safe_ratio(bands["B04"], bands["B02"])

    # Alunite/kaolinite: (B11-B12)/(B11+B12) — high = Al-OH
    alunite_idx = norm_diff(bands["B11"], bands["B12"])

    # NDVI: (B8A-B04)/(B8A+B04)
    ndvi = norm_diff(bands["B8A"], bands["B04"])

    # Silica proxy: B12/B11 — low clay ratio indicates silicification
    silica_proxy = safe_ratio(bands["B12"], bands["B11"])

    # Ferrous minerals: B12/B8A — propylitic indicator
    ferrous = safe_ratio(bands["B12"], bands["B8A"])

    # SWIR brightness (general alteration indicator)
    swir_bright = (bands["B11"] + bands["B12"]) / 2.0

    # Valid data mask (need at least SWIR bands)
    valid = ~np.isnan(bands["B11"]) & ~np.isnan(bands["B12"])

    print(f"Valid pixels: {valid.sum()} ({100*valid.sum()/valid.size:.1f}%)")

    # === CLASSIFICATION BY THRESHOLDS ===
    # Based on spectral properties of alteration minerals in Sentinel-2 bands:
    #
    # Silicic: High overall reflectance, low clay ratio (no deep OH absorption)
    #   → high SWIR brightness + silica_proxy < 0.85 (B12 < B11)
    #
    # Advanced argillic: Strong Al-OH absorption at 2.17-2.20 µm → B12 drops
    #   → high alunite_idx (B11 >> B12) + moderate SWIR brightness
    #
    # Argillic: Moderate Al-OH absorption
    #   → moderate positive alunite_idx + lower iron oxide
    #
    # Propylitic: Mg-OH absorption, chlorite → B12 relatively high, NIR moderate
    #   → clay_ratio < 1.0 (B12 > B11) + low iron + moderate ferrous
    #
    # Iron oxide: High B04/B02, diagnostic of Fe³⁺
    #   → high iron_oxide + moderate/low clay ratio
    #
    # Vegetation: NDVI > 0.3
    #
    # Unaltered: None of the above, moderate values

    print("Classifying...")
    gt = np.zeros(shape, dtype=np.uint8)

    # Print index percentiles to guide thresholds
    print("\nIndex percentiles (valid pixels):")
    for idx_name, idx_data in [("clay_ratio", clay_ratio), ("alunite_idx", alunite_idx),
                                ("iron_oxide", iron_oxide), ("silica_proxy", silica_proxy),
                                ("swir_bright", swir_bright), ("ferrous", ferrous), ("ndvi", ndvi)]:
        v = idx_data[valid & ~np.isnan(idx_data)]
        if len(v) > 0:
            print(f"  {idx_name:15s}: p5={np.percentile(v,5):.3f}  p25={np.percentile(v,25):.3f}  "
                  f"p50={np.percentile(v,50):.3f}  p75={np.percentile(v,75):.3f}  p95={np.percentile(v,95):.3f}")

    # Order matters: later rules overwrite earlier ones
    # Thresholds calibrated from index distributions

    # 6. Unaltered — default for all valid pixels
    gt[valid] = 6

    # 4. Propylitic: B12 > B11 (clay_ratio < 0.90) — Mg-OH absorption
    #    Chlorite, epidote: stronger absorption at B12 relative to B11
    propylitic = valid & (clay_ratio < 0.90) & ~np.isnan(ferrous)
    gt[propylitic] = 4

    # 3. Argillic: moderate Al-OH absorption (positive alunite_idx)
    #    Illite, smectite: moderate B11>B12
    argillic = valid & (alunite_idx > 0.03) & (alunite_idx <= 0.18)
    gt[argillic] = 3

    # 2. Advanced argillic: strong Al-OH (high alunite_idx, top ~5%)
    #    Alunite, kaolinite, pyrophyllite: strong B11>>B12
    adv_argillic = valid & (alunite_idx > 0.18)
    gt[adv_argillic] = 2

    # 1. Silicic: high SWIR brightness + flat SWIR slope (B11 ≈ B12)
    #    Sinter/opal-A: high reflectance, no deep OH absorption
    silicic = valid & (swir_bright > 0.15) & (np.abs(alunite_idx) < 0.05) & (swir_bright > np.nanpercentile(swir_bright[valid], 80))
    gt[silicic] = 1

    # 5. Iron oxide: high Fe³⁺ ratio (where VNIR available)
    iron_mask = ~np.isnan(iron_oxide)
    iron_ox = valid & iron_mask & (iron_oxide > 1.8) & (alunite_idx < 0.10)
    gt[iron_ox] = 5

    # 7. Vegetation: NDVI > 0.25 (arid zone, lower threshold)
    veg_mask = ~np.isnan(ndvi)
    vegetation = valid & veg_mask & (ndvi > 0.25)
    gt[vegetation] = 7

    # === STATISTICS ===
    print("\n=== Ground Truth Statistics ===")
    total_valid = valid.sum()
    for cls_id, (cls_name, _) in CLASSES.items():
        n = (gt == cls_id).sum()
        if n > 0:
            pct = 100 * n / total_valid if cls_id > 0 else 100 * n / gt.size
            print(f"  {cls_id} {cls_name:20s}: {n:8d} pixels ({pct:5.1f}%)")

    # === SAVE RASTER ===
    out_meta = meta.copy()
    out_meta.update(dtype='uint8', count=1, compress='deflate', nodata=0)
    gt_path = GT_DIR / "el_tatio_ground_truth.tif"
    with rasterio.open(gt_path, 'w', **out_meta) as dst:
        dst.write(gt, 1)
    print(f"\nSaved: {gt_path}")

    # === SAVE AS GEOJSON (polygonized, simplified) ===
    # Export class centroids + bounding regions for reference
    features = []
    for cls_id in range(1, 8):
        mask = gt == cls_id
        if not mask.any():
            continue
        rows, cols = np.where(mask)
        # Bounding box of class in geographic coordinates
        min_col, max_col = cols.min(), cols.max()
        min_row, max_row = rows.min(), rows.max()

        # Convert pixel coords to geographic
        west, north = transform * (min_col, min_row)
        east, south = transform * (max_col + 1, max_row + 1)

        # Centroid
        cy, cx = rows.mean(), cols.mean()
        cx_geo, cy_geo = transform * (cx, cy)

        features.append({
            "type": "Feature",
            "geometry": mapping(Point(cx_geo, cy_geo)),
            "properties": {
                "class_id": int(cls_id),
                "class_name": CLASSES[cls_id][0],
                "pixel_count": int(mask.sum()),
                "bbox_west": float(west),
                "bbox_east": float(east),
                "bbox_north": float(north),
                "bbox_south": float(south),
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs.to_epsg()}"}},
        "features": features,
    }
    geojson_path = GT_DIR / "el_tatio_class_centroids.geojson"
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved: {geojson_path}")

    # === FIGURE ===
    print("Generating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 1. Ground truth map
    ax = axes[0]
    cmap_colors = [CLASSES[i][1] for i in range(8)]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)
    im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=7, interpolation='nearest')
    ax.set_title("Ground Truth Classification", fontsize=13)
    patches = [mpatches.Patch(color=CLASSES[i][1], label=f"{i}: {CLASSES[i][0]}")
               for i in range(8) if (gt == i).sum() > 0]
    ax.legend(handles=patches, loc='lower right', fontsize=8)
    ax.set_axis_off()

    # 2. Clay ratio
    ax = axes[1]
    v = clay_ratio.copy()
    v[~valid] = np.nan
    vmin, vmax = np.nanpercentile(v[~np.isnan(v)], [2, 98])
    ax.imshow(v, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    ax.set_title("Clay Ratio (B11/B12)", fontsize=13)
    ax.set_axis_off()

    # 3. Alunite index
    ax = axes[2]
    v = alunite_idx.copy()
    v[~valid] = np.nan
    vmin, vmax = np.nanpercentile(v[~np.isnan(v)], [2, 98])
    ax.imshow(v, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    ax.set_title("Alunite Index (B11-B12)/(B11+B12)", fontsize=13)
    ax.set_axis_off()

    fig.suptitle("El Tatio — Initial Ground Truth from Spectral Indices", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "el_tatio_ground_truth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'el_tatio_ground_truth.png'}")

    # === EXPORT TRAINING DATA ===
    print("\nExtracting training data...")

    # Set 1: 20m bands only (B05-B8A, B11, B12) — maximum coverage
    bands_20m = ["B05", "B06", "B07", "B8A", "B11", "B12"]
    stack_20m = np.stack([bands[b] for b in bands_20m], axis=-1)
    valid_20m = np.all(~np.isnan(stack_20m), axis=-1) & (gt > 0)
    rows_20m, cols_20m = np.where(valid_20m)
    X_20m = stack_20m[rows_20m, cols_20m, :]
    y_20m = gt[rows_20m, cols_20m]

    print(f"\n--- 20m bands (primary set for PySR) ---")
    print(f"Training samples: {len(y_20m)}")
    for cls_id in range(1, 8):
        n = (y_20m == cls_id).sum()
        if n > 0:
            print(f"  {cls_id} {CLASSES[cls_id][0]:20s}: {n:8d}")

    np.savez_compressed(
        GT_DIR / "el_tatio_training_20m.npz",
        X=X_20m, y=y_20m,
        band_names=bands_20m,
        class_names=[CLASSES[i][0] for i in range(8)],
        rows=rows_20m, cols=cols_20m,
        transform=np.array(transform).reshape(3, 3),
        crs_epsg=crs.to_epsg(),
    )
    print(f"Saved: {GT_DIR / 'el_tatio_training_20m.npz'}")

    # Set 2: All 10 bands — where 10m VNIR is available
    bands_all = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    stack_all = np.stack([bands[b] for b in bands_all], axis=-1)
    valid_all = np.all(~np.isnan(stack_all), axis=-1) & (gt > 0)
    rows_all, cols_all = np.where(valid_all)
    X_all = stack_all[rows_all, cols_all, :]
    y_all = gt[rows_all, cols_all]

    print(f"\n--- All 10 bands (secondary set) ---")
    print(f"Training samples: {len(y_all)}")
    for cls_id in range(1, 8):
        n = (y_all == cls_id).sum()
        if n > 0:
            print(f"  {cls_id} {CLASSES[cls_id][0]:20s}: {n:8d}")

    np.savez_compressed(
        GT_DIR / "el_tatio_training_all.npz",
        X=X_all, y=y_all,
        band_names=bands_all,
        class_names=[CLASSES[i][0] for i in range(8)],
        rows=rows_all, cols=cols_all,
        transform=np.array(transform).reshape(3, 3),
        crs_epsg=crs.to_epsg(),
    )
    print(f"Saved: {GT_DIR / 'el_tatio_training_all.npz'}")
    print("\nReady for PySR!")


if __name__ == "__main__":
    main()
