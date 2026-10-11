#!/usr/bin/env python3
"""
PILOT (2026-10): hard-negative sampling for the alteration index question.

Samples, with the SAME Sentinel-2 composite used for the paper's training data
(S2_SR_HARMONIZED, 2024-01-01..2024-04-01, CLOUDY_PIXEL_PERCENTAGE < 20, median/1e4, 20 m):

  Region III (Atacama, Atlas Metalífero polygons)
    ring       : 300 m .. 2 km ring around every alteration polygon (adjacent host rock)
    litho_*    : six hand-placed lithology boxes reused from exp_false_positives.py
    salar_*    : Maricunga, Pedernales (in-region), Atacama, Uyuni
    snow, water, shadow, vegetation, sand
  Region IV (Coquimbo, SERNAGEOMIN sheets M47/M120/M159/M162)
    ring, shadow, vegetation

Positives are NOT resampled: the pilot reuses data/ground_truth/binary_training_s2.csv
(is_altered == 1) and data/ground_truth/region_iv_validation_s2.csv.

Output: data/ground_truth/pilot_hard_negatives.csv  (bands + site + group + lon + lat)
"""
import json, sys, time
from pathlib import Path
import ee, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union

ee.Initialize()

GT_DIR = Path("data/ground_truth")
OUT = GT_DIR / "pilot_hard_negatives.csv"
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
RENAME = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEED = 42
RING_IN, RING_OUT = 300, 2000   # metres


def composite(aoi):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
          .filterDate("2024-01-01", "2024-04-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(S2_BANDS))
    return s2.median().divide(10000).rename(RENAME)


def to_ee(geom):
    return ee.Geometry(mapping(geom), None, False)


def download(fc, max_n):
    size = fc.size().getInfo()
    rows = []
    for start in range(0, min(size, max_n), 1000):
        feats = fc.toList(min(1000, size - start), start).getInfo()
        for f in feats:
            p = f.get("properties", {})
            c = (f.get("geometry") or {}).get("coordinates")
            if c:
                p["lon"], p["lat"] = c[0], c[1]
            rows.append(p)
    return pd.DataFrame(rows)


def sample(img, region, n, site, group, scale=20):
    t = time.time()
    try:
        fc = img.sample(region=region, scale=scale, numPixels=n, seed=SEED, geometries=True)
        df = download(fc, n)
    except Exception as e:
        print(f"  [{site}] {group}: ERROR {e}", flush=True)
        return pd.DataFrame()
    df = df.dropna(subset=[b for b in RENAME if b in df.columns]) if len(df) else df
    df["site"], df["group"] = site, group
    print(f"  [{site}] {group:14s} {len(df):5d} px  ({time.time()-t:.0f}s)", flush=True)
    return df


def ring_geometry(gdf_proj):
    """Ring RING_IN..RING_OUT m around the union of all polygons (projected CRS)."""
    u = unary_union(gdf_proj.geometry.buffer(0))
    ring = u.buffer(RING_OUT).difference(u.buffer(RING_IN)).simplify(30)
    return gpd.GeoSeries([ring], crs=gdf_proj.crs).to_crs(4326).iloc[0]


def terrain_masks(aoi):
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    # S2 overpass ~10:30 local, austral summer: sun roughly NE, high elevation
    shade = ee.Terrain.hillshade(dem, 65, 60)
    shadow = slope.gt(20).And(shade.lt(120))
    return dem, shadow


def main():
    parts = []

    # ------------------------------------------------------------------ Region III
    print("Region III", flush=True)
    atlas = Path("data/external/atlas_metalifero_IIIR")
    g3 = gpd.read_file(atlas / "Geometria/RMM_ALTERACI.shp").set_crs("EPSG:32719", allow_override=True)
    g3_4326 = g3.to_crs(4326)
    b = g3_4326.total_bounds
    # wide enough to cover the lithology boxes (west to -70.5) and the ring (east to -68.37)
    aoi3 = ee.Geometry.Rectangle([min(b[0] - 0.3, -70.6), min(b[1] - 0.3, -28.3), b[2] + 0.3, b[3] + 0.3])
    img3 = composite(aoi3)
    excl3 = to_ee(gpd.GeoSeries([unary_union(g3.geometry.buffer(0)).buffer(RING_IN).simplify(30)],
                                crs=g3.crs).to_crs(4326).iloc[0])
    not_alt3 = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(excl3)]), 1).Not()  # 1 outside buffered polygons
    dem3, shadow3 = terrain_masks(aoi3)
    ndvi = img3.normalizedDifference(["B08", "B04"])
    ndsi = img3.normalizedDifference(["B03", "B11"])
    ndwi = img3.normalizedDifference(["B03", "B08"])

    parts.append(sample(img3, to_ee(ring_geometry(g3)), 8000, "R3", "ring"))

    litho = {  # reused verbatim from exp_false_positives.py
        "litho_andesite": [-69.8, -27.5, -69.7, -27.4],
        "litho_granitoid": [-70.1, -27.8, -70.0, -27.7],
        "litho_sedimentary": [-70.3, -27.3, -70.2, -27.2],
        "litho_basalt": [-69.6, -28.0, -69.5, -27.9],
        "litho_alluvium": [-69.4, -26.7, -69.3, -26.6],
        "litho_greenschist": [-70.5, -28.2, -70.4, -28.1],
    }
    for k, r in litho.items():
        parts.append(sample(img3.updateMask(not_alt3), ee.Geometry.Rectangle(r), 800, "R3", k))

    boxes = {
        "salar_maricunga": [-69.15, -27.12, -69.02, -26.95],
        "salar_pedernales": [-69.20, -26.35, -69.05, -26.20],
    }
    for k, r in boxes.items():
        parts.append(sample(img3.updateMask(not_alt3), ee.Geometry.Rectangle(r), 800, "R3", k))
    # Distant salars: own composites (outside aoi3)
    for k, r in {"salar_atacama": [-68.35, -23.55, -68.25, -23.45],
                 "salar_uyuni": [-67.7, -20.3, -67.6, -20.2],
                 "sand_atacama": [-69.8, -24.5, -69.7, -24.4]}.items():
        geo = ee.Geometry.Rectangle(r)
        parts.append(sample(composite(geo), geo, 800, "R3", k))

    snow_box = ee.Geometry.Rectangle([-68.9, -27.35, -68.3, -26.9])
    parts.append(sample(img3.updateMask(ndsi.gt(0.4).And(dem3.gt(5000)).And(not_alt3)),
                        snow_box, 800, "R3", "snow"))
    water_box = ee.Geometry.MultiPolygon([ee.Geometry.Rectangle([-69.22, -27.12, -69.08, -27.03]).coordinates(),
                                          ee.Geometry.Rectangle([-68.6, -27.0, -68.4, -26.85]).coordinates()])
    parts.append(sample(img3.updateMask(ndwi.gt(0.2).And(not_alt3)), water_box, 800, "R3", "water"))
    parts.append(sample(img3.updateMask(shadow3.And(not_alt3)), aoi3, 1500, "R3", "shadow"))
    veg_box = ee.Geometry.Rectangle([-70.45, -27.5, -70.15, -27.25])
    parts.append(sample(img3.updateMask(ndvi.gt(0.3).And(not_alt3)), veg_box, 800, "R3", "vegetation"))

    # ------------------------------------------------------------------ Region IV
    print("Region IV", flush=True)
    shp_dir = Path("data/external/shapefiles_alteracion")
    g4 = pd.concat([gpd.read_file(p).to_crs("EPSG:32719")[["geometry"]]
                    for p in sorted(shp_dir.glob("*/GB_ALTERACION_P.shp"))], ignore_index=True)
    g4 = gpd.GeoDataFrame(g4, crs="EPSG:32719")
    b = g4.to_crs(4326).total_bounds
    aoi4 = ee.Geometry.Rectangle([b[0] - 0.2, b[1] - 0.2, b[2] + 0.2, b[3] + 0.2])
    img4 = composite(aoi4)
    excl4 = to_ee(gpd.GeoSeries([unary_union(g4.geometry.buffer(0)).buffer(RING_IN).simplify(30)],
                                crs=g4.crs).to_crs(4326).iloc[0])
    not_alt4 = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(excl4)]), 1).Not()
    dem4, shadow4 = terrain_masks(aoi4)
    ndvi4 = img4.normalizedDifference(["B08", "B04"])

    parts.append(sample(img4, to_ee(ring_geometry(g4)), 7000, "R4", "ring"))
    parts.append(sample(img4.updateMask(shadow4.And(not_alt4)), aoi4, 1500, "R4", "shadow"))
    parts.append(sample(img4.updateMask(ndvi4.gt(0.3).And(not_alt4)), aoi4, 800, "R4", "vegetation"))

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    cols = RENAME + ["site", "group", "lon", "lat"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}: {len(df)} rows")
    print(df.groupby(["site", "group"]).size().to_string())


if __name__ == "__main__":
    main()
