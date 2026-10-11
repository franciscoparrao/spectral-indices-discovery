#!/usr/bin/env python3
"""
Recover coordinates and Atlas polygon ids for the Region III positives of
binary_training_s2.csv.

full_pipeline.py sampled the 8 000 altered pixels with
composite.sample(region=all_altered, scale=20, numPixels=8000, seed=42,
geometries=False). GEE sampling is deterministic for a fixed seed, region and
image, so repeating the call with geometries=True returns the same pixels with
their coordinates. Each pixel is then matched back to binary_training_s2.csv by
band values and joined to the Atlas polygon that contains it.

Output: data/ground_truth/binary_training_s2_positives_geo.csv
        (bands, lon, lat, poly_id, INT_ORIG, class_id, matched)
"""
import time
from pathlib import Path
import ee, numpy as np, pandas as pd, geopandas as gpd

ee.Initialize()
ATLAS = Path("data/external/atlas_metalifero_IIIR"); GT = Path("data/ground_truth")
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
RENAME = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ALTERATION_MAP = {
    "Alteración Silicea": 1, "vuggy silica": 1,
    "Alteracion Argilica y Argilica avanzada": 2, "Alteracion Solfatárica": 2,
    "Alteracion Argilica": 3, "Alteracion Sericitica": 3,
    "Alteración Cuarzo-Sericitica(Fílica)": 3,
    "Alteracion Propilitica": 4, "Oxidos e Hidróxidos de Hierro": 5,
    "Alteracion Potasica": 6, "skarn": 6,
}

# --- polygons exactly as full_pipeline.py -----------------------------------
gdf = gpd.read_file(ATLAS / "Geometria/RMM_ALTERACI.shp")
attrs = pd.read_csv(ATLAS / "alteracion.csv")
gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
gdf = gdf.set_crs("EPSG:32719", allow_override=True)
gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)
gdf["poly_id"] = np.arange(len(gdf))
classified = gdf[gdf["class_id"].isin([1, 2, 3, 4, 5, 6])]
undiff = gdf[gdf["ALTERACION"] == "Alteracion hidrotermal indiferenciada (arg, lim y sil)"]
all_altered = pd.concat([classified, undiff], ignore_index=True).to_crs("EPSG:4326")
print(f"polygons: classified {len(classified)}, undifferentiated {len(undiff)}, used {len(all_altered)}")


def polygon_to_ee(geom):
    if geom.geom_type == "Polygon":
        return ee.Feature(ee.Geometry.Polygon([[list(c) for c in geom.exterior.coords]]))
    if geom.geom_type == "MultiPolygon":
        return ee.Feature(ee.Geometry.MultiPolygon([[list(c) for c in p.exterior.coords] for p in geom.geoms]))
    return None


feats = [f for f in (polygon_to_ee(g) for g in all_altered.geometry) if f is not None]
region = ee.FeatureCollection(feats).geometry()

aoi = ee.Geometry.Rectangle([-69.6, -28.1, -68.5, -25.9])
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
      .filterDate("2024-01-01", "2024-04-01")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(S2_BANDS))
composite = s2.median().divide(10000).rename(RENAME)

t = time.time()
fc = composite.sample(region=region, scale=20, numPixels=8000, seed=42, geometries=True)
size = fc.size().getInfo(); rows = []
for start in range(0, min(size, 8000), 1000):
    for f in fc.toList(min(1000, size - start), start).getInfo():
        p = f.get("properties", {}); c = (f.get("geometry") or {}).get("coordinates")
        if c: p["lon"], p["lat"] = c[0], c[1]
        rows.append(p)
new = pd.DataFrame(rows).dropna(subset=RENAME)
print(f"resampled {len(new)} pixels in {time.time()-t:.0f}s")

# --- match to the stored positives by band values ----------------------------
old = pd.read_csv(GT / "binary_training_s2.csv")
old = old[old.is_altered == 1].reset_index(drop=True)
key = lambda d: d[RENAME].round(6).astype(str).agg("|".join, axis=1)
new["k"] = key(new); old["k"] = key(old)
new["matched"] = new.k.isin(set(old.k))
print(f"matched to binary_training_s2 positives: {new.matched.sum()}/{len(old)} "
      f"(unmatched resampled: {(~new.matched).sum()}, duplicate keys in old: {old.k.duplicated().sum()})")

# --- polygon id ----------------------------------------------------------------
pts = gpd.GeoDataFrame(new, geometry=gpd.points_from_xy(new.lon, new.lat), crs=4326).to_crs(32719)
polys = gdf[["poly_id", "INT_ORIG", "class_id", "geometry"]].copy()
polys["geometry"] = polys.geometry.buffer(0)
j = gpd.sjoin(pts, polys, how="left", predicate="within")
j = j[~j.index.duplicated(keep="first")]          # overlapping polygons: keep first
miss = j.poly_id.isna()
if miss.any():                                     # edge pixels: nearest polygon
    near = gpd.sjoin_nearest(j[miss].drop(columns=["index_right", "poly_id", "INT_ORIG", "class_id"]),
                             polys, how="left", distance_col="dist")
    near = near[~near.index.duplicated(keep="first")]
    j.loc[miss, ["poly_id", "INT_ORIG", "class_id"]] = near[["poly_id", "INT_ORIG", "class_id"]].values
    print(f"pixels outside every polygon: {miss.sum()} (nearest at median {near.dist.median():.0f} m, max {near.dist.max():.0f} m)")
out = pd.DataFrame(j.drop(columns=["geometry", "index_right", "k"]))
out["poly_id"] = out.poly_id.astype(int)
print(f"distinct polygons: {out.poly_id.nunique()}; pixels per polygon median {out.poly_id.value_counts().median():.0f}, "
      f"max {out.poly_id.value_counts().max()}")
out.to_csv(GT / "binary_training_s2_positives_geo.csv", index=False)
print("saved", GT / "binary_training_s2_positives_geo.csv")
