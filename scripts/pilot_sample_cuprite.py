#!/usr/bin/env python3
"""
Hard-negative design for Cuprite, Nevada (site R5), mirroring pilot_sample_hard_negatives.py.

Positives : Sentinel-2 pixels inside the USGS alteration classes 1-4 of
            data/ground_truth/cuprite_ground_truth.tif, sampled on GEE with
            coordinates, class id (read from the raster) and a blob id
            (connected component of the altered mask) for grouped splits.
Ring      : 300 m - 2 km around the union of ALL mapped alteration. The stored
            raster is clipped to the Cuprite bbox, so the USGS WMS layers are
            re-downloaded for a bbox enlarged by ~15 km: ring pixels are only
            drawn where the map says "not altered", never on unmapped ground.
Distractors (criteria-based, outside the 2 km buffer, same composite):
            shadow (slope > 20 deg, low hillshade), vegetation (NDVI > 0.3),
            playa (flat, bright: slope < 2, B04 > 0.30, NDVI < 0.10 — the salar
            analogue), alluvium (flat, not bright, NDVI < 0.15), water (NDWI > 0.2).
Composite : same as cuprite_pipeline.py — S2_SR_HARMONIZED, 2024-06-01..09-30,
            cloud < 15 %, median, /10000.

Outputs: data/ground_truth/cuprite_hard_negatives.csv  (bands, site=R5, group, lon, lat)
         data/ground_truth/cuprite_positives_geo.csv   (bands, lon, lat, class_id, poly_id)
         data/ground_truth/cuprite_alteration_expanded.tif (diagnostic, not tracked)
"""
import time
from io import BytesIO
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, requests, ee
from PIL import Image
from rasterio import features
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

ee.Initialize()
GT = Path("data/ground_truth")
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
RENAME = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEED = 42
RING_IN, RING_OUT = 300, 2000
UTM = "EPSG:32611"
BBOX = (-117.28, 37.47, -117.10, 37.60)                 # stored ground truth
EXP = (-117.45, 37.35, -116.95, 37.72)                   # enlarged for ring / distractors
WMS = "https://mrdata.usgs.gov/services/alteration"
LAYERS = ["hydro_silica", "argillic", "phyllic", "carbonate", "epi_chlor"]
RES = 0.00025                                             # deg, ~22 m; keeps WIDTH < 2048


def wms_mask(layer, bbox, w, h):
    p = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer, "CRS": "EPSG:4326",
         "BBOX": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}", "WIDTH": w, "HEIGHT": h,
         "FORMAT": "image/png", "TRANSPARENT": "TRUE"}
    r = requests.get(WMS, params=p, timeout=120); r.raise_for_status()
    return np.array(Image.open(BytesIO(r.content)).convert("RGBA"))[:, :, 3] > 128


# ---------------------------------------------------------------- expanded alteration map (raster only)
# Memory note (2026-10-12): the first version polygonised the enlarged raster and buffered
# thousands of polygons with shapely; that blew the RAM. Everything spatial is now done on
# the raster with a distance transform, and GEE only receives rectangles plus the
# simplified geometry of the stored positives.
EXP_TIF = GT / "cuprite_alteration_expanded.tif"
w, h = int(round((EXP[2] - EXP[0]) / RES)), int(round((EXP[3] - EXP[1]) / RES))
tr = from_bounds(*EXP, w, h)
if EXP_TIF.exists():
    with rasterio.open(EXP_TIF) as src:
        alt = src.read(1).astype(bool); tr = src.transform; h, w = alt.shape
    print(f"reusing {EXP_TIF.name} ({w}x{h})")
else:
    alt = np.zeros((h, w), bool)
    for L in LAYERS:
        m = wms_mask(L, EXP, w, h); alt |= m
        print(f"WMS {L:12s}: {m.sum():7d} px", flush=True)
    with rasterio.open(EXP_TIF, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=tr, compress="deflate") as dst:
        dst.write(alt.astype(np.uint8), 1)
lon = EXP[0] + (np.arange(w) + 0.5) * RES; lat = EXP[3] - (np.arange(h) + 0.5) * RES
inside = (lon[None, :] >= BBOX[0]) & (lon[None, :] <= BBOX[2]) & (lat[:, None] >= BBOX[1]) & (lat[:, None] <= BBOX[3])
print(f"altered px inside stored bbox {np.sum(alt & inside)}, outside {np.sum(alt & ~inside)} "
      f"(outside share {np.sum(alt & ~inside) / max(alt.sum(), 1):.2f})")
# distance (m) from every cell to the nearest mapped alteration cell
px_m = (RES * 111320 * np.cos(np.deg2rad(37.5)), RES * 111320)
dist = ndimage.distance_transform_edt(~alt, sampling=(px_m[1], px_m[0]))
ring_mask = (dist > RING_IN) & (dist < RING_OUT)
print(f"ring area {ring_mask.sum() * px_m[0] * px_m[1] / 1e6:.1f} km2", flush=True)


def dist_at(lons, lats):
    """Distance to mapped alteration (m) for points, nan outside the enlarged raster."""
    c = np.floor((np.asarray(lons) - EXP[0]) / RES).astype(int); r = np.floor((EXP[3] - np.asarray(lats)) / RES).astype(int)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    out = np.full(len(c), np.nan); out[ok] = dist[r[ok], c[ok]]
    return out


# ---------------------------------------------------------------- positives geometry (stored GT)
with rasterio.open(GT / "cuprite_ground_truth.tif") as src:
    gt = src.read(1); gt_tr = src.transform
lab, nlab = ndimage.label(gt > 0)
pos_polys = [shape(g).simplify(2 * RES) for g, v in features.shapes((gt > 0).astype(np.uint8), mask=gt > 0, transform=gt_tr)]
pos_geom = unary_union(pos_polys).simplify(2 * RES)
print(f"positives: {nlab} blobs, {(gt > 0).sum()} px, geometry with "
      f"{sum(len(g.exterior.coords) for g in getattr(pos_geom, 'geoms', [pos_geom]))} vertices", flush=True)


# ---------------------------------------------------------------- GEE
def to_ee(geom): return ee.Geometry(mapping(geom), None, False)


aoi = ee.Geometry.Rectangle(list(EXP))
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
      .filterDate("2024-06-01", "2024-09-30").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15)).select(S2_BANDS))
img = s2.median().divide(10000).rename(RENAME)
dem = ee.Image("USGS/SRTMGL1_003"); slope = ee.Terrain.slope(dem)
shade = ee.Terrain.hillshade(dem, 135, 45)              # S2 ~10:30 local, boreal summer: sun SE; 45 deg keeps shaded NW slopes
ndvi = img.normalizedDifference(["B08", "B04"]).rename("ndvi"); ndwi = img.normalizedDifference(["B03", "B08"]).rename("ndwi")
attrs = img.addBands([slope.rename("slope"), shade.rename("shade"), ndvi, ndwi])
shadow = slope.gt(20).And(shade.lt(110))
playa = slope.lt(2).And(img.select("B04").gt(0.30)).And(ndvi.lt(0.10))
alluv = slope.lt(3).And(img.select("B04").lt(0.30)).And(ndvi.lt(0.15))


def download(fc, n):
    size = fc.size().getInfo(); rows = []
    for start in range(0, min(size, n), 1000):
        for f in fc.toList(min(1000, size - start), start).getInfo():
            p = f.get("properties", {}); c = (f.get("geometry") or {}).get("coordinates")
            if c: p["lon"], p["lat"] = c[0], c[1]
            rows.append(p)
    return pd.DataFrame(rows)


def sample(image, region, n, group, seed=SEED):
    t = time.time()
    try:
        df = download(image.sample(region=region, scale=20, numPixels=n, seed=seed, geometries=True), min(n, 60000))
    except Exception as e:
        print(f"  {group}: ERROR {e}", flush=True); return pd.DataFrame()
    if len(df): df = df.dropna(subset=[b for b in RENAME if b in df.columns])
    df["site"], df["group"] = "R5", group
    print(f"  {group:12s} {len(df):5d} px ({time.time() - t:.0f}s)", flush=True)
    return df


print("sampling", flush=True)
pos = sample(img, to_ee(pos_geom), 4000, "positive")
# ring: uniform sample over the enlarged AOI, kept where the local distance map says 300 m < d < 2 km
ring = sample(attrs, aoi, 60000, "ring_candidates")
ring["dist"] = dist_at(ring.lon, ring.lat)
ring = ring[(ring.dist > RING_IN) & (ring.dist < RING_OUT)].drop(columns="dist").head(4000)
ring["group"] = "ring"; print(f"  ring         {len(ring):5d} px kept", flush=True)
# distractors: image criteria on GEE, mapped-alteration distance checked locally.
# GEE draws numPixels uniformly and only then applies the mask, so sparse classes
# need a dense draw (DENSE) to return a few thousand pixels.
DENSE = 400000
# playas: the salar analogue. Clayton Valley (west) and Stonewall Flat / Lida Valley
# (south) lie outside EXP, so a wider box is used; points outside the enlarged
# alteration raster have no distance (nan) and are accepted — they are > 15 km
# from any mapped alteration of this district by construction of EXP.
playa_aoi = ee.Geometry.Rectangle([-117.80, 37.20, -116.95, 37.95])
parts = [ring]
for group, mask, region, n, dmin in [("shadow", shadow, aoi, 1500, RING_IN), ("vegetation", ndvi.gt(0.3), aoi, 800, RING_IN),
                                     ("playa", playa, playa_aoi, 800, RING_OUT), ("alluvium", alluv, aoi, 800, RING_OUT),
                                     ("water", ndwi.gt(0.2), playa_aoi, 800, RING_IN)]:
    d = sample(attrs.updateMask(mask), region, DENSE, group)
    if len(d):
        dd = dist_at(d.lon, d.lat)
        keep = (dd > dmin) | (np.isnan(dd) if group in ("playa", "water") else False)
        d = d[keep].head(n)
        print(f"  {group:12s} {len(d):5d} px kept (>{dmin} m from mapped alteration)", flush=True)
        parts.append(d)

# positives: class and blob id from the stored raster
with rasterio.open(GT / "cuprite_ground_truth.tif") as src:
    rc = [src.index(x, y) for x, y in zip(pos.lon, pos.lat)]
r = np.clip([i for i, j in rc], 0, gt.shape[0] - 1); c = np.clip([j for i, j in rc], 0, gt.shape[1] - 1)
pos["class_id"] = gt[r, c]; pos["poly_id"] = lab[r, c]
pos = pos[pos.class_id > 0]
print("positives by class:", pos.class_id.value_counts().sort_index().to_dict(), f"; blobs {pos.poly_id.nunique()}")
pos[RENAME + ["lon", "lat", "class_id", "poly_id"]].to_csv(GT / "cuprite_positives_geo.csv", index=False)

neg = pd.concat([p for p in parts if len(p)], ignore_index=True)
neg = neg[RENAME + ["site", "group", "lon", "lat"]].reset_index(drop=True)
neg.to_csv(GT / "cuprite_hard_negatives.csv", index=False)
print(neg.groupby("group").size().to_string())
print("saved", GT / "cuprite_hard_negatives.csv", "and", GT / "cuprite_positives_geo.csv")
