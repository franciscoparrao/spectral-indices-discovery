#!/usr/bin/env python3
"""
(A) Hard-negative design for a third, untouched test site, R6 "Atacama West":
    alteration polygons of five SERNAGEOMIN sheets west of the Maricunga belt
    (La Guardia, Carrera Pinto, Los Loros, Quebrada Salitrosa, El Salvador;
    data/external/sernageomin_atacama_oeste/). Same scheme as Coquimbo/Cuprite:
    positives inside polygons, host-rock ring 0.3-2 km, criteria-based distractors.
    To keep R6 independent of the selection site, every R6 sample within 5 km of an
    Atlas III (selection) alteration polygon is dropped.
(B) Prediction-area inputs for all sites: the 10 bands (+NDVI, NDWI) at every mapped
    deposit/occurrence and at a uniform random sample of each site's area of interest.

Composites: Chile = S2_SR_HARMONIZED Jan-Mar 2024, cloud < 20 %; Cuprite = Jun-Sep 2024,
cloud < 15 % (identical to the existing design).
Outputs (data/ground_truth/):
  site6_positives_geo.csv, site6_hard_negatives.csv
  pa_<site>_deposits.csv, pa_<site>_background.csv   (site in R3, R4, R5, R6)
"""
import time
from pathlib import Path
import ee, numpy as np, pandas as pd, geopandas as gpd
from shapely import wkt
from shapely.geometry import mapping, box
from shapely.ops import unary_union

ee.Initialize()
GT = Path("data/ground_truth"); EXT = Path("data/external")
S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEED, RING_IN, RING_OUT, INDEP = 42, 300, 2000, 5000
UTM = "EPSG:32719"


def composite(aoi, start, end, cloud):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)).select(S2))
    img = s2.median().divide(10000).rename(BANDS)
    return img.addBands([img.normalizedDifference(["B08", "B04"]).rename("ndvi"),
                         img.normalizedDifference(["B03", "B08"]).rename("ndwi")])


def to_ee(g):
    return ee.Geometry(mapping(g), None, False)


def download(fc, n):
    size = fc.size().getInfo(); rows = []
    for s in range(0, min(size, n), 1000):
        for f in fc.toList(min(1000, size - s), s).getInfo():
            p = f.get("properties", {}); c = (f.get("geometry") or {}).get("coordinates")
            if c: p["lon"], p["lat"] = c[0], c[1]
            rows.append(p)
    return pd.DataFrame(rows)


def sample(img, region, n, group, keep=None):
    t = time.time()
    try:
        df = download(img.sample(region=region, scale=20, numPixels=n, seed=SEED, geometries=True), keep or n)
    except Exception as e:
        print(f"  {group}: ERROR {e}", flush=True); return pd.DataFrame()
    df = df.dropna(subset=[b for b in BANDS if b in df.columns]) if len(df) else df
    df["group"] = group
    print(f"  {group:14s} {len(df):6d} px ({time.time() - t:.0f}s)", flush=True)
    return df


def at_points(img, df, batch=3000):
    rows = []
    for s in range(0, len(df), batch):
        ch = df.iloc[s:s + batch]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x, y]), {"i": int(i)})
                                   for i, x, y in zip(ch.index, ch.lon, ch.lat)])
        rows += [f["properties"] for f in img.sampleRegions(collection=fc, scale=20, geometries=False).getInfo()["features"]]
    return df.join(pd.DataFrame(rows).set_index("i"), how="left")


# ------------------------------------------------------------------ geometries
alt6 = gpd.GeoSeries([wkt.loads(w) for w in pd.read_csv(EXT / "sernageomin_atacama_oeste/alteracion_atacama_oeste.csv").wkt_utm19s], crs=UTM)
atlas = gpd.read_file(EXT / "atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp").set_crs(UTM, allow_override=True).geometry.buffer(0)
u6, uA = unary_union(alt6.values), unary_union(atlas.values)
near_atlas = uA.buffer(INDEP)
any_alt = unary_union([u6, uA])
ring6 = u6.buffer(RING_OUT).difference(any_alt.buffer(RING_IN)).difference(near_atlas).simplify(30)
pos6 = u6.difference(near_atlas)
excl = any_alt.buffer(RING_IN).union(near_atlas).simplify(30)      # distractors stay out of this
far = any_alt.buffer(RING_OUT).union(near_atlas).simplify(30)
to4326 = lambda g: gpd.GeoSeries([g], crs=UTM).to_crs(4326).iloc[0]
b6 = gpd.GeoSeries([u6], crs=UTM).to_crs(4326).total_bounds
aoi6 = ee.Geometry.Rectangle([b6[0] - 0.08, b6[1] - 0.08, b6[2] + 0.08, b6[3] + 0.08])
print(f"R6: positives {pos6.area/1e6:.0f} km2 (after 5 km exclusion), ring {ring6.area/1e6:.0f} km2", flush=True)

# ------------------------------------------------------------------ (A) R6 design
img6 = composite(aoi6, "2024-01-01", "2024-04-01", 20)
dem = ee.Image("USGS/SRTMGL1_003"); slope = ee.Terrain.slope(dem)
shadow = slope.gt(20).And(ee.Terrain.hillshade(dem, 65, 60).lt(120))          # same as the Chilean pilot
not_excl = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(to_ee(to4326(excl)))]), 1).Not()
not_far = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(to_ee(to4326(far)))]), 1).Not()
playa = slope.lt(2).And(img6.select("B04").gt(0.30)).And(img6.select("ndvi").lt(0.10))
alluv = slope.lt(3).And(img6.select("B04").lt(0.30)).And(img6.select("ndvi").lt(0.15))
f6 = GT / "site6_positives_geo.csv"
if not f6.exists():
    print("R6 design", flush=True)
    pos = sample(img6, to_ee(to4326(pos6)), 7000, "positive")
    pos.drop(columns=["group"]).to_csv(f6, index=False)
    neg = [sample(img6, to_ee(to4326(ring6)), 7000, "ring"),
           sample(img6.updateMask(shadow.And(not_excl)), aoi6, 400000, "shadow", keep=60000).head(1500),
           sample(img6.updateMask(img6.select("ndvi").gt(0.3).And(not_excl)), aoi6, 400000, "vegetation", keep=60000).head(800),
           sample(img6.updateMask(img6.select("ndwi").gt(0.2).And(not_excl)), aoi6, 400000, "water", keep=60000).head(800),
           sample(img6.updateMask(playa.And(not_far)), aoi6, 400000, "playa", keep=60000).head(800),
           sample(img6.updateMask(alluv.And(not_far)), aoi6, 400000, "alluvium", keep=60000).head(800)]
    neg = pd.concat([d for d in neg if len(d)], ignore_index=True)
    neg["site"] = "R6"; neg[BANDS + ["site", "group", "lon", "lat"]].to_csv(GT / "site6_hard_negatives.csv", index=False)
    print(neg.groupby("group").size().to_string(), flush=True)

# ------------------------------------------------------------------ (B) P-A inputs
def deposits(site):
    if site == "R3":
        a = pd.read_csv(EXT / "atlas_metalifero_IIIR/yacimientos.csv")
        g = gpd.GeoSeries(gpd.points_from_xy(a.UTM_ESTE, a.UTM_NORTE), crs=UTM).to_crs(4326)
        d = pd.DataFrame({"lon": g.x, "lat": g.y, "name": a.NOMBRE.astype(str), "commodity": a.MENA.astype(str), "size": a.TAMANO.astype(str)})
    elif site == "R4":
        d = pd.read_csv(EXT / "yacimientos/coquimbo_rmm_yacimiento.csv")[["lon", "lat", "name", "sheet"]]
    elif site == "R5":
        d = pd.read_csv(EXT / "yacimientos/cuprite_mrds.csv"); d = d[d.dev_stat != "Plant"]
        d = d.assign(lon=d.lon.astype(float), lat=d.lat.astype(float), name=d.site_name, commodity=d.code_list.str.strip(), status=d.dev_stat)[["lon", "lat", "name", "commodity", "status"]]
    else:
        d = pd.read_csv(EXT / "sernageomin_atacama_oeste/yacimientos_atacama_oeste.csv")[["lon", "lat", "sheet"]]
    return d.reset_index(drop=True)


SITES = {  # AOI, composite
    "R3": ([-69.60, -28.05, -68.30, -25.95], ("2024-01-01", "2024-04-01", 20)),
    "R5": ([-117.45, 37.35, -116.95, 37.72], ("2024-06-01", "2024-09-30", 15)),
    "R6": (list(np.r_[b6[:2] - 0.05, b6[2:] + 0.05]), ("2024-01-01", "2024-04-01", 20)),
}
c4 = gpd.read_file  # Coquimbo AOI from its alteration polygons and deposits
coq = pd.concat([gpd.read_file(p).to_crs(4326) for p in sorted((EXT / "shapefiles_alteracion").glob("*/GB_ALTERACION_P.shp"))])
d4 = deposits("R4"); b4 = np.r_[np.minimum(coq.total_bounds[:2], [d4.lon.min(), d4.lat.min()]), np.maximum(coq.total_bounds[2:], [d4.lon.max(), d4.lat.max()])]
SITES["R4"] = (list(np.r_[b4[:2] - 0.03, b4[2:] + 0.03]), ("2024-01-01", "2024-04-01", 20))

for site, (bb, (s, e, cl)) in sorted(SITES.items()):
    fd, fb = GT / f"pa_{site}_deposits.csv", GT / f"pa_{site}_background.csv"
    if fd.exists() and fb.exists():
        print(f"{site}: exists"); continue
    aoi = ee.Geometry.Rectangle([float(x) for x in bb]); img = composite(aoi, s, e, cl)
    d = deposits(site); d = d[(d.lon >= bb[0]) & (d.lon <= bb[2]) & (d.lat >= bb[1]) & (d.lat <= bb[3])].reset_index(drop=True)
    t = time.time(); dv = at_points(img, d); dv.to_csv(fd, index=False)
    bg = sample(img, aoi, 60000, "background"); bg.to_csv(fb, index=False)
    print(f"{site}: AOI {np.round(bb, 3).tolist()}  deposits {len(d)} (with values {dv[BANDS].notna().all(axis=1).sum()})  "
          f"background {len(bg)}  ({time.time() - t:.0f}s)", flush=True)
print("done")
