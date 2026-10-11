#!/usr/bin/env python3
"""
PILOT: resample the sparse distractor groups (snow, water, shadow, vegetation) with
stratifiedSample, which draws inside the mask instead of dropping masked draws.
Replaces those groups in data/ground_truth/pilot_hard_negatives.csv.
Same composite rule as the main sampler (S2_SR_HARMONIZED, 2024-01..03, cloud < 20, median).
"""
import time
from pathlib import Path
import ee, pandas as pd, geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union
ee.Initialize()

GT = Path("data/ground_truth"); CSV = GT / "pilot_hard_negatives.csv"
S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
R = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEED = 42
dem = ee.Image("USGS/SRTMGL1_003"); slope = ee.Terrain.slope(dem); hs = ee.Terrain.hillshade(dem, 65, 60)
shadow_mask = slope.gt(20).And(hs.lt(150))
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

def comp(region):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate("2024-01-01", "2024-04-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(S2)).median().divide(1e4).rename(R)

def to_ee(g): return ee.Geometry(mapping(g), None, False)

def strat(img, mask, region, n, site, group):
    t = time.time()
    m = mask.rename("m")
    fc = (img.addBands(m).updateMask(m)
          .stratifiedSample(numPoints=n, classBand="m", region=region, scale=20, seed=SEED,
                            classValues=[1], classPoints=[n], geometries=True, dropNulls=True))
    rows = []
    for f in fc.toList(n).getInfo():
        p = f["properties"]; c = f["geometry"]["coordinates"]; p["lon"], p["lat"] = c
        rows.append(p)
    df = pd.DataFrame(rows)
    df["site"], df["group"] = site, group
    print(f"  [{site}] {group:11s} {len(df):5d} px ({time.time()-t:.0f}s)", flush=True)
    return df[[c for c in R + ["site", "group", "lon", "lat"] if c in df.columns]]

def geoms(shp_list, crs):
    def rd(p):
        g = gpd.read_file(p)
        return (g if g.crs else g.set_crs("EPSG:32719")).to_crs("EPSG:32719")[["geometry"]]
    g = pd.concat([rd(p) for p in shp_list], ignore_index=True)
    u = unary_union(gpd.GeoDataFrame(g, crs="EPSG:32719").geometry.buffer(0))
    ring = gpd.GeoSeries([u.buffer(2000).difference(u.buffer(300)).simplify(30)], crs="EPSG:32719").to_crs(4326).iloc[0]
    near = gpd.GeoSeries([u.buffer(10000).difference(u.buffer(300)).simplify(100)], crs="EPSG:32719").to_crs(4326).iloc[0]
    return to_ee(ring), to_ee(near)

parts = []
# Region III
ring3, near3 = geoms(["data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp"], None)
aoi3 = ee.Geometry.Rectangle([-70.6, -28.3, -68.07, -25.7]); img3 = comp(aoi3)
ndsi3 = img3.normalizedDifference(["B03", "B11"]); ndvi3 = img3.normalizedDifference(["B08", "B04"])
snow_box = ee.Geometry.Rectangle([-68.9, -27.35, -68.3, -26.9]); veg_box = ee.Geometry.Rectangle([-70.45, -27.5, -70.15, -27.25])
parts.append(strat(img3, ndsi3.gt(0.4).And(dem.gt(4500)), snow_box, 800, "R3", "snow"))
parts.append(strat(img3, jrc.gt(50), near3, 800, "R3", "water"))
parts.append(strat(img3, shadow_mask, ring3, 1500, "R3", "shadow"))
parts.append(strat(img3, ndvi3.gt(0.3), veg_box, 800, "R3", "vegetation"))
# Region IV
shp4 = sorted(Path("data/external/shapefiles_alteracion").glob("*/GB_ALTERACION_P.shp"))
ring4, near4 = geoms(shp4, None)
aoi4 = ee.Geometry.Rectangle([-71.6, -31.2, -70.3, -29.8]); img4 = comp(aoi4)
ndvi4 = img4.normalizedDifference(["B08", "B04"])
parts.append(strat(img4, shadow_mask, ring4, 1500, "R4", "shadow"))
parts.append(strat(img4, ndvi4.gt(0.3), near4, 800, "R4", "vegetation"))

new = pd.concat(parts, ignore_index=True)
old = pd.read_csv(CSV)
keep = ~old.set_index(["site", "group"]).index.isin(new.set_index(["site", "group"]).index.unique())
out = pd.concat([old[keep], new], ignore_index=True)
out.to_csv(CSV, index=False)
print(f"\nSaved {CSV}: {len(out)} rows"); print(out.groupby(["site", "group"]).size().to_string())
