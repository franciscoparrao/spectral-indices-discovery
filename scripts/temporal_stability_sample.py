#!/usr/bin/env python3
"""
Temporal stability: re-sample every stored point of the hard-negative design on
alternative Sentinel-2 composites, so the fixed indices can be re-evaluated on a
different season / year without touching the labels or the design.

Points  : R3 = binary_training_s2_positives_geo + pilot_hard_negatives[R3]
          R4 = region_iv_validation_s2 + pilot_hard_negatives[R4]
          R5 = cuprite_positives_geo + cuprite_hard_negatives
Periods : base (the composite the design was built on, re-sampled as a consistency
          check), same season previous year, opposite season same year.
Output  : data/ground_truth/temporal/{site}_{period}.csv  (lon, lat, role, group, bands)
"""
import time
from pathlib import Path
import ee, numpy as np, pandas as pd

ee.Initialize()
GT = Path("data/ground_truth"); OUT = GT / "temporal"; OUT.mkdir(exist_ok=True)
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
RENAME = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PERIODS = {
    "R3": {"base": ("2024-01-01", "2024-04-01", 20), "prev_summer": ("2023-01-01", "2023-04-01", 20), "winter": ("2024-06-01", "2024-09-30", 20)},
    "R4": {"base": ("2024-01-01", "2024-04-01", 20), "prev_summer": ("2023-01-01", "2023-04-01", 20), "winter": ("2024-06-01", "2024-09-30", 20)},
    "R5": {"base": ("2024-06-01", "2024-09-30", 15), "prev_summer": ("2023-06-01", "2023-09-30", 15), "spring": ("2024-03-01", "2024-05-31", 15)},
}
hard = pd.read_csv(GT / "pilot_hard_negatives.csv")
pts = {
    "R3": pd.concat([pd.read_csv(GT / "binary_training_s2_positives_geo.csv")[["lon", "lat"]].assign(role="pos", group="positive"),
                     hard[hard.site == "R3"][["lon", "lat", "group"]].assign(role="neg")], ignore_index=True),
    "R4": pd.concat([pd.read_csv(GT / "region_iv_validation_s2.csv")[["lon", "lat"]].assign(role="pos", group="positive"),
                     hard[hard.site == "R4"][["lon", "lat", "group"]].assign(role="neg")], ignore_index=True),
    "R5": pd.concat([pd.read_csv(GT / "cuprite_positives_geo.csv")[["lon", "lat"]].assign(role="pos", group="positive"),
                     pd.read_csv(GT / "cuprite_hard_negatives.csv")[["lon", "lat", "group"]].assign(role="neg")], ignore_index=True),
}


def composite(aoi, start, end, cloud):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)).select(S2_BANDS))
    return s2.median().divide(10000).rename(RENAME), s2


def sample_points(img, df, batch=4000):
    rows = []
    for s in range(0, len(df), batch):
        chunk = df.iloc[s:s + batch]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x, y]), {"i": int(i)})
                                   for i, x, y in zip(chunk.index, chunk.lon, chunk.lat)])
        res = img.sampleRegions(collection=fc, scale=20, geometries=False).getInfo()
        for f in res["features"]:
            rows.append(f["properties"])
    r = pd.DataFrame(rows).set_index("i")
    return df.join(r, how="left")


for site, df in pts.items():
    b = df[["lon", "lat"]].agg(["min", "max"])
    aoi = ee.Geometry.Rectangle([b.lon["min"] - 0.05, b.lat["min"] - 0.05, b.lon["max"] + 0.05, b.lat["max"] + 0.05])
    for period, (start, end, cloud) in PERIODS[site].items():
        f = OUT / f"{site}_{period}.csv"
        if f.exists(): print(f"{site} {period}: exists"); continue
        t = time.time(); img, coll = composite(aoi, start, end, cloud)
        n_scenes = coll.size().getInfo()
        out = sample_points(img, df)
        got = out[RENAME].notna().all(axis=1).sum()
        out.to_csv(f, index=False)
        print(f"{site} {period:12s} {start}..{end}: {n_scenes:4d} scenes, {got}/{len(df)} points with values ({time.time()-t:.0f}s)", flush=True)
print("done")
