#!/usr/bin/env python3
"""
Pre-process GEE-derived rasters for the R plotting scripts.

Saves permanent multi-band GeoTIFFs that paper/R/fig_alteration_comparison.R
and paper/R/fig_cuprite_comparison.R read with terra. Decouples GEE download
(Python, ee package) from the actual figure generation (R, ggplot2 + tidyterra).

Outputs (under data/maps/):
  maricunga_composite.tif  — 8-band S2 SR median (B2,B3,B4,B5,B7,B8A,B11,B12), 60 m
  cuprite_composite.tif    — 8-band S2 SR median (B2,B3,B4,B7,B8,B8A,B11,B12), 30 m
"""

import ee
import urllib.request
import tempfile
from pathlib import Path
import shutil

ee.Initialize()

OUT_DIR = Path("data/maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_composite(aoi_dict, bands, date_range, scale, out_path,
                        cloud_pct=15):
    print(f"\n=== {out_path.name} ===")
    aoi = ee.Geometry.Rectangle([
        aoi_dict["west"], aoi_dict["south"],
        aoi_dict["east"], aoi_dict["north"],
    ])
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(*date_range)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(bands)
    )
    n = coll.size().getInfo()
    print(f"  Scenes: {n}")
    composite = coll.median().divide(10000)

    url = composite.getDownloadURL({
        "scale": scale,
        "region": aoi,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    })
    print(f"  Downloading...")
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    print(f"  {len(data)/1e6:.1f} MB")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    shutil.move(tmp_path, out_path)
    print(f"  Saved: {out_path}")


# ===== Maricunga =====
download_composite(
    aoi_dict=dict(west=-69.35, south=-27.05, east=-69.05, north=-26.80),
    bands=["B2", "B3", "B4", "B5", "B7", "B8A", "B11", "B12"],
    date_range=("2024-01-01", "2024-04-01"),
    scale=60,
    out_path=OUT_DIR / "maricunga_composite.tif",
)

# ===== Cuprite =====
download_composite(
    aoi_dict=dict(west=-117.28, south=37.47, east=-117.10, north=37.60),
    bands=["B2", "B3", "B4", "B7", "B8", "B8A", "B11", "B12"],
    date_range=("2024-01-01", "2024-06-01"),
    scale=30,
    out_path=OUT_DIR / "cuprite_composite.tif",
    cloud_pct=10,
)

print("\nAll rasters saved to data/maps/")
