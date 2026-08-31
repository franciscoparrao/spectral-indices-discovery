#!/usr/bin/env python3
"""
Cuprite comparison figure (addresses R#1 critique #7 from the NRR reject).

Six-panel figure for Cuprite, Nevada (USA):
  (a) Sentinel-2 true colour
  (b) Sentinel-2 SWIR false colour
  (c) Clay Ratio (B11/B12)
  (d) SR per-pixel classification using the four locally-rediscovered Cuprite
      formulas (Table tab:cuprite_formulas in the manuscript)
  (e) USGS Rockwell (2017) ASTER-derived alteration map — the reference
      product the reviewer asked us to compare against
  (f) Panel (d) with USGS reference outlines overlaid for visual co-registration

Reads:
  - data/ground_truth/cuprite_ground_truth.tif  (USGS Rockwell raster, EPSG:4326)
Pulls from GEE:
  - Sentinel-2 SR median composite over the same AOI
"""

import ee
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape as shp_shape
import geopandas as gpd
import urllib.request, tempfile
from pathlib import Path

ee.Initialize()

FIG_DIR = Path("figures")
GT_PATH = Path("data/ground_truth/cuprite_ground_truth.tif")
SCALE_M = 30

# USGS Rockwell class mapping (matches the manuscript's Cuprite groupings)
USGS_CLASS_NAMES = {1: "Silicic", 2: "Adv. Argillic", 3: "Argillic-Phyllic", 4: "Propylitic"}
USGS_CLASS_COLORS = {1: "#FF7F00", 2: "#E41A1C", 3: "#4DAF4A", 4: "#00AA00"}

# ---------- 1. Load USGS GT raster + AOI from its bounds ----------
print("Loading USGS Rockwell ground truth...")
with rasterio.open(GT_PATH) as src:
    gt = src.read(1)
    gt_bounds = src.bounds
    gt_transform = src.transform
    gt_crs = src.crs
print(f"  GT shape: {gt.shape}, bounds: {gt_bounds}, CRS: {gt_crs}")
print(f"  Classified pixels: {(gt > 0).sum()} / {gt.size}")

AOI = dict(west=gt_bounds.left, south=gt_bounds.bottom,
           east=gt_bounds.right, north=gt_bounds.top)
print(f"  AOI: {AOI}")

# ---------- 2. Download S2 composite from GEE for the same AOI ----------
print("Downloading Sentinel-2 composite from GEE...")
aoi = ee.Geometry.Rectangle([AOI["west"], AOI["south"], AOI["east"], AOI["north"]])
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2024-01-01", "2024-06-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .select(["B2", "B3", "B4", "B7", "B8", "B8A", "B11", "B12"])
)
print(f"  S2 scenes available: {s2.size().getInfo()}")
composite = s2.median().divide(10000)

url = composite.getDownloadURL({
    "scale": SCALE_M,
    "region": aoi,
    "format": "GEO_TIFF",
    "crs": "EPSG:4326",
})
print(f"  Downloading GeoTIFF...")
with urllib.request.urlopen(url, timeout=180) as resp:
    data = resp.read()
print(f"  Downloaded {len(data)/1e6:.1f} MB")

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
    tmp.write(data)
    tmp_path = tmp.name

BAND_ORDER = ["B2", "B3", "B4", "B7", "B8", "B8A", "B11", "B12"]
with rasterio.open(tmp_path) as src:
    arr = src.read()
    s2_bounds = src.bounds
print(f"  S2 raster shape: {arr.shape}, bounds: {s2_bounds}")
bands = {name: arr[i].astype(np.float32) for i, name in enumerate(BAND_ORDER)}

# ---------- 3. Compute Cuprite SR formulas (Table tab:cuprite_formulas) ----------
eps = 1e-6
sr = {
    1: np.log(np.maximum(bands["B11"], eps) / np.maximum(bands["B12"], eps)),  # Silicic
    2: bands["B11"] - bands["B12"],                                            # Adv. Argillic
    3: np.tanh(bands["B11"]) - bands["B8A"],                                   # Argillic-Phyllic
    4: ((bands["B12"] - bands["B8"]) / np.maximum(bands["B7"], eps)) ** 2,     # Propylitic
}
# Normalize each to [0,1] via p2/p98 percentile stretch, then argmax
sr_norm = {}
for k, v in sr.items():
    p2, p98 = np.nanpercentile(v[np.isfinite(v)], [2, 98])
    sr_norm[k] = np.clip((v - p2) / max(p98 - p2, eps), 0, 1)
sr_stack = np.stack([sr_norm[k] for k in sorted(sr_norm)], axis=0)
classmap = np.argmax(sr_stack, axis=0) + 1  # 1..4

clay_ratio = bands["B11"] / np.maximum(bands["B12"], eps)

# ---------- 4. Vectorize USGS GT for the outline overlay panel ----------
print("Vectorizing USGS classes for outline overlay...")
gt_polys = []
for cls_id in range(1, 5):
    mask = (gt == cls_id).astype("uint8")
    for geom, val in shapes(mask, mask=mask.astype(bool), transform=gt_transform):
        if val == 1:
            gt_polys.append({"class_id": cls_id, "geometry": shp_shape(geom)})
gdf = gpd.GeoDataFrame(gt_polys, crs=gt_crs)
# Simplify a bit so overlay edges are clean
gdf["geometry"] = gdf["geometry"].simplify(0.0003)
print(f"  GT polygons (per pixel cluster): {len(gdf)}")

# ---------- 5. Build figure ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

def style(ax, title):
    ax.set_xlim(AOI["west"], AOI["east"])
    ax.set_ylim(AOI["south"], AOI["north"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°W)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def percentile_stretch(arr, p_lo=2, p_hi=98, gamma=0.85):
    """Linear stretch via p2/p98 percentiles, then gamma."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [p_lo, p_hi])
    norm = np.clip((arr - lo) / max(hi - lo, eps), 0, 1)
    return np.power(norm, gamma)


# Panel A: true colour with p2/p98 stretch
ax = axes[0, 0]
rgb_true = np.dstack([
    percentile_stretch(bands["B4"]),
    percentile_stretch(bands["B3"]),
    percentile_stretch(bands["B2"]),
])
extent_s2 = [s2_bounds.left, s2_bounds.right, s2_bounds.bottom, s2_bounds.top]
ax.imshow(rgb_true, extent=extent_s2, origin="upper")
style(ax, "(a) Sentinel-2 true colour (B4-B3-B2)")
# Scale bar (~3 km at this latitude)
ax.plot([AOI["west"] + 0.005, AOI["west"] + 0.005 + 0.035],
        [AOI["south"] + 0.005] * 2, "k-", linewidth=2.3)
ax.text(AOI["west"] + 0.0225, AOI["south"] + 0.008, "~3 km",
        ha="center", fontsize=7, fontweight="bold", color="white",
        bbox=dict(facecolor="black", alpha=0.5, pad=1))
ax.annotate("N", xy=(0.95, 0.95), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="center", va="top")
ax.annotate("", xy=(0.95, 0.95), xytext=(0.95, 0.85),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

# Panel B: SWIR false colour with p2/p98 stretch
ax = axes[0, 1]
rgb_swir = np.dstack([
    percentile_stretch(bands["B12"]),
    percentile_stretch(bands["B11"]),
    percentile_stretch(bands["B4"]),
])
ax.imshow(rgb_swir, extent=extent_s2, origin="upper")
style(ax, "(b) SWIR false colour (B12-B11-B4)")

# Panel C: Clay Ratio
ax = axes[0, 2]
p2, p98 = np.percentile(clay_ratio[np.isfinite(clay_ratio)], [2, 98])
im = ax.imshow(clay_ratio, extent=extent_s2, origin="upper",
               cmap="RdYlBu_r", vmin=p2, vmax=p98)
style(ax, "(c) Clay Ratio ($B_{11}/B_{12}$)")
plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="$B_{11}/B_{12}$")

# Panel D: SR per-pixel classification with 4 Cuprite-local formulas
ax = axes[1, 0]
cmap_class = ListedColormap([USGS_CLASS_COLORS[k] for k in sorted(USGS_CLASS_COLORS)])
norm_class = BoundaryNorm(np.arange(0.5, 5.5, 1), cmap_class.N)
ax.imshow(classmap, extent=extent_s2, origin="upper",
          cmap=cmap_class, norm=norm_class, interpolation="nearest")
style(ax, "(d) SR per-pixel classification (4 Cuprite formulas)")
patches = [Patch(facecolor=USGS_CLASS_COLORS[k], edgecolor="black",
                 label=USGS_CLASS_NAMES[k]) for k in sorted(USGS_CLASS_COLORS)]
ax.legend(handles=patches, loc="lower left", fontsize=7,
          framealpha=0.92, title="SR predicted class", title_fontsize=7.5)

# Panel E: USGS Rockwell GT raster
ax = axes[1, 1]
gt_display = np.where(gt > 0, gt, np.nan).astype(float)
extent_gt = [gt_bounds.left, gt_bounds.right, gt_bounds.bottom, gt_bounds.top]
ax.imshow(gt_display, extent=extent_gt, origin="upper",
          cmap=cmap_class, norm=norm_class, interpolation="nearest")
style(ax, "(e) USGS Rockwell (2017) ASTER-derived reference")
ax.legend(handles=patches, loc="lower left", fontsize=7,
          framealpha=0.92, title="USGS class", title_fontsize=7.5)

# Panel F: SR classification with USGS outlines overlaid
ax = axes[1, 2]
ax.imshow(classmap, extent=extent_s2, origin="upper",
          cmap=cmap_class, norm=norm_class, interpolation="nearest", alpha=0.85)
gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.4, alpha=0.85)
style(ax, "(f) SR classification with USGS outlines overlaid")

fig.suptitle(
    "Cuprite, Nevada, USA — Sentinel-2 imagery, classical and SR indices, "
    "SR per-pixel classification, and USGS Rockwell (2017) reference comparison\n"
    "Median S2 composite Jan–Jun 2024, 30 m display resolution; "
    f"AOI {(AOI['east']-AOI['west'])*111*0.79:.0f} km × {(AOI['north']-AOI['south'])*111:.0f} km",
    fontsize=11, y=1.005,
)

plt.tight_layout()
out = FIG_DIR / "cuprite_comparison.png"
fig.savefig(out, dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "cuprite_comparison.pdf", bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
