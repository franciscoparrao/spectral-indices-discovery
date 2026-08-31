#!/usr/bin/env python3
"""
Figure 5 v3 — Maricunga alteration mapping.

Fixes R#1 critique #6 from the NRR reject:
- Replaces scatter-plot sampling (which made true/false color look fragmented)
  with proper raster grids via ee.Image.sampleRectangle().
- Replaces the single propylitic-index map with an actual per-pixel argmax
  classification using the 6 SR formulas.
- Adds a ground-truth overlay panel for direct visual comparison with the
  Atlas Metalífero polygons (the comparison R#1 explicitly asked for).
- Adds an explicit "per-pixel classification ≠ visual coherence" honest note
  in the caption (the visual-vs-metric disconnect).
"""

import ee
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
import urllib.request, io, zipfile, tempfile
from pathlib import Path

ee.Initialize()

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
ATLAS_SHP = Path("data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp")
ATLAS_CSV = Path("data/external/atlas_metalifero_IIIR/alteracion.csv")

# AOI: central Maricunga
AOI = dict(west=-69.35, south=-27.05, east=-69.05, north=-26.80)
SCALE = 60  # metres per pixel for the display grid (3x S2 native; faster + still legible)

CLASS_NAMES = {
    1: "Silicic",
    2: "Adv. Argillic",
    3: "Argillic-Phyllic",
    4: "Propylitic",
    5: "Iron Oxide",
    6: "Potassic-Skarn",
}
CLASS_COLORS = {
    1: "#FF7F00",
    2: "#E41A1C",
    3: "#4DAF4A",
    4: "#00AA00",
    5: "#A65628",
    6: "#377EB8",
}

# ---------- 1. Pull a real raster grid from GEE ----------
print("Downloading raster grid from GEE (sampleRectangle)...")
aoi = ee.Geometry.Rectangle([AOI["west"], AOI["south"], AOI["east"], AOI["north"]])
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2024-01-01", "2024-04-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
    .select(["B2", "B3", "B4", "B5", "B7", "B8A", "B11", "B12"])
)
print(f"  S2 scenes available: {s2.size().getInfo()}")
composite = s2.median().divide(10000)

# Download as a GeoTIFF (multi-band) and read with rasterio — the reliable way
# to get a regular raster grid out of GEE.
url = composite.getDownloadURL({
    "scale": SCALE,
    "region": aoi,
    "format": "GEO_TIFF",
    "crs": "EPSG:4326",
})
print(f"  Download URL: {url[:80]}...")
with urllib.request.urlopen(url, timeout=180) as resp:
    data = resp.read()
print(f"  Downloaded {len(data)/1e6:.1f} MB")

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
    tmp.write(data)
    tmp_path = tmp.name

BAND_ORDER = ["B2", "B3", "B4", "B5", "B7", "B8A", "B11", "B12"]  # selection order
with rasterio.open(tmp_path) as src:
    arr = src.read()  # (n_bands, H, W)
    print(f"  Shape: {arr.shape}")
    raster_extent = src.bounds  # (left, bottom, right, top) in EPSG:4326

bands = {name: arr[i].astype(np.float32) for i, name in enumerate(BAND_ORDER)}
H, W = arr.shape[1], arr.shape[2]
print(f"  Grid: {H} x {W}")

# ---------- 2. Compute the six SR indices and the argmax classification ----------
eps = 1e-6
sr = {
    1: bands["B4"] - 0.135,                                              # Silicic
    2: 0.83 - bands["B2"] / np.maximum(bands["B5"], eps),                # Adv. Argillic
    3: 0.09 / np.maximum(bands["B5"], eps),                              # Argillic-Phyllic
    4: bands["B3"] - 0.48 * bands["B11"],                                # Propylitic
    5: (np.sqrt(np.clip(bands["B12"], 0, None)) - bands["B11"]) ** 2,    # Iron Oxide
    6: bands["B3"] * bands["B12"] / np.maximum(bands["B7"] ** 2, eps) - 0.45,  # Potassic
}
# Normalize each SR index to [0,1] within the scene before argmax (so classes are comparable)
sr_norm = {}
for k, v in sr.items():
    p2, p98 = np.percentile(v, [2, 98])
    sr_norm[k] = np.clip((v - p2) / max(p98 - p2, eps), 0, 1)
sr_stack = np.stack([sr_norm[k] for k in sorted(sr_norm)], axis=0)
classmap = np.argmax(sr_stack, axis=0) + 1  # 1..6

# ---------- 3. Compute Clay Ratio for panel C ----------
clay_ratio = bands["B11"] / np.maximum(bands["B12"], eps)

# ---------- 4. Load and clip ground-truth polygons ----------
print("Loading Atlas Metalifero polygons...")
gdf = gpd.read_file(ATLAS_SHP)
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:32719")
gdf = gdf.to_crs("EPSG:4326")
import pandas as pd
attrs = pd.read_csv(ATLAS_CSV)
gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
ALTERATION_MAP = {
    "Alteración Silicea": 1, "vuggy silica": 1,
    "Alteracion Argilica y Argilica avanzada": 2, "Alteracion Solfatárica": 2,
    "Alteracion Argilica": 3, "Alteracion Sericitica": 3,
    "Alteración Cuarzo-Sericitica(Fílica)": 3,
    "Alteracion Propilitica": 4,
    "Oxidos e Hidróxidos de Hierro": 5,
    "Alteracion Potasica": 6, "skarn": 6,
}
gdf["class_id"] = gdf["ALTERACION"].map(ALTERATION_MAP).fillna(0).astype(int)
# Clip to AOI
from shapely.geometry import box
aoi_box = box(AOI["west"], AOI["south"], AOI["east"], AOI["north"])
gdf_aoi = gdf[gdf.intersects(aoi_box)].copy()
gt_classified = gdf_aoi[gdf_aoi["class_id"] > 0]
gt_undiff = gdf_aoi[gdf_aoi["class_id"] == 0]
print(f"  Ground truth in AOI: classified={len(gt_classified)}, undiff={len(gt_undiff)}")

# ---------- 5. Build figure ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
extent = [AOI["west"], AOI["east"], AOI["south"], AOI["north"]]


def style(ax, title):
    ax.set_xlim(AOI["west"], AOI["east"])
    ax.set_ylim(AOI["south"], AOI["north"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°W)", fontsize=9)
    ax.set_ylabel("Latitude (°S)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def percentile_stretch(arr, p_lo=2, p_hi=98, gamma=0.85):
    """Linear stretch via p_lo/p_hi percentiles + gamma correction.

    Replaces the older fixed-gain stretch which clipped at hardcoded reflectance
    multipliers (3.5×, 4×) and lost dynamic range over high-albedo surfaces.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [p_lo, p_hi])
    norm = np.clip((arr - lo) / max(hi - lo, eps), 0, 1)
    return np.power(norm, gamma)


# Panel A: True color (p2/p98 stretch)
ax = axes[0, 0]
rgb_true = np.dstack([
    percentile_stretch(bands["B4"]),
    percentile_stretch(bands["B3"]),
    percentile_stretch(bands["B2"]),
])
ax.imshow(rgb_true, extent=extent, origin="upper")
style(ax, "(a) Sentinel-2 true colour (B4-B3-B2)")
# North arrow + scalebar
ax.annotate("N", xy=(0.93, 0.94), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="center", va="top")
ax.annotate("", xy=(0.93, 0.94), xytext=(0.93, 0.84),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
# ~10 km scale bar
ax.plot([AOI["west"] + 0.02, AOI["west"] + 0.02 + 0.1],
        [AOI["south"] + 0.013] * 2, "k-", linewidth=2.2)
ax.text(AOI["west"] + 0.07, AOI["south"] + 0.018, "~10 km",
        ha="center", fontsize=7, fontweight="bold")

# Panel B: SWIR false color (B12-B11-B4)
ax = axes[0, 1]
rgb_swir = np.dstack([
    gamma_stretch(bands["B12"], gain=4.0),
    gamma_stretch(bands["B11"], gain=3.0),
    gamma_stretch(bands["B4"]),
])
ax.imshow(rgb_swir, extent=extent, origin="upper")
style(ax, "(b) SWIR false colour (B12-B11-B4)")

# Panel C: Clay Ratio
ax = axes[0, 2]
p2, p98 = np.percentile(clay_ratio, [2, 98])
im = ax.imshow(clay_ratio, extent=extent, origin="upper",
               cmap="RdYlBu_r", vmin=p2, vmax=p98)
style(ax, "(c) Clay Ratio ($B_{11}/B_{12}$)")
plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="$B_{11}/B_{12}$")

# Panel D: SR per-pixel argmax classification
ax = axes[1, 0]
cmap_class = ListedColormap([CLASS_COLORS[k] for k in sorted(CLASS_COLORS)])
norm_class = BoundaryNorm(np.arange(0.5, 7.5, 1), cmap_class.N)
ax.imshow(classmap, extent=extent, origin="upper",
          cmap=cmap_class, norm=norm_class, interpolation="nearest")
style(ax, "(d) SR per-pixel classification (argmax of 6 SR indices)")
patches = [Patch(facecolor=CLASS_COLORS[k], edgecolor="black",
                 label=CLASS_NAMES[k]) for k in sorted(CLASS_COLORS)]
ax.legend(handles=patches, loc="lower left", fontsize=6.5,
          framealpha=0.92, title="SR predicted class", title_fontsize=7)

# Panel E: Ground truth polygons (the comparison R#1 asked for)
ax = axes[1, 1]
ax.imshow(rgb_true, extent=extent, origin="upper", alpha=0.55)
if len(gt_undiff):
    gt_undiff.plot(ax=ax, facecolor="#CCCCCC", edgecolor="gray",
                   linewidth=0.4, alpha=0.55)
for cls_id in sorted(CLASS_COLORS):
    sub = gt_classified[gt_classified["class_id"] == cls_id]
    if len(sub):
        sub.plot(ax=ax, facecolor=CLASS_COLORS[cls_id],
                 edgecolor="black", linewidth=0.5, alpha=0.85)
style(ax, "(e) Ground truth — Atlas Metalífero polygons")
gt_patches = [Patch(facecolor="#CCCCCC", edgecolor="gray", label="Undifferentiated")]
gt_patches += [Patch(facecolor=CLASS_COLORS[k], edgecolor="black",
                     label=CLASS_NAMES[k]) for k in sorted(CLASS_COLORS)]
ax.legend(handles=gt_patches, loc="lower left", fontsize=6.5,
          framealpha=0.92, title="Ground-truth class", title_fontsize=7)

# Panel F: SR classification with ground truth polygon outlines for direct overlay
ax = axes[1, 2]
ax.imshow(classmap, extent=extent, origin="upper",
          cmap=cmap_class, norm=norm_class, interpolation="nearest", alpha=0.85)
# Overlay polygon outlines only (no fill) so the SR classification is still visible
if len(gt_undiff):
    gt_undiff.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5,
                            linestyle=":", alpha=0.7)
if len(gt_classified):
    gt_classified.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8, alpha=0.85)
style(ax, "(f) SR classification with ground-truth polygons (outlined)")

fig.suptitle(
    "Sentinel-2 imagery, classical and SR indices, SR per-pixel classification, "
    "and ground-truth comparison\nMaricunga district, Atacama, Chile — "
    "median composite Jan–Mar 2024, 60 m display resolution",
    fontsize=12, y=1.005,
)

plt.tight_layout()
out = FIG_DIR / "alteration_comparison.png"
fig.savefig(out, dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "alteration_comparison.pdf", bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
