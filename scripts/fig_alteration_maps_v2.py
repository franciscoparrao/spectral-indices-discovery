#!/usr/bin/env python3
"""
Figure: Alteration comparison map v2.
Downloads clean S2 RGB from GEE and generates proper cartographic figures
with: scale bar, north arrow, coordinates, colorbar.
"""

import ee
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
import json

ee.Initialize()

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Area of interest — central Maricunga district
# Smaller subset for cleaner visualization
AOI = {
    'west': -69.35, 'south': -27.05,
    'east': -69.05, 'north': -26.80,
}

print("Downloading S2 composite from GEE...")

aoi = ee.Geometry.Rectangle([AOI['west'], AOI['south'], AOI['east'], AOI['north']])

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(aoi)
      .filterDate("2024-01-01", "2024-04-01")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
      .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]))

n_scenes = s2.size().getInfo()
print(f"  S2 scenes: {n_scenes}")

composite = s2.median().divide(10000)

# Sample a grid of pixels for the map
# At 20m resolution, our AOI is ~1500x1400 pixels — sample a dense grid
print("  Sampling pixels for map...")

# Use getRegion for a structured grid
region_pixels = composite.sample(
    region=aoi, scale=30, numPixels=80000, seed=42, geometries=True
)

size = region_pixels.size().getInfo()
print(f"  Sampled {size} pixels")

# Download in batches
all_features = []
batch = 2000
for start in range(0, min(size, 80000), batch):
    feats = region_pixels.toList(min(batch, size - start), start).getInfo()
    all_features.extend(feats)
    print(f"    Downloaded {len(all_features)}/{min(size, 50000)}")

# Extract coordinates and band values
lons, lats = [], []
bands_data = {b: [] for b in ['B2', 'B3', 'B4', 'B5', 'B11', 'B12']}

for feat in all_features:
    props = feat.get('properties', {})
    geom = feat.get('geometry', {})
    if geom and 'coordinates' in geom:
        lons.append(geom['coordinates'][0])
        lats.append(geom['coordinates'][1])
        for b in bands_data:
            bands_data[b].append(props.get(b, 0))

lons = np.array(lons)
lats = np.array(lats)
for b in bands_data:
    bands_data[b] = np.array(bands_data[b])

print(f"  Valid pixels: {len(lons)}")

# ===== Create gridded images via scatter plot =====
eps = 1e-6

# SR indices
sr_propylitic = bands_data['B3'] - 0.48 * bands_data['B11']
sr_adv_argillic = 0.83 - bands_data['B2'] / np.maximum(bands_data['B5'], eps)
sr_swir = (np.sqrt(bands_data['B12']) - bands_data['B11']) ** 2
clay_ratio = bands_data['B11'] / np.maximum(bands_data['B12'], eps)

# ===== Figure: 4-panel comparison =====
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: True Color
ax = axes[0, 0]
# RGB from GEE data — already in reflectance 0-1
r = np.clip(bands_data['B4'] * 3.5, 0, 1)
g = np.clip(bands_data['B3'] * 3.5, 0, 1)
b = np.clip(bands_data['B2'] * 3.5, 0, 1)
# Gamma correction
r, g, b = np.power(r, 0.7), np.power(g, 0.7), np.power(b, 0.7)
colors_rgb = np.column_stack([r, g, b])

ax.scatter(lons, lats, c=colors_rgb, s=1.2, marker='s', edgecolors='none')
ax.set_xlim(AOI['west'], AOI['east'])
ax.set_ylim(AOI['south'], AOI['north'])
ax.set_title('(a) Sentinel-2 True Color (B4-B3-B2)', fontsize=11, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=9)
ax.set_ylabel('Latitude (°S)', fontsize=9)
ax.tick_params(labelsize=7)
ax.set_aspect('equal')
# North arrow
ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=12, fontweight='bold', ha='center', va='top')
ax.annotate('', xy=(0.95, 0.95), xytext=(0.95, 0.85),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
# Scale bar (approximate: 1° lon ≈ 90 km at -27°S)
ax.plot([AOI['west']+0.02, AOI['west']+0.02+0.1], [AOI['south']+0.01]*2,
        'k-', linewidth=2)
ax.text(AOI['west']+0.02+0.05, AOI['south']+0.02, '~10 km',
        ha='center', fontsize=7, fontweight='bold')

# Panel B: SWIR False Color
ax = axes[0, 1]
r_swir = np.clip(bands_data['B12'] * 4, 0, 1)
g_swir = np.clip(bands_data['B11'] * 3, 0, 1)
b_swir = np.clip(bands_data['B4'] * 3.5, 0, 1)
r_swir, g_swir, b_swir = np.power(r_swir, 0.7), np.power(g_swir, 0.7), np.power(b_swir, 0.7)
colors_swir = np.column_stack([r_swir, g_swir, b_swir])

ax.scatter(lons, lats, c=colors_swir, s=1.2, marker='s', edgecolors='none')
ax.set_xlim(AOI['west'], AOI['east'])
ax.set_ylim(AOI['south'], AOI['north'])
ax.set_title('(b) SWIR False Color (B12-B11-B4)', fontsize=11, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=9)
ax.set_ylabel('Latitude (°S)', fontsize=9)
ax.tick_params(labelsize=7)
ax.set_aspect('equal')

# Panel C: Clay Ratio
ax = axes[1, 0]
p2, p98 = np.percentile(clay_ratio, [2, 98])
sc = ax.scatter(lons, lats, c=clay_ratio, s=1.2, marker='s',
                cmap='RdYlBu_r', vmin=p2, vmax=p98, edgecolors='none')
ax.set_xlim(AOI['west'], AOI['east'])
ax.set_ylim(AOI['south'], AOI['north'])
ax.set_title('(c) Clay Ratio ($B_{11}/B_{12}$)', fontsize=11, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=9)
ax.set_ylabel('Latitude (°S)', fontsize=9)
ax.tick_params(labelsize=7)
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label='B11/B12')

# Panel D: SR Propylitic Index
ax = axes[1, 1]
p2, p98 = np.percentile(sr_propylitic, [2, 98])
sc = ax.scatter(lons, lats, c=sr_propylitic, s=1.2, marker='s',
                cmap='RdYlBu_r', vmin=p2, vmax=p98, edgecolors='none')
ax.set_xlim(AOI['west'], AOI['east'])
ax.set_ylim(AOI['south'], AOI['north'])
ax.set_title('(d) SR Propylitic Index ($B_{03} - 0.48 \\cdot B_{11}$)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Longitude (°W)', fontsize=9)
ax.set_ylabel('Latitude (°S)', fontsize=9)
ax.tick_params(labelsize=7)
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label='Index value')

fig.suptitle('Sentinel-2 Imagery and Alteration Mapping — Maricunga District, Atacama, Chile\n'
             'Median composite Jan–Mar 2024, 30 m display resolution',
             fontsize=12, y=1.01)

plt.tight_layout()
fig.savefig(FIG_DIR / 'alteration_comparison.png', dpi=250, bbox_inches='tight')
fig.savefig(FIG_DIR / 'alteration_comparison.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'alteration_comparison.png'}")
