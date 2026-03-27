#!/usr/bin/env python3
"""Figure: Study area map v2 — proper Natural Earth boundaries + coordinates."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds

FIG_DIR = Path("figures")
ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")

# Load boundaries
sa = gpd.read_file("data/external/south_america.geojson")
chile = gpd.read_file("data/external/chile_boundary.geojson")

# S2 extent
with rasterio.open('data/sentinel2/maricunga/B04.tif') as src:
    bounds_m = src.bounds
    bounds_3 = transform_bounds(src.crs, 'EPSG:4326',
                                 bounds_m.left, bounds_m.bottom,
                                 bounds_m.right, bounds_m.top)

# Atlas polygons
gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:32719")
gdf = gdf.to_crs("EPSG:4326")

ALTERATION_MAP = {
    "Alteración Silicea": 1, "vuggy silica": 1,
    "Alteracion Argilica y Argilica avanzada": 2, "Alteracion Solfatárica": 2,
    "Alteracion Argilica": 3, "Alteracion Sericitica": 3,
    "Alteración Cuarzo-Sericitica(Fílica)": 3,
    "Alteracion Propilitica": 4,
    "Oxidos e Hidróxidos de Hierro": 5,
    "Alteracion Potasica": 6, "skarn": 6,
}
CLASS_COLORS = {1: '#FF7F00', 2: '#E41A1C', 3: '#4DAF4A',
                4: '#00AA00', 5: '#A65628', 6: '#377EB8'}
CLASS_NAMES = {1: 'Silicic', 2: 'Adv. Argillic', 3: 'Argillic-Phyllic',
               4: 'Propylitic', 5: 'Iron Oxide', 6: 'Potassic-Skarn'}

gdf['class_id'] = gdf['ALTERACION'].map(ALTERATION_MAP).fillna(0).astype(int)
undiff = gdf[gdf['class_id'] == 0]
classified = gdf[gdf['class_id'] > 0]

# ===== Figure =====
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.25)

# Panel A: South America with Chile highlighted
ax1 = fig.add_subplot(gs[0])
sa.plot(ax=ax1, color='#E8D5B7', edgecolor='#8B7355', linewidth=0.5)
chile.plot(ax=ax1, color='#C4A86B', edgecolor='#5B4513', linewidth=1.2)

# Study areas
ax1.plot(-69.1, -26.85, 's', color='red', markersize=10, markeredgecolor='black',
         markeredgewidth=0.8, zorder=5, label='Region III (Training)')
ax1.plot(-70.5, -30.5, 's', color='blue', markersize=10, markeredgecolor='black',
         markeredgewidth=0.8, zorder=5, label='Region IV (Validation)')
ax1.plot(-117.15, 37.55, 'D', color='green', markersize=8, markeredgecolor='black',
         markeredgewidth=0.8, zorder=5, label='Cuprite, NV (External)')

# S2 extent box
ax1.add_patch(plt.Rectangle((bounds_3[0], bounds_3[1]),
                              bounds_3[2] - bounds_3[0],
                              bounds_3[3] - bounds_3[1],
                              fill=False, edgecolor='red', linewidth=1.5,
                              linestyle='--', zorder=3))

# Cities
cities = {'Copiapó': (-70.33, -27.37), 'La Serena': (-71.25, -29.91),
          'Santiago': (-70.67, -33.45)}
for city, (lon, lat) in cities.items():
    ax1.plot(lon, lat, 'ko', markersize=3, zorder=4)
    ax1.text(lon + 0.5, lat, city, fontsize=6, color='#333333')

ax1.set_xlim(-76, -64)
ax1.set_ylim(-57, -17)
ax1.set_xlabel('Longitude (°W)', fontsize=9)
ax1.set_ylabel('Latitude (°S)', fontsize=9)
ax1.set_title('(a) Study sites in South America', fontsize=10, fontweight='bold')
ax1.legend(loc='lower left', fontsize=6, framealpha=0.9)
ax1.grid(True, alpha=0.2, linewidth=0.5)
ax1.tick_params(labelsize=7)

# Note about map lines (RSE requirement)
ax1.text(0.02, 0.02, 'Map lines delineate study areas\nand do not necessarily depict\naccepted national boundaries.',
         transform=ax1.transAxes, fontsize=5, va='bottom', color='gray', style='italic')

# Panel B: Region III alteration polygons
ax2 = fig.add_subplot(gs[1])

# Plot undifferentiated
if len(undiff) > 0:
    undiff.plot(ax=ax2, color='#CCCCCC', edgecolor='gray', linewidth=0.3, alpha=0.5)

# Plot classified
for cls_id in sorted(CLASS_COLORS.keys()):
    subset = classified[classified['class_id'] == cls_id]
    if len(subset) > 0:
        subset.plot(ax=ax2, color=CLASS_COLORS[cls_id],
                   edgecolor='black', linewidth=0.3, alpha=0.7)

ax2.set_xlim(bounds_3[0], bounds_3[2])
ax2.set_ylim(bounds_3[1], bounds_3[3])

legend_patches = [mpatches.Patch(color='#CCCCCC', label='Undifferentiated')]
for cls_id in sorted(CLASS_COLORS.keys()):
    legend_patches.append(mpatches.Patch(color=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id]))
ax2.legend(handles=legend_patches, loc='lower left', fontsize=6,
           framealpha=0.9, title='Alteration Type', title_fontsize=7)

ax2.set_xlabel('Longitude (°W)', fontsize=9)
ax2.set_ylabel('Latitude (°S)', fontsize=9)
ax2.set_title('(b) Region III — Atlas Metalífero Alteration Polygons',
              fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.2, linewidth=0.5)
ax2.tick_params(labelsize=7)

# Panel C: Training data distribution
ax3 = fig.add_subplot(gs[2])

classes = list(CLASS_NAMES.values())
counts = [2200, 5000, 5000, 495, 500, 2400]
colors = [CLASS_COLORS[i] for i in sorted(CLASS_COLORS.keys())]

bars = ax3.barh(classes, counts, color=colors, edgecolor='black', linewidth=0.5)

for bar, count in zip(bars, counts):
    ax3.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
             f'{count:,}', va='center', fontsize=8)

ax3.set_xlabel('Training Pixels', fontsize=9)
ax3.set_title('(c) Class Distribution', fontsize=10, fontweight='bold')
ax3.set_xlim(0, 6200)
ax3.grid(axis='x', alpha=0.2)
ax3.invert_yaxis()
ax3.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(FIG_DIR / 'study_area.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'study_area.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'study_area.png'}")
