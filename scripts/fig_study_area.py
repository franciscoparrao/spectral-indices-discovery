#!/usr/bin/env python3
"""Figure: Study area map — Regions III and IV in Chile with alteration zones."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from pathlib import Path
import rasterio

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Chile outline (simplified polygon for context)
# Major cities and study area locations
LOCATIONS = {
    'Region III\n(Training)': (-69.1, -26.85, 'red'),
    'Region IV\n(Validation)': (-70.5, -30.5, 'blue'),
}

CITIES = {
    'Copiapó': (-70.33, -27.37),
    'La Serena': (-71.25, -29.91),
    'Santiago': (-70.67, -33.45),
    'Antofagasta': (-70.40, -23.65),
}

# Get actual extent of our S2 data
with rasterio.open('data/sentinel2/maricunga/B04.tif') as src:
    bounds_m = src.bounds
    from rasterio.warp import transform_bounds
    bounds_3 = transform_bounds(src.crs, 'EPSG:4326',
                                 bounds_m.left, bounds_m.bottom,
                                 bounds_m.right, bounds_m.top)

# Load alteration polygons if available
ATLAS_DIR = Path("data/external/atlas_metalifero_IIIR")
has_atlas = (ATLAS_DIR / "Geometria/RMM_ALTERACI.shp").exists()

fig = plt.figure(figsize=(14, 8))

if has_atlas:
    # 3-panel layout: Chile overview + Region III detail + Region IV
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.3)
else:
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)

# ===== Panel A: Chile overview =====
ax1 = fig.add_subplot(gs[0])

# Simplified Chile shape (approximate coastline polygon)
chile_lons = [-70.5, -70.0, -69.5, -69.0, -68.5, -68.0, -67.5, -67.0, -67.5,
              -68.0, -68.5, -69.0, -69.5, -70.0, -70.5, -71.0, -71.5, -72.0,
              -73.0, -73.5, -73.0, -72.5, -72.0, -71.5, -71.0, -70.5]
chile_lats = [-18.5, -19.0, -20.0, -21.0, -22.5, -24.0, -25.5, -27.0, -28.5,
              -30.0, -31.5, -33.0, -34.5, -36.0, -37.5, -38.0, -38.0, -37.5,
              -36.0, -34.0, -32.0, -30.0, -28.0, -26.0, -23.0, -20.0]

ax1.fill(chile_lons, chile_lats, color='#E8D5B7', edgecolor='#8B7355', linewidth=1.5, zorder=1)

# Andes shading (approximate)
andes_lons = [-69.0, -68.5, -68.0, -67.5, -67.0, -67.5, -68.0, -68.5, -69.0, -69.5]
andes_lats = [-18.5, -20.0, -22.0, -25.0, -27.0, -28.5, -31.0, -33.0, -35.0, -37.0]

# Study areas
for name, (lon, lat, color) in LOCATIONS.items():
    ax1.plot(lon, lat, 's', color=color, markersize=12, markeredgecolor='black',
             markeredgewidth=1, zorder=5)
    ax1.annotate(name, xy=(lon, lat), xytext=(lon + 1.0, lat),
                 fontsize=8, fontweight='bold', color=color,
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Cities
for city, (lon, lat) in CITIES.items():
    ax1.plot(lon, lat, 'ko', markersize=4, zorder=4)
    ax1.text(lon - 0.3, lat + 0.3, city, fontsize=7, ha='right', color='#333333')

# S2 extent box for Region III
ax1.add_patch(plt.Rectangle((bounds_3[0], bounds_3[1]),
                              bounds_3[2] - bounds_3[0],
                              bounds_3[3] - bounds_3[1],
                              fill=False, edgecolor='red', linewidth=2,
                              linestyle='--', zorder=3))

ax1.set_xlim(-74, -66)
ax1.set_ylim(-39, -18)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('(a) Location in Chile', fontsize=11, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2)

# Add scale reference
ax1.plot([-73, -72], [-37.5, -37.5], 'k-', linewidth=2)
ax1.text(-72.5, -38, '~100 km', ha='center', fontsize=7)

# ===== Panel B: Region III detail with S2 RGB + alteration polygons =====
ax2 = fig.add_subplot(gs[1])

if has_atlas:
    gdf = gpd.read_file(ATLAS_DIR / "Geometria/RMM_ALTERACI.shp")
    import pandas as pd
    attrs = pd.read_csv(ATLAS_DIR / "alteracion.csv")
    gdf = gdf.merge(attrs[["INT_ORIG", "ALTERACION"]], on="INT_ORIG", how="left")
    # Shapefile has no CRS set but coordinates are UTM 19S
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

    CLASS_COLORS_MAP = {
        1: '#FF7F00', 2: '#E41A1C', 3: '#4DAF4A',
        4: '#00AA00', 5: '#A65628', 6: '#377EB8',
    }
    CLASS_NAMES = {
        1: 'Silicic', 2: 'Adv. Argillic', 3: 'Argillic-Phyllic',
        4: 'Propylitic', 5: 'Iron Oxide', 6: 'Potassic-Skarn',
    }

    gdf['class_id'] = gdf['ALTERACION'].map(ALTERATION_MAP).fillna(0).astype(int)

    # Undifferentiated
    undiff = gdf[gdf['class_id'] == 0]
    classified = gdf[gdf['class_id'] > 0]

    # Plot undifferentiated first (background)
    if len(undiff) > 0:
        undiff.plot(ax=ax2, color='#CCCCCC', edgecolor='gray', linewidth=0.3, alpha=0.5)

    # Plot classified by class
    for cls_id in sorted(CLASS_COLORS_MAP.keys()):
        subset = classified[classified['class_id'] == cls_id]
        if len(subset) > 0:
            subset.plot(ax=ax2, color=CLASS_COLORS_MAP[cls_id],
                       edgecolor='black', linewidth=0.3, alpha=0.7)

    ax2.set_xlim(bounds_3[0], bounds_3[2])
    ax2.set_ylim(bounds_3[1], bounds_3[3])

    # Manual legend
    legend_patches = [mpatches.Patch(color='#CCCCCC', label='Undifferentiated')]
    for cls_id in sorted(CLASS_COLORS_MAP.keys()):
        legend_patches.append(mpatches.Patch(color=CLASS_COLORS_MAP[cls_id],
                                              label=CLASS_NAMES[cls_id]))
    ax2.legend(handles=legend_patches, loc='lower left', fontsize=7,
               framealpha=0.9, title='Alteration Type', title_fontsize=8)
else:
    # Fallback: just show the S2 extent
    ax2.add_patch(plt.Rectangle((bounds_3[0], bounds_3[1]),
                                  bounds_3[2] - bounds_3[0],
                                  bounds_3[3] - bounds_3[1],
                                  fill=True, facecolor='#E8D5B7',
                                  edgecolor='red', linewidth=2))
    ax2.text(np.mean([bounds_3[0], bounds_3[2]]),
             np.mean([bounds_3[1], bounds_3[3]]),
             'Sentinel-2\nCoverage', ha='center', va='center', fontsize=12)
    ax2.set_xlim(bounds_3[0] - 0.1, bounds_3[2] + 0.1)
    ax2.set_ylim(bounds_3[1] - 0.1, bounds_3[3] + 0.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('(b) Region III — Alteration Polygons (Atlas Metalífero)',
              fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2)

# ===== Panel C: Training data summary =====
if has_atlas:
    ax3 = fig.add_subplot(gs[2])

    classes = list(CLASS_NAMES.values())
    counts = [2200, 5000, 5000, 495, 500, 2400]
    colors = [CLASS_COLORS_MAP[i] for i in sorted(CLASS_COLORS_MAP.keys())]

    bars = ax3.barh(classes, counts, color=colors, edgecolor='black', linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax3.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                 f'{count:,}', va='center', fontsize=9)

    ax3.set_xlabel('Number of Training Pixels', fontsize=10)
    ax3.set_title('(c) Training Data Distribution', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 6000)
    ax3.grid(axis='x', alpha=0.2)
    ax3.invert_yaxis()

fig.suptitle('Study Areas and Ground Truth Data\n'
             'Sentinel-2 L2A Median Composite, Jan–Mar 2024',
             fontsize=13, y=1.02)

plt.tight_layout()
fig.savefig(FIG_DIR / 'study_area.png', dpi=250, bbox_inches='tight')
fig.savefig(FIG_DIR / 'study_area.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'study_area.png'}")
