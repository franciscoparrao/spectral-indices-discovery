#!/usr/bin/env python3
"""Figure: Alteration maps — apply SR indices to Maricunga S2 imagery."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
S2_DIR = Path("data/sentinel2/maricunga")

# Load bands (resample all to 20m = B11/B12 resolution)
print("Loading S2 bands...")

# Read B11 as reference grid (20m)
with rasterio.open(S2_DIR / "B11.tif") as ref:
    ref_profile = ref.profile.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = ref.shape
    B11_full = ref.read(1).astype(np.float32) / 10000.0

# For 10m bands, we need to resample to 20m
def load_band(name, ref_shape, ref_transform, ref_crs):
    with rasterio.open(S2_DIR / f"{name}.tif") as src:
        if src.shape == ref_shape:
            return src.read(1).astype(np.float32) / 10000.0
        else:
            # Resample to reference grid
            dst_array = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            return dst_array / 10000.0

bands = {}
for bname in ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B12']:
    print(f"  Loading {bname}...")
    bands[bname] = load_band(bname, ref_shape, ref_transform, ref_crs)
bands['B11'] = B11_full

# Mask nodata
valid = bands['B04'] > 0

# ===== Compute SR indices =====
print("Computing SR indices...")

eps = 1e-6

indices = {
    'Silicic\n$B_{04} - 0.135$': bands['B04'] - 0.135,
    'Adv. Argillic\n$0.83 - B_{02}/B_{05}$': 0.83 - bands['B02'] / np.maximum(bands['B05'], eps),
    'Propylitic\n$B_{03} - 0.48 \\cdot B_{11}$': bands['B03'] - 0.48 * bands['B11'],
    'Iron Oxide\n$(\\sqrt{B_{12}} - B_{11})^2$': (np.sqrt(bands['B12']) - bands['B11']) ** 2,
    'Potassic-Skarn\n$B_{03} B_{12}/B_{07}^2 - 0.45$': (
        bands['B03'] * bands['B12'] / np.maximum(bands['B07'] ** 2, eps) - 0.45
    ),
    'SWIR Composite\nmax(SR scores)': None,  # Will fill below
}

# Also compute classical for comparison
classical = {
    'Clay Ratio\n$B_{11}/B_{12}$': bands['B11'] / np.maximum(bands['B12'], eps),
}

# Create composite: max normalized SR score per pixel
sr_names = [k for k in indices if k != 'SWIR Composite\nmax(SR scores)']
sr_arrays = [indices[k] for k in sr_names]

# Normalize each to [0,1] using percentiles
sr_norm = []
for arr in sr_arrays:
    arr_clean = arr[valid]
    p2, p98 = np.percentile(arr_clean, [2, 98])
    normalized = np.clip((arr - p2) / max(p98 - p2, eps), 0, 1)
    sr_norm.append(normalized)

sr_stack = np.stack(sr_norm, axis=0)
composite = sr_stack.max(axis=0)
composite_class = sr_stack.argmax(axis=0)
indices['SWIR Composite\nmax(SR scores)'] = composite

# Subset to a region of interest (central area with most alteration)
# Use center 40% of the image for detail
h, w = ref_shape
r0, r1 = int(h * 0.25), int(h * 0.65)
c0, c1 = int(w * 0.2), int(w * 0.7)

print(f"Plotting subset: rows {r0}:{r1}, cols {c0}:{c1}")

# ===== Main figure: 6 SR indices =====
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, (name, arr) in enumerate(indices.items()):
    if idx >= 6:
        break
    ax = axes[idx]

    sub = arr[r0:r1, c0:c1].copy()
    sub_valid = valid[r0:r1, c0:c1]
    sub[~sub_valid] = np.nan

    # Use percentile-based color scaling
    vals = sub[sub_valid]
    if len(vals) == 0:
        continue

    if 'Composite' in name:
        cmap = 'hot'
        vmin, vmax = 0, 1
    elif 'Clay' in name:
        cmap = 'RdYlBu_r'
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        cmap = 'RdYlBu_r'
        p2, p98 = np.percentile(vals, [2, 98])
        vmin, vmax = p2, p98

    im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

fig.suptitle('SR-Discovered Alteration Indices Applied to Sentinel-2\n'
             'Region III (Maricunga District), Atacama, Chile',
             fontsize=14, y=1.01)

plt.tight_layout()
fig.savefig(FIG_DIR / 'alteration_maps_sr.png', dpi=200, bbox_inches='tight')
fig.savefig(FIG_DIR / 'alteration_maps_sr.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'alteration_maps_sr.png'}")

# ===== Comparison figure: RGB + Classical + SR Composite + Class Map =====
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: RGB
ax = axes2[0, 0]
rgb = np.stack([bands['B04'][r0:r1, c0:c1],
                bands['B03'][r0:r1, c0:c1],
                bands['B02'][r0:r1, c0:c1]], axis=-1)
# Stretch for visualization
rgb_stretch = np.clip(rgb * 3.5, 0, 1)
ax.imshow(rgb_stretch)
ax.set_title('(a) Sentinel-2 True Color (B04-B03-B02)', fontsize=11, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

# Panel B: SWIR false color
ax = axes2[0, 1]
swir_rgb = np.stack([bands['B12'][r0:r1, c0:c1],
                     bands['B11'][r0:r1, c0:c1],
                     bands['B04'][r0:r1, c0:c1]], axis=-1)
swir_stretch = np.clip(swir_rgb * 3.5, 0, 1)
ax.imshow(swir_stretch)
ax.set_title('(b) SWIR False Color (B12-B11-B04)', fontsize=11, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

# Panel C: Clay Ratio (classical)
ax = axes2[1, 0]
clay = classical['Clay Ratio\n$B_{11}/B_{12}$'][r0:r1, c0:c1].copy()
clay_v = clay[valid[r0:r1, c0:c1]]
clay[~valid[r0:r1, c0:c1]] = np.nan
im = ax.imshow(clay, cmap='RdYlBu_r',
               vmin=np.percentile(clay_v, 2), vmax=np.percentile(clay_v, 98))
ax.set_title('(c) Classical Clay Ratio ($B_{11}/B_{12}$)', fontsize=11, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

# Panel D: SR Alteration class map
ax = axes2[1, 1]
class_map = composite_class[r0:r1, c0:c1].astype(float)
class_map[~valid[r0:r1, c0:c1]] = np.nan

class_colors = ['#FF7F00', '#E41A1C', '#00AA00', '#A65628', '#377EB8']
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_cls = ListedColormap(class_colors)
bounds_cls = np.arange(-0.5, len(class_colors) + 0.5, 1)
norm_cls = BoundaryNorm(bounds_cls, cmap_cls.N)

im = ax.imshow(class_map, cmap=cmap_cls, norm=norm_cls, interpolation='nearest')
ax.set_title('(d) SR-Predicted Dominant Alteration Type', fontsize=11, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

# Legend
import matplotlib.patches as mpatches
sr_short_names = ['Silicic', 'Adv. Argillic', 'Propylitic', 'Iron Oxide', 'Potassic-Skarn']
legend_patches = [mpatches.Patch(color=c, label=n) for c, n in zip(class_colors, sr_short_names)]
ax.legend(handles=legend_patches, loc='lower left', fontsize=8, framealpha=0.9)

fig2.suptitle('Sentinel-2 Imagery and Alteration Mapping Comparison\n'
              'Region III (Maricunga District), Atacama, Chile',
              fontsize=13, y=1.01)

plt.tight_layout()
fig2.savefig(FIG_DIR / 'alteration_comparison.png', dpi=200, bbox_inches='tight')
fig2.savefig(FIG_DIR / 'alteration_comparison.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'alteration_comparison.png'}")
