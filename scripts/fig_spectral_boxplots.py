#!/usr/bin/env python3
"""Figure: Spectral signature boxplots per alteration class."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
GT_DIR = Path("data/ground_truth")

# Load training data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X = data['X']
y = data['y']
band_names = list(data['band_names'])

# Reorder bands by wavelength
wavelength_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
wavelengths_nm = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]

# Reindex columns
col_indices = [band_names.index(b) for b in wavelength_order]
X_ordered = X[:, col_indices]

CLASS_INFO = {
    1: ('Silicic', '#FF7F00'),
    2: ('Adv. Argillic', '#E41A1C'),
    3: ('Argillic-Phyllic', '#4DAF4A'),
    4: ('Propylitic', '#00AA00'),
    5: ('Iron Oxide', '#A65628'),
    6: ('Potassic-Skarn', '#377EB8'),
}

# ===== FIGURE 1: Mean spectral profiles with std bands =====
fig, ax = plt.subplots(figsize=(12, 6))

for cls_id in sorted(CLASS_INFO.keys()):
    mask = y == cls_id
    name, color = CLASS_INFO[cls_id]
    n = mask.sum()

    means = X_ordered[mask].mean(axis=0)
    stds = X_ordered[mask].std(axis=0)
    q25 = np.percentile(X_ordered[mask], 25, axis=0)
    q75 = np.percentile(X_ordered[mask], 75, axis=0)

    ax.plot(wavelengths_nm, means, 'o-', color=color, linewidth=2,
            markersize=5, label=f'{name} (n={n})', zorder=3)
    ax.fill_between(wavelengths_nm, q25, q75, color=color, alpha=0.15, zorder=1)

# Mark SWIR gap
ax.axvspan(900, 1500, alpha=0.03, color='gray')
ax.text(1200, ax.get_ylim()[1] * 0.95, 'No S2\nbands', ha='center', va='top',
        fontsize=8, color='gray', style='italic')

# Band labels
for i, (wl, bname) in enumerate(zip(wavelengths_nm, wavelength_order)):
    ax.text(wl, -0.015, bname, ha='center', va='top', fontsize=7,
            rotation=45, color='#555555')

ax.set_xlabel('Wavelength (nm)', fontsize=11)
ax.set_ylabel('Surface Reflectance', fontsize=11)
ax.set_title('Mean Spectral Profiles by Alteration Class\n'
             'Sentinel-2 L2A, Region III (Maricunga), Jan–Mar 2024\n'
             'Shaded area: interquartile range (Q25–Q75)',
             fontsize=12)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(400, 2300)
ax.set_ylim(-0.02, None)

# Add secondary x-axis with band names
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(wavelengths_nm)
ax2.set_xticklabels(wavelength_order, fontsize=7, rotation=45)

plt.tight_layout()
fig.savefig(FIG_DIR / 'spectral_profiles.png', dpi=250, bbox_inches='tight')
fig.savefig(FIG_DIR / 'spectral_profiles.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'spectral_profiles.png'}")

# ===== FIGURE 2: Boxplots per band, grouped by class =====
fig2, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
axes = axes.flatten()

for band_idx, (bname, wl) in enumerate(zip(wavelength_order, wavelengths_nm)):
    ax = axes[band_idx]

    box_data = []
    box_labels = []
    box_colors = []

    for cls_id in sorted(CLASS_INFO.keys()):
        mask = y == cls_id
        name, color = CLASS_INFO[cls_id]
        vals = X_ordered[mask, band_idx]
        # Subsample for speed if too many
        if len(vals) > 500:
            rng = np.random.RandomState(42)
            vals = rng.choice(vals, 500, replace=False)
        box_data.append(vals)
        box_labels.append(name.split()[0][:6])  # Short labels
        box_colors.append(color)

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.5))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(f'{bname}\n({wl} nm)', fontsize=9, fontweight='bold')
    ax.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.2)

    if band_idx % 5 == 0:
        ax.set_ylabel('Reflectance')

fig2.suptitle('Spectral Band Distributions by Alteration Class\n'
              'Sentinel-2 L2A, Region III (Maricunga)',
              fontsize=13, y=1.02)

plt.tight_layout()
fig2.savefig(FIG_DIR / 'spectral_boxplots.png', dpi=250, bbox_inches='tight')
fig2.savefig(FIG_DIR / 'spectral_boxplots.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'spectral_boxplots.png'}")
