#!/usr/bin/env python3
"""Figure: AUC comparison bar chart — classical vs SR indices, intra-site and cross-site."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("data/results")

with open(RESULTS_DIR / "full_evaluation.json") as f:
    results = json.load(f)

# Cross-site AUC per class for the key methods
# Classes in cross-site: 1=Silicic, 2=Adv_Argillic, 3=Argillic_Phyllic, 4=Propylitic
class_ids = ['1', '2', '3', '4']
class_labels = ['Silicic', 'Adv. Argillic', 'Argillic-\nPhyllic', 'Propylitic']

# SR indices mapped to their target class
sr_target = {
    'SR: B04 - 0.135': '1',
    'SR: 0.83 - B02/B05': '2',
    'SR: 0.09/B05': '3',
    'SR: B03 - B11*0.48': '4',
}

# Methods to compare (classical + SR target)
classical_methods = [
    ('Clay Ratio (B11/B12)', 'Clay Ratio'),
    ('Iron Oxide (B04/B02)', 'Iron Oxide'),
    ('Alunite Idx (B11-B12)/(B11+B12)', 'Alunite Idx'),
    ('NDVI', 'NDVI'),
]

# ===== FIGURE 1: Grouped bar chart — cross-site AUC per class =====
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

for ax_idx, (cls_id, cls_label) in enumerate(zip(class_ids, class_labels)):
    ax = axes[ax_idx]

    # Get the SR index for this class
    sr_key = [k for k, v in sr_target.items() if v == cls_id][0]
    sr_label = sr_key.replace('SR: ', 'SR: ')

    methods = []
    aucs_intra = []
    aucs_cross = []

    # Classical
    for key, label in classical_methods:
        if key in results['intra_site'] and cls_id in results['intra_site'][key]:
            methods.append(label)
            aucs_intra.append(results['intra_site'][key][cls_id])
            aucs_cross.append(results['cross_site'].get(key, {}).get(cls_id, 0.5))

    # SR
    if sr_key in results['intra_site'] and cls_id in results['intra_site'][sr_key]:
        methods.append(sr_label.replace('SR: ', ''))
        aucs_intra.append(results['intra_site'][sr_key][cls_id])
        aucs_cross.append(results['cross_site'].get(sr_key, {}).get(cls_id, 0.5))

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, aucs_intra, width, label='Intra-site',
                   color='#4C72B0', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, aucs_cross, width, label='Cross-site',
                   color='#DD8452', alpha=0.85, edgecolor='white')

    # Highlight the SR bar
    n_classical = len(classical_methods)
    if len(methods) > n_classical:
        bars1[n_classical].set_edgecolor('black')
        bars1[n_classical].set_linewidth(1.5)
        bars2[n_classical].set_edgecolor('black')
        bars2[n_classical].set_linewidth(1.5)

    ax.set_title(cls_label, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.2)

    if ax_idx == 0:
        ax.set_ylabel('AUC (one-vs-rest)')
    if ax_idx == 3:
        ax.legend(loc='upper right', fontsize=8)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0.55:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.55:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)

fig.suptitle('AUC Comparison: Classical Indices vs. SR-Discovered Index per Alteration Class\n'
             'Intra-site (5-fold CV, Region III) and Cross-site (Region III → IV)',
             fontsize=12, y=1.04)

plt.tight_layout()
fig.savefig(FIG_DIR / 'auc_comparison.png', dpi=250, bbox_inches='tight')
fig.savefig(FIG_DIR / 'auc_comparison.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'auc_comparison.png'}")

# ===== FIGURE 2: Summary heatmap — all methods × all classes =====
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Build matrix: methods × classes (cross-site AUC)
all_methods = [
    ('Clay Ratio (B11/B12)', 'Clay Ratio (B11/B12)'),
    ('Iron Oxide (B04/B02)', 'Iron Oxide (B04/B02)'),
    ('Ferrous (B12/B8A)', 'Ferrous (B12/B8A)'),
    ('Alunite Idx (B11-B12)/(B11+B12)', 'Alunite Idx'),
    ('OH Minerals (B02/B11)', 'OH Minerals (B02/B11)'),
    ('Silica (B12/B11)', 'Silica (B12/B11)'),
    ('NDVI', 'NDVI'),
    ('SR: B04 - 0.135', 'SR: B04 − 0.135 (Silicic)'),
    ('SR: 0.83 - B02/B05', 'SR: 0.83 − B02/B05 (Adv. Arg.)'),
    ('SR: 0.09/B05', 'SR: 0.09/B05 (Argillic)'),
    ('SR: B03 - B11*0.48', 'SR: B03 − 0.48·B11 (Propylitic)'),
    ('SR: (sqrt(B12)-B11)²', 'SR: (√B12−B11)² (Iron Oxide)'),
    ('SR: B03*B12/B07² - 0.45', 'SR: B03·B12/B07²−0.45 (Pot.)'),
]

cross = results['cross_site']
matrix = np.zeros((len(all_methods), len(class_ids)))
method_labels = []

for i, (key, label) in enumerate(all_methods):
    method_labels.append(label)
    for j, cid in enumerate(class_ids):
        if key in cross and cid in cross[key]:
            matrix[i, j] = cross[key][cid]
        else:
            matrix[i, j] = np.nan

im = ax2.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.45, vmax=0.90)

# Annotate cells
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.55 or val > 0.82 else 'black'
            ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=8, color=color, fontweight='bold' if val > 0.75 else 'normal')

# Separator line between classical and SR
ax2.axhline(6.5, color='black', linewidth=2)
ax2.text(-0.8, 3, 'Classical', rotation=90, va='center', ha='center',
         fontsize=9, fontweight='bold', color='gray')
ax2.text(-0.8, 10, 'SR', rotation=90, va='center', ha='center',
         fontsize=9, fontweight='bold', color='#E41A1C')

ax2.set_xticks(range(len(class_ids)))
ax2.set_xticklabels(['Silicic', 'Adv. Argillic', 'Argillic-Phyllic', 'Propylitic'],
                     fontsize=10)
ax2.set_yticks(range(len(method_labels)))
ax2.set_yticklabels(method_labels, fontsize=8)

cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Cross-site AUC', fontsize=10)

ax2.set_title('Cross-site Validation AUC: Region III → Region IV\n'
              '(One-versus-rest, per alteration class)',
              fontsize=12, fontweight='bold')

fig2.tight_layout()
fig2.savefig(FIG_DIR / 'auc_heatmap.png', dpi=250, bbox_inches='tight')
fig2.savefig(FIG_DIR / 'auc_heatmap.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'auc_heatmap.png'}")
