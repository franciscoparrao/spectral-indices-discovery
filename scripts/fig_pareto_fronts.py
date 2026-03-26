#!/usr/bin/env python3
"""Figure: Pareto fronts (complexity vs loss) for all 6 alteration classes."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("data/results")

CLASS_COLORS = {
    'Silicic': '#FF7F00',
    'Adv_Argillic': '#E41A1C',
    'Argillic_Phyllic': '#4DAF4A',
    'Propylitic': '#00AA00',
    'Iron_Oxide': '#A65628',
    'Potassic_Skarn': '#377EB8',
}

CLASS_LABELS = {
    'Silicic': 'Silicic',
    'Adv_Argillic': 'Adv. Argillic',
    'Argillic_Phyllic': 'Argillic-Phyllic',
    'Propylitic': 'Propylitic',
    'Iron_Oxide': 'Iron Oxide',
    'Potassic_Skarn': 'Potassic-Skarn',
}

SELECTED_FORMULAS = {
    'Silicic': ('B04 - 0.135', 3),
    'Adv_Argillic': ('0.83 - B02/B05', 5),
    'Argillic_Phyllic': ('0.09/B05', 3),
    'Propylitic': ('B03 - 0.48·B11', 5),
    'Iron_Oxide': ('(√B12 - B11)²', 5),
    'Potassic_Skarn': ('B03·B12/B07² - 0.45', 8),
}

with open(RESULTS_DIR / "pysr_results_gee.json") as f:
    results = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (cls_name, cls_data) in enumerate(
    [(k, v) for k, v in results.items() if k not in ('rf_baseline_balanced_accuracy',)]
):
    ax = axes[idx]
    pareto = cls_data['pareto_front']

    complexities = [p['complexity'] for p in pareto]
    losses = [p['loss'] for p in pareto]

    color = CLASS_COLORS.get(cls_name, '#333333')
    label = CLASS_LABELS.get(cls_name, cls_name)

    # Plot all Pareto points
    ax.plot(complexities, losses, 'o-', color=color, markersize=5,
            linewidth=1.5, alpha=0.7, zorder=2)

    # Highlight selected formula
    sel_compl = SELECTED_FORMULAS[cls_name][1]
    sel_loss = None
    for p in pareto:
        if p['complexity'] == sel_compl:
            sel_loss = p['loss']
            break
    if sel_loss is not None:
        ax.plot(sel_compl, sel_loss, '*', color=color, markersize=18,
                markeredgecolor='black', markeredgewidth=0.8, zorder=3)
        ax.annotate(SELECTED_FORMULAS[cls_name][0],
                    xy=(sel_compl, sel_loss),
                    xytext=(sel_compl + 1.5, sel_loss + (max(losses) - min(losses)) * 0.08),
                    fontsize=7, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              edgecolor='gray', alpha=0.9))

    # Diminishing returns shading
    if len(losses) > 3:
        # Mark the "knee" region
        loss_range = max(losses) - min(losses)
        knee_threshold = min(losses) + loss_range * 0.05
        for i, (c, l) in enumerate(zip(complexities, losses)):
            if l <= knee_threshold and i > 0:
                ax.axvspan(c, max(complexities), alpha=0.05, color='red')
                break

    ax.set_title(f'{label}\n(AUC: {cls_data["auc"]:.3f}, F1: {cls_data["f1"]:.3f})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Complexity (nodes)')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)

fig.suptitle('Pareto Fronts: Complexity vs. Loss for SR-Discovered Indices\n'
             '★ = Selected formula (best trade-off at complexity ≤ 8)',
             fontsize=13, y=1.02)

plt.tight_layout()
fig.savefig(FIG_DIR / 'pareto_fronts.png', dpi=250, bbox_inches='tight')
fig.savefig(FIG_DIR / 'pareto_fronts.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'pareto_fronts.png'}")
