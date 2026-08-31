"""
Detection-rate curves at top-k% area for SR-discovered and classical indices.

Produces Supplementary Figure S3 for the NRR submission.

For each continuous index, sorts pixels by score (descending) and plots
the fraction of true-positive class pixels captured (TPR) as a function
of the fraction of total area flagged (top-k% threshold).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/paper/nrr_submission/detection_curves.png")

df = pd.read_csv(DATA)

# SR-discovered formulas (Region III, Atlas)
sr_formulas = {
    "Silicic":          ("B04 - 0.135",                  lambda d: d.B04 - 0.135),
    "Adv_Argillic":     ("0.83 - B02/B05",               lambda d: 0.83 - d.B02 / d.B05),
    "Argillic_Phyllic": ("0.09 / B05",                   lambda d: 0.09 / d.B05),
    "Propylitic":       ("B03 - 0.48*B11",               lambda d: d.B03 - 0.48 * d.B11),
    "Iron_Oxide":       ("(sqrt(B12) - B11)^2",          lambda d: (np.sqrt(d.B12) - d.B11) ** 2),
    "Potassic_Skarn":   ("B03*B12/B07^2 - 0.45",         lambda d: d.B03 * d.B12 / (d.B07 ** 2) - 0.45),
}

# Classical indices
classical = {
    "Clay Ratio":   ("B11/B12",                lambda d: d.B11 / d.B12),
    "Iron Oxide":   ("B04/B02",                lambda d: d.B04 / d.B02),
    "Ferrous":      ("B12/B8A",                lambda d: d.B12 / d.B8A),
    "Alunite Idx":  ("(B11-B12)/(B11+B12)",    lambda d: (d.B11 - d.B12) / (d.B11 + d.B12)),
    "OH Minerals":  ("B02/B11",                lambda d: d.B02 / d.B11),
    "Silica":       ("B12/B11",                lambda d: d.B12 / d.B11),
    "NDVI":         ("(B8A-B04)/(B8A+B04)",    lambda d: (d.B8A - d.B04) / (d.B8A + d.B04)),
}

# Per-class relevant classical indices (the ones with mineralogical motivation)
relevant_classical_per_class = {
    "Silicic":          ["Silica", "Clay Ratio", "NDVI"],
    "Adv_Argillic":     ["Clay Ratio", "Alunite Idx", "OH Minerals"],
    "Argillic_Phyllic": ["Clay Ratio", "OH Minerals", "Alunite Idx"],
    "Propylitic":       ["Clay Ratio", "OH Minerals", "Iron Oxide"],
    "Iron_Oxide":       ["Iron Oxide", "Ferrous", "Clay Ratio"],
    "Potassic_Skarn":   ["Clay Ratio", "Ferrous", "Silica"],
}

def detection_curve(scores, labels, n_thresholds=200):
    """Compute (area_fraction, tpr) curve from a ranking score."""
    # Try both directions; keep the one with higher initial slope (better)
    for sign in (+1, -1):
        s = sign * scores
        order = np.argsort(-s)  # descending
        y_sorted = labels[order]
        cum_tpr = np.cumsum(y_sorted) / y_sorted.sum()
        cum_area = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        # Integrate to compare directions
        auc_curve = np.trapz(cum_tpr, cum_area)
        if sign == +1:
            best_area, best_tpr, best_auc = cum_area, cum_tpr, auc_curve
        else:
            if auc_curve > best_auc:
                best_area, best_tpr, best_auc = cum_area, cum_tpr, auc_curve
    return best_area, best_tpr

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.flatten()

colors = {"SR": "#d62728", "Clay Ratio": "#1f77b4", "Iron Oxide": "#ff7f0e", "Ferrous": "#2ca02c",
          "Alunite Idx": "#9467bd", "OH Minerals": "#8c564b", "Silica": "#e377c2", "NDVI": "#7f7f7f"}

for ax, cls in zip(axes, sr_formulas.keys()):
    y = (df.class_name == cls).astype(int).values

    # Classical indices for this class
    for cl in relevant_classical_per_class[cls]:
        scores = classical[cl][1](df).values
        valid = np.isfinite(scores)
        a, t = detection_curve(scores[valid], y[valid])
        ax.plot(a, t, color=colors[cl], lw=1.3, alpha=0.85, label=f"{cl}")

    # SR formula
    expr, fn = sr_formulas[cls]
    scores = fn(df).values
    valid = np.isfinite(scores)
    a, t = detection_curve(scores[valid], y[valid])
    ax.plot(a, t, color=colors["SR"], lw=2.4, label=f"SR: {expr}")

    # Diagonal (random)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="Random")

    pretty = cls.replace("_", " ")
    ax.set_title(f"{pretty} (n+={y.sum()}, n-={len(y)-y.sum()})", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

for ax in axes[3:]:
    ax.set_xlabel("Fraction of area flagged (top-$k$%)", fontsize=10)
for ax in axes[::3]:
    ax.set_ylabel("True-positive rate captured", fontsize=10)

fig.suptitle("Detection-rate curves: SR-discovered vs classical spectral indices (Region III)", fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"✓ Saved {OUT} ({OUT.stat().st_size//1024} KB)")
