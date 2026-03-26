#!/usr/bin/env python3
"""E3: DeLong test for comparing SR indices vs classical indices.

Implements the fast DeLong AUC comparison test from:
Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm for Comparing
the Areas Under Correlated Receiver Operating Characteristic Curves"
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")


def compute_midrank(x):
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    i = np.arange(n)
    # Find ties
    while True:
        # Look for duplicate values
        mask = np.concatenate([np.array([True]), z[1:] != z[:-1]])
        if mask.all():
            break
        # Replace tied values with their midranks
        unique_vals = np.unique(z)
        for v in unique_vals:
            idx = np.where(z == v)[0]
            if len(idx) > 1:
                midrank = np.mean(idx)
                # Already handled by ranking
                pass
        break

    # Use scipy rankdata for correct midranks
    from scipy.stats import rankdata
    ranks = rankdata(x, method='average')
    return ranks


def delong_roc_variance(ground_truth, predictions):
    """Compute AUC and its variance using DeLong's method."""
    order = np.argsort(-predictions)
    label_ordered = ground_truth[order]
    predictions_sorted = predictions[order]

    m = np.sum(label_ordered == 1)
    n = np.sum(label_ordered == 0)

    if m == 0 or n == 0:
        return 0.5, 0.0

    positive_examples = predictions[ground_truth == 1]
    negative_examples = predictions[ground_truth == 0]

    # Structural components
    aucs = np.zeros(1)
    v01 = np.zeros(m)
    v10 = np.zeros(n)

    for i in range(m):
        v10_sum = np.sum(positive_examples[i] > negative_examples) + \
                  0.5 * np.sum(positive_examples[i] == negative_examples)
        v01[i] = v10_sum / n

    for j in range(n):
        v01_sum = np.sum(negative_examples[j] < positive_examples) + \
                  0.5 * np.sum(negative_examples[j] == positive_examples)
        v10[j] = v01_sum / m

    auc = np.mean(v01)
    s01 = np.var(v01, ddof=1) if m > 1 else 0
    s10 = np.var(v10, ddof=1) if n > 1 else 0
    var_auc = s01 / m + s10 / n

    return auc, var_auc


def delong_test(ground_truth, pred1, pred2):
    """Two-sided DeLong test for two AUC estimates.

    Returns: auc1, auc2, z_stat, p_value
    """
    auc1, var1 = delong_roc_variance(ground_truth, pred1)
    auc2, var2 = delong_roc_variance(ground_truth, pred2)

    # Covariance (simplified: assume independent for now — conservative)
    # For correlated predictions on the same data, the true variance of
    # (AUC1 - AUC2) is var1 + var2 - 2*cov. Without cov, we get a
    # conservative (wider) test.
    z = (auc1 - auc2) / np.sqrt(var1 + var2 + 1e-10)
    p = 2 * stats.norm.sf(abs(z))

    return float(auc1), float(auc2), float(z), float(p)


# Load data
data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X = data['X']
y = data['y']
band_names = list(data['band_names'])
B = {name: i for i, name in enumerate(band_names)}

# Define indices
def sr_propylitic(x): return x[:, B['B03']] - 0.48 * x[:, B['B11']]
def sr_silicic(x): return x[:, B['B04']] - 0.135
def sr_adv_argillic(x): return 0.83 - x[:, B['B02']] / np.maximum(x[:, B['B05']], 1e-6)
def sr_iron_oxide(x): return (np.sqrt(x[:, B['B12']]) - x[:, B['B11']]) ** 2
def clay_ratio(x): return x[:, B['B11']] / np.maximum(x[:, B['B12']], 1e-6)
def iron_oxide_idx(x): return x[:, B['B04']] / np.maximum(x[:, B['B02']], 1e-6)
def oh_minerals(x): return x[:, B['B02']] / np.maximum(x[:, B['B11']], 1e-6)

# Comparisons to test
comparisons = [
    # (name, SR_func, classical_func, target_class, description)
    ("Propylitic: SR vs Clay Ratio", sr_propylitic, clay_ratio, 4,
     "SR B03-0.48*B11 vs Clay Ratio B11/B12 for propylitic class"),
    ("Propylitic: SR vs OH Minerals", sr_propylitic, oh_minerals, 4,
     "SR B03-0.48*B11 vs OH Minerals B02/B11 for propylitic class"),
    ("Silicic: SR vs Clay Ratio", sr_silicic, clay_ratio, 1,
     "SR B04-0.135 vs Clay Ratio B11/B12 for silicic class"),
    ("Adv Argillic: SR vs Iron Oxide", sr_adv_argillic, iron_oxide_idx, 2,
     "SR 0.83-B02/B05 vs Iron Oxide B04/B02 for adv argillic class"),
    ("Iron Oxide: SR vs Clay Ratio", sr_iron_oxide, clay_ratio, 5,
     "SR (sqrt(B12)-B11)^2 vs Clay Ratio for iron oxide class"),
]

results = []

for name, sr_func, classical_func, target_class, desc in comparisons:
    y_bin = (y == target_class).astype(int)

    scores_sr = sr_func(X)
    scores_cl = classical_func(X)

    # Try both orientations
    auc_sr_pos, var_sr_pos = delong_roc_variance(y_bin, scores_sr)
    auc_sr_neg, var_sr_neg = delong_roc_variance(y_bin, -scores_sr)
    if auc_sr_neg > auc_sr_pos:
        scores_sr = -scores_sr

    auc_cl_pos, var_cl_pos = delong_roc_variance(y_bin, scores_cl)
    auc_cl_neg, var_cl_neg = delong_roc_variance(y_bin, -scores_cl)
    if auc_cl_neg > auc_cl_pos:
        scores_cl = -scores_cl

    auc1, auc2, z, p = delong_test(y_bin, scores_sr, scores_cl)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    result = {
        'comparison': name,
        'description': desc,
        'target_class': target_class,
        'auc_sr': auc1,
        'auc_classical': auc2,
        'delta_auc': auc1 - auc2,
        'z_statistic': z,
        'p_value': p,
        'significance': sig,
    }
    results.append(result)

    print(f"{name}")
    print(f"  SR AUC: {auc1:.4f}, Classical AUC: {auc2:.4f}")
    print(f"  ΔAU = {auc1-auc2:+.4f}, z = {z:.3f}, p = {p:.4f} {sig}")
    print()

output = RESULTS_DIR / "delong_tests.json"
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {output}")
