#!/usr/bin/env python3
"""
Formal polygon-level TOST.

Extends scripts/exp_tost_expanded.py: re-runs the polygon-disjoint OOF analysis,
then computes a formal TOST p-value from the polygon-level paired bootstrap
distribution (in addition to the percentile-based CI already reported).

Two TOST approaches reported:
  1. Empirical bootstrap TOST:
       p_upper = P_boot(Δ ≥ +ε),  p_lower = P_boot(Δ ≤ -ε)
       p_TOST  = max(p_upper, p_lower)
  2. Normal-approximation TOST from bootstrap SE:
       z_upper = (ε - mean) / SE,  z_lower = (mean + ε) / SE
       p_TOST  = max(1-Φ(z_upper), 1-Φ(z_lower))

Both are reported so the reader can see the TOST result is robust to the choice
of inference scheme.

Output: data/results/tost_polygon_formal.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("data/results/tost_polygon_formal.json")

df = pd.read_csv(CSV)
BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
X = df[BANDS].values
y = df['class_id'].values
lon = df['lon'].values
lat = df['lat'].values

# SR features (same six formulas as the manuscript)
eps = 1e-6
b = {n: i for i, n in enumerate(BANDS)}
sr = np.column_stack([
    X[:, b['B04']] - 0.135,
    0.83 - X[:, b['B02']] / np.maximum(X[:, b['B05']], eps),
    0.09 / np.maximum(X[:, b['B05']], eps),
    X[:, b['B03']] - 0.48 * X[:, b['B11']],
    (np.sqrt(np.maximum(X[:, b['B12']], 0)) - X[:, b['B11']]) ** 2,
    X[:, b['B03']] * X[:, b['B12']] / np.maximum(X[:, b['B07']]**2, eps) - 0.45,
])

# Polygon IDs via DBSCAN at ~110 m
poly_id = np.full(len(df), -1, dtype=int)
next_id = 0
EPS_DEG = 0.001
for c in sorted(np.unique(y)):
    mask = y == c
    coords = np.column_stack([lon[mask], lat[mask]])
    if len(coords) < 2:
        poly_id[mask] = next_id; next_id += 1; continue
    lat_mean = coords[:, 1].mean()
    cs = coords.copy(); cs[:, 0] *= np.cos(np.deg2rad(lat_mean))
    labels = DBSCAN(eps=EPS_DEG, min_samples=2).fit(cs).labels_
    for lbl in np.unique(labels):
        if lbl == -1:
            for i in np.where(mask)[0][labels == -1]:
                poly_id[i] = next_id; next_id += 1
        else:
            poly_id[np.where(mask)[0][labels == lbl]] = next_id; next_id += 1
n_polys = len(np.unique(poly_id))
print(f"Reconstructed polygons: {n_polys}")

# OOF predictions with polygon-disjoint GroupKFold(5)
gkf = GroupKFold(n_splits=5)
n_classes = len(np.unique(y))
oof_raw = np.zeros((len(y), n_classes))
oof_sr  = np.zeros((len(y), n_classes))
print("Training polygon-disjoint OOF (5 folds)...")
for fi, (tr, te) in enumerate(gkf.split(X, y, groups=poly_id)):
    rf_raw = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
    rf_sr  = RandomForestClassifier(200, max_depth=15, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
    rf_raw.fit(X[tr], y[tr]); rf_sr.fit(sr[tr], y[tr])
    for i_dst, c in enumerate(np.unique(y)):
        if c in rf_raw.classes_:
            oof_raw[te, i_dst] = rf_raw.predict_proba(X[te])[:, list(rf_raw.classes_).index(c)]
        if c in rf_sr.classes_:
            oof_sr[te, i_dst] = rf_sr.predict_proba(sr[te])[:, list(rf_sr.classes_).index(c)]
    print(f"  fold {fi+1}/5 done")

# Bootstrap polygon-level paired ΔmAUC
rng = np.random.default_rng(42)
B = 2000
polys = np.unique(poly_id)
delta_boot = np.zeros(B)
print(f"Polygon-level paired bootstrap (B={B})...")
for bi in range(B):
    boot = rng.choice(polys, size=len(polys), replace=True)
    idx = np.concatenate([np.where(poly_id == pg)[0] for pg in boot])
    yb_all = y[idx]
    per_class = []
    for i_dst, c in enumerate(np.unique(y)):
        yb = (yb_all == c).astype(int)
        if yb.sum() > 0 and yb.sum() < len(yb):
            if oof_sr[:, i_dst].sum() == 0:
                continue
            a_r = roc_auc_score(yb, oof_raw[idx, i_dst])
            a_s = roc_auc_score(yb, oof_sr[idx, i_dst])
            per_class.append(a_s - a_r)
    delta_boot[bi] = np.mean(per_class) if per_class else np.nan
delta_boot = delta_boot[~np.isnan(delta_boot)]
B_eff = len(delta_boot)
print(f"  Effective replicates: {B_eff}")

# Empirical bootstrap TOST
EPSILON = 0.01
p_upper_emp = float((delta_boot >= +EPSILON).mean())
p_lower_emp = float((delta_boot <= -EPSILON).mean())
p_tost_emp  = max(p_upper_emp, p_lower_emp)

# Normal-approx TOST from bootstrap mean and SE
mean_d = float(delta_boot.mean())
se_d   = float(delta_boot.std(ddof=1))
z_upper = (EPSILON - mean_d) / se_d
z_lower = (mean_d - (-EPSILON)) / se_d
p_upper_n = float(1.0 - stats.norm.cdf(z_upper))
p_lower_n = float(1.0 - stats.norm.cdf(z_lower))
p_tost_n  = max(p_upper_n, p_lower_n)

# Percentiles for reporting
p5  = float(np.percentile(delta_boot, 5))
p25 = float(np.percentile(delta_boot, 2.5))
p975 = float(np.percentile(delta_boot, 97.5))
p95 = float(np.percentile(delta_boot, 95))
p_within = float((np.abs(delta_boot) < EPSILON).mean())

print("\n=== Polygon-level TOST results ===")
print(f"mean Δ      = {mean_d:+.5f}")
print(f"SE          = {se_d:.5f}")
print(f"95% CI      = [{p25:+.5f}, {p975:+.5f}]")
print(f"90% CI      = [{p5:+.5f}, {p95:+.5f}]  (one-sided 95%)")
print(f"P(|Δ|<{EPSILON}) = {p_within:.4f}")
print(f"Empirical TOST  ε={EPSILON}: p_upper={p_upper_emp:.4f}  p_lower={p_lower_emp:.4f}  p_TOST={p_tost_emp:.4f}")
print(f"Normal-approx   ε={EPSILON}: p_upper={p_upper_n:.4f}  p_lower={p_lower_n:.4f}  p_TOST={p_tost_n:.4f}")

results = {
    'description': 'Formal polygon-level TOST equivalence test (ε=0.01)',
    'n_polygons': int(n_polys),
    'B_bootstrap': int(B_eff),
    'epsilon': EPSILON,
    'mean_delta': mean_d,
    'se_delta': se_d,
    'ci_95': [p25, p975],
    'ci_90_one_sided_95': [p5, p95],
    'p_within_epsilon': p_within,
    'empirical_tost': {
        'p_upper': p_upper_emp,
        'p_lower': p_lower_emp,
        'p_tost': p_tost_emp,
        'equivalent_at_alpha_0.05': bool(p_tost_emp < 0.05),
    },
    'normal_approx_tost': {
        'z_upper': z_upper,
        'z_lower': z_lower,
        'p_upper': p_upper_n,
        'p_lower': p_lower_n,
        'p_tost': p_tost_n,
        'equivalent_at_alpha_0.05': bool(p_tost_n < 0.05),
    },
}
OUT.write_text(json.dumps(results, indent=2))
print(f"\n✓ Saved: {OUT}")
