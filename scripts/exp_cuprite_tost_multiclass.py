"""
Multi-class TOST at Cuprite mirroring the Chile analysis.

For each fold (10-fold stratified CV), compute the mean per-class OvR AUC
of RF(10 raw bands) and RF(6 Chilean SR formulas as features) — same metric
used in Table 4 of the main manuscript.

Reports the paired difference and TOST equivalence test at ε=0.01.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

DATA = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/ground_truth/cuprite_training_s2.npz")
OUT = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/results/cuprite_tost_multiclass.json")

d = np.load(DATA)
X, y, bands = d["X"], d["y"], list(d["band_names"])
b = {n: bands.index(n) for n in bands}
def B(name): return X[:, b[name]]

# 6 Chilean SR formulas applied to Cuprite data
SR = np.column_stack([
    B("B04") - 0.135,                                          # Silicic
    0.83 - B("B02") / B("B05"),                                # Adv Argillic
    0.09 / B("B05"),                                           # Argillic_Phyllic
    B("B03") - 0.48 * B("B11"),                                # Propylitic
    (np.sqrt(B("B12")) - B("B11"))**2,                         # Iron Oxide
    B("B03") * B("B12") / (B("B07")**2) - 0.45,                # Potassic_Skarn
])
mask = np.all(np.isfinite(SR), axis=1) & np.all(np.isfinite(X), axis=1)
X, y, SR = X[mask], y[mask], SR[mask]

# Keep only altered classes (1..4) + unaltered (0). Multi-class OvR will compute AUC per class.
print(f"After filtering: n={len(X)}, classes={dict(zip(*np.unique(y, return_counts=True)))}")

cls_present = [c for c in [1, 2, 3, 4] if (y == c).sum() >= 50]
print(f"Classes with sufficient positives: {cls_present}")

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_mean_raw, fold_mean_sr = [], []
for fi, (tr, te) in enumerate(skf.split(X, y)):
    rf_raw = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    rf_sr = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    rf_raw.fit(X[tr], y[tr])
    rf_sr.fit(SR[tr], y[tr])

    aucs_raw, aucs_sr = [], []
    for cls in cls_present:
        y_te = (y[te] == cls).astype(int)
        if y_te.sum() == 0:
            continue
        # OvR probability via the trained multi-class classifier
        cls_idx_raw = list(rf_raw.classes_).index(cls)
        cls_idx_sr = list(rf_sr.classes_).index(cls)
        p_raw = rf_raw.predict_proba(X[te])[:, cls_idx_raw]
        p_sr = rf_sr.predict_proba(SR[te])[:, cls_idx_sr]
        aucs_raw.append(roc_auc_score(y_te, p_raw))
        aucs_sr.append(roc_auc_score(y_te, p_sr))
    mraw, msr = np.mean(aucs_raw), np.mean(aucs_sr)
    fold_mean_raw.append(mraw); fold_mean_sr.append(msr)
    print(f"  Fold {fi+1}: raw={mraw:.4f}  SR={msr:.4f}  Δ={msr-mraw:+.4f}")

raw, sr = np.array(fold_mean_raw), np.array(fold_mean_sr)
diffs = sr - raw
mean_diff = diffs.mean()
sem = diffs.std(ddof=1) / np.sqrt(n_folds)
df_t = n_folds - 1
eps = 0.01
t1 = (mean_diff - (-eps)) / sem
p_lower = 1 - stats.t.cdf(t1, df=df_t)
t2 = (mean_diff - eps) / sem
p_upper = stats.t.cdf(t2, df=df_t)
p_tost = max(p_lower, p_upper)
t_c = stats.t.ppf(0.975, df=df_t)
ci_lo, ci_hi = mean_diff - t_c * sem, mean_diff + t_c * sem

print(f"\n=== Cuprite multi-class TOST (10-fold, ε=0.01) ===")
print(f"mean RF(raw)  = {raw.mean():.4f} ± {raw.std(ddof=1):.4f}")
print(f"mean RF(SR)   = {sr.mean():.4f} ± {sr.std(ddof=1):.4f}")
print(f"Δ = SR − raw  = {mean_diff:+.5f}")
print(f"95% CI of Δ   = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
print(f"TOST p_lower = {p_lower:.3e}, p_upper = {p_upper:.3e}, max(p) = {p_tost:.3e}")
print(f"→ Equivalence at ±0.01: {'YES (rejected H0)' if p_tost < 0.05 else 'NO'}")

# Also try wider ε to characterise the data
for eps_try in [0.005, 0.01, 0.015, 0.02]:
    t1_ = (mean_diff - (-eps_try)) / sem
    p_low_ = 1 - stats.t.cdf(t1_, df=df_t)
    t2_ = (mean_diff - eps_try) / sem
    p_up_ = stats.t.cdf(t2_, df=df_t)
    p_ = max(p_low_, p_up_)
    print(f"  ε={eps_try:.3f}: p_max={p_:.3e}, equivalent={'YES' if p_ < 0.05 else 'NO'}")

out = {"site": "Cuprite, NV (USA)", "task": "multi-class OvR mean AUC", "n_folds": n_folds, "epsilon": eps,
       "fold_auc_raw": raw.tolist(), "fold_auc_sr": sr.tolist(),
       "mean_auc_raw": float(raw.mean()), "mean_auc_sr": float(sr.mean()),
       "mean_diff": float(mean_diff), "ci95_lower": float(ci_lo), "ci95_upper": float(ci_hi),
       "tost_p_lower": float(p_lower), "tost_p_upper": float(p_upper), "tost_p": float(p_tost),
       "equivalent_at_eps_0.01": bool(p_tost < 0.05)}
OUT.write_text(json.dumps(out, indent=2))
print(f"✓ Wrote {OUT}")
