"""
Cuprite ablation: RF(4 SR target-relevant) vs RF(6 SR full set) vs RF(10 raw).

Checks whether the two Chilean SR formulas (Iron-Oxide and Potassic-Skarn) with
no target class at Cuprite contribute to or degrade the RF(6 SR) result.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/ground_truth/cuprite_training_s2.npz")
OUT = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/results/cuprite_ablation_4vs6.json")

d = np.load(DATA)
X, y, bands = d["X"], d["y"], list(d["band_names"])
b = {n: bands.index(n) for n in bands}
def B(n): return X[:, b[n]]

# Six Chilean SR formulas applied at Cuprite
sr_full = np.column_stack([
    B("B04") - 0.135,                                # Silicic
    0.83 - B("B02") / B("B05"),                      # Adv Argillic
    0.09 / B("B05"),                                 # Argillic_Phyllic
    B("B03") - 0.48 * B("B11"),                      # Propylitic
    (np.sqrt(B("B12")) - B("B11"))**2,               # Iron Oxide  (no target at Cuprite)
    B("B03") * B("B12") / (B("B07")**2) - 0.45,      # Potassic_Skarn (no target at Cuprite)
])
sr_4target = sr_full[:, :4]  # only the formulas whose targets exist at Cuprite

mask = np.all(np.isfinite(sr_full), axis=1) & np.all(np.isfinite(X), axis=1)
X, y, sr_full, sr_4 = X[mask], y[mask], sr_full[mask], sr_4target[mask]
print(f"After filtering: n={len(X)}, classes={dict(zip(*np.unique(y, return_counts=True)))}")

cls_present = [c for c in [1, 2, 3, 4] if (y == c).sum() >= 50]
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

def fold_mauc(features):
    out = []
    for tr, te in skf.split(features, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(features[tr], y[tr])
        aucs = []
        for cls in cls_present:
            y_te = (y[te] == cls).astype(int)
            if y_te.sum() == 0: continue
            idx = list(rf.classes_).index(cls)
            p = rf.predict_proba(features[te])[:, idx]
            aucs.append(roc_auc_score(y_te, p))
        out.append(np.mean(aucs))
    return np.array(out)

m6 = fold_mauc(sr_full)
m4 = fold_mauc(sr_4)
mraw = fold_mauc(X)

print(f"\n=== Cuprite ablation (10-fold, OvR mAUC) ===")
print(f"RF(6 SR full)   = {m6.mean():.4f} ± {m6.std(ddof=1):.4f}")
print(f"RF(4 SR target) = {m4.mean():.4f} ± {m4.std(ddof=1):.4f}")
print(f"RF(10 raw)      = {mraw.mean():.4f} ± {mraw.std(ddof=1):.4f}")
print(f"|Δ(6,4)| = {abs(m6.mean()-m4.mean()):.4f}")
print(f"|Δ(6,raw)| = {abs(m6.mean()-mraw.mean()):.4f}")
print(f"|Δ(4,raw)| = {abs(m4.mean()-mraw.mean()):.4f}")

out = {
    "site": "Cuprite, NV",
    "n_folds": n_folds,
    "mean_auc_6sr_full": float(m6.mean()),
    "mean_auc_4sr_target": float(m4.mean()),
    "mean_auc_10raw": float(mraw.mean()),
    "abs_delta_6_minus_4": float(abs(m6.mean() - m4.mean())),
    "abs_delta_4_minus_raw": float(abs(m4.mean() - mraw.mean())),
    "abs_delta_6_minus_raw": float(abs(m6.mean() - mraw.mean())),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"✓ Wrote {OUT}")
