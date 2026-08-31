"""
Phase 4.7 — Cuprite TOST equivalence test + detection-rate curves.

Replicates the Chile equivalence framework at Cuprite to symmetrize the
statistical evidence behind the method-transferability claim.

Outputs:
  - cuprite_tost.json   (per-fold AUCs + TOST p-values)
  - cuprite_detection_curves.png  (Supplementary Figure S4)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

DATA = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/ground_truth/cuprite_training_s2.npz")
OUT_JSON = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/data/results/cuprite_tost.json")
OUT_PNG = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/paper/nrr_submission/cuprite_detection_curves.png")

d = np.load(DATA)
X, y, bands = d["X"], d["y"], list(d["band_names"])
print(f"Loaded: X{X.shape}, classes={sorted(set(y.tolist()))}, bands={bands}")

# Map band names to column indices
b = {name: bands.index(name) for name in bands}
def B(name): return X[:, b[name]]

# Locally-rediscovered Cuprite SR formulas (from §4.5, Table 5)
# Note: signs / directions handled by detection_curve()
sr_formulas_cuprite = {
    "Silicic":          ("log(B11/B12)",            lambda: np.log(B("B11") / B("B12"))),
    "Adv_Argillic":     ("B11 - B12",                lambda: B("B11") - B("B12")),
    "Argillic_Phyllic": ("tanh(B11) - B8A",          lambda: np.tanh(B("B11")) - B("B8A")),
    "Propylitic":       ("((B12-B08)/B07)^2",        lambda: ((B("B12") - B("B08")) / B("B07")) ** 2),
}

# Classical indices for Cuprite (4 classes available)
classical = {
    "Clay Ratio":  ("B11/B12",                 lambda: B("B11") / B("B12")),
    "Iron Oxide":  ("B04/B02",                 lambda: B("B04") / B("B02")),
    "OH Minerals": ("B02/B11",                 lambda: B("B02") / B("B11")),
    "Alunite Idx": ("(B11-B12)/(B11+B12)",     lambda: (B("B11") - B("B12")) / (B("B11") + B("B12"))),
    "Silica":      ("B12/B11",                 lambda: B("B12") / B("B11")),
}

# USGS class mapping: 1=Silicic, 2=Adv_Argillic, 3=Argillic_Phyllic, 4=Propylitic, 0=Unaltered
class_names_cuprite = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic", 4: "Propylitic"}

# ---------- Per-fold equivalence: RF(SR) vs RF(raw) ----------
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Build SR feature matrix (only the 4 Cuprite formulas)
SR_features = np.column_stack([fn() for _, fn in sr_formulas_cuprite.values()])
mask = np.all(np.isfinite(SR_features), axis=1)
X_clean, y_clean, SR_clean = X[mask], y[mask], SR_features[mask]
print(f"After filtering NaN/Inf: n={len(X_clean)}")

# Binary alteration target: y > 0 means altered
y_bin = (y_clean > 0).astype(int)

fold_aucs_raw, fold_aucs_sr = [], []
for fold_idx, (tr, te) in enumerate(skf.split(X_clean, y_bin)):
    # RF on raw bands
    rf_raw = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    rf_raw.fit(X_clean[tr], y_bin[tr])
    auc_raw = roc_auc_score(y_bin[te], rf_raw.predict_proba(X_clean[te])[:, 1])

    # RF on SR features
    rf_sr = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    rf_sr.fit(SR_clean[tr], y_bin[tr])
    auc_sr = roc_auc_score(y_bin[te], rf_sr.predict_proba(SR_clean[te])[:, 1])

    fold_aucs_raw.append(auc_raw)
    fold_aucs_sr.append(auc_sr)
    print(f"  Fold {fold_idx+1}: raw={auc_raw:.4f}, SR={auc_sr:.4f}")

raw = np.array(fold_aucs_raw)
sr = np.array(fold_aucs_sr)
diffs = sr - raw
mean_diff = diffs.mean()
sem = diffs.std(ddof=1) / np.sqrt(n_folds)

# TOST at epsilon = 0.01
eps = 0.01
df_tost = n_folds - 1
# H0_lower: mean_diff <= -eps;  reject if t1 > t_crit
t1 = (mean_diff - (-eps)) / sem
p_lower = 1 - stats.t.cdf(t1, df=df_tost)
# H0_upper: mean_diff >=  eps;  reject if t2 < -t_crit
t2 = (mean_diff - eps) / sem
p_upper = stats.t.cdf(t2, df=df_tost)
p_tost = max(p_lower, p_upper)

# 95% CI of the mean difference
t_crit_95 = stats.t.ppf(0.975, df=df_tost)
ci_lo = mean_diff - t_crit_95 * sem
ci_hi = mean_diff + t_crit_95 * sem

print(f"\n=== Cuprite equivalence (ε=0.01, 10-fold) ===")
print(f"mean RF(raw)  = {raw.mean():.4f} ± {raw.std(ddof=1):.4f}")
print(f"mean RF(SR)   = {sr.mean():.4f} ± {sr.std(ddof=1):.4f}")
print(f"Δ = RF(SR) − RF(raw) = {mean_diff:+.5f}")
print(f"95% CI of Δ = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
print(f"TOST p_lower = {p_lower:.3e}, p_upper = {p_upper:.3e}, max(p) = {p_tost:.3e}")
print(f"→ Equivalence within ±{eps}: {'YES' if p_tost < 0.05 else 'NO'}")

out = {
    "site": "Cuprite, Nevada, USA",
    "n_folds": n_folds,
    "epsilon": eps,
    "fold_auc_raw": raw.tolist(),
    "fold_auc_sr": sr.tolist(),
    "mean_auc_raw": float(raw.mean()),
    "mean_auc_sr": float(sr.mean()),
    "mean_diff": float(mean_diff),
    "ci95_lower": float(ci_lo),
    "ci95_upper": float(ci_hi),
    "tost_p_lower": float(p_lower),
    "tost_p_upper": float(p_upper),
    "tost_p": float(p_tost),
    "equivalent_at_eps": bool(p_tost < 0.05),
}
OUT_JSON.write_text(json.dumps(out, indent=2))
print(f"✓ Wrote {OUT_JSON}")

# ---------- Detection-rate curves at Cuprite ----------
def detection_curve(scores, labels):
    for sign in (+1, -1):
        s = sign * scores
        order = np.argsort(-s)
        y_sorted = labels[order]
        cum_tpr = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
        cum_area = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        auc_curve = np.trapz(cum_tpr, cum_area)
        if sign == +1:
            best_area, best_tpr, best_auc = cum_area, cum_tpr, auc_curve
        else:
            if auc_curve > best_auc:
                best_area, best_tpr, best_auc = cum_area, cum_tpr, auc_curve
    return best_area, best_tpr

# Build DataFrame for cleaner indexing
df = pd.DataFrame(X_clean, columns=bands)
df["class_id"] = y_clean

# Refresh band accessors for filtered data
b2 = {name: df.columns.tolist().index(name) for name in bands}
def Bdf(name): return df[name].values

relevant_classical_per_class = {
    1: ["Clay Ratio", "OH Minerals", "Silica"],            # Silicic
    2: ["Clay Ratio", "Alunite Idx", "OH Minerals"],       # Adv Argillic
    3: ["Clay Ratio", "OH Minerals", "Alunite Idx"],       # Argillic_Phyllic
    4: ["Clay Ratio", "Iron Oxide", "Silica"],             # Propylitic
}
colors = {"SR": "#d62728", "Clay Ratio": "#1f77b4", "Iron Oxide": "#ff7f0e",
          "Alunite Idx": "#9467bd", "OH Minerals": "#8c564b", "Silica": "#e377c2"}

# Re-evaluate SR formulas on filtered df
sr_eval_cuprite = {
    "Silicic":          ("log(B11/B12)",        np.log(Bdf("B11") / Bdf("B12"))),
    "Adv_Argillic":     ("B11 - B12",            Bdf("B11") - Bdf("B12")),
    "Argillic_Phyllic": ("tanh(B11) - B8A",      np.tanh(Bdf("B11")) - Bdf("B8A")),
    "Propylitic":       ("((B12-B08)/B07)^2",    ((Bdf("B12") - Bdf("B08")) / Bdf("B07")) ** 2),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
for ax, (cls_id, cls_name) in zip(axes, class_names_cuprite.items()):
    y_cls = (df.class_id == cls_id).astype(int).values
    n_pos = y_cls.sum()

    for cl in relevant_classical_per_class[cls_id]:
        expr, fn = classical[cl]
        scores = fn()[mask]  # use filtered
        valid = np.isfinite(scores)
        a, t = detection_curve(scores[valid], y_cls[valid])
        ax.plot(a, t, color=colors[cl], lw=1.3, alpha=0.85, label=cl)

    expr_sr, scores_sr = sr_eval_cuprite[cls_name]
    valid = np.isfinite(scores_sr)
    a, t = detection_curve(scores_sr[valid], y_cls[valid])
    ax.plot(a, t, color=colors["SR"], lw=2.4, label=f"SR (local): {expr_sr}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="Random")
    pretty = cls_name.replace("_", " ")
    ax.set_title(f"{pretty} (n+={n_pos})", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_xlabel("Fraction of area flagged (top-$k$%)", fontsize=10)

axes[0].set_ylabel("True-positive rate captured", fontsize=10)
fig.suptitle("Detection-rate curves at Cuprite, Nevada (locally-rediscovered SR formulas vs classical indices)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"✓ Wrote {OUT_PNG} ({OUT_PNG.stat().st_size//1024} KB)")
