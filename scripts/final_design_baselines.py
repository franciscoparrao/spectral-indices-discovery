#!/usr/bin/env python3
"""
Baselines and ROC curves on the final hard-negative design (positives geo, R3 split by
10 km blocks, Cuprite as second cross-site test). Everything is trained on the R3
selection half only; R4 and R5 are never touched.

Closed-form indices : clay B11/B12, winner m4, winner m5, m2, ferric, paper SR
Models (on sel)     : logistic on 9 log-ratios (vs B08), RF on 10 raw bands, RF on log-ratios
Per evaluation set  : AUC, ROC curve, TPR at FPR 5/10/20 %, FPR at TPR 80 %
Uncertainty         : block bootstrap (10 km cells, positives and negatives resampled
                      jointly) of AUC and of the delta vs the clay ratio, 1000 draws.
Output: data/results/final_design_baselines.json, figures/roc_final_design.png
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, str(Path(__file__).parent))
from hardneg_design import load_design, logb, block_id, BANDS, INDICES

RNG = np.random.default_rng(0)
OUT = Path("data/results/final_design_baselines.json")
D = load_design(verbose=True)
sets, sel_w = D["sets"], D["sel_w"]
EVAL = [k for k in sets if k != "sel"]


def logratios(df):
    L = logb(df); return L - L[:, [BANDS.index("B08")]]


# ------------------------------------------------------------------ models on the selection half
p, n = sets["sel"]
Xs = {"logistic 9 log-ratios": (logratios, LogisticRegression(C=1.0, max_iter=2000)),
      "RF 10 raw bands": (lambda d: d[BANDS].values, RandomForestClassifier(300, min_samples_leaf=5, n_jobs=6, random_state=0)),
      "RF 9 log-ratios": (logratios, RandomForestClassifier(300, min_samples_leaf=5, n_jobs=6, random_state=0))}
y = np.r_[np.ones(len(p)), np.zeros(len(n))]
w = np.r_[np.full(len(p), 1 / len(p)), sel_w]; w = w / w.sum() * len(w)
models = {}
for name, (feat, clf) in Xs.items():
    t = time.time(); clf.fit(np.vstack([feat(p), feat(n)]), y, sample_weight=w)
    models[name] = (feat, clf); print(f"fitted {name} ({time.time()-t:.0f}s)", flush=True)


def scores(name, df):
    if name in INDICES:
        return INDICES[name](df).values.astype(float)
    feat, clf = models[name]; return clf.predict_proba(feat(df))[:, 1]


# orientation of closed-form indices fixed on the selection set (as enum_logspace does)
orient = {}
for name in INDICES:
    s = np.r_[scores(name, p), scores(name, n)]
    orient[name] = 1.0 if roc_auc_score(y, s, sample_weight=w) >= 0.5 else -1.0
for name in models: orient[name] = 1.0


def oriented(name, df): return orient[name] * scores(name, df)


# ------------------------------------------------------------------ evaluation
def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    grid = np.linspace(0, 1, 201)
    return {"auc": float(roc_auc_score(y, s)),
            "tpr_at_fpr": {f: float(np.interp(f, fpr, tpr)) for f in (0.05, 0.10, 0.20)},
            "fpr_at_tpr80": float(np.interp(0.80, tpr, fpr)),
            "roc": {"fpr": grid.round(4).tolist(), "tpr": np.interp(grid, fpr, tpr).round(4).tolist()}}


BOOT_KM = {"R3": D["block_km"], "R4": D["block_km"], "R5": 2.0}   # Cuprite district spans ~16 km: 10 km cells would leave ~4 blocks


def blocks(df, site):
    return block_id(df.lon, df.lat, BOOT_KM[site]) if "lon" in df else np.arange(len(df))


results = {}
names = list(INDICES) + list(models)
for k in EVAL:
    P, N = sets[k]; yk = np.r_[np.ones(len(P)), np.zeros(len(N))]
    S = {name: np.r_[oriented(name, P), oriented(name, N)] for name in names}
    results[k] = {name: metrics(yk, s) for name, s in S.items()}
    # block bootstrap of AUC and delta vs clay
    b = np.r_[blocks(P, k[:2]), blocks(N, k[:2])]; ub, inv = np.unique(b, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ub))]
    boot = {name: [] for name in names}
    for _ in range(1000):
        draw = RNG.integers(0, len(ub), len(ub))
        idx = np.concatenate([members[i] for i in draw])
        yb = yk[idx]
        if yb.min() == yb.max(): continue
        for name in names: boot[name].append(roc_auc_score(yb, S[name][idx]))
    for name in names:
        a = np.asarray(boot[name]); d = a - np.asarray(boot["clay B11/B12"])
        results[k][name]["auc_ci95"] = [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
        results[k][name]["delta_vs_clay"] = float(results[k][name]["auc"] - results[k]["clay B11/B12"]["auc"])
        results[k][name]["delta_ci95"] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    print(f"{k:12s} blocks {len(ub):4d} | " + " ".join(f"{name.split()[0][:8]}={results[k][name]['auc']:.3f}" for name in names), flush=True)

summary = {"design": {"positives": "geo", "split": D["split"], "block_km": D["block_km"], "cuprite": True,
                      "exclude": sorted(D["excl"]), "bootstrap": "1000 draws, pos+neg jointly, block km per site: " + json.dumps(BOOT_KM)},
           "sets": {k: {"pos": len(sets[k][0]), "neg": len(sets[k][1])} for k in sets},
           "orientation": orient, "results": results}
OUT.write_text(json.dumps(summary, indent=1)); print("saved", OUT)

# ------------------------------------------------------------------ table + exploratory figure
print("\nAUC [95% block-bootstrap CI]; delta vs clay [CI]")
for k in EVAL:
    print(f"\n{k}")
    for name in names:
        r = results[k][name]
        print(f"  {name:26s} {r['auc']:.3f} [{r['auc_ci95'][0]:.3f},{r['auc_ci95'][1]:.3f}]  "
              f"Δclay {r['delta_vs_clay']:+.3f} [{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]  "
              f"TPR@FPR10 {r['tpr_at_fpr'][0.1]:.2f}  FPR@TPR80 {r['fpr_at_tpr80']:.2f}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = ["clay B11/B12", "winner m4", "logistic 9 log-ratios", "RF 10 raw bands"]
panels = ["R3_ho_hard", "R4_hard", "R5_hard"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, k in zip(axes, panels):
    for name in show:
        r = results[k][name]; ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{name} ({r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_title(k); ax.set_xlabel("FPR"); ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("TPR"); fig.suptitle("Final design (exploratory figure)"); fig.tight_layout()
Path("figures").mkdir(exist_ok=True); fig.savefig("figures/roc_final_design.png", dpi=130); print("saved figures/roc_final_design.png")
