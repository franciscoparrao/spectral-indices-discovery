#!/usr/bin/env python3
"""
NRR R1.8 — the missing Cuprite comparison: RF using the four LOCALLY
discovered Cuprite SR formulas (Table 10) versus RF using the ten raw
Sentinel-2 bands.

Leakage-free design: the local formulas were discovered on the 70% partition
of the 15,000-pixel Cuprite subsample (run_pysr_cuprite.py; both splits
random_state=42). We reproduce the identical subsample and split, train all
RFs on that same 70% partition, and evaluate exclusively on the locked 30%
hold-out — which never influenced formula discovery.

Feature sets: raw(10) · localSR(4) · classical(7) · localSR+classical(11).
Output: data/results/cuprite_local_sr_rf.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = Path("data/ground_truth/cuprite_training_s2.npz")
OUT = Path("data/results/cuprite_local_sr_rf.json")

d = np.load(DATA)
X, y, bands = d["X"], d["y"], [str(b) for b in d["band_names"]]
b = {n: i for i, n in enumerate(bands)}


def B(n, M):
    return M[:, b[n]]


def local_sr(M):
    # Four formulas discovered locally at Cuprite (Table 10), protected ops
    eps = 1e-9
    return np.column_stack([
        np.log(np.maximum(B("B11", M) / np.maximum(B("B12", M), eps), eps)),  # Silicic
        B("B11", M) - B("B12", M),                                            # Adv. Argillic
        np.tanh(B("B11", M)) - B("B8A", M),                                   # Argillic-Phyllic
        np.square((B("B12", M) - B("B08", M)) / np.maximum(B("B07", M), eps)),# Propylitic
    ])


def classical(M):
    eps = 1e-9
    return np.column_stack([
        B("B11", M) / np.maximum(B("B12", M), eps),
        B("B04", M) / np.maximum(B("B02", M), eps),
        B("B12", M) / np.maximum(B("B8A", M), eps),
        (B("B11", M) - B("B12", M)) / np.maximum(B("B11", M) + B("B12", M), eps),
        B("B02", M) / np.maximum(B("B11", M), eps),
        B("B12", M) / np.maximum(B("B11", M), eps),
        (B("B8A", M) - B("B04", M)) / np.maximum(B("B8A", M) + B("B04", M), eps),
    ])


# Reproduce the exact discovery subsample and split of run_pysr_cuprite.py
if len(y) > 15000:
    X, _, y, _ = train_test_split(X, y, train_size=15000, stratify=y, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(f"Discovery partition: {len(y_tr)}  ·  Locked hold-out: {len(y_te)}")

FEATS = {
    "raw10": (X_tr, X_te),
    "localSR4": (local_sr(X_tr), local_sr(X_te)),
    "classical7": (classical(X_tr), classical(X_te)),
    "localSR4_classical7": (np.hstack([local_sr(X_tr), classical(X_tr)]),
                            np.hstack([local_sr(X_te), classical(X_te)])),
}
CLS = [c for c in [1, 2, 3, 4] if (y_te == c).sum() > 0]

results = {"design": ("RF trained on the 70% Cuprite discovery partition, "
                      "evaluated on the locked 30% hold-out (leakage-free: "
                      "local formulas were discovered on the same 70%)"),
           "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
           "sets": {}}
for name, (Ftr, Fte) in FEATS.items():
    rf = RandomForestClassifier(200, max_depth=15, random_state=42, n_jobs=-1,
                                class_weight="balanced").fit(Ftr, y_tr)
    proba = rf.predict_proba(Fte)
    aucs = {}
    for c in CLS:
        yb = (y_te == c).astype(int)
        aucs[int(c)] = float(roc_auc_score(yb, proba[:, list(rf.classes_).index(c)]))
    mean = float(np.mean(list(aucs.values())))
    results["sets"][name] = {"mean_auc": mean, "per_class_auc": aucs}
    print(f"{name:22s} mAUC={mean:.4f}  " +
          " ".join(f"c{c}={a:.3f}" for c, a in aucs.items()))

results["delta_localSR_minus_raw"] = (results["sets"]["localSR4"]["mean_auc"]
                                      - results["sets"]["raw10"]["mean_auc"])
print(f"\nΔ(localSR4 − raw10) = {results['delta_localSR_minus_raw']:+.4f}")
OUT.write_text(json.dumps(results, indent=2))
print(f"Saved → {OUT}")
