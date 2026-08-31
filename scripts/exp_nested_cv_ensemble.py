#!/usr/bin/env python3
"""
Extension of exp_nested_cv.py: evaluate the combined feature sets under the
same nested folds, WITHOUT re-running PySR — the per-fold equations saved in
nested_cv.json are re-evaluated as closed-form expressions.

Answers: does the manuscript's secondary claim (SR + classical indices is the
best feature set, non-nested AUC 0.920) survive leakage-free discovery?

Feature sets per fold:
  RF(raw)            — 10 bands (re-computed as sanity check vs nested_cv.json)
  RF(SR nested)      — per-fold discovered formulas (sanity check)
  RF(7 classical)    — fixed literature formulas (leakage-free by construction)
  RF(SR + classical) — nested SR features + 7 classical
  RF(raw + classical)— fair upper reference for the combination claim

Output: data/results/nested_cv_ensemble.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
NESTED = Path("data/results/nested_cv.json")
OUT = Path("data/results/nested_cv_ensemble.json")

BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
CLASS_NAMES = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
               4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn"}


def eval_equation(eq, X):
    """Evaluate a PySR equation string on the band matrix (protected ops)."""
    env = {b: X[:, i] for i, b in enumerate(BANDS)}
    env.update({
        "sqrt": lambda x: np.sqrt(np.clip(x, 0, None)),
        "log": lambda x: np.log(np.maximum(x, 1e-9)),
        "square": lambda x: np.square(x),
        "tanh": np.tanh,
        "exp": np.exp,
    })
    expr = eq.replace("^", "**")
    if not re.fullmatch(r"[\w\s\.\+\-\*\/\(\)]+", expr):
        raise ValueError(f"Unsafe equation string: {eq}")
    return np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=float)


def classical_features(X):
    b = {n: i for i, n in enumerate(BANDS)}
    eps = 1e-9
    B = lambda n: X[:, b[n]]
    return np.column_stack([
        B('B11') / np.maximum(B('B12'), eps),                       # Clay Ratio
        B('B04') / np.maximum(B('B02'), eps),                       # Iron Oxide
        B('B12') / np.maximum(B('B8A'), eps),                       # Ferrous
        (B('B11') - B('B12')) / np.maximum(B('B11') + B('B12'), eps),  # Alunite
        B('B02') / np.maximum(B('B11'), eps),                       # OH minerals
        B('B12') / np.maximum(B('B11'), eps),                       # Silica
        (B('B8A') - B('B04')) / np.maximum(B('B8A') + B('B04'), eps),  # NDVI
    ])


def reconstruct_polygons(y, lon, lat):
    poly_id = np.full(len(y), -1, dtype=int)
    next_id = 0
    for c in sorted(np.unique(y)):
        mask = y == c
        coords = np.column_stack([lon[mask], lat[mask]])
        if len(coords) < 2:
            poly_id[mask] = next_id
            next_id += 1
            continue
        cs = coords.copy()
        cs[:, 0] *= np.cos(np.deg2rad(coords[:, 1].mean()))
        labels = DBSCAN(eps=0.001, min_samples=2).fit(cs).labels_
        for lbl in np.unique(labels):
            if lbl == -1:
                for i in np.where(mask)[0][labels == -1]:
                    poly_id[i] = next_id
                    next_id += 1
            else:
                poly_id[np.where(mask)[0][labels == lbl]] = next_id
                next_id += 1
    return poly_id


def mean_auc(rf, Xte, yte, all_classes):
    proba = rf.predict_proba(Xte)
    aucs = {}
    for c in all_classes:
        yb = (yte == c).astype(int)
        if 0 < yb.sum() < len(yb) and c in rf.classes_:
            aucs[int(c)] = float(
                roc_auc_score(yb, proba[:, list(rf.classes_).index(c)]))
    return aucs


def main():
    df = pd.read_csv(CSV)
    X = df[BANDS].values
    y = df["class_id"].values
    poly_id = reconstruct_polygons(y, df["lon"].values, df["lat"].values)
    all_classes = sorted(np.unique(y))
    nested = json.load(open(NESTED))
    cls_feats = classical_features(X)

    rf_kw = dict(n_estimators=200, max_depth=15, random_state=42,
                 n_jobs=-1, class_weight="balanced")
    SETS = ["raw", "sr", "classical", "sr_classical", "raw_classical"]
    results = {"description": ("Nested-fold ensemble comparison using per-fold "
                               "SR equations from nested_cv.json (no PySR re-run)"),
               "schemes": {}}

    for scheme in ["stratified", "polygon"]:
        if scheme == "stratified":
            splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
        else:
            splits = list(GroupKFold(5).split(X, y, groups=poly_id))
        folds_meta = nested["schemes"][scheme]["folds"]
        assert len(splits) == len(folds_meta)

        rows = []
        for (tr, te), meta in zip(splits, folds_meta):
            # Rebuild the fold's SR features from saved equations
            sr = np.zeros((len(y), len(all_classes)))
            for j, c in enumerate(all_classes):
                info = meta["classes"].get(CLASS_NAMES[c], {})
                if "equation" in info:
                    sr[:, j] = eval_equation(info["equation"], X)
            feats = {
                "raw": X,
                "sr": sr,
                "classical": cls_feats,
                "sr_classical": np.hstack([sr, cls_feats]),
                "raw_classical": np.hstack([X, cls_feats]),
            }
            row = {"fold": meta["fold"]}
            for name, F in feats.items():
                rf = RandomForestClassifier(**rf_kw).fit(F[tr], y[tr])
                aucs = mean_auc(rf, F[te], y[te], all_classes)
                row[name] = float(np.mean(list(aucs.values())))
            # Only classes evaluable in every set are comparable; sets share
            # the same test partition so the class subsets coincide.
            rows.append(row)
            print(f"[{scheme}] fold {row['fold']}: " +
                  "  ".join(f"{s}={row[s]:.4f}" for s in SETS), flush=True)

        summary = {}
        for s in SETS:
            vals = np.array([r[s] for r in rows])
            summary[s] = {"mean": float(vals.mean()),
                          "per_fold": vals.round(4).tolist()}
        d_comb = np.array([r["sr_classical"] - r["raw"] for r in rows])
        n = len(d_comb)
        se = float(d_comb.std(ddof=1) / np.sqrt(n))
        t = float(d_comb.mean() / se) if se > 0 else float("nan")
        p_gt = float(1 - stats.t.cdf(t, df=n - 1))
        summary["sr_classical_minus_raw"] = {
            "mean": float(d_comb.mean()), "se": se,
            "p_one_sided_greater": p_gt,
        }
        results["schemes"][scheme] = {"folds": rows, "summary": summary}
        print(f"[{scheme}] SUMMARY " +
              "  ".join(f"{s}={summary[s]['mean']:.4f}" for s in SETS) +
              f"  Δ(SR+cl − raw)={d_comb.mean():+.4f} (p={p_gt:.3f})", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
