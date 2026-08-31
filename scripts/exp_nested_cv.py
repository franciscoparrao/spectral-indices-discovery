#!/usr/bin/env python3
"""
Nested CV — fix for NRR R1.1 (leakage at the feature-engineering stage).

The published pipeline discovered SR formulas on a fixed 70% split and then
ran downstream CV on the full dataset, so test pixels had influenced formula
selection. Here the complete pipeline (PySR discovery -> SR features -> RF)
is treated as a single predictive unit: PySR runs exclusively on the outer
training partition of each fold, and the outer test partition is only used
for evaluation.

Schemes:
  stratified — 5-fold StratifiedKFold (mirrors Section 4.7)
  polygon    — 5-fold polygon-disjoint GroupKFold (DBSCAN clusters, as GR3)

Outputs: data/results/nested_cv.json (incremental), per-fold equations CSV,
TOST (eps=0.01) on per-fold deltas, formula stability across folds.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from pysr import PySRRegressor

CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("data/results/nested_cv.json")
EQ_DIR = Path("data/results/nested_cv_equations")

BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
CLASS_NAMES = {1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
               4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn"}
MIN_SAMPLES = 100

# Non-nested reference values from the manuscript (fixed-split formulas)
REFERENCE = {
    "stratified": {"sr": 0.898, "raw": 0.899},
    "polygon": {"sr": 0.815, "raw": 0.822},
}


def make_pysr(tempdir):
    # Identical configuration to run_pysr_atlas_gee.py (source of the
    # published formulas), so the only change is where discovery happens.
    return PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "square", "tanh"],
        maxsize=15,
        populations=30,
        population_size=50,
        parsimony=0.004,
        weight_optimize=0.001,
        constraints={"sqrt": 5, "log": 5, "square": 5, "tanh": 5},
        nested_constraints={
            "sqrt": {"sqrt": 0, "log": 0},
            "log": {"log": 0, "sqrt": 0},
            "square": {"square": 0},
        },
        select_k_features=5,
        progress=False,
        temp_equation_file=True,
        tempdir=str(tempdir),
        turbo=True,
        bumper=True,
    )


def reconstruct_polygons(y, lon, lat):
    # Same DBSCAN reconstruction as exp_tost_expanded.py / GR3
    poly_id = np.full(len(y), -1, dtype=int)
    next_id = 0
    EPS_DEG = 0.001
    for c in sorted(np.unique(y)):
        mask = y == c
        coords = np.column_stack([lon[mask], lat[mask]])
        if len(coords) < 2:
            poly_id[mask] = next_id
            next_id += 1
            continue
        lat_mean = coords[:, 1].mean()
        cs = coords.copy()
        cs[:, 0] *= np.cos(np.deg2rad(lat_mean))
        labels = DBSCAN(eps=EPS_DEG, min_samples=2).fit(cs).labels_
        for lbl in np.unique(labels):
            if lbl == -1:
                for i in np.where(mask)[0][labels == -1]:
                    poly_id[i] = next_id
                    next_id += 1
            else:
                poly_id[np.where(mask)[0][labels == lbl]] = next_id
                next_id += 1
    return poly_id


def per_class_auc(y_true, proba, classes, all_classes):
    aucs = {}
    for c in all_classes:
        yb = (y_true == c).astype(int)
        if 0 < yb.sum() < len(yb) and c in classes:
            aucs[int(c)] = float(
                roc_auc_score(yb, proba[:, list(classes).index(c)]))
    return aucs


def save(results):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


def run_scheme(scheme, X, y, poly_id, results):
    all_classes = sorted(np.unique(y))
    if scheme == "stratified":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(X, y, groups=poly_id)

    scheme_res = {"folds": [], "reference_non_nested": REFERENCE[scheme]}
    results["schemes"][scheme] = scheme_res

    for k, (tr, te) in enumerate(splits):
        print(f"\n{'='*70}\n[{scheme}] FOLD {k+1}/5 — "
              f"train {len(tr)}, test {len(te)}\n{'='*70}", flush=True)
        fold = {"fold": k + 1, "n_train": int(len(tr)), "n_test": int(len(te)),
                "classes": {}}
        t0_fold = time.time()

        # --- PySR discovery on outer-train ONLY ---
        sr_tr = np.zeros((len(tr), len(all_classes)))
        sr_te = np.zeros((len(te), len(all_classes)))
        for j, c in enumerate(all_classes):
            name = CLASS_NAMES[c]
            n_pos = int((y[tr] == c).sum())
            if n_pos < MIN_SAMPLES:
                print(f"  {name}: only {n_pos} positives in train — "
                      f"feature left at zero", flush=True)
                fold["classes"][name] = {"skipped": True, "n_pos_train": n_pos}
                continue
            y_bin = (y[tr] == c).astype(np.float64)
            tempdir = EQ_DIR / f"tmp_{scheme}_f{k+1}_{name}"
            tempdir.mkdir(parents=True, exist_ok=True)
            model = make_pysr(tempdir)
            t0 = time.time()
            model.fit(X[tr], y_bin, variable_names=BANDS)
            elapsed = time.time() - t0
            best = model.get_best()
            sr_tr[:, j] = model.predict(X[tr])
            sr_te[:, j] = model.predict(X[te])
            eq = str(best["equation"])
            print(f"  {name}: {eq}  ({elapsed:.0f}s)", flush=True)
            fold["classes"][name] = {
                "equation": eq,
                "complexity": int(best["complexity"]),
                "loss": float(best["loss"]),
                "n_pos_train": n_pos,
                "time_s": round(elapsed, 1),
            }
            if model.equations_ is not None:
                EQ_DIR.mkdir(parents=True, exist_ok=True)
                model.equations_.to_csv(
                    EQ_DIR / f"{scheme}_fold{k+1}_{name}.csv", index=False)

        # --- Downstream RF, trained on outer-train, evaluated on outer-test ---
        rf_kw = dict(n_estimators=200, max_depth=15, random_state=42,
                     n_jobs=-1, class_weight="balanced")
        rf_raw = RandomForestClassifier(**rf_kw).fit(X[tr], y[tr])
        rf_sr = RandomForestClassifier(**rf_kw).fit(sr_tr, y[tr])
        auc_raw = per_class_auc(y[te], rf_raw.predict_proba(X[te]),
                                rf_raw.classes_, all_classes)
        auc_sr = per_class_auc(y[te], rf_sr.predict_proba(sr_te),
                               rf_sr.classes_, all_classes)
        common = sorted(set(auc_raw) & set(auc_sr))
        fold["auc_raw"] = auc_raw
        fold["auc_sr"] = auc_sr
        fold["mean_auc_raw"] = float(np.mean([auc_raw[c] for c in common]))
        fold["mean_auc_sr"] = float(np.mean([auc_sr[c] for c in common]))
        fold["delta_sr_minus_raw"] = fold["mean_auc_sr"] - fold["mean_auc_raw"]
        fold["time_fold_s"] = round(time.time() - t0_fold, 1)
        print(f"  fold {k+1}: raw={fold['mean_auc_raw']:.4f} "
              f"sr={fold['mean_auc_sr']:.4f} "
              f"delta={fold['delta_sr_minus_raw']:+.4f}", flush=True)

        scheme_res["folds"].append(fold)
        save(results)  # incremental — partial results survive a kill

    # --- Aggregate + TOST across folds ---
    deltas = np.array([f["delta_sr_minus_raw"] for f in scheme_res["folds"]])
    n = len(deltas)
    mean_d = float(deltas.mean())
    se_d = float(deltas.std(ddof=1) / np.sqrt(n))
    epsilon = 0.01
    if se_d > 0:
        p_upper = stats.t.cdf((mean_d - epsilon) / se_d, df=n - 1)
        p_lower = 1 - stats.t.cdf((mean_d + epsilon) / se_d, df=n - 1)
        p_tost = float(max(p_upper, p_lower))
    else:
        p_tost = float("nan")
    scheme_res["summary"] = {
        "mean_auc_raw": float(np.mean([f["mean_auc_raw"] for f in scheme_res["folds"]])),
        "mean_auc_sr": float(np.mean([f["mean_auc_sr"] for f in scheme_res["folds"]])),
        "mean_delta": mean_d,
        "se_delta": se_d,
        "per_fold_deltas": deltas.tolist(),
        "tost_epsilon_0.01_p": p_tost,
        "tost_reject_nonequivalence": bool(p_tost < 0.05) if not np.isnan(p_tost) else None,
    }
    save(results)
    print(f"\n[{scheme}] SUMMARY: raw={scheme_res['summary']['mean_auc_raw']:.4f} "
          f"sr={scheme_res['summary']['mean_auc_sr']:.4f} "
          f"mean delta={mean_d:+.4f}  TOST p={p_tost:.4g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["stratified", "polygon", "both"],
                    default="both")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    X = df[BANDS].values
    y = df["class_id"].values
    print(f"Loaded {len(y)} pixels, {len(BANDS)} bands", flush=True)

    poly_id = reconstruct_polygons(y, df["lon"].values, df["lat"].values)
    print(f"Reconstructed {len(np.unique(poly_id))} DBSCAN clusters", flush=True)

    results = {
        "description": ("Nested CV: PySR discovery inside each outer training "
                        "fold; the complete SR+RF pipeline evaluated as a "
                        "single predictive unit (fix for NRR R1.1)"),
        "pysr_config": "identical to run_pysr_atlas_gee.py",
        "schemes": {},
    }
    schemes = ["stratified", "polygon"] if args.scheme == "both" else [args.scheme]
    t0 = time.time()
    for s in schemes:
        run_scheme(s, X, y, poly_id, results)
    print(f"\nTOTAL {time.time()-t0:.0f}s — saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
