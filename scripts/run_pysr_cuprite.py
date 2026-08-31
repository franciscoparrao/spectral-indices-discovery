#!/usr/bin/env python3
"""
GR1 — PySR over Cuprite, Nevada data to recover local formulas.
Same configuration as run_pysr_atlas_gee.py but using USGS-derived
Cuprite ground truth. Enables direct comparison of locally discovered
formulas vs. the Chilean ones, to substantiate the method-transferability
claim.
"""

import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time
import json

GT_DIR = Path("data/ground_truth")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic", 4: "Propylitic",
}
MIN_SAMPLES = 100


def main():
    print("Loading Cuprite S2 + USGS training data...")
    data = np.load(GT_DIR / "cuprite_training_s2.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    band_names = [str(b) for b in data["band_names"]]
    print(f"Total samples: {X.shape[0]}, bands: {band_names}")
    print(f"Class distribution (including Unaltered=0):")
    for c in sorted(np.unique(y)):
        print(f"  {int(c)} {CLASS_NAMES.get(int(c), 'Unaltered'):20s}: {(y == c).sum()}")

    for cls_id in np.unique(y):
        n = (y == cls_id).sum()
        if n < MIN_SAMPLES:
            print(f"Removing class {cls_id}: {n} < {MIN_SAMPLES}")
            mask = y != cls_id
            X, y = X[mask], y[mask]

    MAX = 15000
    if len(y) > MAX:
        X, _, y, _ = train_test_split(X, y, train_size=MAX, stratify=y, random_state=42)
        print(f"Subsampled to {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # RF baseline
    print("\n" + "=" * 60)
    print("RF BASELINE — Cuprite, 10 bands, 70/30 split")
    print("=" * 60)
    rf = RandomForestClassifier(200, max_depth=15, random_state=42, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    ba = balanced_accuracy_score(y_test, rf.predict(X_test))
    print(f"  Balanced Accuracy: {ba:.4f}")

    y_proba = rf.predict_proba(X_test)
    print("  Per-class AUC (one-vs-rest):")
    for cls_id in sorted(np.unique(y_test)):
        y_bin = (y_test == cls_id).astype(int)
        idx = list(rf.classes_).index(cls_id)
        auc = roc_auc_score(y_bin, y_proba[:, idx])
        print(f"    {int(cls_id)} {CLASS_NAMES.get(int(cls_id), 'Unaltered'):20s}: {auc:.4f}")

    all_results = {
        "site": "Cuprite, Nevada, USA",
        "ground_truth": "USGS hydrothermal alteration map (Rockwell 2017)",
        "s2_source": "GEE COPERNICUS/S2_SR_HARMONIZED median Jun-Sep 2024",
        "rf_baseline_balanced_accuracy": float(ba),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    for cls_id, cls_name in CLASS_NAMES.items():
        n_pos = int((y_train == cls_id).sum())
        if n_pos < MIN_SAMPLES:
            continue
        print(f"\n{'='*60}\nPySR CLASS {cls_id}: {cls_name} ({n_pos} positives in training)\n{'='*60}")

        y_bin_train = (y_train == cls_id).astype(np.float64)
        y_bin_test = (y_test == cls_id).astype(np.float64)

        model = PySRRegressor(
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
            tempdir=str(RESULTS_DIR / "pysr_temp_cuprite"),
            turbo=True,
            bumper=True,
        )

        t0 = time.time()
        model.fit(X_train, y_bin_train, variable_names=band_names)
        elapsed = time.time() - t0

        best = model.get_best()
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_bin_test, y_pred) if len(np.unique(y_bin_test)) > 1 else 0
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(y_bin_test, (y_pred > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        print(f"Best equation: {best['equation']}")
        print(f"  AUC (30% test, unseen): {auc:.4f}")
        print(f"  F1: {best_f1:.4f} (thresh={best_thresh:.2f})")
        print(f"  Complexity: {best['complexity']}, Loss: {best['loss']:.4f}")
        print(f"  Time: {elapsed:.0f}s")

        equations_df = model.equations_
        if equations_df is not None:
            equations_df.to_csv(RESULTS_DIR / f"equations_cuprite_{cls_name}.csv", index=False)

        entry = {
            "class_id": int(cls_id), "n_train": n_pos,
            "best_equation": str(best["equation"]),
            "complexity": int(best["complexity"]),
            "loss": float(best["loss"]),
            "auc": float(auc), "f1": float(best_f1),
            "threshold": float(best_thresh),
            "time_s": float(elapsed),
        }
        if equations_df is not None:
            entry["pareto_front"] = [
                {"equation": str(r.get("equation", "")),
                 "complexity": int(r.get("complexity", 0)),
                 "loss": float(r.get("loss", 0))}
                for _, r in equations_df.iterrows()
            ]
        all_results[cls_name] = entry

    with open(RESULTS_DIR / "pysr_results_cuprite.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY — Cuprite PySR formulas")
    print("=" * 70)
    for name, r in all_results.items():
        if isinstance(r, dict) and "best_equation" in r:
            print(f"\n{name}:")
            print(f"  {r['best_equation']}")
            print(f"  AUC = {r['auc']:.4f}, complexity = {r['complexity']}")


if __name__ == "__main__":
    main()
