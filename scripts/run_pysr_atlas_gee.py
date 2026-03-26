#!/usr/bin/env python3
"""
PySR with Atlas ground truth + S2 bands extracted via GEE.
Same pixels as embedding comparison — fair benchmarking.
"""

import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time
import json

GT_DIR = Path("data/ground_truth")
RESULTS_DIR = Path("data/results")

CLASS_NAMES = {
    1: "Silicic", 2: "Adv_Argillic", 3: "Argillic_Phyllic",
    4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic_Skarn",
}
MIN_SAMPLES = 100


def main():
    print("Loading S2 GEE training data...")
    data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    band_names = [str(b) for b in data["band_names"]]

    print(f"Samples: {X.shape[0]}, Bands: {band_names}")

    # Remove tiny classes
    for cls_id in np.unique(y):
        n = (y == cls_id).sum()
        if n < MIN_SAMPLES:
            print(f"Removing class {cls_id}: {n} samples")
            mask = y != cls_id
            X, y = X[mask], y[mask]

    # Subsample
    MAX = 15000
    if len(y) > MAX:
        X, _, y, _ = train_test_split(X, y, train_size=MAX, stratify=y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    for cls_id, name in CLASS_NAMES.items():
        n = (y_train == cls_id).sum()
        if n > 0:
            print(f"  {cls_id} {name:25s}: {n}")

    # === RF baseline on S2 bands ===
    print("\n" + "=" * 60)
    print("RF BASELINE ON S2 10 BANDS")
    print("=" * 60)
    rf = RandomForestClassifier(200, max_depth=15, random_state=42, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    from sklearn.metrics import balanced_accuracy_score
    y_pred_rf = rf.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred_rf)
    print(f"  Balanced Accuracy: {ba:.4f}")

    y_proba = rf.predict_proba(X_test)
    print("  Per-class AUC:")
    for cls_id in sorted(np.unique(y_test)):
        y_bin = (y_test == cls_id).astype(int)
        idx = list(rf.classes_).index(cls_id)
        auc = roc_auc_score(y_bin, y_proba[:, idx])
        print(f"    {cls_id} {CLASS_NAMES.get(cls_id,'?'):25s}: {auc:.4f}")

    # === PySR per class ===
    all_results = {"rf_baseline_balanced_accuracy": float(ba)}

    for cls_id, cls_name in CLASS_NAMES.items():
        n_pos = (y_train == cls_id).sum()
        if n_pos < MIN_SAMPLES:
            continue

        print(f"\n{'='*60}")
        print(f"PySR CLASS {cls_id}: {cls_name} ({n_pos} positive)")
        print(f"{'='*60}")

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
            progress=True,
            temp_equation_file=True,
            tempdir=str(RESULTS_DIR / "pysr_temp_gee"),
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

        print(f"\nBest: {best['equation']}")
        print(f"AUC: {auc:.4f}, F1: {best_f1:.4f} (thresh={best_thresh:.2f}), Time: {elapsed:.0f}s")

        equations_df = model.equations_
        if equations_df is not None:
            equations_df.to_csv(RESULTS_DIR / f"equations_gee_{cls_name}.csv", index=False)

        all_results[cls_name] = {
            "class_id": cls_id, "n_train": int(n_pos),
            "best_equation": str(best["equation"]),
            "complexity": int(best["complexity"]),
            "loss": float(best["loss"]),
            "auc": float(auc), "f1": float(best_f1),
            "threshold": float(best_thresh),
            "time_s": float(elapsed),
            "ground_truth": "Atlas Metalífero III Región (field-mapped)",
            "s2_source": "GEE COPERNICUS/S2_SR_HARMONIZED median Jan-Mar 2024",
        }

        if equations_df is not None:
            pareto = []
            for _, row in equations_df.iterrows():
                pareto.append({
                    "equation": str(row.get("equation", "")),
                    "complexity": int(row.get("complexity", 0)),
                    "loss": float(row.get("loss", 0)),
                })
            all_results[cls_name]["pareto_front"] = pareto

    with open(RESULTS_DIR / "pysr_results_gee.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY — S2 bands via GEE + Atlas GT (field-mapped)")
    print("=" * 70)
    print(f"\nRF baseline (10 bands): {all_results['rf_baseline_balanced_accuracy']:.4f}")
    for name, r in all_results.items():
        if isinstance(r, dict) and "best_equation" in r:
            print(f"\n{name}:")
            print(f"  {r['best_equation']}")
            print(f"  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  complexity={r['complexity']}")


if __name__ == "__main__":
    main()
