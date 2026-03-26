#!/usr/bin/env python3
"""
Run PySR with all 10 Sentinel-2 bands.
Focuses on classes that benefit from VNIR: Iron Oxide, Vegetation,
and re-runs alteration classes to see if VNIR improves formulas.
"""

import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
import time
import json

GT_DIR = Path("data/ground_truth")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    2: "Adv_Argillic",
    3: "Argillic",
    4: "Propylitic",
    5: "Iron_Oxide",
    6: "Unaltered",
    7: "Vegetation",
}

MIN_SAMPLES = 50


def main():
    print("Loading 10-band training data...")
    data = np.load(GT_DIR / "el_tatio_training_all.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    band_names = list(data["band_names"])

    print(f"Total samples: {X.shape[0]}")
    print(f"Bands: {band_names}")

    # Remove any class with < MIN_SAMPLES before stratified split
    for cls_id in np.unique(y):
        n = (y == cls_id).sum()
        if n < MIN_SAMPLES:
            name = CLASSES.get(cls_id, f"Class_{cls_id}")
            print(f"Removing {name} (id={cls_id}): only {n} samples")
            mask = y != cls_id
            X, y = X[mask], y[mask]

    # Subsample
    MAX_SAMPLES = 15000
    if len(y) > MAX_SAMPLES:
        print(f"Subsampling to {MAX_SAMPLES}...")
        X, _, y, _ = train_test_split(X, y, train_size=MAX_SAMPLES,
                                       stratify=y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    print("\nClass distribution (train):")
    for cls_id, cls_name in CLASSES.items():
        n = (y_train == cls_id).sum()
        print(f"  {cls_id} {cls_name:20s}: {n:6d}")

    all_results = {}

    for cls_id, cls_name in CLASSES.items():
        n_pos = (y_train == cls_id).sum()
        if n_pos < MIN_SAMPLES:
            print(f"\nSKIP {cls_id}: {cls_name} — {n_pos} samples")
            continue

        print(f"\n{'='*60}")
        print(f"CLASS {cls_id}: {cls_name} ({n_pos} positive)")
        print(f"{'='*60}")

        y_bin_train = (y_train == cls_id).astype(np.float64)
        y_bin_test = (y_test == cls_id).astype(np.float64)

        model = PySRRegressor(
            niterations=80,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "square", "tanh"],
            maxsize=15,
            populations=20,
            population_size=40,
            parsimony=0.005,
            weight_optimize=0.001,
            constraints={
                "sqrt": 5, "log": 5, "square": 5, "tanh": 5,
            },
            nested_constraints={
                "sqrt": {"sqrt": 0, "log": 0},
                "log": {"log": 0, "sqrt": 0},
                "square": {"square": 0},
            },
            select_k_features=4,
            progress=True,
            temp_equation_file=True,
            tempdir=str(RESULTS_DIR / "pysr_temp_10b"),
            turbo=True,
            bumper=True,
        )

        t0 = time.time()
        model.fit(X_train, y_bin_train, variable_names=band_names)
        elapsed = time.time() - t0

        best = model.get_best()
        print(f"\nBest: {best['equation']}")
        print(f"Complexity: {best['complexity']}, Loss: {best['loss']:.6f}")

        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_bin_test, y_pred) if len(np.unique(y_bin_test)) > 1 else 0
        # Optimize threshold by testing several
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_bin_test, (y_pred > thresh).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        print(f"AUC: {auc:.4f}, Best F1: {best_f1:.4f} (thresh={best_thresh:.2f})")

        equations_df = model.equations_
        if equations_df is not None:
            equations_df.to_csv(RESULTS_DIR / f"equations_10b_{cls_name}.csv", index=False)

        all_results[cls_name] = {
            "class_id": cls_id,
            "n_train": int(n_pos),
            "best_equation": str(best["equation"]),
            "complexity": int(best["complexity"]),
            "loss": float(best["loss"]),
            "auc": float(auc),
            "f1": float(best_f1),
            "threshold": float(best_thresh),
            "time_s": float(elapsed),
            "top_equations": [],
        }
        if equations_df is not None:
            for _, row in equations_df.tail(5).iterrows():
                all_results[cls_name]["top_equations"].append({
                    "equation": str(row.get("equation", "")),
                    "complexity": int(row.get("complexity", 0)),
                    "loss": float(row.get("loss", 0)),
                })

    with open(RESULTS_DIR / "pysr_results_10bands.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY — 10-Band Spectral Indices")
    print("=" * 70)
    for name, r in all_results.items():
        print(f"\n{name}:")
        print(f"  {r['best_equation']}")
        print(f"  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  complexity={r['complexity']}")


if __name__ == "__main__":
    main()
