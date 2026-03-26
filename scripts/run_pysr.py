#!/usr/bin/env python3
"""
Run PySR symbolic regression to discover spectral indices
for hydrothermal alteration classes at El Tatio.

One-vs-rest: trains a separate SR model per alteration class.
Uses 20m bands (B05, B06, B07, B8A, B11, B12) as primary features.
"""

import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from pathlib import Path
import time
import json

GT_DIR = Path("data/ground_truth")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    1: "Silicic",
    2: "Adv_Argillic",
    3: "Argillic",
    4: "Propylitic",
    5: "Iron_Oxide",
    6: "Unaltered",
    7: "Vegetation",
}

# Skip classes with too few samples
MIN_SAMPLES = 50


def main():
    # Load training data
    print("Loading training data...")
    data = np.load(GT_DIR / "el_tatio_training_20m.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    band_names = list(data["band_names"])

    print(f"Total samples: {X.shape[0]}")
    print(f"Bands: {band_names}")
    print(f"Classes: {np.unique(y)}")

    # Subsample for speed: SR doesn't need 275K samples
    # Use stratified sampling to keep class proportions
    MAX_SAMPLES = 20000
    if len(y) > MAX_SAMPLES:
        print(f"\nSubsampling to {MAX_SAMPLES} for SR efficiency...")
        X, _, y, _ = train_test_split(X, y, train_size=MAX_SAMPLES,
                                       stratify=y, random_state=42)
    print(f"Working set: {X.shape[0]} samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Print class distribution
    print("\nClass distribution (train):")
    for cls_id, cls_name in CLASSES.items():
        n = (y_train == cls_id).sum()
        print(f"  {cls_id} {cls_name:20s}: {n:6d}")

    # === PySR Configuration ===
    # Optimized for spectral index discovery:
    # - Binary ops: +, -, *, / (standard index building blocks)
    # - Unary ops: sqrt, log, square, tanh (common in spectral indices)
    # - Max formula size: 15 nodes (keep interpretable)
    # - Parsimony: penalize complexity to favor simple indices
    # - Select up to 4 features: realistic for a spectral index

    all_results = {}

    for cls_id, cls_name in CLASSES.items():
        n_pos = (y_train == cls_id).sum()
        if n_pos < MIN_SAMPLES:
            print(f"\n{'='*60}")
            print(f"SKIP {cls_id}: {cls_name} — only {n_pos} samples (< {MIN_SAMPLES})")
            continue

        print(f"\n{'='*60}")
        print(f"CLASS {cls_id}: {cls_name} ({n_pos} positive samples)")
        print(f"{'='*60}")

        # Binary target
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
                "sqrt": 5,
                "log": 5,
                "square": 5,
                "tanh": 5,
            },
            nested_constraints={
                "sqrt": {"sqrt": 0, "log": 0},
                "log": {"log": 0, "sqrt": 0},
                "square": {"square": 0},
            },
            select_k_features=4,
            progress=True,
            temp_equation_file=True,
            tempdir=str(RESULTS_DIR / "pysr_temp"),
            turbo=True,
            bumper=True,
        )

        t0 = time.time()
        model.fit(X_train, y_bin_train, variable_names=band_names)
        elapsed = time.time() - t0

        # Results
        print(f"\nTraining time: {elapsed:.1f}s")
        print(f"\nTop equations for {cls_name}:")
        print(model)

        # Best equation
        best = model.get_best()
        print(f"\nBest equation: {best['equation']}")
        print(f"Complexity: {best['complexity']}")
        print(f"Loss: {best['loss']:.6f}")

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        auc = roc_auc_score(y_bin_test, y_pred) if len(np.unique(y_bin_test)) > 1 else 0
        f1 = f1_score(y_bin_test, y_pred_binary)

        print(f"\nTest AUC: {auc:.4f}")
        print(f"Test F1:  {f1:.4f}")

        # Save model equations
        equations_df = model.equations_
        if equations_df is not None:
            eq_path = RESULTS_DIR / f"equations_{cls_name}.csv"
            equations_df.to_csv(eq_path, index=False)
            print(f"Saved: {eq_path}")

        # Store results
        all_results[cls_name] = {
            "class_id": cls_id,
            "n_train_positive": int(n_pos),
            "best_equation": str(best["equation"]),
            "best_complexity": int(best["complexity"]),
            "best_loss": float(best["loss"]),
            "test_auc": float(auc),
            "test_f1": float(f1),
            "training_time_s": float(elapsed),
            "top_5_equations": [],
        }

        # Top 5 equations from Pareto front
        if equations_df is not None:
            for _, row in equations_df.tail(5).iterrows():
                all_results[cls_name]["top_5_equations"].append({
                    "equation": str(row.get("equation", "")),
                    "complexity": int(row.get("complexity", 0)),
                    "loss": float(row.get("loss", 0)),
                })

    # Save summary
    summary_path = RESULTS_DIR / "pysr_results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved summary: {summary_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY — Discovered Spectral Indices")
    print("=" * 70)
    for cls_name, res in all_results.items():
        print(f"\n{cls_name}:")
        print(f"  Formula:    {res['best_equation']}")
        print(f"  Complexity: {res['best_complexity']}")
        print(f"  AUC:        {res['test_auc']:.4f}")
        print(f"  F1:         {res['test_f1']:.4f}")


if __name__ == "__main__":
    main()
