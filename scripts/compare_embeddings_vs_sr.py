#!/usr/bin/env python3
"""
Compare Google Satellite Embeddings (64-dim) vs SR-discovered indices (6 bands)
for hydrothermal alteration classification.

Benchmark:
1. Random Forest on embeddings (upper bound - rich representation)
2. Random Forest on S2 bands (traditional ML baseline)
3. PySR-discovered formulas (our contribution - interpretable)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json

EMB_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")

ALTERATION_MAP = {
    "Alteración Silicea": 1, "vuggy silica": 1,
    "Alteracion Argilica y Argilica avanzada": 2, "Alteracion Solfatárica": 2,
    "Alteracion Argilica": 3, "Alteracion Sericitica": 3,
    "Alteración Cuarzo-Sericitica(Fílica)": 3,
    "Alteracion Propilitica": 4,
    "Oxidos e Hidróxidos de Hierro": 5,
    "Alteracion Potasica": 6, "skarn": 6,
    "Alteracion hidrotermal indiferenciada (arg, lim y sil)": 7,
}

CLASS_NAMES = {
    1: "Silicic", 2: "Adv_Argillic", 3: "Argillic/Phyllic",
    4: "Propylitic", 5: "Iron_Oxide", 6: "Potassic/Skarn",
    7: "Hydrothermal_Indiferenciada",
}

MIN_CLASS_SIZE = 10


def main():
    print("Loading per-pixel embeddings + alteration data...")
    pixel_path = EMB_DIR / "atlas_alteration_embeddings_pixels.csv"
    centroid_path = EMB_DIR / "atlas_alteration_embeddings.csv"
    df = pd.read_csv(pixel_path if pixel_path.exists() else centroid_path)

    emb_cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    print(f"Embedding dimensions: {len(emb_cols)}")
    print(f"Total samples: {len(df)}")

    # Filter out no-data, "indiferenciada" (ambiguous), and tiny classes
    df = df[(df["class_id"] > 0) & (df["class_id"] != 7)].copy()
    df = df.dropna(subset=emb_cols)

    # Remove classes with too few samples
    class_counts = df["class_id"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SIZE].index
    df = df[df["class_id"].isin(valid_classes)]

    print(f"\nFiltered samples: {len(df)}")
    print("Class distribution:")
    for cls_id, count in df["class_id"].value_counts().sort_index().items():
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        print(f"  {cls_id} {name:30s}: {count}")

    X_emb = df[emb_cols].values
    y = df["class_id"].values

    # === 1. RANDOM FOREST ON EMBEDDINGS (64-dim) ===
    print("\n" + "=" * 60)
    print("1. RANDOM FOREST ON GOOGLE EMBEDDINGS (64-dim)")
    print("=" * 60)

    rf_emb = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_emb = cross_val_score(rf_emb, X_emb, y, cv=cv, scoring="balanced_accuracy")

    print(f"  Balanced Accuracy (5-fold CV): {scores_emb.mean():.4f} ± {scores_emb.std():.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in scores_emb]}")

    # Fit on all data for feature importance
    rf_emb.fit(X_emb, y)
    importances = pd.Series(rf_emb.feature_importances_, index=emb_cols)
    print(f"\n  Top 10 important embedding dimensions:")
    for dim, imp in importances.nlargest(10).items():
        print(f"    {dim}: {imp:.4f}")

    # Per-class AUC (one-vs-rest)
    from sklearn.model_selection import cross_val_predict
    y_pred_proba = cross_val_predict(rf_emb, X_emb, y, cv=cv, method="predict_proba")
    print(f"\n  Per-class AUC (one-vs-rest):")
    for cls_id in sorted(df["class_id"].unique()):
        y_binary = (y == cls_id).astype(int)
        cls_idx = list(rf_emb.classes_).index(cls_id)
        auc = roc_auc_score(y_binary, y_pred_proba[:, cls_idx])
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        print(f"    {cls_id} {name:30s}: AUC={auc:.4f}")

    # === 2. RANDOM FOREST ON SIMULATED S2 BANDS ===
    # The embeddings don't include raw bands, so we compute proxy S2 features
    # from the embedding data's lat/lon to check if positional info helps
    print("\n" + "=" * 60)
    print("2. RANDOM FOREST ON LOCATION FEATURES (lat/lon baseline)")
    print("=" * 60)

    if "_lat" in df.columns and "_lon" in df.columns:
        X_loc = df[["_lat", "_lon"]].values
        X_loc = X_loc[~np.isnan(X_loc).any(axis=1)]
        y_loc = y[~np.isnan(df[["_lat", "_lon"]].values).any(axis=1)]

        if len(y_loc) > 20:
            rf_loc = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
                class_weight="balanced",
            )
            scores_loc = cross_val_score(rf_loc, X_loc, y_loc, cv=cv, scoring="balanced_accuracy")
            print(f"  Balanced Accuracy (5-fold CV): {scores_loc.mean():.4f} ± {scores_loc.std():.4f}")
            print("  (This is the 'spatial autocorrelation' baseline)")

    # === 3. COMPARISON SUMMARY ===
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    results = {
        "embedding_rf_balanced_accuracy": float(scores_emb.mean()),
        "embedding_rf_std": float(scores_emb.std()),
        "embedding_dimensions": len(emb_cols),
        "n_samples": len(df),
        "n_classes": len(df["class_id"].unique()),
        "classes": {str(k): v for k, v in df["class_id"].value_counts().to_dict().items()},
    }

    # Load PySR Atlas results for comparison
    pysr_path = RESULTS_DIR / "pysr_results_atlas.json"
    if pysr_path.exists():
        with open(pysr_path) as f:
            pysr_results = json.load(f)

        print(f"\n{'Method':<45} {'Accuracy/AUC':<15} {'Interpretable':<15} {'Dims'}")
        print("-" * 90)
        print(f"{'Google Embeddings + RF (64-dim)':<45} {scores_emb.mean():.4f}{'':>10} {'No':<15} 64")
        if "_lat" in df.columns:
            print(f"{'Location only (lat/lon) + RF':<45} {scores_loc.mean():.4f}{'':>10} {'N/A':<15} 2")

        for cls_name, res in pysr_results.items():
            formula = res["best_equation"][:40]
            auc = res["auc"]
            comp = res["complexity"]
            print(f"{'PySR: ' + cls_name + ' (' + formula + '...)':<45} {auc:.4f}{'':>10} {'Yes':<15} {comp}")

        results["pysr_comparison"] = pysr_results

    with open(RESULTS_DIR / "embedding_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'embedding_comparison.json'}")


if __name__ == "__main__":
    main()
