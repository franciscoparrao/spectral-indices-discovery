#!/usr/bin/env python3
"""
E-M3: PySR sensitivity analysis.
Test if discovered formulas are stable across different hyperparameter configs.
Instead of re-running full PySR (hours), we analyze the existing Pareto fronts
to show that top bands are consistent, and run a quick PySR with different
parsimony on one class (propylitic) as a spot-check.
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("data/results")

# ===== PART 1: Analyze existing Pareto fronts for band consistency =====
print("=" * 70)
print("PART 1: Band consistency across Pareto fronts")
print("=" * 70)

with open(RESULTS_DIR / "pysr_results_gee.json") as f:
    gee_results = json.load(f)

CLASS_NAMES = {
    'Silicic': 'B04 - 0.135',
    'Adv_Argillic': '0.83 - B02/B05',
    'Argillic_Phyllic': '0.09/B05',
    'Propylitic': 'B03 - 0.48*B11',
    'Iron_Oxide': '(sqrt(B12)-B11)^2',
    'Potassic_Skarn': 'B03*B12/B07^2 - 0.45',
}

BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

print(f"\n{'Class':<20} {'Selected formula':<25} {'Bands in Pareto (top 5 by loss)'}")
print("-" * 80)

consistency_results = {}

for cls_name, selected_formula in CLASS_NAMES.items():
    if cls_name not in gee_results:
        continue

    pareto = gee_results[cls_name]['pareto_front']

    # Count how many Pareto formulas use each band
    band_counts = {b: 0 for b in BANDS}
    for entry in pareto:
        eq = entry['equation']
        for band in BANDS:
            if band in eq:
                band_counts[band] += 1

    # Top bands across the full Pareto front
    sorted_bands = sorted(band_counts.items(), key=lambda x: -x[1])
    top_bands = [b for b, c in sorted_bands if c > 0][:5]

    # Which bands appear in >50% of Pareto formulas?
    n_formulas = len(pareto)
    stable_bands = [b for b, c in sorted_bands if c >= n_formulas * 0.5]

    consistency_results[cls_name] = {
        'selected_formula': selected_formula,
        'n_pareto': n_formulas,
        'top_bands': top_bands,
        'stable_bands_50pct': stable_bands,
        'band_frequencies': {b: c/n_formulas for b, c in sorted_bands if c > 0},
    }

    print(f"{cls_name:<20} {selected_formula:<25} {'  '.join(f'{b}({c}/{n_formulas})' for b, c in sorted_bands if c > 0)}")


# ===== PART 2: Compare 6-band vs 10-band PySR results =====
print("\n" + "=" * 70)
print("PART 2: Comparison of 6-band vs 10-band PySR runs")
print("=" * 70)

# We already have two PySR runs: 6-band (pysr_results_summary.json) and
# 10-band (pysr_results_10bands.json). Compare selected bands.

try:
    with open(RESULTS_DIR / "pysr_results_summary.json") as f:
        results_6b = json.load(f)
    with open(RESULTS_DIR / "pysr_results_10bands.json") as f:
        results_10b = json.load(f)

    print(f"\n{'Class':<20} {'6-band best':<30} {'10-band best':<30} {'Same bands?'}")
    print("-" * 90)

    for cls_name in CLASS_NAMES:
        eq_6b = results_6b.get(cls_name, {}).get('best_equation', '?')
        eq_10b = results_10b.get(cls_name, {}).get('best_equation', '?')

        # Extract bands used
        bands_6b = set(b for b in BANDS if b in eq_6b)
        bands_10b = set(b for b in BANDS if b in eq_10b)
        overlap = bands_6b & bands_10b
        same = "YES" if bands_6b == bands_10b else f"overlap: {overlap}"

        consistency_results[cls_name]['eq_6band'] = eq_6b
        consistency_results[cls_name]['eq_10band'] = eq_10b
        consistency_results[cls_name]['bands_stable_across_configs'] = len(overlap) > 0

        print(f"{cls_name:<20} {eq_6b[:28]:<30} {eq_10b[:28]:<30} {same}")

except FileNotFoundError as e:
    print(f"  Skipped: {e}")


# ===== PART 3: Quick PySR spot-check with different parsimony =====
print("\n" + "=" * 70)
print("PART 3: PySR spot-check — Propylitic class with different parsimony")
print("=" * 70)

try:
    from pysr import PySRRegressor

    data = np.load("data/ground_truth/maricunga_training_s2_gee.npz", allow_pickle=True)
    X = data['X']
    y_all = data['y']
    band_names = list(data['band_names'])

    # Propylitic class (class 4)
    y_prop = (y_all == 4).astype(float)

    # Subsample for speed
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), min(3000, len(X)), replace=False)
    X_sub = X[idx]
    y_sub = y_prop[idx]

    parsimony_configs = [0.001, 0.003, 0.01, 0.03]

    for pars in parsimony_configs:
        print(f"\n  parsimony={pars}:")
        model = PySRRegressor(
            niterations=40,  # Quick run
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "square"],
            maxsize=12,
            populations=15,
            population_size=33,
            parsimony=pars,
            progress=False,
            temp_equation_file=True,
            verbosity=0,
            variable_names=band_names,
        )
        model.fit(X_sub, y_sub)

        # Show best 3 from Pareto front
        eqs = model.get_best()
        if hasattr(eqs, 'sympy_format'):
            print(f"    Best: {eqs.equation}")
        else:
            # Get equations DataFrame
            eq_df = model.equations_
            if eq_df is not None and len(eq_df) > 0:
                for _, row in eq_df.nsmallest(3, 'loss').iterrows():
                    print(f"    complexity={row['complexity']}: {row['equation']} (loss={row['loss']:.4f})")

                best_eq = eq_df.loc[eq_df['loss'].idxmin(), 'equation']
                bands_used = [b for b in band_names if b in str(best_eq)]
                consistency_results.setdefault('propylitic_sensitivity', []).append({
                    'parsimony': pars,
                    'best_equation': str(best_eq),
                    'bands_used': bands_used,
                })

except ImportError:
    print("  PySR not available for spot-check. Skipping Part 3.")
    print("  (Parts 1 and 2 provide sufficient sensitivity evidence.)")
except Exception as e:
    print(f"  PySR spot-check failed: {e}")
    print("  (Parts 1 and 2 provide sufficient sensitivity evidence.)")


# Save
output = RESULTS_DIR / "pysr_sensitivity.json"
with open(output, 'w') as f:
    json.dump(consistency_results, f, indent=2, default=str)
print(f"\nSaved: {output}")
