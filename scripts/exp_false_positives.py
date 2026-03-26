#!/usr/bin/env python3
"""
E-M8: False positive analysis by lithology.
Apply SR indices to known non-altered lithologies and report FP rates.
Uses the geological map WMS from SERNAGEOMIN or samples from
areas clearly outside alteration zones.
"""

import ee
import numpy as np
import pandas as pd
from pathlib import Path
import json

ee.Initialize()

RESULTS_DIR = Path("data/results")

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_RENAME = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
               "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
               "B11": "B11", "B12": "B12"}

# Define lithological units outside alteration zones in Region III
# These represent common non-altered host rocks
LITHOLOGIES = {
    'Andesite (Tertiary volcanics, fresh)': {
        'geometry': ee.Geometry.Rectangle([-69.8, -27.5, -69.7, -27.4]),
        'description': 'Fresh andesitic volcanics, no mapped alteration',
    },
    'Granite/Granodiorite (intrusive)': {
        'geometry': ee.Geometry.Rectangle([-70.1, -27.8, -70.0, -27.7]),
        'description': 'Cretaceous-Tertiary granitoids',
    },
    'Sedimentary (marine/continental)': {
        'geometry': ee.Geometry.Rectangle([-70.3, -27.3, -70.2, -27.2]),
        'description': 'Mesozoic sedimentary sequence',
    },
    'Basalt (mafic volcanics)': {
        'geometry': ee.Geometry.Rectangle([-69.6, -28.0, -69.5, -27.9]),
        'description': 'Mafic volcanic flows',
    },
    'Alluvium/Colluvium (Quaternary)': {
        'geometry': ee.Geometry.Rectangle([-69.4, -26.7, -69.3, -26.6]),
        'description': 'Recent sediments, fans',
    },
    'Greenschist (metamorphic)': {
        'geometry': ee.Geometry.Rectangle([-70.5, -28.2, -70.4, -28.1]),
        'description': 'Low-grade metamorphic — chlorite-bearing but non-hydrothermal',
    },
    'Altered rock (ground truth positive)': {
        'geometry': ee.Geometry.Rectangle([-69.25, -26.95, -69.15, -26.85]),
        'description': 'Known alteration zone (Maricunga)',
    },
}

eps = 1e-6

def compute_sr_indices(df):
    """Compute all 6 SR indices from a DataFrame with band columns."""
    b02 = df['B02'].values
    b03 = df['B03'].values
    b04 = df['B04'].values
    b05 = df['B05'].values
    b07 = df['B07'].values
    b11 = df['B11'].values
    b12 = df['B12'].values

    return {
        'Silicic (B04-0.135)': b04 - 0.135,
        'Adv.Argillic (0.83-B02/B05)': 0.83 - b02 / np.maximum(b05, eps),
        'Argillic (0.09/B05)': 0.09 / np.maximum(b05, eps),
        'Propylitic (B03-0.48*B11)': b03 - 0.48 * b11,
        'Iron Oxide ((√B12-B11)²)': (np.sqrt(b12) - b11) ** 2,
        'Potassic (B03*B12/B07²-0.45)': b03 * b12 / np.maximum(b07**2, eps) - 0.45,
    }


# Build S2 composite
chile_aoi = ee.Geometry.Rectangle([-70.6, -28.5, -69.0, -26.5])
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(chile_aoi)
      .filterDate("2024-01-01", "2024-04-01")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
      .select(S2_BANDS))

composite = s2.median().divide(10000).rename([BAND_RENAME[b] for b in S2_BANDS])

print("=" * 70)
print("FALSE POSITIVE ANALYSIS BY LITHOLOGY")
print("=" * 70)

results = {}

for lith_name, info in LITHOLOGIES.items():
    print(f"\n  Sampling: {lith_name}...")

    sampled = composite.sample(
        region=info['geometry'], scale=20, numPixels=500, seed=42, geometries=False
    )

    size = sampled.size().getInfo()
    if size == 0:
        print(f"    No pixels, skipping")
        continue

    feats = sampled.toList(min(size, 500)).getInfo()
    rows = [f.get('properties', {}) for f in feats]
    df = pd.DataFrame(rows)

    keep_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
    df = df.dropna(subset=[b for b in keep_bands if b in df.columns])

    if len(df) < 10:
        print(f"    Insufficient pixels ({len(df)}), skipping")
        continue

    indices = compute_sr_indices(df)

    # For each index, compute: mean score, % positive (potential FP), std
    lith_result = {
        'n_pixels': len(df),
        'description': info['description'],
        'indices': {},
    }

    for idx_name, scores in indices.items():
        positive_rate = float((scores > 0).mean())
        lith_result['indices'][idx_name] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'positive_rate': positive_rate,
            'median': float(np.median(scores)),
        }

    results[lith_name] = lith_result
    print(f"    n={len(df)} pixels")
    for idx_name, scores in indices.items():
        pr = (scores > 0).mean()
        print(f"      {idx_name:35s}: mean={scores.mean():+.3f}, FP={pr:.0%}")


# Summary table
print("\n" + "=" * 70)
print("SUMMARY: False Positive Rates by Lithology and SR Index")
print("=" * 70)

idx_names = list(list(results.values())[0]['indices'].keys()) if results else []
header = f"{'Lithology':<35}" + "".join(f"{n[:12]:>13}" for n in idx_names)
print(header)
print("-" * len(header))

for lith_name, r in results.items():
    row = f"{lith_name[:35]:<35}"
    for idx_name in idx_names:
        fp = r['indices'][idx_name]['positive_rate']
        row += f"{fp:>12.0%} "
    print(row)

# Save
output = RESULTS_DIR / "false_positives.json"
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output}")
