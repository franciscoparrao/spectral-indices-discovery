#!/usr/bin/env python3
"""
E-M5: Silicic index specificity test.
Test B04 - 0.135 against known non-silicic bright surfaces.
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

# Define test surfaces — known non-silicic bright areas
TEST_SURFACES = {
    'Salar de Atacama (salt flat)': {
        'geometry': ee.Geometry.Rectangle([-68.35, -23.55, -68.25, -23.45]),
        'expected': 'bright, non-silicic',
    },
    'Salar de Uyuni (salt flat, Bolivia)': {
        'geometry': ee.Geometry.Rectangle([-67.7, -20.3, -67.6, -20.2]),
        'expected': 'very bright, non-silicic',
    },
    'Urban Santiago (Chile)': {
        'geometry': ee.Geometry.Rectangle([-70.68, -33.47, -70.62, -33.43]),
        'expected': 'mixed bright, non-silicic',
    },
    'Sandy desert (Atacama)': {
        'geometry': ee.Geometry.Rectangle([-69.8, -24.5, -69.7, -24.4]),
        'expected': 'bright soil, non-silicic',
    },
    'Maricunga silicic alteration (GT)': {
        'geometry': ee.Geometry.Rectangle([-69.25, -26.95, -69.15, -26.85]),
        'expected': 'true silicic alteration',
    },
    'Maricunga unaltered rock (GT)': {
        'geometry': ee.Geometry.Rectangle([-69.5, -27.2, -69.4, -27.1]),
        'expected': 'unaltered rock, dark',
    },
    'Snow/ice (high Andes)': {
        'geometry': ee.Geometry.Rectangle([-69.9, -27.1, -69.85, -27.05]),
        'expected': 'very bright, non-silicic',
    },
    'Lake/water (Laguna Verde)': {
        'geometry': ee.Geometry.Rectangle([-68.55, -22.8, -68.5, -22.75]),
        'expected': 'dark, non-silicic',
    },
}

print("=" * 70)
print("SILICIC INDEX SPECIFICITY TEST")
print("B04 - 0.135 tested against non-silicic bright surfaces")
print("=" * 70)

results = {}

for name, info in TEST_SURFACES.items():
    print(f"\n  Testing: {name}...")

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(info['geometry'])
          .filterDate("2024-01-01", "2024-04-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .select(S2_BANDS))

    n_scenes = s2.size().getInfo()
    if n_scenes == 0:
        # Try summer dates for southern hemisphere
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(info['geometry'])
              .filterDate("2024-06-01", "2024-09-30")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
              .select(S2_BANDS))
        n_scenes = s2.size().getInfo()

    if n_scenes == 0:
        print(f"    No scenes found, skipping")
        continue

    composite = s2.median().divide(10000).rename([BAND_RENAME[b] for b in S2_BANDS])

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

    if 'B04' not in df.columns:
        print(f"    No B04 data, skipping")
        continue

    b04 = df['B04'].values
    silicic_score = b04 - 0.135

    # Also compute other SR indices for comparison
    b02 = df.get('B02', pd.Series(np.zeros(len(df)))).values
    b03 = df.get('B03', pd.Series(np.zeros(len(df)))).values
    b05 = df.get('B05', pd.Series(np.zeros(len(df)))).values
    b11 = df.get('B11', pd.Series(np.zeros(len(df)))).values
    b12 = df.get('B12', pd.Series(np.zeros(len(df)))).values
    b07 = df.get('B07', pd.Series(np.zeros(len(df)))).values

    eps = 1e-6
    propylitic_score = b03 - 0.48 * b11
    adv_argillic_score = 0.83 - b02 / np.maximum(b05, eps)
    swir_alt = (np.sqrt(b12) - b11) ** 2

    fp_rate = (silicic_score > 0).mean()

    result = {
        'n_pixels': len(df),
        'n_scenes': n_scenes,
        'expected': info['expected'],
        'B04_mean': float(b04.mean()),
        'B04_std': float(b04.std()),
        'silicic_score_mean': float(silicic_score.mean()),
        'silicic_positive_rate': float(fp_rate),
        'propylitic_score_mean': float(propylitic_score.mean()),
        'adv_argillic_score_mean': float(adv_argillic_score.mean()),
    }
    results[name] = result

    print(f"    n={len(df)}, B04 mean={b04.mean():.3f}")
    print(f"    Silicic score: mean={silicic_score.mean():.3f}, positive rate={fp_rate:.1%}")
    print(f"    Propylitic: mean={propylitic_score.mean():.3f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Surface':<40} {'B04':>6} {'Score':>7} {'FP Rate':>8}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<40} {r['B04_mean']:>6.3f} {r['silicic_score_mean']:>7.3f} {r['silicic_positive_rate']:>8.1%}")

output = RESULTS_DIR / "silicic_specificity.json"
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output}")
