# NRR Paper — R Figure Pipeline

ggplot2 + patchwork + tidyterra implementation of every figure in the NRR
submission. Replaces the matplotlib originals (`scripts/fig_*.py`).

## Files

| File | Output | Notes |
|---|---|---|
| `theme_paper.R` | shared theme + palettes + `save_paper()` | source at top of every script |
| `fig_study_area.R` | `study_area.pdf/.png` | 3-panel: hemisphere + Region III polygons + class distribution |
| `fig_spectral_profiles.R` | `spectral_profiles.pdf/.png` | mean S2 reflectance per class with IQR ribbons |
| `fig_pareto_fronts.R` | `pareto_fronts.pdf/.png` | PySR Pareto fronts, 2×3 panels |
| `fig_auc_heatmap.R` | `auc_heatmap.pdf/.png` | cross-site AUC heatmap, methods × classes |
| `fig_physical_interpretation.R` | `physical_interpretation.pdf/.png` | 5 stacked panels: mineral spectra + S2 bands per formula |
| `fig_alteration_comparison.R` | `alteration_comparison.pdf/.png` | Maricunga 6-panel raster + GT polygons |
| `fig_cuprite_comparison.R` | `cuprite_comparison.pdf/.png` | Cuprite 6-panel raster + USGS Rockwell ref |
| `Makefile` | rebuild any/all figures | `make -C paper/R all` |

## Quick start

```bash
# From project root:
cd paper/R && make all

# Sync into the LaTeX submission folder:
make sync-submission
```

## Dependencies

- R 4.3+
- CRAN: `ggplot2`, `patchwork`, `ggrepel`, `dplyr`, `tidyr`, `readr`, `purrr`,
  `tibble`, `jsonlite`, `sf`, `terra`, `tidyterra`, `rnaturalearth`,
  `rnaturalearthdata`, `ggspatial`, `scico`, `ggsci`, `khroma`,
  `systemfonts`, `ragg`, `cowplot`
- Python (for raster prep only): `ee` (earthengine-api), `rasterio`

## Data prerequisites

| File | Source |
|---|---|
| `data/results/pysr_results_gee.json` | PySR run, produced by `scripts/run_pysr_*.py` |
| `data/results/full_evaluation.json` | evaluation pipeline |
| `data/ground_truth/maricunga_training_s2.csv` | dumped from `*.npz` (Makefile target) |
| `data/maps/maricunga_composite.tif` | GEE download via `scripts/prep_rasters_for_R.py` |
| `data/maps/cuprite_composite.tif` | same script |
| `data/ground_truth/cuprite_ground_truth.tif` | USGS Rockwell (2017) ASTER raster |
| `data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp` | SERNAGEOMIN |

## Design choices

- **Theme**: Springer-style — Helvetica base font, white background, axis lines only.
- **Palettes**: semantic, defined once in `theme_paper.R`:
  - `pal_alteration` — six alteration classes (matches Python originals for visual consistency across the historical figure trail)
  - `pal_methods` — raw bands / SR / classical / combined
- **Output**: PDF (vector primary, via cairo_pdf) + PNG (preview, 600 dpi via ragg::agg_png).
- **Sizes**: double-column = 18 cm wide; single-column variant available via `save_paper_single()`.
- **Spatial figures**: GEE download stays in Python (`scripts/prep_rasters_for_R.py`); R reads GeoTIFFs via `terra` and renders with `tidyterra::geom_spatraster*`. This avoids requiring `rgee` auth setup and keeps the GEE side reproducible from existing tooling.
