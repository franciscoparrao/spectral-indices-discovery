# Spectral Indices Discovery for Hydrothermal Alteration Mapping

Code, data, and discovered formulas for the paper:

> Parra O., F., Ríos, G., Latorre, M. *Symbolic Regression as Spectral Feature Engineering: Discovering Interpretable Indices for Hydrothermal Alteration Mapping from Sentinel-2.* Submitted to ISPRS Journal of Photogrammetry and Remote Sensing (2026).

## Overview

This repository accompanies a methodological study that uses symbolic regression (PySR) to discover compact, interpretable spectral indices for hydrothermal alteration mapping with Sentinel-2 imagery. The framework is validated at three sites across two continents (Atacama and Coquimbo regions in Chile, and Cuprite, Nevada, USA), and a cross-domain proof-of-concept on land cover classification (central Germany) is included.

## Repository structure

```
spectral-indices-discovery/
├── data/
│   ├── ground_truth/        Alteration polygons (SERNAGEOMIN, USGS)
│   ├── sentinel2/           Sentinel-2 L2A stacks per study site
│   └── results/             JSON/CSV outputs of all experiments and discovered formulas
├── scripts/                 Discovery, validation, and figure-generation scripts
├── notebooks/               Exploratory analyses
├── figures/                 Final figures used in the manuscript
├── paper/                   LaTeX sources (main + supplementary)
└── src/                     Shared utilities (loaders, metrics, plotting)
```

Discovered formulas are stored as plain CSV in `data/results/equations_*.csv` (one file per class per site).

## Requirements

- Python 3.10+
- [PySR](https://github.com/MilesCranmer/PySR) (Julia backend, installed automatically)
- Standard scientific Python stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Geospatial: `rasterio`, `geopandas`, `shapely`
- For Sentinel-2 access: `earthengine-api` (Google Earth Engine account) or `planetary-computer` + `pystac-client`

Install with:

```bash
pip install pysr scikit-learn numpy pandas matplotlib rasterio geopandas earthengine-api
```

## Reproducing the key results

The pipeline is split into independent scripts in `scripts/`. The minimum sequence to reproduce the main results is:

```bash
# 1. Acquire Sentinel-2 pixels for the training site (Atacama Region III)
python scripts/extract_s2_pixels_gee.py

# 2. Run symbolic regression with the 70/30 locked split
python scripts/run_pysr_atlas_gee.py

# 3. Evaluate SR features against raw bands and classical indices
python scripts/exp_sr_rf_ensemble.py

# 4. External validation at Cuprite, Nevada
python scripts/run_pysr_cuprite.py
python scripts/cuprite_pipeline.py

# 5. Cross-domain proof-of-concept on land cover
python scripts/land_cover_poc.py
```

Robustness experiments referenced in the paper:

```bash
python scripts/exp_polygon_disjoint_cv.py    # Polygon-disjoint spatial CV
python scripts/exp_spatial_block_cv.py       # Spatial block CV
python scripts/exp_tost_expanded.py          # TOST equivalence test
python scripts/exp_delong_test.py            # DeLong AUC test
python scripts/exp_dimred_baselines.py       # PCA / mutual-information baselines
python scripts/exp_multi_classifier.py       # RF / XGBoost / SVM comparison
python scripts/exp_dl_comparison.py          # MLP and 1D-CNN baselines
```

Outputs are written to `data/results/` and consumed by figure-generation scripts (`scripts/fig_*.py`).

## Discovered formulas

Final formulas reported in the manuscript are listed in `data/results/equations_atlas_*.csv` (training site) and `data/results/equations_cuprite_*.csv` (external validation site). The land cover formulas are in `data/results/equations_lc_*.csv`.

## Data sources

- **Sentinel-2 L2A**: Copernicus / Google Earth Engine.
- **Chilean alteration ground truth**: SERNAGEOMIN, *Atlas Metalífero de la III Región*.
- **Cuprite ground truth**: USGS, accessed via WMS at `mrdata.usgs.gov`.
- **Mineral spectral references**: USGS Spectral Library v7.
- **Land cover labels**: ESA WorldCover 2021.

## Contact

Francisco Parra O. — `francisco.parra.o@usach.cl`
