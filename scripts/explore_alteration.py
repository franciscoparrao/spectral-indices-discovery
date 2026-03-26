#!/usr/bin/env python3
"""
Exploratory analysis of El Tatio Sentinel-2 composite.
Generates classical alteration indices and RGB composites
to guide manual ground truth delineation.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

DATA_DIR = Path("data/sentinel2/el_tatio")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# S2 L2A scale factor (DN to reflectance)
SCALE = 1.0 / 10000.0
# Valid pixel range for S2 L2A
VALID_MAX = 10000


def load_band(name, target_20m_shape=None):
    """Load a band, resample 10m to 20m via 2x2 block mean if needed."""
    path = DATA_DIR / f"{name}.tif"
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    if target_20m_shape and data.shape != target_20m_shape:
        # 10m -> 20m: block average of 2x2, then trim/pad to match target
        h, w = data.shape
        h2, w2 = h // 2, w // 2
        data = data[:h2*2, :w2*2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))
        th, tw = target_20m_shape
        # Trim or pad to match exactly
        result = np.zeros((th, tw), dtype=np.float32)
        mh, mw = min(h2, th), min(w2, tw)
        result[:mh, :mw] = data[:mh, :mw]
        data = result
        meta.update(height=th, width=tw,
                    transform=rasterio.transform.Affine(
                        20.0, 0.0, meta['transform'].c,
                        0.0, -20.0, meta['transform'].f))

    return data, meta


def mask_invalid(data):
    """Mask pixels outside valid S2 L2A range."""
    mask = (data > 0) & (data <= VALID_MAX)
    return np.where(mask, data * SCALE, np.nan)


def safe_ratio(a, b):
    """Safe division avoiding div by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, np.nan)
    return result


def normalized_diff(a, b):
    """Normalized difference index."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((a + b) != 0, (a - b) / (a + b), np.nan)
    return result


def main():
    print("Loading bands...")
    # Use B11 (20m) as reference grid
    b11_raw, meta_20m = load_band("B11")
    shape_20m = b11_raw.shape
    transform_20m = meta_20m['transform']
    crs_20m = meta_20m['crs']

    # Load all bands, resampling 10m bands to 20m grid via block average
    bands = {}
    bands_10m = {"B02", "B03", "B04", "B08"}
    for name in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]:
        target = shape_20m if name in bands_10m else None
        raw, _ = load_band(name, target_20m_shape=target)
        bands[name] = mask_invalid(raw)
        valid = np.count_nonzero(~np.isnan(bands[name]))
        print(f"  {name}: shape={bands[name].shape} valid={valid} ({100*valid/bands[name].size:.1f}%)")

    # === CLASSICAL ALTERATION INDICES ===
    print("\nComputing alteration indices...")

    # 1. Clay ratio: B11/B12 (SWIR1/SWIR2)
    # High values indicate OH-bearing minerals (clays, phyllosilicates)
    clay_ratio = safe_ratio(bands["B11"], bands["B12"])

    # 2. Iron oxide ratio: B04/B02 (Red/Blue)
    # High values indicate Fe³⁺ oxides (goethite, hematite)
    iron_oxide = safe_ratio(bands["B04"], bands["B02"])

    # 3. Ferrous iron: B12/B08 + B03/B04 (adapted from ASTER)
    # Ferrous minerals (chlorite, epidote in propylitic)
    ferrous = safe_ratio(bands["B12"], bands["B8A"]) + safe_ratio(bands["B03"], bands["B04"])

    # 4. Alunite/kaolinite index: (B11 - B12) / (B11 + B12)
    # Normalized SWIR difference - sensitive to Al-OH absorption
    alunite_idx = normalized_diff(bands["B11"], bands["B12"])

    # 5. NDVI - to identify vegetated vs bare/altered areas
    ndvi = normalized_diff(bands["B8A"], bands["B04"])

    # 6. B11/(B8A + B12) - alteration intensity
    # Highlights areas where SWIR1 is anomalously high relative to NIR and SWIR2
    alteration_intensity = safe_ratio(bands["B11"], bands["B8A"] + bands["B12"])

    # 7. Silica index proxy: B12/B11
    # Low clay ratio (inverse) may indicate silicification
    silica_proxy = safe_ratio(bands["B12"], bands["B11"])

    # === SAVE INDEX GEOTIFFS ===
    print("Saving index GeoTIFFs...")
    out_meta = meta_20m.copy()
    out_meta.update(dtype='float32', count=1, compress='deflate', nodata=np.nan)

    indices = {
        "clay_ratio": clay_ratio,
        "iron_oxide": iron_oxide,
        "ferrous": ferrous,
        "alunite_idx": alunite_idx,
        "ndvi": ndvi,
        "alteration_intensity": alteration_intensity,
        "silica_proxy": silica_proxy,
    }

    for name, data in indices.items():
        path = DATA_DIR / f"idx_{name}.tif"
        with rasterio.open(path, 'w', **out_meta) as dst:
            dst.write(data.astype(np.float32), 1)
        print(f"  Saved {path}")

    # === FIGURES ===
    print("\nGenerating figures...")

    # Helper to plot index
    def plot_index(data, title, filename, cmap='RdYlBu_r', vmin=None, vmax=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            print(f"  Skipping {filename} - no valid data")
            return
        if vmin is None:
            vmin = np.percentile(valid, 2)
        if vmax is None:
            vmax = np.percentile(valid, 98)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_axis_off()
        fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {FIG_DIR / filename}")

    # RGB composites
    def make_rgb(r, g, b, title, filename, percentile=2):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        rgb = np.stack([r, g, b], axis=-1)
        for i in range(3):
            ch = rgb[:, :, i]
            valid = ch[~np.isnan(ch)]
            if len(valid) == 0:
                continue
            lo, hi = np.nanpercentile(ch, [percentile, 100 - percentile])
            rgb[:, :, i] = np.clip((ch - lo) / (hi - lo + 1e-10), 0, 1)
        rgb = np.nan_to_num(rgb, nan=0)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {FIG_DIR / filename}")

    # 1. True color (B04, B03, B02)
    make_rgb(bands["B04"], bands["B03"], bands["B02"],
             "El Tatio - True Color (B4, B3, B2)", "el_tatio_rgb.png")

    # 2. SWIR composite (B12, B11, B04) - geology standard
    make_rgb(bands["B12"], bands["B11"], bands["B04"],
             "El Tatio - SWIR Composite (B12, B11, B4)", "el_tatio_swir.png")

    # 3. Alteration composite (B11, B12, B04) - highlights clay vs iron
    make_rgb(bands["B11"], bands["B12"], bands["B04"],
             "El Tatio - Alteration Composite (B11, B12, B4)", "el_tatio_alteration.png")

    # 4. Iron oxide composite (B04, B03, B02 with enhancement)
    make_rgb(bands["B04"], bands["B02"], bands["B8A"],
             "El Tatio - Iron/NIR Composite (B4, B2, B8A)", "el_tatio_iron.png")

    # 5. Index maps
    plot_index(clay_ratio, "Clay Ratio (B11/B12)\nHigh = OH-bearing minerals",
               "el_tatio_clay_ratio.png", cmap='RdYlBu_r')

    plot_index(iron_oxide, "Iron Oxide Ratio (B4/B2)\nHigh = Fe³⁺ oxides",
               "el_tatio_iron_oxide.png", cmap='YlOrRd')

    plot_index(alunite_idx, "Alunite/Kaolinite Index (B11-B12)/(B11+B12)\nHigh = Al-OH absorption",
               "el_tatio_alunite.png", cmap='RdYlBu_r')

    plot_index(ndvi, "NDVI\nGreen = vegetation, Brown = bare/altered",
               "el_tatio_ndvi.png", cmap='RdYlGn', vmin=-0.1, vmax=0.5)

    plot_index(alteration_intensity, "Alteration Intensity B11/(B8A+B12)\nHigh = SWIR1 anomaly",
               "el_tatio_alteration_intensity.png", cmap='hot')

    plot_index(silica_proxy, "Silica Proxy (B12/B11)\nLow clay = potential silicification",
               "el_tatio_silica.png", cmap='RdYlBu')

    # 6. Multi-index composite for alteration mapping
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = [
        (clay_ratio, "Clay Ratio (B11/B12)", 'RdYlBu_r'),
        (iron_oxide, "Iron Oxide (B4/B2)", 'YlOrRd'),
        (alunite_idx, "Alunite/Kaolinite Idx", 'RdYlBu_r'),
        (ndvi, "NDVI", 'RdYlGn'),
        (alteration_intensity, "Alteration Intensity", 'hot'),
        (silica_proxy, "Silica Proxy (B12/B11)", 'RdYlBu'),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            ax.set_title(title)
            continue
        vmin, vmax = np.percentile(valid, [2, 98])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_axis_off()

    fig.suptitle("El Tatio — Alteration Indices from Sentinel-2 Composite", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "el_tatio_all_indices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'el_tatio_all_indices.png'}")

    # === STATISTICS ===
    print("\n=== Index Statistics (valid pixels) ===")
    for name, data in indices.items():
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            print(f"{name:25s}: n={len(valid):6d}  "
                  f"min={valid.min():.3f}  p25={np.percentile(valid, 25):.3f}  "
                  f"med={np.median(valid):.3f}  p75={np.percentile(valid, 75):.3f}  "
                  f"max={valid.max():.3f}")

    print("\nDone! Check figures/ for visualizations.")


if __name__ == "__main__":
    main()
