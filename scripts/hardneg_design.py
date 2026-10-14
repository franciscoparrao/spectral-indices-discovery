#!/usr/bin/env python3
"""
Shared loader for the hard-negative design (same RNG order as enum_logspace.py, so the
selection / held-out halves are identical for a given seed, split and positives).

load_design(...) -> dict with
  sets   : name -> (pos DataFrame, neg DataFrame) with BANDS (+ lon/lat/block where known)
  sel_w  : weights of the selection negatives (weighted AUC, as in enum_logspace)
  groups : "R3_ring", "R3_salar_uyuni", ... -> DataFrame (held-out halves for R3, all for R4/R5)
Verified against data/results/enum_logspace_m4_all_geo_block_cuprite.json (set sizes and AUCs).
"""
from pathlib import Path
import numpy as np, pandas as pd

GT = Path("data/ground_truth")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def logb(df):
    return np.log(np.clip(df[BANDS].values.astype(np.float64), 1e-4, None))


def block_id(lon, lat, km):
    import geopandas as gpd
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(32719)
    k = km * 1000
    return (np.floor(pts.x / k).astype(int) * 100000 + np.floor(pts.y / k).astype(int)).values


def nearest_polygon(lon, lat):
    import geopandas as gpd
    g = gpd.read_file("data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp").set_crs("EPSG:32719", allow_override=True)
    g = gpd.GeoDataFrame({"poly_id": np.arange(len(g))}, geometry=g.geometry.buffer(0), crs=g.crs)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs(32719)
    j = gpd.sjoin_nearest(pts, g, how="left")
    return j[~j.index.duplicated(keep="first")].poly_id.values


def load_design(positives="geo", split="block", block_km=10.0, distractor_block_km=2.0, cuprite=True,
                seed=0, exclude="vegetation,water", ring_weight=0.5, select_on="all", verbose=False):
    RNG = np.random.default_rng(seed)
    EXCL = set(filter(None, exclude.split(",")))

    def split_df(df, groups=None):
        if groups is None:
            m = RNG.random(len(df)) < 0.5
        else:
            g = pd.factorize(np.asarray(groups))[0]
            order = RNG.permutation(g.max() + 1)
            side = np.zeros(g.max() + 1, bool); side[order[: (len(order) + 1) // 2]] = True
            m = side[g]
        return df[m].reset_index(drop=True), df[~m].reset_index(drop=True)

    old = pd.read_csv(GT / "binary_training_s2.csv")
    if positives == "legacy":
        pos3 = old[old.is_altered == 1][BANDS].reset_index(drop=True); pos3_grp = None
    else:
        geo = pd.read_csv(GT / "binary_training_s2_positives_geo.csv")
        pos3 = geo[BANDS + ["lon", "lat"]].reset_index(drop=True)
        pos3_grp = (geo.poly_id.values if split == "polygon"
                    else block_id(geo.lon, geo.lat, block_km) if split == "block" else None)
    oldneg3 = old[old.is_altered == 0][BANDS].reset_index(drop=True)
    pos4 = pd.read_csv(GT / "region_iv_validation_s2.csv")[BANDS + ["lon", "lat"]].reset_index(drop=True)
    hard = pd.read_csv(GT / "pilot_hard_negatives.csv")
    hard = hard[(hard[BANDS] > 0).all(axis=1)]
    neg = {(s, g): d[BANDS + ["lon", "lat"]].reset_index(drop=True) for (s, g), d in hard.groupby(["site", "group"])}

    def grp(d, kind):
        if split == "pixel": return None
        if kind == "distractor": return block_id(d.lon, d.lat, distractor_block_km)
        if split == "block": return block_id(d.lon, d.lat, block_km)
        return nearest_polygon(d.lon, d.lat)

    pos3_sel, pos3_ho = split_df(pos3, pos3_grp)
    ring3_sel, ring3_ho = split_df(neg[("R3", "ring")], grp(neg[("R3", "ring")], "ring"))
    distract3 = {g: d for (s, g), d in sorted(neg.items()) if s == "R3" and g != "ring"}
    d3_sel, d3_ho = {}, {}
    for g, d in distract3.items():
        d3_sel[g], d3_ho[g] = split_df(d, grp(d, "distractor"))
    ring4 = neg[("R4", "ring")]
    distract4 = {g: d for (s, g), d in neg.items() if s == "R4" and g != "ring"}
    out_sites = ["R3", "R4"]
    if cuprite:
        c5 = pd.read_csv(GT / "cuprite_positives_geo.csv")
        pos5 = c5[BANDS + ["lon", "lat"]].reset_index(drop=True)
        hard5 = pd.read_csv(GT / "cuprite_hard_negatives.csv")
        hard5 = hard5[(hard5[BANDS] > 0).all(axis=1)]
        neg5 = {g: d[BANDS + ["lon", "lat"]].reset_index(drop=True) for g, d in hard5.groupby("group")}
        ring5 = neg5.pop("ring"); distract5 = neg5
        out_sites.append("R5")

    def cat(dfs): return pd.concat(list(dfs), ignore_index=True)
    keep3 = {g: d for g, d in d3_ho.items() if g not in EXCL}
    keep4 = {g: d for g, d in distract4.items() if g not in EXCL}

    if select_on == "ring":
        sel_neg = ring3_sel; sel_w = np.ones(len(sel_neg)) / len(sel_neg)
    else:
        groups_sel = {"ring": ring3_sel, **{g: d for g, d in d3_sel.items() if g not in EXCL}}
        nd = len(groups_sel) - 1
        sel_neg = cat(groups_sel.values())
        parts = []
        for g, d in groups_sel.items():
            gw = ring_weight if g == "ring" else (1 - ring_weight) / nd
            parts.append(np.full(len(d), gw / len(d)))
        sel_w = np.concatenate(parts); sel_w /= sel_w.sum()

    sets = {"sel": (pos3_sel, sel_neg),
            "R3_ho_ring": (pos3_ho, ring3_ho),
            "R3_ho_hard": (pos3_ho, cat([ring3_ho, *keep3.values()])),
            "R3_oldboxes": (pos3_ho, oldneg3),
            "R4_ring": (pos4, ring4),
            "R4_hard": (pos4, cat([ring4, *keep4.values()]))}
    groups = {f"R3_{g}": d for g, d in {"ring": ring3_ho, **d3_ho}.items()}
    groups.update({f"R4_{g}": d for g, d in {"ring": ring4, **distract4}.items()})
    if cuprite:
        keep5 = {g: d for g, d in distract5.items() if g not in EXCL}
        sets["R5_ring"] = (pos5, ring5)
        sets["R5_hard"] = (pos5, cat([ring5, *keep5.values()]))
        groups.update({f"R5_{g}": d for g, d in {"ring": ring5, **distract5}.items()})
    if verbose:
        for k, (p, n) in sets.items():
            print(f"  {k:12s} pos {len(p):6d}  neg {len(n):6d}")
    return {"sets": sets, "sel_w": sel_w, "groups": groups, "sites": out_sites, "excl": EXCL,
            "block_km": block_km, "split": split}


# index formulas on raw reflectance DataFrames ---------------------------------------------
INDICES = {
    "clay B11/B12":            lambda d: d.B11 / d.B12,
    "winner m4":               lambda d: d.B02 * d.B8A ** 2 * d.B12 / (d.B03 * d.B04 * d.B11 ** 2),
    "winner m5":               lambda d: d.B02 * d.B8A ** 3 * d.B12 / (d.B03 * d.B04 * d.B06 * d.B11 ** 2),
    "m2 B02·B12/(B03·B11)":    lambda d: d.B02 * d.B12 / (d.B03 * d.B11),
    "ferric B04/B02":          lambda d: d.B04 / d.B02,
    "paper SR (sqrtB12-B11)^2": lambda d: (np.sqrt(d.B12) - d.B11) ** 2,
}


if __name__ == "__main__":
    import json
    from sklearn.metrics import roc_auc_score
    D = load_design(verbose=True)
    ref = json.load(open("data/results/enum_logspace_m4_all_geo_block_cuprite.json"))
    ok = True
    for k, (p, n) in D["sets"].items():
        r = ref["sets"][k]
        if (len(p), len(n)) != (r["pos"], r["neg"]):
            ok = False; print("SIZE MISMATCH", k, (len(p), len(n)), r)
    for name, w in [("winner m4", ref["winner_by_selection"]), ("clay B11/B12", ref["clay_B11_B12"])]:
        for k, (p, n) in D["sets"].items():
            if k == "sel": continue
            s = np.r_[INDICES[name](p), INDICES[name](n)]; y = np.r_[np.ones(len(p)), np.zeros(len(n))]
            a = max(roc_auc_score(y, s), roc_auc_score(y, -s))
            if abs(a - w[f"auc_{k}"]) > 1e-6:
                ok = False; print("AUC MISMATCH", name, k, round(a, 6), round(w[f"auc_{k}"], 6))
    print("design reproduces enum_logspace output:", ok)
