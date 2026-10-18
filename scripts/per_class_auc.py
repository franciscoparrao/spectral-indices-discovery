#!/usr/bin/env python3
"""AUC of the selected index and the clay ratio against the host-rock ring, per alteration class,
at Atacama (held-out half of the final design), Coquimbo and Cuprite, with 10 km (Chile) / 2 km
(Cuprite) block-bootstrap CIs of the difference. Output: data/results/per_class_auc.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from hardneg_design import load_design, block_id, BANDS
RNG = np.random.default_rng(0)
I = lambda d: -np.log(d.B02 * d.B8A ** 2 * d.B12 / (d.B03 * d.B04 * d.B11 ** 2)).values
C = lambda d: np.log(d.B11 / d.B12).values
NAMES = {0: "Undifferentiated", 1: "Silicic", 2: "Advanced argillic", 3: "Argillic-phyllic", 4: "Propylitic",
         5: "Iron oxide", 6: "Potassic-skarn"}
D = load_design()
geo = pd.read_csv("data/ground_truth/binary_training_s2_positives_geo.csv")
p3, ring3 = D["sets"]["R3_ho_ring"]
p3 = p3.merge(geo[["lon", "lat", "class_id"]].drop_duplicates(["lon", "lat"]), on=["lon", "lat"], how="left")
p4, ring4 = D["sets"]["R4_ring"]
r4 = pd.read_csv("data/ground_truth/region_iv_validation_s2.csv")
p4 = p4.merge(r4[["lon", "lat", "class_id"]].drop_duplicates(["lon", "lat"]), on=["lon", "lat"], how="left")
p5, ring5 = D["sets"]["R5_ring"]
c5 = pd.read_csv("data/ground_truth/cuprite_positives_geo.csv")
p5 = p5.merge(c5[["lon", "lat", "class_id"]].drop_duplicates(["lon", "lat"]), on=["lon", "lat"], how="left")
out = {}
for site, P, N, km in [("Atacama (held-out)", p3, ring3, 10.0), ("Coquimbo", p4, ring4, 10.0), ("Cuprite", p5, ring5, 2.0)]:
    out[site] = {}
    for cid, g in P.groupby("class_id"):
        if len(g) < 30: continue
        y = np.r_[np.ones(len(g)), np.zeros(len(N))]; si, sc = np.r_[I(g), I(N)], np.r_[C(g), C(N)]
        b = np.r_[block_id(g.lon, g.lat, km), block_id(N.lon, N.lat, km)]; ub, inv = np.unique(b, return_inverse=True)
        mem = [np.flatnonzero(inv == i) for i in range(len(ub))]; dl = []
        for _ in range(500):
            idx = np.concatenate([mem[i] for i in RNG.integers(0, len(ub), len(ub))])
            if y[idx].min() == y[idx].max(): continue
            dl.append(roc_auc_score(y[idx], si[idx]) - roc_auc_score(y[idx], sc[idx]))
        ai, ac = roc_auc_score(y, si), roc_auc_score(y, sc)
        out[site][NAMES[int(cid)]] = {"n": int(len(g)), "auc_index": ai, "auc_clay": ac, "delta": ai - ac,
                                       "delta_ci95": np.quantile(dl, [.025, .975]).tolist()}
        print(f"{site:20s} {NAMES[int(cid)]:18s} n={len(g):5d} index {ai:.3f} clay {ac:.3f} Δ {ai-ac:+.3f} [{np.quantile(dl,.025):+.3f},{np.quantile(dl,.975):+.3f}]")
json.dump(out, open("data/results/per_class_auc.json", "w"), indent=1); print("saved")
