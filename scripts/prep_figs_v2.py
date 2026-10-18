#!/usr/bin/env python3
"""Tidy CSVs for the figures of paper/main_v2.tex (read by paper/R/fig_v2_*.R).
All numbers come from versioned results; nothing is recomputed except the selection
AUC of the m=5 winner on the final design (weighted AUC, as enum_logspace)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
R = Path("data/results"); OUT = Path("paper/data/figs_v2"); OUT.mkdir(parents=True, exist_ok=True)
SITE = {"R3": "Atacama (held-out)", "R4": "Coquimbo", "R5": "Cuprite"}

# F2: selection AUC vs test AUC, 87 000 expressions, three criteria
final = pd.read_csv(R / "enum_logspace_m4_all_geo_block_cuprite.csv.gz")
f2 = final[["expr", "m", "auc_R4_hard", "auc_R5_hard"]].copy()
for k, f in [("boxes", "enum_logspace_m4_boxes"), ("ring", "enum_logspace_m4_hard")]:
    f2 = f2.merge(pd.read_csv(R / f"{f}.csv.gz")[["expr", "auc_sel"]].rename(columns={"auc_sel": f"sel_{k}"}), on="expr")
f2 = f2.merge(final[["expr", "auc_sel"]].rename(columns={"auc_sel": "sel_all"}), on="expr")
assert len(f2) == 87000
long = f2.melt(id_vars=["expr", "m", "auc_R4_hard", "auc_R5_hard"], value_vars=["sel_boxes", "sel_ring", "sel_all"],
               var_name="criterion", value_name="auc_sel")
long["criterion"] = long.criterion.str.replace("sel_", "")
long["is_winner"] = long.groupby("criterion").auc_sel.transform("max").eq(long.auc_sel)
long["is_clay"] = long.expr.eq("B11 / B12")
long.to_csv(OUT / "f2_scatter.csv", index=False)

# F3: exact Pareto front (m1-4 final design) + m5 winner on the final design
E4 = json.load(open(R / "enum_logspace_m4_all_geo_block_cuprite.json"))
BL = json.load(open(R / "final_design_baselines.json"))["results"]
rows = []
for p in E4["pareto_front_exact"]:
    for s, k in [("Selection (Atacama)", "auc_sel"), ("Atacama (held-out)", "auc_R3_ho_hard"),
                 ("Coquimbo", "auc_R4_hard"), ("Cuprite", "auc_R5_hard")]:
        rows.append({"m": p["m"], "set": s, "auc": p[k], "expr": p["expr"]})
from hardneg_design import load_design, logb, BANDS
D = load_design(); P, N = D["sets"]["sel"]
c = np.zeros(10)
for b, v in {"B02": 1, "B8A": 3, "B12": 1, "B03": -1, "B04": -1, "B06": -1, "B11": -2}.items(): c[BANDS.index(b)] = v
sp, sn = logb(P) @ c, logb(N) @ c
from scipy.stats import rankdata
r = rankdata(np.r_[sp, sn])[len(sp):]; rn = rankdata(sn)
a = float(D["sel_w"] @ (1 - (r - rn) / len(sp))); a = max(a, 1 - a)
m5 = "B02·B8A^3·B12 / (B03·B04·B06·B11^2)"
for s, v in [("Selection (Atacama)", a), ("Atacama (held-out)", BL["R3_ho_hard"]["winner m5"]["auc"]),
             ("Coquimbo", BL["R4_hard"]["winner m5"]["auc"]), ("Cuprite", BL["R5_hard"]["winner m5"]["auc"])]:
    rows.append({"m": 5, "set": s, "auc": v, "expr": m5})
pd.DataFrame(rows).to_csv(OUT / "f3_pareto.csv", index=False)
print("m5 selection AUC on final design:", round(a, 4))

# F4: ROC curves (hard sets)
show = {"clay B11/B12": "Clay ratio", "winner m4": "Selected index", "logistic 9 log-ratios": "Logistic (log-ratios)",
        "RF 10 raw bands": "Random forest (bands)"}
rows = []
for k, site in [("R3_ho_hard", "Atacama (held-out)"), ("R4_hard", "Coquimbo"), ("R5_hard", "Cuprite")]:
    for name, lab in show.items():
        rr = BL[k][name]
        for x, y in zip(rr["roc"]["fpr"], rr["roc"]["tpr"]):
            rows.append({"site": site, "model": lab, "fpr": x, "tpr": y, "auc": rr["auc"]})
pd.DataFrame(rows).to_csv(OUT / "f4_roc.csv", index=False)

# F5: FPR per negative group at the Youden threshold of the selection set
w, cl = E4["winner_by_selection"]["fpr@t"], E4["clay_B11_B12"]["fpr@t"]
rows = []
for g in w:
    site, grp = g.split("_", 1)
    rows += [{"site": SITE[site], "group": grp, "index": "Selected index", "fpr": w[g]},
             {"site": SITE[site], "group": grp, "index": "Clay ratio", "fpr": cl[g]}]
pd.DataFrame(rows).to_csv(OUT / "f5_fpr.csv", index=False)

# F6: temporal stability (hard sets)
TS = json.load(open(R / "temporal_stability.json"))["sites"]
PER = {"base": "Base (2024)", "prev_summer": "Same season 2023", "winter": "Opposite season", "spring": "Opposite season"}
rows = []
for site, periods in TS.items():
    k = {"R3": "R3_ho_hard", "R4": "R4_hard", "R5": "R5_hard"}[site]
    for per, rec in periods.items():
        for name, lab in [("clay B11/B12", "Clay ratio"), ("winner m4", "Selected index")]:
            rows.append({"site": SITE[site], "period": PER[per], "index": lab, "auc": rec["auc"][k][name]})
pd.DataFrame(rows).to_csv(OUT / "f6_temporal.csv", index=False)
print("saved", sorted(p.name for p in OUT.glob("*.csv")))

# ---------------------------------------------------------------- Atacama West (R6) additions, 2026-10-18
import json as _json
from sklearn.metrics import roc_curve as _roc, roc_auc_score as _auc
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.ensemble import RandomForestClassifier as _RF
_s6 = _json.load(open(R / "site6_and_pa.json"))["site6"]
# F5: FPR per group at R6 (selected index vs clay ratio)
_f = _s6["fpr_at_selection_threshold"]
_rows = [{"site": "Atacama West", "group": g, "index": lab, "fpr": _f[k][g]}
         for g in _f["clay B11/B12"] for k, lab in [("winner ring+distractors", "Selected index"), ("clay B11/B12", "Clay ratio")]]
pd.concat([pd.read_csv(OUT / "f5_fpr.csv"), pd.DataFrame(_rows)]).to_csv(OUT / "f5_fpr.csv", index=False)
# F4: ROC at R6 (ring + distractors) for index, clay, logistic and RF on raw bands
_pos = pd.read_csv("data/ground_truth/site6_positives_geo.csv"); _neg = pd.read_csv("data/ground_truth/site6_hard_negatives.csv")
_pos = _pos[(_pos[BANDS] > 0).all(axis=1)]; _neg = _neg[(_neg[BANDS] > 0).all(axis=1) & ~_neg.group.isin(["vegetation", "water"])]
_y = np.r_[np.ones(len(_pos)), np.zeros(len(_neg))]
_lr = lambda d: logb(d) - logb(d)[:, [BANDS.index("B08")]]
_ys = np.r_[np.ones(len(P)), np.zeros(len(N))]; _w = np.r_[np.full(len(P), 1 / len(P)), D["sel_w"]]; _w = _w / _w.sum() * len(_w)
_log = _LR(C=1.0, max_iter=2000).fit(np.vstack([_lr(P), _lr(N)]), _ys, sample_weight=_w)
_rf = _RF(300, min_samples_leaf=5, n_jobs=6, random_state=0).fit(np.vstack([P[BANDS].values, N[BANDS].values]), _ys, sample_weight=_w)
_sc = {"Selected index": -np.r_[logb(_pos) @ np.array([1, -1, -1, 0, 0, 0, 0, 2, -2, 1.]), logb(_neg) @ np.array([1, -1, -1, 0, 0, 0, 0, 2, -2, 1.])],
       "Clay ratio": np.r_[np.log(_pos.B11 / _pos.B12), np.log(_neg.B11 / _neg.B12)],
       "Logistic (log-ratios)": np.r_[_log.predict_proba(_lr(_pos))[:, 1], _log.predict_proba(_lr(_neg))[:, 1]],
       "Random forest (bands)": np.r_[_rf.predict_proba(_pos[BANDS].values)[:, 1], _rf.predict_proba(_neg[BANDS].values)[:, 1]]}
_grid = np.linspace(0, 1, 201); _rows = []
for lab, s in _sc.items():
    fpr, tpr, _ = _roc(_y, s); a = _auc(_y, s)
    _rows += [{"site": "Atacama West", "model": lab, "fpr": x, "tpr": t, "auc": a} for x, t in zip(_grid, np.interp(_grid, fpr, tpr))]
    print("R6 ROC", lab, round(a, 3))
pd.concat([pd.read_csv(OUT / "f4_roc.csv"), pd.DataFrame(_rows)]).to_csv(OUT / "f4_roc.csv", index=False)
# F2: Atacama West row (needs data/results/enum_logspace_m4_R6.csv.gz from selection_resampling.py)
_r6f = R / "enum_logspace_m4_R6.csv.gz"
if _r6f.exists():
    _r6 = pd.read_csv(_r6f); _sg = final.set_index("expr").loc[_r6.expr, "sign"].values
    _r6["auc_R6_hard"] = np.where(_sg > 0, _r6.auc_R6_hard_raw, 1 - _r6.auc_R6_hard_raw)
    pd.read_csv(OUT / "f2_scatter.csv").merge(_r6[["expr", "auc_R6_hard"]], on="expr").to_csv(OUT / "f2_scatter.csv", index=False)
    print("F2: Atacama West column added")
