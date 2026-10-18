#!/usr/bin/env python3
"""
(1) Third test site R6 (Atacama West): fixed indices and models, never re-selected.
    AUC against ring / ring + distractors, block-bootstrap CIs (10 km), FPR per group at the
    selection-set Youden threshold, transfer gate. Also the winners of the three selection
    criteria (distant boxes, ring, ring + distractors), which extends RQ1 to a third site.
(2) Prediction-area (P-A) analysis against mapped deposits/occurrences at R3, R4, R5, R6
    (Yousefi & Carranza 2015): for each threshold t of an index, the prediction rate
    P(t) = % of deposits with index >= t and the occupied area A(t) = % of the AOI with
    index >= t (estimated from a uniform random sample). The curves P(t) and 100 - A(t)
    intersect where P% of the deposits fall in (100 - P)% of the area; normalised density
    Nd = P / (100 - P), weight We = ln(Nd). Success-rate curve = P against A; its AUC.
    NDVI > 0.3 and NDWI > 0.2 are masked (as recommended for the index).
    CIs: 1000 bootstrap replicates resampling deposits in 2 km clusters and the background.
Output: data/results/site6_and_pa.json, paper/data/figs_v2/f7_pa.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, str(Path(__file__).parent))
from hardneg_design import load_design, logb, block_id, BANDS, INDICES

RNG = np.random.default_rng(0)
GT = Path("data/ground_truth"); OUT = Path("data/results/site6_and_pa.json")
D = load_design(); P, N = D["sets"]["sel"]; sel_w = D["sel_w"]
y_sel = np.r_[np.ones(len(P)), np.zeros(len(N))]; w_sel = np.r_[np.full(len(P), 1 / len(P)), sel_w]
crit = json.load(open("data/results/selection_criterion_comparison.json"))["criteria"]


def vec(expr):
    c = np.zeros(10); num, den = (expr.split(" / ") + [""])[:2]
    for side, sg in [(num, 1), (den.strip("()"), -1)]:
        for t in filter(None, (x.strip() for x in side.split("·"))):
            b, _, p = t.partition("^"); c[BANDS.index(b)] += sg * (int(p) if p else 1)
    return c


IDX = {k: (lambda c: (lambda d: logb(d) @ c))(vec(v["winner"])) for k, v in
       [("winner ring+distractors", crit["all"]), ("winner ring only", crit["ring"]), ("winner distant boxes", crit["boxes"])]}
IDX["clay B11/B12"] = lambda d: np.log(d.B11 / d.B12).values
IDX["winner m5"] = lambda d: np.log(INDICES["winner m5"](d)).values
IDX["m2 B02·B12/(B03·B11)"] = lambda d: np.log(INDICES["m2 B02·B12/(B03·B11)"](d)).values
# orientation and Youden threshold on the selection set (weighted, as in enum_logspace)
ORIENT, THR = {}, {}
for k, f in IDX.items():
    s = np.r_[f(P), f(N)]
    ORIENT[k] = 1.0 if roc_auc_score(y_sel, s, sample_weight=w_sel) >= 0.5 else -1.0
    fpr, tpr, thr = roc_curve(y_sel, ORIENT[k] * s, sample_weight=w_sel); THR[k] = thr[np.argmax(tpr - fpr)]
score = lambda k, d: ORIENT[k] * IDX[k](d)

# ------------------------------------------------------------------ (1) site R6
pos = pd.read_csv(GT / "site6_positives_geo.csv"); neg = pd.read_csv(GT / "site6_hard_negatives.csv")
pos = pos[(pos[BANDS] > 0).all(axis=1)].reset_index(drop=True); neg = neg[(neg[BANDS] > 0).all(axis=1)].reset_index(drop=True)
ring = neg[neg.group == "ring"]; hard = neg[~neg.group.isin(["vegetation", "water"])]
sets6 = {"R6_ring": (pos, ring), "R6_hard": (pos, hard)}


def boot_auc(Pd, Nd, keys, n=1000, km=10.0):
    y = np.r_[np.ones(len(Pd)), np.zeros(len(Nd))]
    S = {k: np.r_[score(k, Pd), score(k, Nd)] for k in keys}
    b = np.r_[block_id(Pd.lon, Pd.lat, km), block_id(Nd.lon, Nd.lat, km)]; ub, inv = np.unique(b, return_inverse=True)
    mem = [np.flatnonzero(inv == i) for i in range(len(ub))]
    point = {k: roc_auc_score(y, S[k]) for k in keys}; reps = {k: [] for k in keys}
    for _ in range(n):
        idx = np.concatenate([mem[i] for i in RNG.integers(0, len(ub), len(ub))])
        if y[idx].min() == y[idx].max(): continue
        for k in keys: reps[k].append(roc_auc_score(y[idx], S[k][idx]))
    out = {}
    for k in keys:
        a = np.asarray(reps[k]); dlt = a - np.asarray(reps["clay B11/B12"])
        out[k] = {"auc": float(point[k]), "ci95": np.quantile(a, [.025, .975]).tolist(),
                  "delta_vs_clay": float(point[k] - point["clay B11/B12"]), "delta_ci95": np.quantile(dlt, [.025, .975]).tolist()}
    out["_blocks"] = int(len(ub)); return out


r6 = {"n": {"positives": len(pos), **neg.groupby("group").size().to_dict()}}
for name, (Pd, Nd) in sets6.items():
    r6[name] = boot_auc(Pd, Nd, list(IDX))
fpr = {k: {g: float((score(k, d) > THR[k]).mean()) for g, d in neg.groupby("group")} for k in IDX}
dist = lambda k: float(np.mean([v for g, v in fpr[k].items() if g not in ("ring", "vegetation", "water")]))
r6["fpr_at_selection_threshold"] = fpr; r6["recall_at_selection_threshold"] = {k: float((score(k, pos) > THR[k]).mean()) for k in IDX}
r6["gate"] = {}
for k in [x for x in IDX if x.startswith("winner")]:
    dr, dh = r6["R6_ring"][k]["delta_vs_clay"], r6["R6_hard"][k]["delta_vs_clay"]
    r6["gate"][k] = {"delta_ring": dr, "delta_hard": dh, "fpr_distractors": dist(k), "fpr_distractors_clay": dist("clay B11/B12"),
                     "pass": bool(dr >= 0.03 and dh >= 0.03 and dist(k) <= dist("clay B11/B12"))}
print("== R6 (Atacama West), n =", r6["n"])
for k in IDX:
    a, b = r6["R6_ring"][k], r6["R6_hard"][k]
    print(f"  {k:26s} ring {a['auc']:.3f} Δ{a['delta_vs_clay']:+.3f} [{a['delta_ci95'][0]:+.3f},{a['delta_ci95'][1]:+.3f}]  "
          f"hard {b['auc']:.3f} Δ{b['delta_vs_clay']:+.3f} [{b['delta_ci95'][0]:+.3f},{b['delta_ci95'][1]:+.3f}]  "
          f"recall {r6['recall_at_selection_threshold'][k]:.2f}" + (f"  gate {r6['gate'][k]['pass']}" if k in r6["gate"] else ""))
print("  FPR by group (selected | clay):", {g: (round(fpr['winner ring+distractors'][g], 2), round(fpr['clay B11/B12'][g], 2)) for g in fpr["clay B11/B12"]})

# ------------------------------------------------------------------ (2) P-A analysis
PA_IDX = ["winner ring+distractors", "clay B11/B12"]
LAB = {"winner ring+distractors": "Selected index", "clay B11/B12": "Clay ratio"}


def pa_curves(dep_s, bg_s, q=np.linspace(0, 100, 201)):
    t = np.percentile(bg_s, q)
    Pr = np.array([(dep_s >= x).mean() * 100 for x in t]); Oa = np.array([(bg_s >= x).mean() * 100 for x in t])
    i = int(np.argmin(np.abs(Pr - (100 - Oa))))
    pr_int = float((Pr[i] + 100 - Oa[i]) / 2)
    # success-rate curve: deposits captured vs area, highest values first
    order = np.sort(bg_s)[::-1]; area = np.linspace(0, 1, 101)
    cut = np.r_[np.inf, np.quantile(bg_s, 1 - area[1:])]
    capt = np.array([(dep_s >= c).mean() for c in cut])
    return {"pct": q, "thr": t, "Pr": Pr, "Oa": Oa, "pr_intersection": pr_int,
            "Nd": pr_int / max(100 - pr_int, 1e-9), "success_area": area, "success_capt": capt,
            "success_auc": float(np.trapezoid(capt, area))}


pa, rows = {}, []
for site in ["R3", "R4", "R5", "R6"]:
    dep = pd.read_csv(GT / f"pa_{site}_deposits.csv"); bg = pd.read_csv(GT / f"pa_{site}_background.csv")
    ok = lambda d: (d[BANDS] > 0).all(axis=1) & (d.ndvi <= 0.3) & (d.ndwi <= 0.2)
    n_dep0 = len(dep); dep = dep[ok(dep)].reset_index(drop=True); bg = bg[ok(bg)].reset_index(drop=True)
    cl = block_id(dep.lon, dep.lat, 2.0); ucl, inv = np.unique(cl, return_inverse=True); mem = [np.flatnonzero(inv == i) for i in range(len(ucl))]
    res = {"n_deposits": int(len(dep)), "n_deposits_masked_out": int(n_dep0 - len(dep)), "n_deposit_clusters_2km": int(len(ucl)), "n_background": int(len(bg))}
    sc = {k: (score(k, dep), score(k, bg)) for k in PA_IDX}
    for k in PA_IDX:
        c = pa_curves(*sc[k]); res[k] = {"pr_intersection": c["pr_intersection"], "Nd": c["Nd"], "We": float(np.log(c["Nd"])),
                                          "success_auc": c["success_auc"]}
        for pc, t, pr, oa in zip(c["pct"], c["thr"], c["Pr"], c["Oa"]):
            rows.append({"site": site, "index": LAB[k], "curve": "pa", "x": pc, "pr": pr, "oa": oa})
        for a_, cp in zip(c["success_area"], c["success_capt"]):
            rows.append({"site": site, "index": LAB[k], "curve": "success", "x": a_ * 100, "pr": cp * 100, "oa": np.nan})
    reps = {k: {"pr": [], "auc": []} for k in PA_IDX}
    for _ in range(1000):
        di = np.concatenate([mem[i] for i in RNG.integers(0, len(ucl), len(ucl))]); bi = RNG.integers(0, len(bg), len(bg))
        for k in PA_IDX:
            c = pa_curves(sc[k][0][di], sc[k][1][bi], q=np.linspace(0, 100, 101))
            reps[k]["pr"].append(c["pr_intersection"]); reps[k]["auc"].append(c["success_auc"])
    for k in PA_IDX:
        res[k]["pr_intersection_ci95"] = np.quantile(reps[k]["pr"], [.025, .975]).tolist()
        res[k]["success_auc_ci95"] = np.quantile(reps[k]["auc"], [.025, .975]).tolist()
    dpr = np.array(reps[PA_IDX[0]]["pr"]) - np.array(reps[PA_IDX[1]]["pr"]); dau = np.array(reps[PA_IDX[0]]["auc"]) - np.array(reps[PA_IDX[1]]["auc"])
    res["delta_pr_intersection"] = {"point": res[PA_IDX[0]]["pr_intersection"] - res[PA_IDX[1]]["pr_intersection"], "ci95": np.quantile(dpr, [.025, .975]).tolist()}
    res["delta_success_auc"] = {"point": res[PA_IDX[0]]["success_auc"] - res[PA_IDX[1]]["success_auc"], "ci95": np.quantile(dau, [.025, .975]).tolist()}
    pa[site] = res
    print(f"== P-A {site}: deposits {len(dep)} ({res['n_deposits_masked_out']} masked out, {len(ucl)} clusters), background {len(bg)}")
    for k in PA_IDX:
        r = res[k]; print(f"  {LAB[k]:15s} intersection Pr {r['pr_intersection']:.1f}% [{r['pr_intersection_ci95'][0]:.1f},{r['pr_intersection_ci95'][1]:.1f}]  "
                          f"Nd {r['Nd']:.2f}  success AUC {r['success_auc']:.3f} [{r['success_auc_ci95'][0]:.3f},{r['success_auc_ci95'][1]:.3f}]")
    print(f"  Δ intersection {res['delta_pr_intersection']['point']:+.1f} [{res['delta_pr_intersection']['ci95'][0]:+.1f},{res['delta_pr_intersection']['ci95'][1]:+.1f}]  "
          f"Δ success AUC {res['delta_success_auc']['point']:+.3f} [{res['delta_success_auc']['ci95'][0]:+.3f},{res['delta_success_auc']['ci95'][1]:+.3f}]")

OUT.write_text(json.dumps({"site6": r6, "pa": pa, "orientation": ORIENT,
                           "notes": "R6 never used for selection; samples within 5 km of Atlas III alteration dropped. "
                                    "P-A masks NDVI>0.3 and NDWI>0.2; background = uniform random sample of the AOI."}, indent=1, default=float))
Path("paper/data/figs_v2").mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv("paper/data/figs_v2/f7_pa.csv", index=False)
print("saved", OUT)
