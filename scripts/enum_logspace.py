#!/usr/bin/env python3
"""
Exhaustive enumeration of scale-invariant band-ratio indices in log space.

Grammar
-------
In log space a product of band ratios is an integer vector c over the 10 bands:
    index(c) = prod_b B_b^{c_b}  <=>  log index = sum_b c_b * log B_b
with sum(c) = 0 (scale invariant) and complexity m = |c|_1 / 2 = number of
ratio factors. AUC depends only on the ranking of pixels, so a normalised
difference (Bi-Bj)/(Bi+Bj) = tanh(0.5*log(Bi/Bj)) is the same expression as
Bi/Bj, k*c is the same as c, and c and -c differ only by orientation. The
enumeration (sign-canonical, primitive, disjoint supports) is therefore
complete for {ratios, products/quotients of ratios, normalised differences}
up to m factors: the best expression per complexity is the global optimum of
that family on the selection set. Counts: m<=2 1 035, m<=4 87 000, m<=5 452 706.

Selection designs (--select-on)
-------------------------------
ring : pilot design. Selection = half of Region III positives vs half of the
       Region III ring (adjacent host rock). Everything else held out.
all  : redesign (2026-10-09). Selection = half of Region III positives vs half
       of EVERY Region III negative group (ring + distractors), with a weighted
       AUC: the ring carries --ring-weight of the negative mass and the
       distractor groups share the rest equally (each group weighted as a
       whole, not per pixel). Groups in --exclude-groups (default: vegetation,
       water, which are masked with NDVI/NDWI in practice) get weight 0 and
       are dropped from the *_hard sets, but their FPR is still reported.
       Region IV is never touched by selection: it is the cross-site test.

Outputs: every expression with AUC on every set (csv.gz), exact Pareto front,
top-20, whole-family held-out distribution, post-hoc ceilings, R4 gate (json).
"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

GT = Path("data/ground_truth"); RES = Path("data/results")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
NB = len(BANDS)

ap = argparse.ArgumentParser()
ap.add_argument("--max-ratios", type=int, default=4)
ap.add_argument("--batch", type=int, default=1000)
ap.add_argument("--perms", type=int, default=200, help="label permutations for the null of max AUC (ring design only)")
ap.add_argument("--select-on", choices=["ring", "all"], default="all")
ap.add_argument("--ring-weight", type=float, default=0.5)
ap.add_argument("--exclude-groups", default="vegetation,water")
ap.add_argument("--seed", type=int, default=0, help="split seed (0 = same split as the pilot)")
ap.add_argument("--tag", default="")
ap.add_argument("--positives", choices=["legacy", "geo"], default="legacy",
                help="legacy: binary_training_s2.csv (no coordinates); geo: binary_training_s2_positives_geo.csv "
                     "(same composite, resampled 2026-10-10 with coordinates and Atlas polygon ids)")
ap.add_argument("--split", choices=["pixel", "polygon", "block"], default="pixel",
                help="how the Region III selection / held-out halves are drawn: pixel (pilot design, inflated by "
                     "autocorrelation), polygon (positives and ring grouped by Atlas polygon, distractors by block), "
                     "block (every Region III pixel grouped by --block-km grid cell). polygon/block need --positives geo")
ap.add_argument("--block-km", type=float, default=10.0)
ap.add_argument("--distractor-block-km", type=float, default=2.0,
                help="block size for the distractor boxes (~10 km wide, so 10 km blocks would leave a side empty)")
args = ap.parse_args()
RNG = np.random.default_rng(args.seed)
EXCL = set(filter(None, args.exclude_groups.split(",")))


# ------------------------------------------------------------------ data
def logb(df):
    return np.log(np.clip(df[BANDS].values.astype(np.float64), 1e-4, None))


def split(df, groups=None):
    """Random 50/50 split. With `groups` (one id per row) whole groups go to one side."""
    if groups is None:
        m = RNG.random(len(df)) < 0.5
    else:
        g = pd.factorize(np.asarray(groups))[0]
        order = RNG.permutation(g.max() + 1)                 # balanced: half of the groups per side
        side = np.zeros(g.max() + 1, bool); side[order[: (len(order) + 1) // 2]] = True
        m = side[g]
    return df[m].reset_index(drop=True), df[~m].reset_index(drop=True)


def block_id(lon, lat, km=None):
    """Grid-cell id (--block-km, or `km`) in UTM 19S for Region III pixels."""
    import geopandas as gpd
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(32719)
    k = (km or args.block_km) * 1000
    return (np.floor(pts.x / k).astype(int) * 100000 + np.floor(pts.y / k).astype(int)).values


def nearest_polygon(lon, lat):
    """Atlas polygon id nearest to each point (ring pixels lie 300 m - 2 km from one)."""
    import geopandas as gpd
    g = gpd.read_file("data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp").set_crs("EPSG:32719", allow_override=True)
    g = gpd.GeoDataFrame({"poly_id": np.arange(len(g))}, geometry=g.geometry.buffer(0), crs=g.crs)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs(32719)
    j = gpd.sjoin_nearest(pts, g, how="left")
    return j[~j.index.duplicated(keep="first")].poly_id.values


if args.split != "pixel" and args.positives != "geo":
    ap.error("--split polygon/block requires --positives geo")

old = pd.read_csv(GT / "binary_training_s2.csv")
if args.positives == "legacy":
    pos3 = old[old.is_altered == 1][BANDS].reset_index(drop=True)
    pos3_grp = None
else:
    geo = pd.read_csv(GT / "binary_training_s2_positives_geo.csv")
    pos3 = geo[BANDS].reset_index(drop=True)
    pos3_grp = (geo.poly_id.values if args.split == "polygon"
                else block_id(geo.lon, geo.lat) if args.split == "block" else None)
oldneg3 = old[old.is_altered == 0][BANDS].reset_index(drop=True)
pos4 = pd.read_csv(GT / "region_iv_validation_s2.csv")[BANDS].reset_index(drop=True)
hard = pd.read_csv(GT / "pilot_hard_negatives.csv")
hard = hard[(hard[BANDS] > 0).all(axis=1)]
neg = {(s, g): d[BANDS + ["lon", "lat"]].reset_index(drop=True) for (s, g), d in hard.groupby(["site", "group"])}


def grp(d, kind):
    """Group ids for a Region III negative table under the chosen split."""
    if args.split == "pixel": return None
    if kind == "distractor": return block_id(d.lon, d.lat, args.distractor_block_km)   # boxes are ~10 km wide
    if args.split == "block": return block_id(d.lon, d.lat)
    return nearest_polygon(d.lon, d.lat)                      # polygon split, ring


# same RNG order as the pilot: positives, then ring, then the other R3 groups alphabetically
pos3_sel, pos3_ho = split(pos3, pos3_grp)
ring3_sel, ring3_ho = split(neg[("R3", "ring")], grp(neg[("R3", "ring")], "ring"))
distract3 = {g: d for (s, g), d in sorted(neg.items()) if s == "R3" and g != "ring"}
d3_sel, d3_ho = {}, {}
for g, d in distract3.items():
    d3_sel[g], d3_ho[g] = split(d, grp(d, "distractor"))
ring4 = neg[("R4", "ring")]
distract4 = {g: d for (s, g), d in neg.items() if s == "R4" and g != "ring"}
neg = {k: v[BANDS] for k, v in neg.items()}
for _d in (pos3_sel, pos3_ho, ring3_sel, ring3_ho, ring4):
    _d.drop(columns=[c for c in ("lon", "lat") if c in _d], inplace=True)
d3_sel = {g: d[BANDS] for g, d in d3_sel.items()}; d3_ho = {g: d[BANDS] for g, d in d3_ho.items()}
distract4 = {g: d[BANDS] for g, d in distract4.items()}
if args.split != "pixel":
    print(f"split={args.split}: positives sel/ho {len(pos3_sel)}/{len(pos3_ho)} "
          f"({pd.Series(pos3_grp).nunique()} groups), ring sel/ho {len(ring3_sel)}/{len(ring3_ho)}")

def cat(dfs): return pd.concat(list(dfs), ignore_index=True)
keep3 = {g: d for g, d in d3_ho.items() if g not in EXCL}
keep4 = {g: d for g, d in distract4.items() if g not in EXCL}

# selection set (+ weights) --------------------------------------------------
if args.select_on == "ring":
    sel_pos, sel_neg = logb(pos3_sel), logb(ring3_sel)
    sel_w = np.ones(len(sel_neg))
else:
    groups_sel = {"ring": ring3_sel, **{g: d for g, d in d3_sel.items() if g not in EXCL}}
    nd = len(groups_sel) - 1
    sel_pos = logb(pos3_sel)
    sel_neg = logb(cat(groups_sel.values()))
    parts = []
    for g, d in groups_sel.items():
        gw = args.ring_weight if g == "ring" else (1 - args.ring_weight) / nd
        parts.append(np.full(len(d), gw / len(d)))
    sel_w = np.concatenate(parts); sel_w /= sel_w.sum()
    print("selection negative groups:", {g: len(d) for g, d in groups_sel.items()})

sets = {"sel": (sel_pos, sel_neg)}
sets["R3_ho_ring"] = (logb(pos3_ho), logb(ring3_ho))
sets["R3_ho_hard"] = (logb(pos3_ho), logb(cat([ring3_ho, *keep3.values()])))
sets["R3_oldboxes"] = (logb(pos3_ho), logb(oldneg3))
sets["R4_ring"] = (logb(pos4), logb(ring4))
sets["R4_hard"] = (logb(pos4), logb(cat([ring4, *keep4.values()])))
groups = {f"R3_{g}": logb(d) for g, d in {"ring": ring3_ho, **d3_ho}.items()}
groups.update({f"R4_{g}": logb(d) for g, d in {"ring": ring4, **distract4}.items()})

print(f"design: select-on={args.select_on}  ring-weight={args.ring_weight}  excluded={sorted(EXCL)}  positives={args.positives}  split={args.split}")
for k, (p, n) in sets.items():
    print(f"  {k:12s} pos {len(p):6d}  neg {len(n):6d}")


# ------------------------------------------------------------------ enumeration
def enumerate_vectors(max_m):
    out, comp = [], []
    for m in range(1, max_m + 1):
        for P in itertools.combinations_with_replacement(range(NB), m):
            rest = [b for b in range(NB) if b not in set(P)]
            for N in itertools.combinations_with_replacement(rest, m):
                c = np.zeros(NB, dtype=np.int8)
                for b in P: c[b] += 1
                for b in N: c[b] -= 1
                if c[np.flatnonzero(c)[0]] < 0: continue              # canonical sign
                if np.gcd.reduce(np.abs(c[c != 0])) > 1: continue     # primitive only
                out.append(c); comp.append(m)
    return np.array(out), np.array(comp)


def expr_str(c):
    def side(idx):
        return "·".join(BANDS[b] + (f"^{abs(int(c[b]))}" if abs(c[b]) > 1 else "") for b in idx)
    num, den = side(np.flatnonzero(c > 0)), side(np.flatnonzero(c < 0))
    return f"{num} / ({den})" if "·" in den else f"{num} / {den}"


t0 = time.time()
C, M = enumerate_vectors(args.max_ratios); E = len(C); Cf = C.astype(np.float64)
print(f"\n{E} expressions up to m={args.max_ratios} ({time.time()-t0:.1f}s): "
      + ", ".join(f"m={m}: {(M == m).sum()}" for m in range(1, args.max_ratios + 1)))


# ------------------------------------------------------------------ AUC
def prep(Lp, Ln):
    return np.vstack([Lp, Ln]), np.r_[np.ones(len(Lp)), np.zeros(len(Ln))], len(Lp), len(Ln)


def auc_batch(L, y, npos, nneg, Cb):
    """Unweighted AUC of every column of S = L @ Cb.T via ranks (Mann-Whitney)."""
    R = rankdata(L @ Cb.T, axis=0, method="average")
    return (y @ R - npos * (npos + 1) / 2) / (npos * nneg), R


def wauc_batch(Lp, Ln, w, Cb):
    """Weighted AUC: sum_j w_j * P(score_pos > score_neg_j), ties count 1/2.
    Vectorised: rank of negative j in the pooled sample minus its rank among
    negatives = (#positives below j) + 0.5 * (#positives tied with j)."""
    Sp, Sn = Lp @ Cb.T, Ln @ Cb.T
    R_all = rankdata(np.vstack([Sp, Sn]), axis=0, method="average")[len(Sp):]
    R_neg = rankdata(Sn, axis=0, method="average")
    frac_above = 1 - (R_all - R_neg) / len(Sp)          # (nneg, batch)
    return w @ frac_above


prepped = {k: prep(*v) for k, v in sets.items() if k != "sel"}
weighted = args.select_on == "all"
if not weighted:
    Lsel, ysel, npsel, nnsel = prep(*sets["sel"])
    Yperm = np.stack([RNG.permutation(ysel) for _ in range(args.perms)]) if args.perms else None
else:
    Yperm = None
null_max = np.zeros(args.perms)

auc = {k: np.zeros(E) for k in sets}; sign = np.zeros(E)
# checkpoint (2026-10-10): the m=5 run was lost twice to a machine freeze; save partial
# AUCs every 10 batches and resume from them if the configuration matches.
RES.mkdir(exist_ok=True)
DESIGN = (("_geo" if args.positives == "geo" else "") + (f"_{args.split}" if args.split != "pixel" else "")
          + (f"_{args.tag}" if args.tag else ""))
CKPT = RES / f".ckpt_enum_logspace_m{args.max_ratios}_{args.select_on}{DESIGN}.npz"
start = 0
if CKPT.exists():
    ck = np.load(CKPT, allow_pickle=False)
    if int(ck["E"]) == E and int(ck["seed"]) == args.seed and float(ck["ring_weight"]) == args.ring_weight:
        start = int(ck["done"]); sign[:start] = ck["sign"][:start]
        for k in sets:
            auc[k][:start] = ck[f"auc_{k}"][:start]
        null_max = ck["null_max"]
        print(f"resuming from checkpoint {CKPT.name}: {start}/{E} done", flush=True)
    else:
        print(f"checkpoint {CKPT.name} ignored (configuration differs)", flush=True)


def save_ckpt(done):
    tmp = CKPT.with_suffix(".tmp.npz")
    np.savez(tmp, E=E, seed=args.seed, ring_weight=args.ring_weight, done=done, sign=sign,
             null_max=null_max, **{f"auc_{k}": v for k, v in auc.items()})
    tmp.replace(CKPT)


t0 = time.time()
for s in range(start, E, args.batch):
    Cb = Cf[s:s + args.batch]; sl = slice(s, s + len(Cb))
    if weighted:
        a = wauc_batch(sets["sel"][0], sets["sel"][1], sel_w, Cb)
    else:
        a, R = auc_batch(Lsel, ysel, npsel, nnsel, Cb)
        if Yperm is not None:
            ap_ = (Yperm @ R - npsel * (npsel + 1) / 2) / (npsel * nnsel)
            null_max = np.maximum(null_max, np.maximum(ap_, 1 - ap_).max(axis=1))
    sg = np.where(a >= 0.5, 1.0, -1.0); sign[sl] = sg
    auc["sel"][sl] = np.where(sg > 0, a, 1 - a)
    for k, (L, y, np_, nn_) in prepped.items():
        a, _ = auc_batch(L, y, np_, nn_, Cb)
        auc[k][sl] = np.where(sg > 0, a, 1 - a)
    if (s // args.batch) % 10 == 0:
        print(f"  {s+len(Cb):7d}/{E}  {time.time()-t0:6.0f}s  best sel so far {auc['sel'][:s+len(Cb)].max():.4f}", flush=True)
        save_ckpt(s + len(Cb))
print(f"evaluation done in {time.time()-t0:.0f}s")
if CKPT.exists():
    CKPT.unlink()

df = pd.DataFrame({"expr": [expr_str(c) for c in C], "m": M, "sign": sign,
                   **{f"auc_{k}": v for k, v in auc.items()},
                   "coef": [",".join(map(str, c)) for c in C]})


# ------------------------------------------------------------------ threshold metrics
def youden_threshold(c, sg):
    Lp, Ln = sets["sel"]; s_ = sg * np.r_[Lp @ c, Ln @ c]
    y = np.r_[np.ones(len(Lp)), np.zeros(len(Ln))]
    w = np.r_[np.full(len(Lp), 1 / len(Lp)), sel_w] if weighted else None
    fpr, tpr, thr = roc_curve(y, s_, sample_weight=w); return thr[np.argmax(tpr - fpr)]


def fpr_report(i):
    c, sg = Cf[i], sign[i]; t = youden_threshold(c, sg)
    out = {"recall_R3_ho@t": float((sg * (sets["R3_ho_ring"][0] @ c) > t).mean()),
           "recall_R4@t": float((sg * (sets["R4_ring"][0] @ c) > t).mean()),
           "fpr@t": {g: float((sg * (L @ c) > t).mean()) for g, L in groups.items()}}
    for site in ("R3", "R4"):
        d = [v for g, v in out["fpr@t"].items() if g.startswith(site) and "ring" not in g
             and g.split("_", 1)[1] not in EXCL]
        out[f"fpr_distractors_mean_{site}"] = float(np.mean(d)) if d else None
    return out


# ------------------------------------------------------------------ summaries
def row(i):
    return {"expr": df.expr[i], "m": int(M[i]), **{f"auc_{k}": float(auc[k][i]) for k in sets}}

order = np.argsort(-auc["sel"]); winner, runner = int(order[0]), int(order[1])
pareto, best = [], -1
for m in range(1, args.max_ratios + 1):
    idx = np.flatnonzero(M == m); i = int(idx[np.argmax(auc["sel"][idx])])
    pareto.append(row(i) | {"improves": bool(auc["sel"][i] > best)}); best = max(best, auc["sel"][i])

clay_i = int(np.flatnonzero(df.expr == "B11 / B12")[0])
w_, c_ = row(winner) | fpr_report(winner), row(clay_i) | fpr_report(clay_i)
gate = {"best": df.expr[winner],
        "delta_auc_R4_ring": w_["auc_R4_ring"] - c_["auc_R4_ring"],
        "delta_auc_R4_hard": w_["auc_R4_hard"] - c_["auc_R4_hard"],
        "delta_auc_R3_ho_hard": w_["auc_R3_ho_hard"] - c_["auc_R3_ho_hard"],
        "fpr_distractors_R4_best": w_["fpr_distractors_mean_R4"], "fpr_distractors_R4_clay": c_["fpr_distractors_mean_R4"],
        "fpr_distractors_R3_best": w_["fpr_distractors_mean_R3"], "fpr_distractors_R3_clay": c_["fpr_distractors_mean_R3"]}
gate["pass_R4"] = bool(gate["delta_auc_R4_ring"] >= 0.03 and gate["delta_auc_R4_hard"] >= 0.03
                       and (gate["fpr_distractors_R4_best"] or 0) <= (gate["fpr_distractors_R4_clay"] or 0))

summary = {
    "config": vars(args) | {"n_expressions": int(E), "counts_per_m": {int(m): int((M == m).sum()) for m in range(1, args.max_ratios + 1)}},
    "sets": {k: {"pos": len(p), "neg": len(n)} for k, (p, n) in sets.items()},
    "winner_by_selection": w_, "runner_up_by_selection": row(runner),
    "gap_sel_winner_vs_runner": float(auc["sel"][winner] - auc["sel"][runner]),
    "clay_B11_B12": c_, "gate_R4": gate,
    "pareto_front_exact": pareto, "top20_by_selection": [row(int(i)) for i in order[:20]],
    "spearman_sel_vs": {k: float(pd.Series(auc["sel"]).corr(pd.Series(auc[k]), method="spearman")) for k in sets if k != "sel"},
    "winner_percentile_in_family": {k: float((auc[k] < auc[k][winner]).mean() * 100) for k in sets if k != "sel"},
    "posthoc_best_per_set": {k: row(int(np.argmax(auc[k]))) for k in sets if k != "sel"},
    "family_quantiles": {k: {q: float(np.quantile(auc[k], q)) for q in (0.5, 0.99, 0.999, 1.0)} for k in sets},
}
if Yperm is not None:
    summary["null_max_sel_auc_under_label_permutation"] = {
        "perms": args.perms, "mean": float(null_max.mean()), "q95": float(np.quantile(null_max, 0.95)),
        "q99": float(np.quantile(null_max, 0.99)), "max": float(null_max.max())}

# continuous relaxation: logistic on log-ratios, same selection set and weights
Lp, Ln = sets["sel"]; ref = BANDS.index("B08")
X = np.vstack([Lp, Ln]); X = X - X[:, [ref]]; y = np.r_[np.ones(len(Lp)), np.zeros(len(Ln))]
sw = np.r_[np.full(len(Lp), 1 / len(Lp)), sel_w / sel_w.sum()] * len(y) / 2 if weighted else None
lr = LogisticRegression(max_iter=5000).fit(X, y, sample_weight=sw)
def lr_auc(k):
    p, n = sets[k]; sp = lr.decision_function(p - p[:, [ref]]); sn = lr.decision_function(n - n[:, [ref]])
    R = rankdata(np.r_[sp, sn]); return float((R[:len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))
summary["continuous_relaxation_logistic_logratios"] = {f"auc_{k}": lr_auc(k) for k in sets if k != "sel"}
summary["logistic_exponents_relative_to_B08"] = dict(zip(BANDS, np.round(lr.coef_[0], 3).tolist()))

# ------------------------------------------------------------------ report
cols = [f"auc_{k}" for k in sets]
def table(title, rows):
    print(f"\n== {title}\n{'expr':40s} {'m':>2s} " + " ".join(f"{c.replace('auc_',''):>11s}" for c in cols))
    for r in rows: print(f"{r['expr'][:40]:40s} {r['m']:2d} " + " ".join(f"{r[c]:11.3f}" for c in cols))
table("Exact Pareto front (best per complexity on 'sel')", pareto)
table("Top-20 by selection AUC", summary["top20_by_selection"])
table("Reference: clay ratio", [c_])
table("Post-hoc best per set (optimistic ceilings of the family)", list(summary["posthoc_best_per_set"].values()))
print("\n== Logistic on log-ratios (continuous relaxation):", {k: round(v, 3) for k, v in summary["continuous_relaxation_logistic_logratios"].items()})
print("== Spearman(sel, set):", {k: round(v, 3) for k, v in summary["spearman_sel_vs"].items()})
print("== Winner percentile in family:", {k: round(v, 2) for k, v in summary["winner_percentile_in_family"].items()})
print("== Winner vs runner-up gap on sel:", round(summary["gap_sel_winner_vs_runner"], 4))
if Yperm is not None: print("== Null max sel AUC:", summary["null_max_sel_auc_under_label_permutation"])
print("\n== FPR per group at Youden threshold (winner | clay):")
for g in groups: print(f"  {g:22s} {w_['fpr@t'][g]:6.2f} | {c_['fpr@t'][g]:6.2f}")
print(f"  recall R3 ho / R4 at t: {w_['recall_R3_ho@t']:.2f} {w_['recall_R4@t']:.2f} | {c_['recall_R3_ho@t']:.2f} {c_['recall_R4@t']:.2f}")
print("\n== GATE (R4 cross-site):", json.dumps(gate, indent=1))

RES.mkdir(exist_ok=True)
name = f"enum_logspace_m{args.max_ratios}_{args.select_on}{DESIGN}"
df.to_csv(RES / f"{name}.csv.gz", index=False, compression="gzip")
json.dump(summary, open(RES / f"{name}.json", "w"), indent=1)
print(f"\nSaved {RES}/{name}.{{csv.gz,json}}")
