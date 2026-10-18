#!/usr/bin/env python3
"""
Selection variance (review issue: one argmax per criterion). For each split seed and each of
the three selection criteria (distant boxes, host-rock ring, ring + distractors):
  split Maricunga into selection / held-out halves (same block rules as hardneg_design),
  score all 87,000 expressions (m <= 4) on the selection half, take the argmax,
  and read its AUC at the three test sites (Coquimbo, Cuprite, Atacama West).
Test-site AUCs do not depend on the split, so they are computed once per expression
(Coquimbo/Cuprite from the final enumeration table, Atacama West here).
Rows are appended per (seed, criterion) to data/results/selection_resampling.csv (resumable).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).parent))
from hardneg_design import load_design, logb, BANDS

SEEDS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(0, 21))   # seed 0 = paper split (control)
OUT_W = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = Path("data/results/selection_resampling.csv")  # merged output; workers write their own file
R6F = Path("data/results/enum_logspace_m4_R6.csv.gz")
fam = pd.read_csv("data/results/enum_logspace_m4_all_geo_block_cuprite.csv.gz")


def parse(expr):
    c = np.zeros(10); num, den = (expr.split(" / ") + [""])[:2]
    for side, sg in [(num, 1), (den.strip("()"), -1)]:
        for t in filter(None, (x.strip() for x in side.split("·"))):
            b, _, p = t.partition("^"); c[BANDS.index(b)] += sg * (int(p) if p else 1)
    return c


C = np.vstack([parse(e) for e in fam.expr]); E = len(C)
clay = int(np.flatnonzero(fam.expr == "B11 / B12")[0])


def auc_all(Lp, Ln, w=None, batch=4000):
    """AUC of every expression, ties counted 1/2 (Mann-Whitney), fully vectorised:
    per batch, sort positive scores per column, shift column j by j*OFF so all columns live in
    one sorted 1-D array, and count positives below / tied with each negative with searchsorted.
    w: per-negative weights summing to 1 (weighted AUC as in enum_logspace), or None."""
    out = np.zeros(E); n1 = len(Lp)
    for s in range(0, E, batch):
        Cb = C[s:s + batch].T; k = Cb.shape[1]
        Sp, Sn = Lp @ Cb, Ln @ Cb
        lo = min(Sp.min(), Sn.min()); OFF = (max(Sp.max(), Sn.max()) - lo) + 1.0
        sh = np.arange(k) * OFF - lo
        flat = np.sort(Sp, axis=0).T.ravel() + np.repeat(sh, n1)
        q = (Sn + sh).T.ravel()
        base = np.repeat(np.arange(k) * n1, Sn.shape[0])
        lt = np.searchsorted(flat, q, "left") - base
        le = np.searchsorted(flat, q, "right") - base
        frac = (1 - (lt + le) / (2 * n1)).reshape(k, Sn.shape[0])   # P(pos > neg) + 0.5 P(tie)
        out[s:s + k] = frac.mean(1) if w is None else frac @ w
    return out


# Atacama West per-expression AUC (once)
if not R6F.exists():
    t = time.time()
    pos = pd.read_csv("data/ground_truth/site6_positives_geo.csv"); neg = pd.read_csv("data/ground_truth/site6_hard_negatives.csv")
    pos = pos[(pos[BANDS] > 0).all(axis=1)]; neg = neg[(neg[BANDS] > 0).all(axis=1)]
    Lp = logb(pos)
    a_ring = auc_all(Lp, logb(neg[neg.group == "ring"]))
    a_hard = auc_all(Lp, logb(neg[~neg.group.isin(["vegetation", "water"])]))
    pd.DataFrame({"expr": fam.expr, "auc_R6_ring_raw": a_ring, "auc_R6_hard_raw": a_hard}).to_csv(R6F, index=False, compression="gzip")
    print(f"R6 family AUC computed ({time.time()-t:.0f}s)", flush=True)
r6 = pd.read_csv(R6F)
# raw (unoriented) test AUCs; fam stores oriented AUCs + sign from the final selection, so un-orient them
sign_final = fam["sign"].values
TEST = {"R4_ring": np.where(sign_final > 0, fam.auc_R4_ring, 1 - fam.auc_R4_ring),
        "R4_hard": np.where(sign_final > 0, fam.auc_R4_hard, 1 - fam.auc_R4_hard),
        "R5_ring": np.where(sign_final > 0, fam.auc_R5_ring, 1 - fam.auc_R5_ring),
        "R5_hard": np.where(sign_final > 0, fam.auc_R5_hard, 1 - fam.auc_R5_hard),
        "R6_ring": r6.auc_R6_ring_raw.values, "R6_hard": r6.auc_R6_hard_raw.values}

OUT = OUT_W or OUT
done = set()
if OUT.exists():
    d0 = pd.read_csv(OUT); done = set(zip(d0.seed, d0.criterion))
old = pd.read_csv("data/ground_truth/binary_training_s2.csv"); oldneg = logb(old[old.is_altered == 0])
for seed in SEEDS:
    if all((seed, c) in done for c in ("boxes", "ring", "all")): continue
    t = time.time()
    D = load_design(seed=seed)
    P, Nall = D["sets"]["sel"]; Lp = logb(P)
    crit = {}
    crit["all"] = auc_all(Lp, logb(Nall), w=D["sel_w"])
    crit["ring"] = auc_all(Lp, logb(D["sel_ring"]))
    # distant boxes have no coordinates: pixel split with the same seed
    rng = np.random.default_rng(seed); m = rng.random(len(oldneg)) < 0.5
    crit["boxes"] = auc_all(Lp, oldneg[m])
    rows = []
    for c, a in crit.items():
        sg = np.where(a >= 0.5, 1.0, -1.0); ao = np.where(sg > 0, a, 1 - a); w = int(np.argmax(ao))
        rec = {"seed": seed, "criterion": c, "winner": fam.expr[w], "m": int(fam.m[w]), "auc_sel": ao[w]}
        for k, v in TEST.items():
            vw = v[w] if sg[w] > 0 else 1 - v[w]
            vc = max(v[clay], 1 - v[clay])  # clay ratio, B11/B12 high = altered
            rec[f"auc_{k}"] = vw; rec[f"delta_{k}"] = vw - vc
        rec["gate_R4"] = rec["delta_R4_ring"] >= 0.03 and rec["delta_R4_hard"] >= 0.03
        rec["gate_R5"] = rec["delta_R5_ring"] >= 0.03 and rec["delta_R5_hard"] >= 0.03
        rec["gate_R6"] = rec["delta_R6_ring"] >= 0.03 and rec["delta_R6_hard"] >= 0.03
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"seed {seed:2d} ({time.time()-t:.0f}s): " + " | ".join(f"{r['criterion']}: {r['winner']} "
          f"ΔR4 {r['delta_R4_hard']:+.3f} ΔR5 {r['delta_R5_hard']:+.3f} ΔR6 {r['delta_R6_hard']:+.3f}" for r in rows), flush=True)
print("done")
