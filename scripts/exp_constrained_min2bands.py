#!/usr/bin/env python3
"""
Min-2-bands constrained selection (ESIN review Issue 2 / NRR R2).

Two of the reported formulas are monotone-equivalent to raw bands under an
AUC/threshold protocol: B04 - 0.135 == B04, and 0.09/B05 == -B05. Their fitted
constants are inert, so for those classes SR performed band selection rather
than discovering a multi-band contrast.

This experiment applies a selection constraint -- the returned expression must
use at least two DISTINCT bands -- to the Pareto fronts already produced by the
identical PySR searches (no re-run needed; the constraint acts on selection, not
on the search). It quantifies what the constraint costs:

  (A) standalone indices on the locked 30% hold-out   -> Table 3 counterpart
  (B) downstream RF under nested CV, both schemes     -> Table 6 counterpart

Output: data/results/constrained_min2bands.json
"""
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

BANDS = ['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']
CLASS_NAMES = {1:"Silicic",2:"Adv_Argillic",3:"Argillic_Phyllic",
               4:"Propylitic",5:"Iron_Oxide",6:"Potassic_Skarn"}
MAXCOMPLEX = 8
CSV = Path("data/ground_truth/maricunga_training_s2_gee.csv")
OUT = Path("data/results/constrained_min2bands.json")
BAND_RE = re.compile(r'B(?:02|03|04|05|06|07|08|11|12|8A)')


def bands_used(eq):
    return set(BAND_RE.findall(eq))


def evaluate(eq, X):
    env = {b: X[:, i] for i, b in enumerate(BANDS)}
    env.update({"sqrt": lambda x: np.sqrt(np.clip(x, 0, None)),
                "log": lambda x: np.log(np.maximum(x, 1e-9)),
                "square": np.square, "tanh": np.tanh, "exp": np.exp})
    expr = eq.replace("^", "**")
    if not re.fullmatch(r"[\w\s\.\+\-\*\/\(\)]+", expr):
        raise ValueError(f"unsafe: {eq}")
    return np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=float)


def pick(front, min_bands):
    """Max-score expression with complexity <= 8 and >= min_bands distinct bands."""
    cand = front[front.complexity <= MAXCOMPLEX].copy()
    cand = cand[cand.equation.map(lambda e: len(bands_used(e)) >= min_bands)]
    if cand.empty:
        return None
    return cand.loc[cand.score.idxmax()]


def classical(M):
    b = {n: i for i, n in enumerate(BANDS)}; eps = 1e-9; B = lambda n: M[:, b[n]]
    return np.column_stack([
        B('B11')/np.maximum(B('B12'),eps), B('B04')/np.maximum(B('B02'),eps),
        B('B12')/np.maximum(B('B8A'),eps),
        (B('B11')-B('B12'))/np.maximum(B('B11')+B('B12'),eps),
        B('B02')/np.maximum(B('B11'),eps), B('B12')/np.maximum(B('B11'),eps),
        (B('B8A')-B('B04'))/np.maximum(B('B8A')+B('B04'),eps)])


def polygons(y, lon, lat):
    pid = np.full(len(y), -1, int); nid = 0
    for c in sorted(np.unique(y)):
        m = y == c; co = np.column_stack([lon[m], lat[m]])
        if len(co) < 2:
            pid[m] = nid; nid += 1; continue
        cs = co.copy(); cs[:, 0] *= np.cos(np.deg2rad(co[:, 1].mean()))
        lab = DBSCAN(eps=0.001, min_samples=2).fit(cs).labels_
        for l in np.unique(lab):
            if l == -1:
                for i in np.where(m)[0][lab == -1]: pid[i] = nid; nid += 1
            else:
                pid[np.where(m)[0][lab == l]] = nid; nid += 1
    return pid


df = pd.read_csv(CSV)
X = df[BANDS].values; y = df['class_id'].values
poly = polygons(y, df['lon'].values, df['lat'].values)
res = {"selection_rule": f"max PySR score, complexity <= {MAXCOMPLEX}, >= 2 distinct bands",
       "note": "constraint applied at SELECTION over the same Pareto fronts; no PySR re-run"}

# ---------- (A) standalone on locked 30% hold-out ----------
X15, _, y15, _ = train_test_split(X, y, train_size=15000, stratify=y, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X15, y15, test_size=0.3, stratify=y15, random_state=42)
print(f"Locked hold-out: {len(yte)} px\n")
print(f"{'class':18s} {'unconstrained':40s} {'AUC':6s} | {'constrained (>=2 bands)':42s} {'AUC':6s}  d")
standalone = {}
for cid, name in CLASS_NAMES.items():
    f = pd.read_csv(f"data/results/equations_gee_{name}.csv")
    u, c = pick(f, 1), pick(f, 2)
    yb = (yte == cid).astype(int)
    row = {}
    for tag, sel in (("unconstrained", u), ("constrained", c)):
        if sel is None: row[tag] = None; continue
        s = evaluate(sel.equation, Xte)
        auc = max(roc_auc_score(yb, s), roc_auc_score(yb, -s))
        row[tag] = {"equation": sel.equation, "complexity": int(sel.complexity),
                    "loss": float(sel.loss), "auc": float(auc),
                    "n_bands": len(bands_used(sel.equation))}
    standalone[name] = row
    du, dc = row["unconstrained"], row["constrained"]
    d = (dc["auc"] - du["auc"]) if (du and dc) else float('nan')
    print(f"{name:18s} {du['equation'][:38]:40s} {du['auc']:.3f} | "
          f"{dc['equation'][:40]:42s} {dc['auc']:.3f}  {d:+.3f}")
res["standalone_locked_holdout"] = standalone

# ---------- (B) downstream RF under nested CV ----------
rf_kw = dict(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
cls_feats = classical(X); allc = sorted(np.unique(y))
res["nested"] = {}
print()
for scheme in ["stratified", "polygon"]:
    splits = (list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
              if scheme == "stratified" else list(GroupKFold(5).split(X, y, groups=poly)))
    acc = {k: [] for k in ["sr_unconstrained", "sr_constrained", "srC_classical"]}
    used_eqs = []
    for k, (tr, te) in enumerate(splits, 1):
        srU = np.zeros((len(y), 6)); srC = np.zeros((len(y), 6)); fold_eqs = {}
        for j, cid in enumerate(allc):
            p = Path(f"data/results/nested_cv_equations/{scheme}_fold{k}_{CLASS_NAMES[cid]}.csv")
            if not p.exists(): continue
            f = pd.read_csv(p)
            su, sc = pick(f, 1), pick(f, 2)
            if su is not None: srU[:, j] = evaluate(su.equation, X)
            if sc is not None:
                srC[:, j] = evaluate(sc.equation, X)
                fold_eqs[CLASS_NAMES[cid]] = sc.equation
        used_eqs.append(fold_eqs)
        for tag, F in (("sr_unconstrained", srU), ("sr_constrained", srC),
                       ("srC_classical", np.hstack([srC, cls_feats]))):
            rf = RandomForestClassifier(**rf_kw).fit(F[tr], y[tr])
            pr = rf.predict_proba(F[te]); a = []
            for cc in allc:
                yb = (y[te] == cc).astype(int)
                if 0 < yb.sum() < len(yb) and cc in rf.classes_:
                    a.append(roc_auc_score(yb, pr[:, list(rf.classes_).index(cc)]))
            acc[tag].append(float(np.mean(a)))
    res["nested"][scheme] = {t: {"per_fold": [round(v,4) for v in vs],
                                 "mean": float(np.mean(vs))} for t, vs in acc.items()}
    res["nested"][scheme]["equations_per_fold"] = used_eqs
    m = res["nested"][scheme]
    print(f"[{scheme}] SR uncon={m['sr_unconstrained']['mean']:.4f}  "
          f"SR con={m['sr_constrained']['mean']:.4f}  "
          f"SRcon+classical={m['srC_classical']['mean']:.4f}  "
          f"(delta con-uncon = {m['sr_constrained']['mean']-m['sr_unconstrained']['mean']:+.4f})")

OUT.write_text(json.dumps(res, indent=2))
print(f"\nSaved -> {OUT}")
