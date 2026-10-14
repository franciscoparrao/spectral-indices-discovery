#!/usr/bin/env python3
"""
Evaluate the fixed indices on the alternative composites sampled by
temporal_stability_sample.py. Labels, sets and orientation are those of the final
design (hardneg_design.load_design); only the reflectance values change.

Per site x period x set : AUC of each closed-form index.
Per site x period       : Spearman correlation of each index between base and alternative
                          composite (same pixels), share of points with valid values.
Base period re-sample   : median relative difference against the stored reflectances
                          (checks whether the GEE collection moved since the design was built).
Output: data/results/temporal_stability.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from hardneg_design import load_design, BANDS, INDICES

TDIR = Path("data/ground_truth/temporal"); OUT = Path("data/results/temporal_stability.json")
D = load_design(); sets = D["sets"]
NAMES = ["clay B11/B12", "winner m4", "winner m5", "m2 B02·B12/(B03·B11)", "paper SR (sqrtB12-B11)^2"]
EVAL = {"R3": ["R3_ho_ring", "R3_ho_hard"], "R4": ["R4_ring", "R4_hard"], "R5": ["R5_ring", "R5_hard"]}
key = lambda df: pd.Series(list(zip(df.lon.round(6), df.lat.round(6))), index=df.index)

# orientation fixed on the selection set of the base design
p, n = sets["sel"]
orient = {}
for name in NAMES:
    s = np.r_[INDICES[name](p), INDICES[name](n)]; y = np.r_[np.ones(len(p)), np.zeros(len(n))]
    orient[name] = 1.0 if roc_auc_score(y, s, sample_weight=np.r_[np.full(len(p), 1 / len(p)), D["sel_w"]]) >= 0.5 else -1.0

out = {"orientation": orient, "sites": {}}
for site, evals in EVAL.items():
    files = sorted(TDIR.glob(f"{site}_*.csv"))
    if not files: continue
    out["sites"][site] = {}
    base = None
    for f in files:
        period = f.stem.split("_", 1)[1]
        t = pd.read_csv(f); t["k"] = key(t)
        valid = t[BANDS].notna().all(axis=1) & (t[BANDS] > 0).all(axis=1)
        t = t[valid].drop_duplicates("k").set_index("k")
        rec = {"valid_share": float(valid.mean()), "n_scenes": None, "auc": {}, "spearman_vs_base": {}}
        for k in evals:
            P, N = sets[k]
            Pk, Nk = key(P), key(N)
            Pt, Nt = t.reindex(Pk.values), t.reindex(Nk.values)
            ok = np.r_[Pt[BANDS].notna().all(axis=1).values, Nt[BANDS].notna().all(axis=1).values]
            y = np.r_[np.ones(len(P)), np.zeros(len(N))][ok]
            rec["auc"][k] = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
            for name in NAMES:
                s = orient[name] * np.r_[INDICES[name](Pt), INDICES[name](Nt)][ok]
                rec["auc"][k][name] = float(roc_auc_score(y, s))
        if period == "base":
            base = t
            # consistency against stored reflectance (positives + negatives of this site)
            stored = pd.concat([P.assign(_s="p") for kk in evals for P in sets[kk][:1]] + [sets[evals[0]][1]], ignore_index=True)
            stored = stored.drop_duplicates(subset=["lon", "lat"]); stored.index = key(stored)
            common = stored.index.intersection(t.index)
            rel = (np.abs(t.loc[common, BANDS].values - stored.loc[common, BANDS].values) / stored.loc[common, BANDS].values)
            rec["base_vs_stored"] = {"n_common": int(len(common)), "median_rel_diff": float(np.median(rel)),
                                     "share_rel_diff_gt_1pct": float((rel > 0.01).mean())}
        elif base is not None:
            common = base.index.intersection(t.index)
            for name in NAMES:
                a, b = INDICES[name](base.loc[common]), INDICES[name](t.loc[common])
                rec["spearman_vs_base"][name] = float(spearmanr(a, b).statistic)
        out["sites"][site][period] = rec
        print(f"{site} {period:12s} valid {rec['valid_share']:.3f} | " +
              " | ".join(f"{k}: " + " ".join(f"{nm.split()[0][:7]}={rec['auc'][k][nm]:.3f}" for nm in NAMES[:3]) for k in evals), flush=True)
        if "base_vs_stored" in rec: print("   base vs stored:", rec["base_vs_stored"])
        if rec["spearman_vs_base"]: print("   spearman vs base:", {k: round(v, 3) for k, v in rec["spearman_vs_base"].items()})
OUT.write_text(json.dumps(out, indent=1)); print("saved", OUT)
