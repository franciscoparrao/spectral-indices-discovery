#!/usr/bin/env python3
"""
PILOT (2026-10): can a scale-invariant closed-form index beat the clay ratio
once negatives are adjacent host rock + named distractors, and across sites?

Selection set : random half of Region III positives vs Region III ring negatives.
Everything else (other half, lithology boxes, distractors, Region IV) is held out.
Threshold     : Youden on the selection set; applied unchanged everywhere else.

Gate (pre-registered in conversation): a scale-invariant combination must beat
B11/B12 by >= 0.03 AUC on Region IV (altered vs ring) AND on Region III hard
negatives, with no worse distractor false-positive rate.
"""
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

GT = Path("data/ground_truth"); RES = Path("data/results")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
RNG = np.random.default_rng(0)
eps = 1e-6

# ------------------------------------------------------------------ data
old = pd.read_csv(GT / "binary_training_s2.csv")
pos3 = old[old.is_altered == 1][BANDS].reset_index(drop=True)
oldneg3 = old[old.is_altered == 0][BANDS].reset_index(drop=True)
pos4 = pd.read_csv(GT / "region_iv_validation_s2.csv")[BANDS].reset_index(drop=True)
hard = pd.read_csv(GT / "pilot_hard_negatives.csv")
hard = hard[(hard[BANDS] > 0).all(axis=1)]

neg = {(s, g): d[BANDS].reset_index(drop=True) for (s, g), d in hard.groupby(["site", "group"])}
ring3, ring4 = neg[("R3", "ring")], neg[("R4", "ring")]

# selection / held-out split of R3 positives and ring
def split(df):
    m = RNG.random(len(df)) < 0.5
    return df[m].reset_index(drop=True), df[~m].reset_index(drop=True)
pos3_sel, pos3_ho = split(pos3)
ring3_sel, ring3_ho = split(ring3)

distract3 = {g: d for (s, g), d in neg.items() if s == "R3" and g != "ring"}
distract4 = {g: d for (s, g), d in neg.items() if s == "R4" and g != "ring"}
allneg3_ho = pd.concat([ring3_ho] + list(distract3.values()), ignore_index=True)
allneg4 = pd.concat([ring4] + list(distract4.values()), ignore_index=True)

# ------------------------------------------------------------------ candidates
def B(df): return {b: df[b].values for b in BANDS}
def r(a, b): return lambda d: d[a] / np.maximum(d[b], eps)

classical = {
    "clay B11/B12": r("B11", "B12"),
    "ferric B04/B02": r("B04", "B02"),
    "ferrous B11/B8A": r("B11", "B8A"),
    "NDSI (B03-B11)/(B03+B11)": lambda d: (d["B03"] - d["B11"]) / (d["B03"] + d["B11"] + eps),
    "albedo mean10": lambda d: np.mean([d[b] for b in BANDS], axis=0),
    "B04 alone": lambda d: d["B04"],
}
paper_sr = {
    "SR binary (sqrtB12-B11)^2": lambda d: (np.sqrt(d["B12"]) - d["B11"]) ** 2,
    "SR propylitic B03-0.48B11": lambda d: d["B03"] - 0.48 * d["B11"],
    "SR silicic B03^2/B12": lambda d: d["B03"] ** 2 / np.maximum(d["B12"], eps),
    "SR adv.arg 0.83-B02/B05": lambda d: 0.83 - d["B02"] / np.maximum(d["B05"], eps),
    "SR argillic (B03+.105)/B04-.83": lambda d: (d["B03"] + 0.105) / np.maximum(d["B04"], eps) - 0.83,
    "SR potassic B03B12/B07^2-.45": lambda d: d["B03"] * d["B12"] / np.maximum(d["B07"] ** 2, eps) - 0.45,
}
pairs = list(itertools.combinations(BANDS, 2))
brute = {f"{a}/{b}": r(a, b) for a, b in pairs}
for (a, b), (c, d) in itertools.combinations(pairs, 2):
    brute[f"({a}/{b})*({c}/{d})"] = (lambda a, b, c, d: lambda x: x[a] / np.maximum(x[b], eps) * x[c] / np.maximum(x[d], eps))(a, b, c, d)
    brute[f"({a}/{b})/({c}/{d})"] = (lambda a, b, c, d: lambda x: x[a] / np.maximum(x[b], eps) / np.maximum(x[c] / np.maximum(x[d], eps), eps))(a, b, c, d)

# ------------------------------------------------------------------ evaluation
def auc(sp, sn):
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; s = np.r_[sp, sn]
    ok = np.isfinite(s); return roc_auc_score(y[ok], s[ok])

def orient_and_threshold(f):
    """Sign so that altered scores high (on selection set); Youden threshold there."""
    sp, sn = f(B(pos3_sel)), f(B(ring3_sel))
    sign = 1.0 if auc(sp, sn) >= 0.5 else -1.0
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; s = sign * np.r_[sp, sn]
    fpr, tpr, thr = roc_curve(y, s); t = thr[np.argmax(tpr - fpr)]
    return sign, t

def evaluate(f, full=True):
    sign, t = orient_and_threshold(f)
    g = lambda df: sign * f(B(df))
    out = {"auc_R3_sel_ring": auc(g(pos3_sel), g(ring3_sel)),
           "auc_R3_ho_ring": auc(g(pos3_ho), g(ring3_ho)),
           "auc_R3_ho_hard": auc(g(pos3_ho), g(allneg3_ho)),
           "auc_R3_oldboxes": auc(g(pos3_ho), g(oldneg3)),
           "auc_R4_ring": auc(g(pos4), g(ring4)),
           "auc_R4_hard": auc(g(pos4), g(allneg4))}
    if full:
        out["recall_R3_ho@t"] = float((g(pos3_ho) > t).mean())
        out["recall_R4@t"] = float((g(pos4) > t).mean())
        out["fpr@t"] = {f"R3_{k}": float((g(d) > t).mean()) for k, d in {"ring": ring3_ho, **distract3}.items()}
        out["fpr@t"].update({f"R4_{k}": float((g(d) > t).mean()) for k, d in {"ring": ring4, **distract4}.items()})
        out["fpr_distractors_mean"] = float(np.mean([v for k, v in out["fpr@t"].items() if "ring" not in k]))
    return out

print(f"R3 positives {len(pos3)} (sel {len(pos3_sel)} / ho {len(pos3_ho)}), ring {len(ring3)}; "
      f"R3 distractor groups {len(distract3)} ({sum(len(d) for d in distract3.values())} px); "
      f"R4 positives {len(pos4)}, ring {len(ring4)}, distractors {sum(len(d) for d in distract4.values())} px\n")

results = {"reference": {}, "paper_sr": {}, "brute_top": {}, "models": {}}
for k, f in classical.items(): results["reference"][k] = evaluate(f)
for k, f in paper_sr.items(): results["paper_sr"][k] = evaluate(f)

# brute force: rank on SELECTION set only, then evaluate the top ones held-out
sel_auc = {k: evaluate(f, full=False)["auc_R3_sel_ring"] for k, f in brute.items()}
top = sorted(sel_auc, key=sel_auc.get, reverse=True)[:15]
for k in top: results["brute_top"][k] = evaluate(brute[k])

# model ceilings: train on selection set (altered vs ring), test held-out
def feats_logratio(df):
    L = np.log(np.clip(df[BANDS].values, 1e-4, None)); return L - L[:, [BANDS.index("B08")]]
def feats_raw(df): return df[BANDS].values
Xs = {"logistic 9 log-ratios": (LogisticRegression(max_iter=3000), feats_logratio),
      "RF 9 log-ratios": (RandomForestClassifier(300, n_jobs=-1, random_state=0), feats_logratio),
      "RF 10 raw bands": (RandomForestClassifier(300, n_jobs=-1, random_state=0), feats_raw)}
for k, (m, fx) in Xs.items():
    X = np.vstack([fx(pos3_sel), fx(ring3_sel)]); y = np.r_[np.ones(len(pos3_sel)), np.zeros(len(ring3_sel))]
    m.fit(X, y); p = lambda df: m.predict_proba(fx(df))[:, 1]
    results["models"][k] = {"auc_R3_ho_ring": auc(p(pos3_ho), p(ring3_ho)),
                            "auc_R3_ho_hard": auc(p(pos3_ho), p(allneg3_ho)),
                            "auc_R4_ring": auc(p(pos4), p(ring4)), "auc_R4_hard": auc(p(pos4), p(allneg4))}

# R4 in-site ceiling: is the cross-site collapse a transfer problem or a Region IV separability problem?
from sklearn.model_selection import StratifiedKFold, cross_val_predict
X4 = np.vstack([feats_logratio(pos4), feats_logratio(ring4)]); y4 = np.r_[np.ones(len(pos4)), np.zeros(len(ring4))]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
results["R4_insite"] = {
    "logistic 9 log-ratios (5-fold)": roc_auc_score(y4, cross_val_predict(LogisticRegression(max_iter=3000), X4, y4, cv=cv, method="predict_proba")[:, 1]),
    "RF 10 raw bands (5-fold)": roc_auc_score(y4, cross_val_predict(RandomForestClassifier(300, n_jobs=-1, random_state=0),
                                              np.vstack([feats_raw(pos4), feats_raw(ring4)]), y4, cv=cv, method="predict_proba")[:, 1]),
    "clay B11/B12": auc(classical["clay B11/B12"](B(pos4)), classical["clay B11/B12"](B(ring4))),
    "albedo mean10": auc(classical["albedo mean10"](B(pos4)), classical["albedo mean10"](B(ring4))),
}
# ------------------------------------------------------------------ report
cols = ["auc_R3_oldboxes", "auc_R3_ho_ring", "auc_R3_ho_hard", "auc_R4_ring", "auc_R4_hard", "recall_R4@t", "fpr_distractors_mean"]
def table(title, d):
    print(f"\n== {title}")
    print(f"{'index':34s} " + " ".join(f"{c.replace('auc_','').replace('_ho',''):>12s}" for c in cols))
    for k, v in d.items():
        print(f"{k[:34]:34s} " + " ".join(f"{v.get(c, float('nan')):12.3f}" for c in cols))
table("Reference indices (threshold = Youden on R3 selection set)", results["reference"])
table("Paper SR indices", results["paper_sr"])
table("Brute-force top-15 by SELECTION AUC (ratios, ratio products/quotients)", results["brute_top"])
print("\n== Model ceilings (trained on R3 selection set)")
for k, v in results["models"].items(): print(f"{k:34s} " + " ".join(f"{c}={v[c]:.3f}" for c in v))

print("\n== Region IV IN-SITE (altered vs ring, no transfer)")
for k, v in results["R4_insite"].items(): print(f"{k:34s} {v:.3f}")

print("\n== False-positive rate per group at the R3-calibrated threshold")
groups = list(results["reference"]["clay B11/B12"]["fpr@t"].keys())
show = {"clay B11/B12": results["reference"]["clay B11/B12"], "SR binary (sqrtB12-B11)^2": results["paper_sr"]["SR binary (sqrtB12-B11)^2"],
        "SR propylitic B03-0.48B11": results["paper_sr"]["SR propylitic B03-0.48B11"], "albedo mean10": results["reference"]["albedo mean10"]}
for k in top[:3]: show[k] = results["brute_top"][k]
print(f"{'group':20s} " + " ".join(f"{k[:14]:>14s}" for k in show))
for g in groups: print(f"{g:20s} " + " ".join(f"{v['fpr@t'][g]:14.2f}" for v in show.values()))

# gate
clay = results["reference"]["clay B11/B12"]; best_k = top[0]; best = results["brute_top"][best_k]
gate = {"best_scale_invariant": best_k,
        "delta_auc_R4_ring": best["auc_R4_ring"] - clay["auc_R4_ring"],
        "delta_auc_R3_hard": best["auc_R3_ho_hard"] - clay["auc_R3_ho_hard"],
        "fpr_distractors_best": best["fpr_distractors_mean"], "fpr_distractors_clay": clay["fpr_distractors_mean"]}
gate["pass"] = bool(gate["delta_auc_R4_ring"] >= 0.03 and gate["delta_auc_R3_hard"] >= 0.03
                    and gate["fpr_distractors_best"] <= gate["fpr_distractors_clay"])
results["gate"] = gate
print("\n== GATE:", json.dumps(gate, indent=1))
RES.mkdir(exist_ok=True)
json.dump(results, open(RES / "pilot_hard_negatives.json", "w"), indent=1)
print(f"\nSaved {RES/'pilot_hard_negatives.json'}")
