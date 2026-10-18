#!/usr/bin/env python3
"""
RQ1: does the selection criterion decide which index wins and whether it transfers?
Joins the per-expression AUC tables of three enumerations that differ ONLY in the
selection set (same 87 000 expressions, m<=4):
  boxes : Region III positives vs distant boxes (the protocol of the rejected paper)
  ring  : Region III positives vs host-rock ring (pilot)
  all   : Region III positives vs ring + distractors (final design)
and evaluates every winner on the SAME test sets of the final design
(enum_logspace_m4_all_geo_block_cuprite: R4 and R5 ring / hard). Applies the AUC part
of the transfer gate (>= clay + 0.03 in ring and hard, per site).
Output: data/results/selection_criterion_comparison.json
"""
import json
import pandas as pd
R = "data/results/"
final = pd.read_csv(R + "enum_logspace_m4_all_geo_block_cuprite.csv.gz")
TEST = ["auc_R4_ring", "auc_R4_hard", "auc_R5_ring", "auc_R5_hard"]
sel = {"boxes": pd.read_csv(R + "enum_logspace_m4_boxes.csv.gz")[["expr", "auc_sel"]],
       "ring": pd.read_csv(R + "enum_logspace_m4_hard.csv.gz")[["expr", "auc_sel"]],
       "all": final[["expr", "auc_sel"]]}
m = final[["expr", "m"] + TEST]
for k, d in sel.items():
    m = m.merge(d.rename(columns={"auc_sel": f"sel_{k}"}), on="expr")
assert len(m) == 87000, len(m)
clay = m[m.expr == "B11 / B12"].iloc[0]
out = {"n_expressions": len(m), "clay": {t: float(clay[t]) for t in TEST}, "criteria": {}}
for k in sel:
    w = m.loc[m[f"sel_{k}"].idxmax()]
    rec = {"winner": w.expr, "m": int(w.m), "auc_sel": float(w[f"sel_{k}"]),
           **{t: float(w[t]) for t in TEST},
           **{f"delta_{t[4:]}": float(w[t] - clay[t]) for t in TEST},
           "spearman_sel_vs": {t[4:]: float(m[f"sel_{k}"].corr(m[t], method="spearman")) for t in TEST}}
    rec["auc_gate_R4"] = bool(rec["delta_R4_ring"] >= 0.03 and rec["delta_R4_hard"] >= 0.03)
    rec["auc_gate_R5"] = bool(rec["delta_R5_ring"] >= 0.03 and rec["delta_R5_hard"] >= 0.03)
    out["criteria"][k] = rec
    print(f"{k:5s} {w.expr:38s} ΔR4 {rec['delta_R4_ring']:+.3f}/{rec['delta_R4_hard']:+.3f} "
          f"ΔR5 {rec['delta_R5_ring']:+.3f}/{rec['delta_R5_hard']:+.3f} gate R4 {rec['auc_gate_R4']} R5 {rec['auc_gate_R5']} "
          f"ρ(sel,R4hard) {rec['spearman_sel_vs']['R4_hard']:+.2f} ρ(sel,R5hard) {rec['spearman_sel_vs']['R5_hard']:+.2f}")
json.dump(out, open(R + "selection_criterion_comparison.json", "w"), indent=1)
print("saved", R + "selection_criterion_comparison.json")
