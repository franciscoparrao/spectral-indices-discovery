#!/usr/bin/env python3
"""Logistic (9 log-ratios) and random forests (raw bands, log-ratios) trained on the Atacama
selection half exactly as final_design_baselines.py, evaluated at R6 (Atacama West) with 10 km
block-bootstrap CIs. Output: data/results/site6_models.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from hardneg_design import load_design, logb, block_id, BANDS
RNG = np.random.default_rng(0)
D = load_design(); P, N = D["sets"]["sel"]
lr = lambda d: logb(d) - logb(d)[:, [BANDS.index("B08")]]
y = np.r_[np.ones(len(P)), np.zeros(len(N))]; w = np.r_[np.full(len(P), 1 / len(P)), D["sel_w"]]; w = w / w.sum() * len(w)
M = {"logistic 9 log-ratios": (lr, LogisticRegression(C=1.0, max_iter=2000)),
     "RF 10 raw bands": (lambda d: d[BANDS].values, RandomForestClassifier(300, min_samples_leaf=5, n_jobs=6, random_state=0)),
     "RF 9 log-ratios": (lr, RandomForestClassifier(300, min_samples_leaf=5, n_jobs=6, random_state=0))}
for f, m in M.values(): m.fit(np.vstack([f(P), f(N)]), y, sample_weight=w)
pos = pd.read_csv("data/ground_truth/site6_positives_geo.csv"); neg = pd.read_csv("data/ground_truth/site6_hard_negatives.csv")
pos = pos[(pos[BANDS] > 0).all(axis=1)]; neg = neg[(neg[BANDS] > 0).all(axis=1)]
out = {}
for name, Nd in [("R6_ring", neg[neg.group == "ring"]), ("R6_hard", neg[~neg.group.isin(["vegetation", "water"])])]:
    yy = np.r_[np.ones(len(pos)), np.zeros(len(Nd))]
    b = np.r_[block_id(pos.lon, pos.lat, 10), block_id(Nd.lon, Nd.lat, 10)]; ub, inv = np.unique(b, return_inverse=True)
    mem = [np.flatnonzero(inv == i) for i in range(len(ub))]; out[name] = {}
    for k, (f, m) in M.items():
        s = np.r_[m.predict_proba(f(pos))[:, 1], m.predict_proba(f(Nd))[:, 1]]; reps = []
        for _ in range(1000):
            idx = np.concatenate([mem[i] for i in RNG.integers(0, len(ub), len(ub))])
            if yy[idx].min() != yy[idx].max(): reps.append(roc_auc_score(yy[idx], s[idx]))
        out[name][k] = {"auc": roc_auc_score(yy, s), "ci95": np.quantile(reps, [.025, .975]).tolist()}
        print(f"{name} {k:24s} {out[name][k]['auc']:.3f} [{out[name][k]['ci95'][0]:.3f},{out[name][k]['ci95'][1]:.3f}]")
json.dump(out, open("data/results/site6_models.json", "w"), indent=1)
