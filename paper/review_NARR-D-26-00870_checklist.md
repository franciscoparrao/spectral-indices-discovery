# Reviewer checklist — NARR-D-26-00870

**Manuscript**: "Sparse Spectral Unmixing with Nonlinear Mixing Model Applied to HISUI Imagery for Accurate Identification and Mapping of Hydrothermal Alteration Minerals"
**Journal**: Natural Resources Research (Springer). **EIC**: Carranza.
**Deadline**: 30 days from acceptance.
**Your angle**: Cuprite validation + multispectral SR expertise. You are NOT an unmixing theory expert; frame reviewer voice as *applied remote-sensing methodologist*, not *sparse-optimization mathematician*.

---

## 0. First pass — read abstract + figures + captions + tables only (60 min)

Before touching the equations, decide the top-level story and pin the claims:

- [ ] **Central claim**: "1.4–1.5× higher κ than LMM-SUnSAL, across HISUI + AVIRIS + Hyperion + WV3, at Cuprite."
- [ ] **Novel contribution**: LCRM (log + continuum removal) + sparse regularization (L1 SUnSAL / L2,1 CLSUnSAL). Is the novelty the *combination*, or a *new solver*, or a *new physical model*?
- [ ] **Failure mode acknowledged**: noise amplification, worst for calcite. Is this the only limitation, or is it a symptom of a deeper issue?

Pin what MUST be true for the paper to hold up:
1. LCRM preprocessing genuinely captures nonlinear mixing better than LMM.
2. The sparse regularization on the preprocessed signal is the right way to invert.
3. κ improvement is real, not a baseline-implementation artefact.
4. Cross-sensor consistency claim (HISUI + AVIRIS + Hyperion) is not driven by shared preprocessing pipeline.

If any of the four collapses, most of the paper collapses.

---

## 1. Novelty & prior art (highest-leverage red flags)

- [ ] **LCRM is not new**. Log transform + continuum removal is standard hyperspectral preprocessing (Clark & Roush 1984, Kruse et al. 1993). Ask: what is genuinely new about *this* LCRM formulation vs. classical CR? Check §Methods for a clear mathematical statement of what LCRM adds.
- [ ] **SUnSAL/CLSUnSAL are well-established** (Iordache, Bioucas-Dias & Plaza 2011, 2014). Applying them to a preprocessed signal is a preprocessing choice, not a new unmixing algorithm. If the authors present it as a new algorithm, push back.
- [ ] **Nonlinear mixing literature** — check that they engage with:
  - Halimi et al. (bilinear / MLM models)
  - Heylen et al. (nonlinear unmixing survey)
  - Dobigeon et al. (Bayesian nonlinear unmixing)
  - Zare & Ho 2014 (endmember variability review)
  If they cite only the linear-unmixing pantheon, they've missed a decade.
- [ ] **The CR paradox**: continuum removal was invented precisely to *linearise* absorption-feature mixing so linear unmixing works. Claiming that CR + sparse LMM = "nonlinear mixing model" is conceptually slippery. Force the authors to justify why applying LMM in the CR-domain is *nonlinear*, and to distinguish this from bilinear/multi-linear models.

## 2. Methodological rigor

- [ ] **Spectral library**: which one? Size? USGS SPECPR? ECOSTRESS? Curated subset? Endmember library size directly affects the L1/L2,1 sparsity behaviour.
- [ ] **Regularization hyperparameter** (λ for L1, and for the L2,1 group): how selected? Cross-validated on ground truth? Fixed empirically? If fixed by hand-tuning against the same test set, that's optimistic.
- [ ] **Preprocessing consistency across sensors**: was every dataset atmospherically corrected the same way? HISUI L2A? AVIRIS through ATREM/ACORN? Hyperion FLAASH? WV3 how? Different atmospheric corrections → different residual absorption features → different LCRM behaviour. If the paper doesn't declare this, ask for the pipeline.
- [ ] **Spatial resampling**: HISUI 20-30 m, AVIRIS 15-20 m, Hyperion 30 m, WV3 ~1.24 m MS / 3.7 m SWIR. Were they resampled to a common grid? How? Nearest-neighbour would destroy the CR signal; bilinear/cubic would smooth spectra.
- [ ] **WV3 multispectral in the same framework**: WV3 has 4 SWIR bands vs ~100 useful SWIR bands in HISUI/AVIRIS/Hyperion. Continuum removal on 4 SWIR bands is not the same operation as on 100. Is the LCRM step even well-defined on WV3? If yes, why did they include WV3?
- [ ] **Nonlinear model formalism**: is there an explicit forward model (r = f(a; E) + n) written down, or does "nonlinear" just mean "we applied a log"? A log is a nonlinear preprocessing, not a nonlinear mixing model. Push for the equation.

## 3. Validation & statistics

- [ ] **Ground truth for Cuprite**: which reference product? Rockwell 2017 (USGS DataSeries)? Swayze 2014 (USGS 2014-5100)? Clark et al. Tetracorder maps? Each has different mineral classes and coverage. If they don't name it precisely, request.
- [ ] **κ definition** in an unmixing context: unmixing produces abundance maps (continuous), not classification (categorical). κ presupposes a categorical prediction. So they must be doing hard-assignment (argmax over endmember abundances) or thresholding. That collapses their unmixing signal into a classifier. Two consequences:
  1. The "1.4–1.5×" improvement in κ is a *classification* improvement, not an *unmixing* improvement. Reframing needed.
  2. Standard unmixing metrics (RMSE on abundance, spectral angle mapper, reconstruction error) should also be reported. If they aren't, ask.
- [ ] **Uncertainty and significance**: is the κ improvement reported with a confidence interval? Bootstrap, McNemar test, or paired κ over CV folds? A 1.4–1.5× ratio without CIs is a point estimate; on hundreds of pixels it may or may not be significant.
- [ ] **Baseline choice**: only LMM-SUnSAL is compared. Reasonable additions:
  - **VCA + FCLS** (unregularised, non-sparse — the workhorse)
  - **N-FINDR + FCLS**
  - **Nonlinear MLM / bilinear** (would test whether *their* nonlinear model beats *established* nonlinear models)
  - Deep unmixing (autoencoder-based) — optional but current
  If they compare only against LMM-SUnSAL and win, the natural reviewer question is: "did you beat LMM because you preprocessed, or because you regularised?" Ask for the LMM-SUnSAL + CR baseline (LMM on the *same* CR-preprocessed signal). If that ablation is missing, it's a major hole.
- [ ] **Confusion matrix**: is it per-mineral? Diagonal vs off-diagonal per-class error rates matter more than aggregate κ.
- [ ] **Cross-sensor consistency**: they claim the method produces consistent maps across HISUI/AVIRIS/Hyperion. How is *consistency* quantified? Pixel-by-pixel agreement (Cohen's κ between two sensor maps)? Just visual? Push for a number.

## 4. Reproducibility

- [ ] **Code**: is there a repo? SUnSAL/CLSUnSAL implementations exist (Iordache MATLAB code). If theirs is a wrapper + preprocessing, releasing it is trivial.
- [ ] **Data**: HISUI is JAXA — check the data policy. Is the exact scene ID + processing level reported? AVIRIS + Hyperion are public (JPL / USGS). WV3 is commercial.
- [ ] **Spectral library**: cite exact version and subset used.
- [ ] **Hyperparameter values**: report λ for L1 and L2,1 explicitly.

## 5. Cuprite-specific (your expertise)

- [ ] **Alteration nomenclature**: check they don't conflate *mineral identification* (their claim, hyperspectral scale) with *alteration-zone mapping* (broader classes, what your paper does with S2). Both are legitimate, but they must be honest about which one κ is measuring.
- [ ] **USGS Rockwell 2017 reference**: is it used as ground truth or just as a visual comparator? Rockwell is derived from ASTER + Tetracorder; using it as ground truth for HISUI validation is not neutral. If they treat it as gold-standard, flag that Rockwell itself has classification uncertainty.
- [ ] **Cuprite has been beaten to death**. Its role in the paper is a benchmark, not a discovery site. Fine — but the paper should acknowledge this and cite the last 10 years of Cuprite unmixing work (Ma et al., Kruse et al., Rockwell, Swayze, Halimi, Bioucas-Dias). If the lit review is thin, request.
- [ ] **Calcite noise amplification** — they flag this. Calcite absorption is centred at ~2.34 μm and is relatively broad and shallow. CR division by a small continuum near a shallow feature blows up noise. This is a known CR pathology, not a new discovery. Ask whether they discuss it in context or present it as novel.

## 6. Writing & framing

- [ ] **Title mismatch check**: title says "HISUI Imagery" but abstract says "three hyperspectral datasets (HISUI, AVIRIS, and Hyperion) and one multispectral (WV3)". Title oversells HISUI-specificity. Either the paper is a HISUI application (drop AVIRIS/Hyperion/WV3 to the discussion), or it's a cross-sensor benchmark (retitle). Cannot be both.
- [ ] **"Accurate identification and mapping"** in the title — a strong claim. "Accurate" against what? If the reference is Rockwell 2017 (itself a remote sensing product), accuracy is relative agreement, not ground-truth accuracy. Soften request.
- [ ] **Kappa reporting**: what are the *absolute* κ values (not just ratios)? A ratio of 1.5 could mean 0.20 → 0.30 (both poor) or 0.55 → 0.83 (poor → strong). Very different stories.

## 7. NRR fit

- [ ] **Natural Resources positioning**: NRR is about resource exploration, not sensor benchmarking. Does the paper explicitly discuss the value for exploration (target mineralogy for porphyry Cu, epithermal Au, etc.)? If it stays purely at "sensor + method", it's more a *Remote Sensing* or *IJRS* fit than NRR. Not a rejection reason, but flag for the AE.

## 8. Recommendation logic (Chile-honest)

Map issues into the four recommendation buckets before writing:

- **Reject / Reject-and-resubmit** if: (a) LCRM turns out to be a repackaging of standard CR + LMM in disguise, (b) LMM-SUnSAL + CR ablation is missing and would erase the 1.5× gain, (c) κ is used but the manuscript is fundamentally unmixing (metric mismatch) and the authors do not report abundance-space metrics.
- **Major revision** if: (a) the novelty story survives but needs sharper framing, (b) ablations and CIs missing, (c) validation methodology needs clarification (which Cuprite reference, how κ is computed, cross-sensor consistency quantified), (d) title/scope misaligned.
- **Minor revision** if: only baseline-choice, additional citations, and clarifications remain.
- **Accept** as-is: extremely unlikely on a submitted (not revised) NRR manuscript.

Default in your first cycle: **Major revision**, unless something in §1 or §3 kills it (then Reject-and-resubmit).

---

## 9. Reviewer-report scaffolding

Write in this order to keep the tone constructive:

1. **Summary** (3–4 sentences): what the paper does, what claim, why it matters. Show you read it.
2. **General comments** (1 paragraph): where the paper stands relative to the field. Positive framing, then the biggest concern.
3. **Major issues** (numbered, 3–6 items): the novelty framing, the ablation, the κ-in-unmixing question, cross-sensor consistency quantification, baseline choice. One paragraph per item.
4. **Minor issues** (bulleted): title mismatch, citations missing, absolute κ values, calcite noise as known pathology.
5. **Recommendation** and confidence statement.

Sign convention: NRR is *single-anonymised* — the authors see your report but not your identity. Write firmly but respectfully. No sarcasm, no rhetorical questions.

## 10. Time budget (30 days)

- Day 1–2: first pass (abstract + figures + tables + conclusion). Decide the top-level story.
- Day 3–7: read Methods carefully with this checklist open.
- Day 8–12: chase the 5–8 most important prior-art citations.
- Day 13–20: buffer / re-read.
- Day 21–28: draft the report.
- Day 29–30: polish and submit.

Do the first pass this week while your own submission is fresh in EM — you'll be efficient. Don't leave it to day 28.
