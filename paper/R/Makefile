# Makefile to regenerate every figure of the NRR paper from data sources.
# Run from project root: make -C paper/R all
#
# Pipeline:
#   data sources --> (optional) Python GEE prep --> R fig_*.R --> PDF + PNG
#
# Dependencies: R 4.3+, the packages declared in theme_paper.R, and (for spatial
# figures) data/maps/*.tif produced by ../../scripts/prep_rasters_for_R.py.

R = Rscript
PYTHON = python3
ROOT = ../..
FIGS = $(ROOT)/figures
DATA = $(ROOT)/data

# ---- Data prerequisites ----

$(DATA)/maps/maricunga_composite.tif $(DATA)/maps/cuprite_composite.tif:
	cd $(ROOT) && $(PYTHON) scripts/prep_rasters_for_R.py

$(DATA)/ground_truth/maricunga_training_s2.csv:
	cd $(ROOT) && $(PYTHON) -c "import numpy as np, pandas as pd; \
		d = np.load('data/ground_truth/maricunga_training_s2_gee.npz', allow_pickle=True); \
		pd.DataFrame(d['X'], columns=list(d['band_names'])).assign(y=d['y']).to_csv('data/ground_truth/maricunga_training_s2.csv', index=False)"

# ---- Individual figures ----

$(FIGS)/study_area.pdf: fig_study_area.R theme_paper.R
	cd $(ROOT) && $(R) paper/R/fig_study_area.R

$(FIGS)/spectral_profiles.pdf: fig_spectral_profiles.R theme_paper.R \
                                $(DATA)/ground_truth/maricunga_training_s2.csv
	cd $(ROOT) && $(R) paper/R/fig_spectral_profiles.R

$(FIGS)/pareto_fronts.pdf: fig_pareto_fronts.R theme_paper.R
	cd $(ROOT) && $(R) paper/R/fig_pareto_fronts.R

$(FIGS)/auc_heatmap.pdf: fig_auc_heatmap.R theme_paper.R
	cd $(ROOT) && $(R) paper/R/fig_auc_heatmap.R

$(FIGS)/physical_interpretation.pdf: fig_physical_interpretation.R theme_paper.R
	cd $(ROOT) && $(R) paper/R/fig_physical_interpretation.R

$(FIGS)/alteration_comparison.pdf: fig_alteration_comparison.R theme_paper.R \
                                    $(DATA)/maps/maricunga_composite.tif
	cd $(ROOT) && $(R) paper/R/fig_alteration_comparison.R

$(FIGS)/cuprite_comparison.pdf: fig_cuprite_comparison.R theme_paper.R \
                                 $(DATA)/maps/cuprite_composite.tif
	cd $(ROOT) && $(R) paper/R/fig_cuprite_comparison.R

# ---- Top-level targets ----

ALL_FIGS = \
  $(FIGS)/study_area.pdf \
  $(FIGS)/spectral_profiles.pdf \
  $(FIGS)/pareto_fronts.pdf \
  $(FIGS)/auc_heatmap.pdf \
  $(FIGS)/physical_interpretation.pdf \
  $(FIGS)/alteration_comparison.pdf \
  $(FIGS)/cuprite_comparison.pdf

all: $(ALL_FIGS)
	@echo "Built $(words $(ALL_FIGS)) figures."

# Copy PNG mirrors of the PDFs into paper/nrr_submission for the LaTeX includes
sync-submission: all
	@for fig in $(ALL_FIGS); do \
	  bn=$$(basename $$fig .pdf); \
	  cp $(FIGS)/$$bn.png $(ROOT)/paper/nrr_submission/$$bn.png; \
	  cp $(FIGS)/$$bn.pdf $(ROOT)/paper/nrr_submission/$$bn.pdf; \
	done
	@echo "Synced PDF+PNG to paper/nrr_submission/"

clean:
	rm -f $(ALL_FIGS) $(ALL_FIGS:.pdf=.png)

.PHONY: all clean sync-submission
