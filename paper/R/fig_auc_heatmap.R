# fig_auc_heatmap.R --- Cross-site validation AUC heatmap (methods x classes)
# Migration of scripts/fig_auc_comparison.py (the heatmap part).
# Run: Rscript paper/R/fig_auc_heatmap.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(scico)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

RESULTS <- "data/results/full_evaluation.json"
OUT_PDF <- "figures/auc_heatmap.pdf"

# ---- Load + tidy ----

raw   <- read_json(RESULTS, simplifyVector = TRUE)
cross <- raw$cross_site

# Method definitions: (key in json, display label, family)
methods <- tibble::tribble(
  ~key,                                         ~label,                                                 ~family,
  "Clay Ratio (B11/B12)",                        "[Classical] Clay Ratio (B11/B12)",                     "Classical",
  "Iron Oxide (B04/B02)",                        "[Classical] Iron Oxide (B04/B02)",                     "Classical",
  "Ferrous (B12/B8A)",                           "[Classical] Ferrous (B12/B8A)",                        "Classical",
  "Alunite Idx (B11-B12)/(B11+B12)",             "[Classical] Alunite Idx (ASTER nomenclature)",         "Classical",
  "OH Minerals (B02/B11)",                       "[Classical] OH Minerals (ASTER nomenclature)",         "Classical",
  "Silica (B12/B11)",                            "[Classical] Silica (ASTER nomenclature)",              "Classical",
  "NDVI",                                        "[Classical] NDVI",                                     "Classical",
  "SR: B04 - 0.135",                             "[SR] B04 − 0.135 (Silicic)",                           "SR",
  "SR: 0.83 - B02/B05",                          "[SR] 0.83 − B02/B05 (Adv. Argillic)",                  "SR",
  "SR: 0.09/B05",                                "[SR] 0.09/B05 (Argillic-Phyllic)",                     "SR",
  "SR: B03 - B11*0.48",                          "[SR] B03 − 0.48·B11 (Propylitic)",                     "SR",
  "SR: (sqrt(B12)-B11)²",                        "[SR] (√B12−B11)² (Iron Oxide)",                        "SR",
  "SR: B03*B12/B07² - 0.45",                     "[SR] B03·B12/B07²−0.45 (Pot.-Skarn)",                  "SR"
)

class_levels <- c("Silicic", "Adv. Argillic", "Argillic-Phyllic", "Propylitic")
class_keys   <- c("1", "2", "3", "4")

cells <- crossing(method_key = methods$key, class_key = class_keys) |>
  rowwise() |>
  mutate(auc = {
    v <- cross[[method_key]][[class_key]]
    if (is.null(v)) NA_real_ else as.numeric(v)
  }) |>
  ungroup() |>
  mutate(
    class  = factor(class_levels[match(class_key, class_keys)], levels = class_levels),
    label  = methods$label[match(method_key, methods$key)],
    family = methods$family[match(method_key, methods$key)]
  )

# Order rows: Classical first then SR, in the order defined in methods
cells$label <- factor(cells$label, levels = rev(methods$label))

# ---- Plot ----

p <- ggplot(cells, aes(class, label, fill = auc)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(is.na(auc), "—", sprintf("%.3f", auc)),
                colour = auc > 0.75),
            size = 2.5) +
  # Horizontal separator between classical and SR families
  geom_hline(yintercept = 6.5, colour = "black", linewidth = 0.8) +
  scale_fill_scico(palette = "vik", direction = -1,
                   limits = c(0.45, 0.95), midpoint = 0.70,
                   na.value = "grey90",
                   name = "Cross-site AUC",
                   breaks = c(0.5, 0.6, 0.7, 0.8, 0.9),
                   guide = guide_colorbar(barwidth = unit(4, "cm"),
                                          barheight = unit(0.3, "cm"),
                                          title.position = "top",
                                          title.hjust = 0.5)) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "black"),
                      guide = "none") +
  scale_x_discrete(position = "top", expand = expansion(0)) +
  scale_y_discrete(expand = expansion(0)) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(clip = "off") +
  theme(
    axis.text.x = element_text(angle = 0, face = "bold", size = 8),
    axis.text.y = element_text(size = 7.5),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    plot.margin = margin(8, 8, 8, 8, "pt")
  )

dir.create("figures", showWarnings = FALSE)
save_paper(p, OUT_PDF, width_cm = 16.0, height_cm = 10.0)
