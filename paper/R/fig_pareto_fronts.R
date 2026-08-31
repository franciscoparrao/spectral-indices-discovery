# fig_pareto_fronts.R --- Pareto fronts (complexity vs loss) per alteration class
# Migration of scripts/fig_pareto_fronts.py to ggplot2 + patchwork.
# Run: Rscript paper/R/fig_pareto_fronts.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(ggrepel)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

RESULTS <- "data/results/pysr_results_gee.json"
OUT_PDF <- "figures/pareto_fronts.pdf"

# ---- Load + tidy ----

raw <- read_json(RESULTS, simplifyVector = FALSE)
class_keys <- setdiff(names(raw), "rf_baseline_balanced_accuracy")

CLASS_LABEL <- c(
  Silicic           = "Silicic",
  Adv_Argillic      = "Adv. Argillic",
  Argillic_Phyllic  = "Argillic-Phyllic",
  Propylitic        = "Propylitic",
  Iron_Oxide        = "Iron Oxide",
  Potassic_Skarn    = "Potassic-Skarn"
)

# Selected formula per class (matches Table tab:sr_formulas in the manuscript)
SELECTED <- tibble::tribble(
  ~class,            ~label,                       ~complexity,
  "Silicic",          "italic(B[04]) - 0.135",                          3,
  "Adv_Argillic",     "0.83 - italic(B[02]/B[05])",                     5,
  "Argillic_Phyllic", "0.09 / italic(B[05])",                           3,
  "Propylitic",       "italic(B[03]) - 0.48 %.% italic(B[11])",         5,
  "Iron_Oxide",       "(sqrt(italic(B[12])) - italic(B[11]))^2",        5,
  "Potassic_Skarn",   "italic(B[03] %.% B[12] / B[07]^2) - 0.45",       8
)

pareto_df <- purrr::map_dfr(class_keys, function(k) {
  pareto <- raw[[k]]$pareto_front
  tibble::tibble(
    class      = k,
    complexity = vapply(pareto, function(p) p$complexity, integer(1)),
    loss       = vapply(pareto, function(p) p$loss, numeric(1))
  )
})

meta_df <- tibble::tibble(
  class       = class_keys,
  class_label = CLASS_LABEL[class_keys],
  auc         = vapply(class_keys, function(k) raw[[k]]$auc, numeric(1)),
  f1          = vapply(class_keys, function(k) raw[[k]]$f1, numeric(1)),
  title       = sprintf("%s (AUC %.2f, F1 %.2f)",
                        CLASS_LABEL[class_keys], auc, f1)
)

selected_pts <- SELECTED |>
  inner_join(pareto_df, by = c("class", "complexity")) |>
  inner_join(meta_df, by = "class")

# ---- One panel per class ----

panel_for <- function(cls) {
  d <- pareto_df |> filter(class == cls)
  m <- meta_df  |> filter(class == cls)
  s <- selected_pts |> filter(class == cls)
  col <- unname(pal_alteration[m$class_label])

  # Diminishing-returns shading: shade x >= complexity at which loss reaches
  # 5 % of (max-min) above min, so visual cue for "knee".
  shade_x <- NA_real_
  if (nrow(d) > 3) {
    lo <- min(d$loss); rng <- max(d$loss) - lo
    knee <- d |> filter(loss <= lo + 0.05 * rng) |> slice_min(complexity)
    if (nrow(knee) > 0) shade_x <- knee$complexity[1]
  }

  p <- ggplot(d, aes(complexity, loss)) +
    {if (!is.na(shade_x))
      annotate("rect", xmin = shade_x, xmax = 16,
               ymin = -Inf, ymax = Inf,
               fill = col, alpha = 0.05)} +
    geom_line(colour = col, linewidth = 0.5, alpha = 0.85) +
    geom_point(colour = col, size = 1.6, alpha = 0.9) +
    geom_point(data = s, aes(complexity, loss),
               shape = 23, size = 3.6, fill = col, colour = "black",
               stroke = 0.5) +
    geom_label_repel(
      data = s, aes(complexity, loss, label = label),
      parse = TRUE, size = 2.4, fontface = "plain",
      box.padding = 0.45, point.padding = 0.4,
      min.segment.length = 0, segment.colour = "grey50",
      segment.size = 0.3,
      fill = "white", colour = "black",
      label.r = unit(0.1, "lines"),
      label.size = 0.2,
      nudge_x = 2, nudge_y = 0.06 * diff(range(d$loss))
    ) +
    scale_x_continuous(limits = c(0, 16),
                       breaks = c(1, 4, 8, 12, 16),
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.10))) +
    labs(x = "Complexity (nodes)", y = "Loss (MSE)",
         subtitle = m$title) +
    theme(plot.subtitle = element_text(face = "bold", size = 7,
                                       hjust = 0, margin = margin(b = 2)),
          plot.margin = margin(4, 8, 4, 4, "pt"))
  p
}

panels <- lapply(class_keys, panel_for)
names(panels) <- class_keys

# ---- Compose 2x3 with patchwork ----
fig <- (panels$Silicic         | panels$Adv_Argillic   | panels$Argillic_Phyllic) /
       (panels$Propylitic      | panels$Iron_Oxide     | panels$Potassic_Skarn) +
  plot_annotation(
    tag_levels = "a", tag_suffix = ")",
    caption = "Diamonds: selected formula at the chosen complexity-accuracy balance (complexity ≤ 8)."
  ) &
  theme(plot.tag = element_text(face = "bold", size = 9),
        plot.caption = element_text(size = 7, hjust = 0,
                                    margin = margin(t = 6)))

dir.create("figures", showWarnings = FALSE)
save_paper(fig, OUT_PDF, width_cm = 18.0, height_cm = 11.0)
