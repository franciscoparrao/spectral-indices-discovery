# fig_v2_scatter.R --- F2: selection AUC vs test AUC for 87 000 expressions, three criteria
# Run: Rscript paper/R/fig_v2_scatter.R  (cwd = project root). Data: scripts/prep_figs_v2.py
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(dplyr); library(readr); library(tidyr)})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")

d <- read_csv("paper/data/figs_v2/f2_scatter.csv", show_col_types = FALSE) |>
  pivot_longer(c(auc_R4_hard, auc_R5_hard, auc_R6_hard), names_to = "site", values_to = "auc_test") |>
  mutate(site = factor(recode(site, auc_R4_hard = "Coquimbo", auc_R5_hard = "Cuprite", auc_R6_hard = "Atacama West"), c("Coquimbo", "Cuprite", "Atacama West")),
         criterion = factor(recode(criterion, boxes = "Distant negatives", ring = "Host-rock ring",
                                   all = "Ring + distractors"),
                            c("Distant negatives", "Host-rock ring", "Ring + distractors")))
rho <- d |> group_by(criterion, site) |>
  summarise(rho = cor(auc_sel, auc_test, method = "spearman"), .groups = "drop") |>
  mutate(lab = sprintf("rho == '%.2f'", rho))
clay_y <- d |> filter(is_clay) |> distinct(site, auc_test)
win <- d |> filter(is_winner)

p <- ggplot(d, aes(auc_sel, auc_test)) +
  geom_bin_2d(bins = 70) +
  scale_fill_gradient(low = "grey88", high = "grey15", trans = "log10", name = "Expressions") +
  geom_hline(data = clay_y, aes(yintercept = auc_test), linetype = "22", linewidth = 0.35, colour = "#0072B2") +
  geom_point(data = win, colour = "#D55E00", size = 1.8) +
  geom_point(data = filter(d, is_clay), colour = "#0072B2", size = 1.6, shape = 17) +
  geom_text(data = rho, aes(x = -Inf, y = Inf, label = lab), parse = TRUE, hjust = -0.15, vjust = 1.5, size = 2.6) +
  facet_grid(site ~ criterion, scales = "free") +
  labs(x = "Selection AUC (Maricunga)", y = "Test AUC (ring + distractors)") +
  theme(legend.position = "top", legend.key.height = unit(0.25, "cm"), legend.key.width = unit(1.2, "cm"),
        strip.text = element_text(face = "bold"), plot.margin = margin(4, 6, 4, 4))
save_paper(p, "figures/v2/fig2_selection_vs_test.pdf", width_cm = 18, height_cm = 15.5)
