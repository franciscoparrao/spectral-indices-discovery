# fig_v2_temporal.R --- F6: AUC per composite and site, selected index vs clay ratio
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")
d <- read_csv("paper/data/figs_v2/f6_temporal.csv", show_col_types = FALSE) |>
  mutate(period = factor(period, c("Base (2024)", "Same season 2023", "Opposite season")),
         site = factor(site, c("Atacama (held-out)", "Coquimbo", "Cuprite")))
p <- ggplot(d, aes(period, auc, colour = index, shape = index, group = index)) +
  geom_line(linewidth = 0.5) + geom_point(size = 1.9) +
  facet_wrap(~site, nrow = 1) +
  scale_colour_manual(values = c("Clay ratio" = "#0072B2", "Selected index" = "#D55E00")) +
  scale_shape_manual(values = c("Clay ratio" = 17, "Selected index" = 16)) +
  scale_x_discrete(labels = c("Base\n2024", "Same season\n2023", "Opposite\nseason")) +
  labs(x = NULL, y = "AUC (ring + distractors)", colour = NULL, shape = NULL) +
  theme(legend.position = "top", strip.text = element_text(face = "bold"), plot.margin = margin(4, 6, 4, 4))
save_paper(p, "figures/v2/fig6_temporal.pdf", width_cm = 18, height_cm = 7)
