# fig_v2_pareto.R --- F3: exact Pareto front m = 1..5 (best selection AUC per complexity)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")
d <- read_csv("paper/data/figs_v2/f3_pareto.csv", show_col_types = FALSE) |>
  mutate(set = factor(set, c("Selection (Atacama)", "Atacama (held-out)", "Cuprite", "Coquimbo"),
                      labels = c("Selection", "Atacama\nheld-out", "Cuprite", "Coquimbo")))
lab <- d |> filter(m == 5)
p <- ggplot(d, aes(m, auc, colour = set, shape = set)) +
  annotate("rect", xmin = 4.5, xmax = 5.5, ymin = -Inf, ymax = Inf, fill = "grey93") +
  annotate("text", x = 5, y = 0.62, label = "m = 5:\nsaturation", size = 2.3, colour = "grey35", lineheight = 0.9) +
  geom_line(linewidth = 0.5) + geom_point(size = 1.8) +
  geom_text(data = lab, aes(label = set), hjust = -0.15, size = 2.4, lineheight = 0.85, show.legend = FALSE) +
  scale_x_continuous(breaks = 1:5, limits = c(1, 6.6), expand = expansion(mult = c(0.02, 0))) +
  scale_y_continuous(limits = c(0.6, 1)) +
  scale_colour_manual(values = c("#000000", "#999999", "#009E73", "#D55E00")) +
  scale_shape_manual(values = c(15, 0, 17, 16)) +
  labs(x = "Complexity m (number of ratio factors)", y = "AUC (ring + distractors)") +
  theme(legend.position = "none", plot.margin = margin(4, 4, 4, 4))
save_paper_single(p, "figures/v2/fig3_pareto.pdf", aspect = 1.1)
