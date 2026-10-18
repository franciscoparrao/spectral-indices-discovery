# fig_v2_roc.R --- F4: ROC curves against ring + distractors, three sites
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")
lev <- c("Selected index", "Logistic (log-ratios)", "Random forest (bands)", "Clay ratio")
d <- read_csv("paper/data/figs_v2/f4_roc.csv", show_col_types = FALSE) |>
  mutate(site = factor(site, c("Atacama (held-out)", "Coquimbo", "Cuprite", "Atacama West"),
                       labels = c("Maricunga (held-out)", "Coquimbo", "Cuprite", "Atacama West")), model = factor(model, lev))
aucs <- d |> distinct(site, model, auc) |> arrange(site, model) |> group_by(site) |>
  mutate(y = 0.05 + 0.075 * (n() - row_number()), lab = sprintf("%.3f", auc))
cols <- c("#D55E00", "#009E73", "#CC79A7", "#0072B2")
p <- ggplot(d, aes(fpr, tpr, colour = model, linetype = model)) +
  geom_abline(linetype = "dotted", colour = "grey60", linewidth = 0.3) +
  geom_line(linewidth = 0.55) +
  geom_text(data = aucs, aes(x = 0.97, y = y, label = lab), hjust = 1, size = 2.4, show.legend = FALSE) +
  annotate("text", x = 0.97, y = 0.36, label = "AUC", hjust = 1, size = 2.4, fontface = "bold") +
  facet_wrap(~site, nrow = 1) + coord_equal() +
  scale_colour_manual(values = cols) + scale_linetype_manual(values = c("solid", "longdash", "dotdash", "solid")) +
  scale_x_continuous(breaks = c(0, .5, 1), labels = c("0", "0.5", "1"), expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = c(0, .5, 1), labels = c("0", "0.5", "1"), expand = expansion(mult = 0.01)) +
  labs(x = "False-positive rate", y = "True-positive rate", colour = NULL, linetype = NULL) +
  theme(legend.position = "top", strip.text = element_text(face = "bold"), panel.spacing.x = unit(0.5, "cm"), plot.margin = margin(4, 6, 4, 4))
save_paper(p, "figures/v2/fig4_roc.pdf", width_cm = 18, height_cm = 6.6)
