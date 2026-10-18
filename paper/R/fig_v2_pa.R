# fig_v2_pa.R --- F7: prediction-area (P-A) plots and success-rate curves against mapped deposits
# Data: scripts/evaluate_site6_and_pa.py -> paper/data/figs_v2/f7_pa.csv, f7_pa_summary.csv
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(dplyr); library(readr)})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")
SITES <- c(R3 = "Maricunga\n(selection, epithermal Au)", R5 = "Cuprite\n(epithermal Au-Ag)",
           R4 = "Coquimbo\n(Cu-Au-Fe, Cretaceous)", R6 = "Atacama West\n(Cu-Fe-Au, Cretaceous)")
COL <- c("Selected index" = "#D55E00", "Clay ratio" = "#0072B2")
d <- read_csv("paper/data/figs_v2/f7_pa.csv", show_col_types = FALSE) |>
  mutate(site = factor(SITES[site], SITES), index = factor(index, names(COL)))
s <- read_csv("paper/data/figs_v2/f7_pa_summary.csv", show_col_types = FALSE) |>
  mutate(site = factor(SITES[site], SITES), index = factor(index, names(COL)))
pa <- d |> filter(curve == "pa")
area <- pa |> filter(index == "Clay ratio") |> mutate(v = 100 - oa)
lab_pa <- s |> group_by(site) |> arrange(index) |>
  mutate(txt = sprintf("%s: %.0f%% in %.0f%% of area", ifelse(index == "Selected index", "Index", "Clay"), pr_int, 100 - pr_int),
         y = c(14, 5))
p1 <- ggplot(pa, aes(x, pr, colour = index)) +
  geom_line(data = area, aes(x, v, linetype = "100 - occupied area (%)"), colour = "grey45", linewidth = 0.45) +
  geom_line(linewidth = 0.55) +
  geom_point(data = s, aes(x = pr_int, y = pr_int, colour = index), size = 1.9, inherit.aes = FALSE, show.legend = FALSE) +
  facet_wrap(~site, nrow = 1) +
  scale_colour_manual(values = COL, name = "Deposits predicted (%):") +
  scale_linetype_manual(values = c("100 - occupied area (%)" = "22"), name = NULL) +
  scale_x_continuous(breaks = c(0, 50, 100), expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = c(0, 50, 100), limits = c(0, 100)) +
  labs(x = "Prediction value (percentile of the area of interest)", y = "%") +
  theme(legend.position = "top", strip.text = element_text(face = "bold", size = 6.8, lineheight = 0.9),
        panel.spacing.x = unit(0.6, "cm"))
lab_sr <- s |> group_by(site) |> arrange(index) |>
  mutate(txt = sprintf("AUC %.2f", sauc), y = c(16, 7))
p2 <- ggplot(filter(d, curve == "success"), aes(x, pr, colour = index)) +
  geom_abline(linetype = "dotted", colour = "grey60", linewidth = 0.3) +
  geom_line(linewidth = 0.55) +
  geom_text(data = lab_sr, aes(x = 97, y = y, label = txt, colour = index), hjust = 1, size = 2.2,
            inherit.aes = FALSE, show.legend = FALSE) +
  facet_wrap(~site, nrow = 1) + coord_equal() +
  scale_colour_manual(values = COL, guide = "none") +
  scale_x_continuous(breaks = c(0, 50, 100), expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = c(0, 50, 100), expand = expansion(mult = 0.01)) +
  labs(x = "Area classified as altered, highest values first (%)", y = "Deposits captured (%)") +
  theme(strip.text = element_blank(), panel.spacing.x = unit(0.6, "cm"), plot.margin = margin(12, 4, 2, 4))
fig <- (p1 / p2) + plot_layout(heights = c(1, 1)) +
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 8), plot.tag.location = "plot", plot.tag.position = c(0.005, 0.995))
save_paper(fig, "figures/v2/fig7_pa_success.pdf", width_cm = 18, height_cm = 12)
