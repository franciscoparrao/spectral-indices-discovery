# fig_class_distribution.R --- Training-pixel class distribution (split from fig_study_area.R)
# Run: Rscript paper/R/fig_class_distribution.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

OUT_PDF <- "figures/class_distribution.pdf"

class_counts <- tibble::tibble(
  class = factor(c("Silicic", "Adv. Argillic", "Argillic-Phyllic",
                   "Propylitic", "Iron Oxide", "Potassic-Skarn"),
                 levels = rev(c("Silicic", "Adv. Argillic", "Argillic-Phyllic",
                                "Propylitic", "Iron Oxide", "Potassic-Skarn"))),
  count = c(2200, 5000, 5000, 495, 500, 2400)
)

p <- ggplot(class_counts, aes(count, class, fill = class)) +
  geom_col(colour = "black", linewidth = 0.3, width = 0.72) +
  geom_text(aes(label = format(count, big.mark = ",")),
            hjust = -0.15, size = 3, fontface = "bold") +
  scale_fill_alteration() +
  scale_x_continuous(limits = c(0, 6300),
                     breaks = c(0, 1000, 2000, 3000, 4000, 5000),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Training pixels (n)", y = NULL) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3),
    axis.text.y = element_text(size = 8.5),
    axis.text.x = element_text(size = 7.5),
    axis.title.x = element_text(size = 9),
    plot.margin = margin(6, 12, 6, 6, "pt")
  )

dir.create("figures", showWarnings = FALSE)
save_paper(p, OUT_PDF, width_cm = 8.8, height_cm = 6.0)
