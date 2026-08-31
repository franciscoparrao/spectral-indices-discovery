# fig_spectral_profiles.R --- Mean spectral profiles by alteration class
# Migration of scripts/fig_spectral_boxplots.py (the profile figure).
# Run: Rscript paper/R/fig_spectral_profiles.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

CSV <- "data/ground_truth/maricunga_training_s2.csv"
OUT_PDF <- "figures/spectral_profiles.pdf"

# ---- Load + tidy ----

raw <- read_csv(CSV, show_col_types = FALSE)

# Wavelength mapping (nm) — keep only the 10 alteration-relevant S2 bands
WAVELENGTHS <- tibble::tribble(
  ~band,  ~nm,
  "B02",  490,
  "B03",  560,
  "B04",  665,
  "B05",  705,
  "B06",  740,
  "B07",  783,
  "B08",  842,
  "B8A",  865,
  "B11",  1610,
  "B12",  2190
)

CLASS_LABEL <- c(
  `1` = "Silicic",
  `2` = "Adv. Argillic",
  `3` = "Argillic-Phyllic",
  `4` = "Propylitic",
  `5` = "Iron Oxide",
  `6` = "Potassic-Skarn"
)

long <- raw |>
  filter(y %in% as.integer(names(CLASS_LABEL))) |>
  pivot_longer(cols = -y, names_to = "band", values_to = "refl") |>
  inner_join(WAVELENGTHS, by = "band") |>
  mutate(class = factor(CLASS_LABEL[as.character(y)],
                        levels = unname(CLASS_LABEL)))

class_n <- long |> distinct(y, class) |>
  inner_join(raw |> count(y, name = "n"), by = "y") |>
  mutate(label_n = sprintf("%s (n=%d)", class, n))

stats <- long |>
  group_by(class, nm) |>
  summarise(
    mean = mean(refl, na.rm = TRUE),
    q25  = quantile(refl, 0.25, na.rm = TRUE),
    q75  = quantile(refl, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) |>
  inner_join(class_n |> select(class, label_n), by = "class") |>
  mutate(label_n = factor(label_n, levels = unique(class_n$label_n)))

# ---- Plot ----

# Build a band-position table for the secondary axis labels
band_pos <- WAVELENGTHS$nm
band_lab <- WAVELENGTHS$band

# SWIR gap shading (900-1500 nm: where S2 has no bands)
gap_shade <- data.frame(xmin = 900, xmax = 1500, ymin = -Inf, ymax = Inf)

p <- ggplot(stats, aes(nm, mean, colour = label_n, fill = label_n)) +
  geom_rect(data = gap_shade,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            inherit.aes = FALSE,
            fill = "grey85", alpha = 0.35) +
  annotate("text", x = 1200, y = Inf,
           label = "No S2\nbands", colour = "grey45",
           fontface = "italic", size = 2.4, vjust = 1.4) +
  geom_ribbon(aes(ymin = q25, ymax = q75), colour = NA, alpha = 0.13) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.4, shape = 21, colour = "white", stroke = 0.25) +
  scale_x_continuous(
    breaks = c(500, 700, 900, 1500, 1700, 1900, 2100),
    expand = expansion(mult = c(0.01, 0.01)),
    sec.axis = dup_axis(name = NULL, breaks = band_pos, labels = band_lab)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.08))) +
  scale_colour_manual(values = unname(pal_alteration[unique(class_n$class)]),
                      name = NULL) +
  scale_fill_manual(values = unname(pal_alteration[unique(class_n$class)]),
                    name = NULL) +
  labs(x = "Wavelength (nm)",
       y = "Surface reflectance") +
  theme(
    legend.position = "top",
    legend.text = element_text(size = 7.5),
    legend.key.size = unit(0.5, "cm"),
    axis.text.x.top = element_text(size = 6.5, colour = "grey30",
                                    angle = 60, hjust = 0, vjust = 0,
                                    margin = margin(b = 1)),
    axis.ticks.x.top = element_line(colour = "grey60", linewidth = 0.25),
    plot.margin = margin(8, 10, 6, 6, "pt")
  )

dir.create("figures", showWarnings = FALSE)
save_paper(p, OUT_PDF, width_cm = 18.0, height_cm = 10.0)
