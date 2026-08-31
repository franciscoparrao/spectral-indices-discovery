# fig_physical_interpretation.R --- Physical interpretation of SR formulas
# Migration of scripts/physical_interpretation.py.
# Stacked panels (one per SR formula): mineral spectral curves + S2 bands
# highlighted for the formula + formula text + cross-site AUC info.
# Run: Rscript paper/R/fig_physical_interpretation.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

OUT_PDF <- "figures/physical_interpretation.pdf"

# ============================================================
# Sentinel-2 bands: center (um) + bandwidth (um)
# ============================================================
S2 <- tribble(
  ~band, ~center, ~width,
  "B02",  0.490,  0.065,
  "B03",  0.560,  0.035,
  "B04",  0.665,  0.030,
  "B05",  0.705,  0.015,
  "B06",  0.740,  0.015,
  "B07",  0.783,  0.020,
  "B08",  0.842,  0.115,
  "B8A",  0.865,  0.020,
  "B11",  1.610,  0.090,
  "B12",  2.190,  0.180
)

# ============================================================
# Mineral spectra (simplified from USGS Spectral Library v7)
# ============================================================
mineral_list <- list(
  Chlorite = list(
    colour = "#00AA00", style = "solid", alteration = "Propylitic",
    curve = matrix(c(
      0.35,0.02, 0.40,0.03, 0.50,0.05, 0.55,0.07, 0.60,0.08,
      0.65,0.09, 0.70,0.12, 0.75,0.15, 0.80,0.17, 0.85,0.16,
      0.90,0.14, 0.95,0.15, 1.00,0.18, 1.10,0.22, 1.20,0.24,
      1.30,0.24, 1.38,0.14, 1.42,0.12, 1.50,0.18, 1.60,0.22,
      1.70,0.24, 1.80,0.24, 1.90,0.18, 2.00,0.22, 2.10,0.24,
      2.20,0.22, 2.25,0.18, 2.32,0.10, 2.35,0.12, 2.40,0.16,
      2.50,0.14, 2.55,0.13
    ), ncol = 2, byrow = TRUE)
  ),
  Epidote = list(
    colour = "#66AA00", style = "dashed", alteration = "Propylitic",
    curve = matrix(c(
      0.35,0.02, 0.40,0.03, 0.50,0.08, 0.55,0.12, 0.60,0.14,
      0.65,0.13, 0.70,0.15, 0.75,0.18, 0.80,0.20, 0.90,0.22,
      1.00,0.24, 1.10,0.26, 1.20,0.27, 1.30,0.27, 1.40,0.22,
      1.50,0.25, 1.60,0.27, 1.70,0.28, 1.80,0.28, 1.90,0.24,
      2.00,0.27, 2.10,0.28, 2.20,0.26, 2.25,0.22, 2.33,0.15,
      2.35,0.17, 2.40,0.20, 2.50,0.18, 2.55,0.17
    ), ncol = 2, byrow = TRUE)
  ),
  Kaolinite = list(
    colour = "#E41A1C", style = "solid", alteration = "Adv. Argillic",
    curve = matrix(c(
      0.35,0.42, 0.40,0.48, 0.50,0.62, 0.60,0.70, 0.70,0.74,
      0.80,0.76, 0.90,0.77, 1.00,0.78, 1.10,0.79, 1.20,0.80,
      1.30,0.80, 1.38,0.55, 1.42,0.52, 1.50,0.65, 1.60,0.78,
      1.70,0.80, 1.80,0.80, 1.90,0.76, 2.00,0.78, 2.10,0.78,
      2.16,0.60, 2.20,0.52, 2.25,0.72, 2.30,0.76, 2.40,0.70,
      2.50,0.68, 2.55,0.66
    ), ncol = 2, byrow = TRUE)
  ),
  Alunite = list(
    colour = "#377EB8", style = "solid", alteration = "Adv. Argillic",
    curve = matrix(c(
      0.35,0.30, 0.40,0.38, 0.50,0.58, 0.60,0.70, 0.70,0.74,
      0.80,0.76, 0.90,0.77, 1.00,0.78, 1.10,0.79, 1.20,0.79,
      1.30,0.78, 1.43,0.50, 1.48,0.45, 1.55,0.65, 1.60,0.72,
      1.70,0.74, 1.80,0.73, 1.90,0.74, 2.00,0.76, 2.10,0.76,
      2.15,0.58, 2.17,0.50, 2.25,0.70, 2.30,0.72, 2.40,0.62,
      2.50,0.58, 2.55,0.55
    ), ncol = 2, byrow = TRUE)
  ),
  Goethite = list(
    colour = "#A65628", style = "solid", alteration = "Iron Oxide",
    curve = matrix(c(
      0.35,0.04, 0.40,0.06, 0.48,0.04, 0.52,0.08, 0.55,0.15,
      0.60,0.30, 0.65,0.38, 0.70,0.40, 0.75,0.48, 0.80,0.52,
      0.85,0.50, 0.90,0.42, 0.95,0.48, 1.00,0.55, 1.10,0.60,
      1.20,0.62, 1.30,0.63, 1.40,0.60, 1.50,0.63, 1.60,0.64,
      1.70,0.65, 1.80,0.65, 1.90,0.62, 2.00,0.64, 2.10,0.64,
      2.20,0.63, 2.30,0.61, 2.40,0.59, 2.50,0.57, 2.55,0.56
    ), ncol = 2, byrow = TRUE)
  ),
  Quartz = list(
    colour = "#FF7F00", style = "solid", alteration = "Silicic",
    curve = matrix(c(
      0.35,0.60, 0.40,0.70, 0.50,0.82, 0.60,0.86, 0.70,0.88,
      0.80,0.89, 0.90,0.89, 1.00,0.89, 1.10,0.89, 1.20,0.89,
      1.40,0.87, 1.60,0.88, 1.80,0.87, 2.00,0.87, 2.20,0.86,
      2.40,0.84, 2.55,0.81
    ), ncol = 2, byrow = TRUE)
  ),
  Illite = list(
    colour = "#4DAF4A", style = "dashed", alteration = "Argillic",
    curve = matrix(c(
      0.35,0.20, 0.40,0.28, 0.50,0.42, 0.60,0.55, 0.70,0.62,
      0.80,0.66, 0.90,0.68, 1.00,0.70, 1.10,0.71, 1.20,0.72,
      1.30,0.72, 1.38,0.50, 1.42,0.48, 1.55,0.65, 1.60,0.68,
      1.70,0.70, 1.80,0.71, 1.88,0.55, 1.92,0.50, 2.00,0.65,
      2.10,0.69, 2.18,0.48, 2.20,0.45, 2.30,0.65, 2.34,0.52,
      2.40,0.58, 2.55,0.58
    ), ncol = 2, byrow = TRUE)
  )
)

# Interpolate every mineral to a regular wavelength grid for cleaner lines
WL_GRID <- seq(0.35, 2.55, by = 0.005)
mineral_df <- map_dfr(names(mineral_list), function(nm) {
  m <- mineral_list[[nm]]
  rf <- approx(m$curve[, 1], m$curve[, 2], xout = WL_GRID, rule = 2)$y
  tibble(mineral = nm,
         alteration = m$alteration,
         colour = m$colour,
         style = m$style,
         wl = WL_GRID,
         refl = rf,
         label = sprintf("%s (%s)", nm, m$alteration))
})

# Mineral palette by name (to keep consistent across panels)
mineral_pal <- vapply(mineral_list, function(m) m$colour, character(1))
names(mineral_pal) <- vapply(names(mineral_list),
                              function(nm) sprintf("%s (%s)", nm,
                                                   mineral_list[[nm]]$alteration),
                              character(1))
mineral_lty <- vapply(mineral_list, function(m) m$style, character(1))
names(mineral_lty) <- names(mineral_pal)

# ============================================================
# SR formulas
# ============================================================
formulas <- list(
  list(
    name = "Propylitic Index",
    formula_expr = "italic(B[3]) - 0.48 %.% italic(B[11])",
    target = "Propylitic (chlorite, epidote)",
    auc_in = 0.91, auc_cross = 0.82,
    bands = c("B03", "B11"),
    rationale = paste(
      "Chlorite and epidote have moderate Green reflectance (B03 ~ 560 nm) and",
      "an Mg-OH absorption near 2.32 um that depresses SWIR1 (B11 ~ 1610 nm) reflectance.",
      "The contrast B03 - 0.48 B11 captures this chlorite-epidote VNIR-vs-SWIR signature."
    )
  ),
  list(
    name = "Adv. Argillic Index",
    formula_expr = "0.83 - italic(B[2]/B[5])",
    target = "Adv. Argillic (kaolinite, alunite)",
    auc_in = 0.72, auc_cross = 0.70,
    bands = c("B02", "B05"),
    rationale = paste(
      "Kaolinite and alunite show a steep VNIR slope between Blue (B02 ~ 490 nm)",
      "and Red Edge 1 (B05 ~ 705 nm). The inverse ratio B02/B05 is low for altered",
      "surfaces; the additive shift 0.83 centres the index near zero."
    )
  ),
  list(
    name = "SWIR Alteration Index",
    formula_expr = "(sqrt(italic(B[12])) - italic(B[11]))^2",
    target = "General alteration / Iron Oxide",
    auc_in = 0.75, auc_cross = 0.69,
    bands = c("B11", "B12"),
    rationale = paste(
      "Non-linear contrast in the two SWIR bands.",
      "sqrt(B12) compresses dynamic range; the squared difference amplifies any",
      "deviation from a baseline B11/B12 ratio, so OH-bearing or Fe-oxide perturbations",
      "are detected as a general altered/unaltered cue."
    )
  ),
  list(
    name = "Potassic / Skarn Index",
    formula_expr = "italic(B[3] %.% B[12] / B[7]^2) - 0.45",
    target = "Potassic-Skarn",
    auc_in = 0.77, auc_cross = 0.54,
    bands = c("B03", "B07", "B12"),
    rationale = paste(
      "Three-band non-linear combination (B03 Green, B07 Red Edge 3, B12 SWIR2).",
      "B12/B07^2 amplifies the SWIR vs Red Edge contrast; B03 normalises overall",
      "brightness. No diagnostic single-mineral absorption is targeted."
    )
  ),
  list(
    name = "Silicic Index",
    formula_expr = "italic(B[4]) - 0.135",
    target = "Silicic (quartz, opal)",
    auc_in = 0.66, auc_cross = 0.86,
    bands = c("B04"),
    rationale = paste(
      "Single-band red threshold. Quartz / opal surfaces have high overall reflectance;",
      "the threshold 0.135 separates bright siliceous surfaces from darker rocks.",
      "Brightness proxy with no spectral contrast (limitation; see manuscript)."
    )
  )
)

# ============================================================
# Panel builder
# ============================================================

panel_for <- function(f) {
  used_bands <- S2 |> filter(band %in% f$bands)
  unused_bands <- S2 |> filter(!band %in% f$bands)

  subtitle <- sprintf(
    "%s\nIntra-site AUC = %.2f • Cross-site AUC = %.2f",
    f$name, f$auc_in, f$auc_cross
  )

  p <- ggplot() +
    # Highlighted bands (gold rectangles)
    geom_rect(data = used_bands,
              aes(xmin = center - width / 2, xmax = center + width / 2,
                  ymin = 0, ymax = 1),
              fill = "#F0C040", alpha = 0.18, inherit.aes = FALSE) +
    # Labels for used bands
    geom_text(data = used_bands,
              aes(x = center, y = 0.94, label = band),
              fontface = "bold", size = 2.4, colour = "#333333") +
    # Other S2 bands as light vertical lines (context)
    geom_vline(data = unused_bands, aes(xintercept = center),
               colour = "grey80", linetype = "dotted", linewidth = 0.25) +
    # Mineral curves
    geom_line(data = mineral_df,
              aes(wl, refl, colour = label, linetype = label),
              linewidth = 0.5, alpha = 0.85) +
    # SWIR atmospheric gap (1.35-1.42 and 1.85-1.95 um) — out of S2 anyway, but
    # visually informative
    annotate("rect", xmin = 1.35, xmax = 1.42, ymin = 0, ymax = 1,
             fill = "grey92", alpha = 0.6) +
    annotate("rect", xmin = 1.85, xmax = 1.95, ymin = 0, ymax = 1,
             fill = "grey92", alpha = 0.6) +
    scale_x_continuous(
      limits = c(0.35, 2.55),
      breaks = seq(0.5, 2.5, by = 0.5),
      expand = expansion(0),
      name = "Wavelength (µm)"
    ) +
    scale_y_continuous(limits = c(0, 1),
                       breaks = c(0, 0.25, 0.5, 0.75, 1.0),
                       expand = expansion(0),
                       name = "Reflectance") +
    scale_colour_manual(values = mineral_pal, name = NULL,
                        guide = guide_legend(nrow = 1)) +
    scale_linetype_manual(values = mineral_lty, name = NULL,
                          guide = guide_legend(nrow = 1)) +
    labs(subtitle = subtitle) +
    theme(
      plot.subtitle = element_text(face = "bold", size = 8.5,
                                   margin = margin(b = 3)),
      legend.position = "none",
      plot.margin = margin(4, 6, 4, 4, "pt")
    )
  p
}

panels <- lapply(formulas, panel_for)

# Shared legend at top
legend_plot <- ggplot(mineral_df, aes(wl, refl, colour = label, linetype = label)) +
  geom_line() +
  scale_colour_manual(values = mineral_pal, name = NULL,
                      guide = guide_legend(nrow = 1)) +
  scale_linetype_manual(values = mineral_lty, name = NULL,
                        guide = guide_legend(nrow = 1)) +
  theme(legend.position = "top",
        legend.text = element_text(size = 7),
        legend.key.size = unit(0.5, "cm"))

# Extract just the legend grob
legend_grob <- cowplot::get_plot_component(legend_plot, "guide-box-top",
                                            return_all = FALSE)

# Compose: legend + 5 panels stacked
fig <- wrap_plots(panels, ncol = 1) +
  plot_annotation(
    tag_levels = "a", tag_suffix = ")"
  ) &
  theme(plot.tag = element_text(face = "bold", size = 9))

dir.create("figures", showWarnings = FALSE)

# Build composite: legend on top, then panels
# Place a slim legend strip as the first row of patchwork
legend_strip <- patchwork::wrap_elements(full = legend_grob)
final <- legend_strip / fig + plot_layout(heights = c(0.08, 1))

save_paper(final, OUT_PDF, width_cm = 18.0, height_cm = 28.0,
           also_png = TRUE, dpi_png = 600)
