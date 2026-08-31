# theme_paper.R --- publication-quality theme + helpers for the NRR paper
# Stack: R 4.3+, ggplot2 3.5+, patchwork, scico, ggsci
# Journal target: Natural Resources Research (Springer); fallback Elsevier
# Source this at the top of every fig_*.R script.

suppressPackageStartupMessages({
  library(ggplot2)
  library(systemfonts)
  library(scico)
  library(patchwork)
})

# ============================================================
# Paletas categoriales (colorblind-safe)
# ============================================================

# Wong (8): default. Codifica las 6 clases de alteración + extras.
pal_wong <- c(
  "#000000", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
)

# Tol Bright (7): alternativa cuando Wong colisiona.
pal_tol_bright <- c(
  "#4477AA", "#EE6677", "#228833", "#CCBB44",
  "#66CCEE", "#AA3377", "#BBBBBB"
)

# ============================================================
# Paletas SEMÁNTICAS del proyecto
# ============================================================

# Clases de alteración (matchea fig_*.py originales para consistencia visual
# entre runs Python y R durante la migración).
pal_alteration <- c(
  "Silicic"          = "#FF7F00",
  "Adv. Argillic"    = "#E41A1C",
  "Argillic-Phyllic" = "#4DAF4A",
  "Propylitic"       = "#00AA00",
  "Iron Oxide"       = "#A65628",
  "Potassic-Skarn"   = "#377EB8",
  "Undifferentiated" = "#CCCCCC"
)

# Métodos comparados (raw bands, SR, classical, combined)
pal_methods <- c(
  "Raw bands"  = "#000000",
  "SR"         = "#0072B2",
  "Classical"  = "#D55E00",
  "SR + Class" = "#009E73",
  "PCA(6)"     = "#CC79A7",
  "MI top-6"   = "#999999"
)

# Sentinel-2 band colours (approximate true VIS-NIR-SWIR mapping)
pal_s2_bands <- c(
  B02 = "#3A56A3",   # Blue
  B03 = "#5AAA5A",   # Green
  B04 = "#C24446",   # Red
  B05 = "#A53A60",   # Red Edge 1
  B06 = "#933A6C",   # Red Edge 2
  B07 = "#7F3A78",   # Red Edge 3
  B08 = "#6B3A84",   # NIR
  B8A = "#5E3A8E",   # NIR narrow
  B11 = "#A66800",   # SWIR1
  B12 = "#7A3D00"    # SWIR2
)

# ============================================================
# Theme
# ============================================================
#
# Single-column figs in Springer NRR are ~88mm wide, double-column ~180mm.
# Base font size 9 pt para que el texto sea legible al ancho de columna final.
# Helvetica fallback Liberation Sans (siempre disponible en Linux).

theme_paper <- function(base_size = 9, base_family = "Helvetica") {
  fonts_available <- systemfonts::system_fonts()$family
  if (!base_family %in% fonts_available) {
    if ("Liberation Sans" %in% fonts_available) {
      base_family <- "Liberation Sans"
    } else {
      base_family <- ""
    }
  }
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),

      axis.line        = element_line(color = "black", linewidth = 0.4),
      axis.ticks       = element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      axis.text        = element_text(color = "black", size = base_size - 1),
      axis.title       = element_text(color = "black", size = base_size),

      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),

      legend.background = element_rect(fill = "white", color = NA),
      legend.key        = element_rect(fill = "white", color = NA),
      legend.text       = element_text(size = base_size - 1),
      legend.title      = element_text(size = base_size, face = "plain"),
      legend.position   = "top",
      legend.margin     = margin(0, 0, 0, 0, "pt"),
      legend.box.spacing = unit(2, "pt"),

      plot.tag          = element_text(face = "bold", size = base_size + 1,
                                       family = base_family),
      plot.tag.position = c(0.02, 0.98),
      plot.margin       = margin(4, 6, 4, 4, "pt"),

      strip.background  = element_blank(),
      strip.text        = element_text(size = base_size, face = "plain")
    )
}

# Set as global default; configure discrete palette globally
setup_paper_theme <- function(journal = "springer") {
  fontfam <- switch(journal,
    "springer" = "Helvetica",
    "elsevier" = "Helvetica",
    "nature"   = "Helvetica",
    "ieee"     = "Helvetica",
    "agu"      = "Helvetica",
    "Helvetica"
  )
  theme_set(theme_paper(base_family = fontfam))
  options(
    ggplot2.discrete.colour = pal_wong,
    ggplot2.discrete.fill   = pal_wong
  )
  invisible(fontfam)
}

# ============================================================
# Save helpers
# ============================================================

# Defaults: Springer NRR double-column = 18 cm wide; aspect 1.618 (golden).
# Always export both PDF (vector, primary) and PNG (preview, 600 dpi).
save_paper <- function(plot, filename,
                       width_cm = 18.0,
                       aspect = 1.618,
                       height_cm = NULL,
                       also_png = TRUE,
                       dpi_png = 600) {
  if (is.null(height_cm)) height_cm <- width_cm / aspect

  ggsave(
    filename = filename, plot = plot,
    width = width_cm, height = height_cm, units = "cm",
    device = cairo_pdf, bg = "white"
  )
  cat(sprintf("Saved: %s (%.1f x %.1f cm)\n", filename, width_cm, height_cm))

  if (also_png) {
    png_file <- sub("\\.pdf$", ".png", filename)
    ggsave(
      filename = png_file, plot = plot,
      width = width_cm, height = height_cm, units = "cm",
      dpi = dpi_png, device = ragg::agg_png, bg = "white"
    )
    cat(sprintf("Saved: %s (%d dpi)\n", png_file, dpi_png))
  }
}

# Single column (88 mm)
save_paper_single <- function(plot, filename, ...) {
  save_paper(plot, filename, width_cm = 8.8, ...)
}

# ============================================================
# Helpers de uso recurrente
# ============================================================

# Scale colour/fill por nombre de paleta semántica
scale_color_alteration <- function(...) {
  scale_colour_manual(values = pal_alteration, name = NULL, ...)
}
scale_fill_alteration <- function(...) {
  scale_fill_manual(values = pal_alteration, name = NULL, ...)
}
scale_color_methods <- function(...) {
  scale_colour_manual(values = pal_methods, name = NULL, ...)
}
scale_fill_methods <- function(...) {
  scale_fill_manual(values = pal_methods, name = NULL, ...)
}

# Tag suffix consistente para patchwork (a), b), ...)
tag_paper <- function() {
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
    theme(plot.tag = element_text(face = "bold"))
}
