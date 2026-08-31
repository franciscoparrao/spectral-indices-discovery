# fig_cuprite_comparison.R --- Cuprite 6-panel figure with USGS Rockwell ref
# Migration of scripts/fig_cuprite_comparison.py.
# Reads composite from prep_rasters_for_R.py + USGS Rockwell raster.
# Run: Rscript paper/R/fig_cuprite_comparison.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(terra)
  library(tidyterra)
  library(sf)
  library(dplyr)
  library(tibble)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

S2_TIF  <- "data/maps/cuprite_composite.tif"
GT_TIF  <- "data/ground_truth/cuprite_ground_truth.tif"
OUT_PDF <- "figures/cuprite_comparison.pdf"

# ---- Load rasters ----

r <- rast(S2_TIF)
names(r) <- c("B2", "B3", "B4", "B7", "B8", "B8A", "B11", "B12")
ext_r <- ext(r)
AOI <- list(west = ext_r[1], east = ext_r[2],
            south = ext_r[3], north = ext_r[4])

gt <- rast(GT_TIF)
# Ensure same crs as composite
if (st_crs(gt)$epsg != 4326) gt <- project(gt, "EPSG:4326")

# ---- SR formulas locally-rediscovered at Cuprite (Tabla tab:cuprite_formulas) ----

eps <- 1e-6
clamp_lo <- function(x, lo) clamp(x, lower = lo, upper = Inf, values = TRUE)
b2 <- r$B2; b3 <- r$B3; b4 <- r$B4; b7 <- r$B7
b8 <- r$B8; b8a <- r$B8A; b11 <- r$B11; b12 <- r$B12

sr1 <- log(clamp_lo(b11, eps) / clamp_lo(b12, eps))  # Silicic
sr2 <- b11 - b12                                       # Adv. Argillic
sr3 <- tanh(b11) - b8a                                 # Argillic-Phyllic
sr4 <- ((b12 - b8) / clamp_lo(b7, eps))^2              # Propylitic

percentile_stretch <- function(x, p_lo = 2, p_hi = 98) {
  v <- values(x); v <- v[is.finite(v)]
  q <- quantile(v, probs = c(p_lo, p_hi) / 100, na.rm = TRUE)
  (x - q[1]) / max(q[2] - q[1], eps)
}

sr_stack <- rast(list(
  Silicic         = percentile_stretch(sr1),
  AdvArgillic     = percentile_stretch(sr2),
  ArgillicPhyllic = percentile_stretch(sr3),
  Propylitic      = percentile_stretch(sr4)
))
sr_stack <- clamp(sr_stack, 0, 1)
classmap <- which.max(sr_stack)
levels(classmap) <- data.frame(
  ID = 1:4,
  class = c("Silicic", "Adv. Argillic", "Argillic-Phyllic", "Propylitic")
)

clay_ratio <- b11 / clamp_lo(b12, eps)

# USGS GT raster — encode as factor matching same class labels
gt_factor <- gt
levels(gt_factor) <- data.frame(
  ID = 1:4,
  class = c("Silicic", "Adv. Argillic", "Argillic-Phyllic", "Propylitic")
)
gt_factor[gt_factor == 0] <- NA

# Vectorize GT for outline overlay
gt_polys <- as.polygons(gt_factor, dissolve = TRUE, values = TRUE, aggregate = TRUE)
gt_sf <- st_as_sf(gt_polys)

# ---- True / false colour ----

stretch_rgb <- function(b1, b2, b3, gamma = 0.85) {
  do_one <- function(x) {
    s <- percentile_stretch(x); s <- clamp(s, 0, 1); s^gamma
  }
  rast(list(R = do_one(b1), G = do_one(b2), B = do_one(b3)))
}
rgb_true <- stretch_rgb(b4, b3, b2)
rgb_swir <- stretch_rgb(b12, b11, b4)

# ---- Plot helpers ----

map_theme <- function() {
  theme(
    legend.position = c(0.02, 0.02),
    legend.justification = c(0, 0),
    legend.background = element_rect(fill = "white", colour = "grey80",
                                     linewidth = 0.2),
    legend.title = element_text(size = 6, face = "bold"),
    legend.text = element_text(size = 5.5),
    legend.key.size = unit(0.3, "cm"),
    plot.subtitle = element_text(face = "bold", size = 8.5,
                                 hjust = 0.5, margin = margin(b = 2)),
    axis.text = element_text(size = 6.5),
    axis.title = element_text(size = 7),
    plot.margin = margin(4, 4, 4, 4, "pt")
  )
}

style_axes <- function(subtitle) {
  list(
    scale_x_continuous(name = "Longitude (°W)",
                       limits = c(AOI$west, AOI$east),
                       breaks = seq(round(AOI$west, 2), round(AOI$east, 2), 0.05),
                       expand = expansion(0)),
    scale_y_continuous(name = "Latitude (°N)",
                       limits = c(AOI$south, AOI$north),
                       breaks = seq(round(AOI$south, 2), round(AOI$north, 2), 0.05),
                       expand = expansion(0)),
    labs(subtitle = subtitle),
    coord_sf(crs = 4326, expand = FALSE)
  )
}

# ---- Build 6 panels ----

p_a <- ggplot() +
  geom_spatraster_rgb(data = rgb_true, max_col_value = 1) +
  annotate("segment", x = AOI$west + 0.005,
           xend = AOI$west + 0.005 + 0.035,
           y = AOI$south + 0.006,
           yend = AOI$south + 0.006,
           colour = "black", linewidth = 0.9) +
  annotate("text", x = AOI$west + 0.0225,
           y = AOI$south + 0.011,
           label = "~3 km", colour = "black",
           fontface = "bold", size = 2.2) +
  style_axes("Sentinel-2 true colour (B4-B3-B2)") +
  map_theme()

p_b <- ggplot() +
  geom_spatraster_rgb(data = rgb_swir, max_col_value = 1) +
  style_axes("SWIR false colour (B12-B11-B4)") +
  map_theme()

p_c <- ggplot() +
  geom_spatraster(data = clay_ratio) +
  scale_fill_scico(palette = "vik", direction = -1,
                   limits = quantile(values(clay_ratio),
                                     c(0.02, 0.98), na.rm = TRUE),
                   oob = scales::squish, na.value = NA,
                   name = expression(B[11]/B[12]),
                   guide = guide_colorbar(barwidth = unit(2.5, "cm"),
                                          barheight = unit(0.25, "cm"))) +
  style_axes("Clay Ratio (B11/B12)") +
  map_theme() +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.title = element_text(size = 6, face = "plain", vjust = 0.85))

p_d <- ggplot() +
  geom_spatraster(data = classmap) +
  scale_fill_manual(values = pal_alteration, na.value = NA, guide = "none") +
  style_axes("SR classification (Cuprite formulas)") +
  map_theme()

p_e <- ggplot() +
  geom_spatraster(data = gt_factor) +
  scale_fill_manual(values = pal_alteration, na.value = NA, guide = "none") +
  style_axes("USGS Rockwell (2017) reference") +
  map_theme()

p_f <- ggplot() +
  geom_spatraster(data = classmap, alpha = 0.85) +
  scale_fill_manual(values = pal_alteration, na.value = NA, guide = "none") +
  geom_sf(data = gt_sf, fill = NA, colour = "black",
          linewidth = 0.3) +
  style_axes("SR classification + USGS outlines") +
  map_theme()

fig <- (p_a | p_b | p_c) / (p_d | p_e | p_f) +
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

dir.create("figures", showWarnings = FALSE)
save_paper(fig, OUT_PDF, width_cm = 18.0, height_cm = 12.0)
