# fig_alteration_comparison.R --- Maricunga 6-panel alteration map figure
# Migration of scripts/fig_alteration_maps_v3.py.
# Reads pre-computed S2 composite from scripts/prep_rasters_for_R.py.
# Run: Rscript paper/R/fig_alteration_comparison.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(terra)
  library(tidyterra)
  library(sf)
  library(dplyr)
  library(readr)
  library(tibble)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

TIF      <- "data/maps/maricunga_composite.tif"
ATLAS    <- "data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp"
ATLAS_CSV <- "data/external/atlas_metalifero_IIIR/alteracion.csv"
OUT_PDF  <- "figures/alteration_comparison.pdf"

# ---- Load raster ----

r <- rast(TIF)
# Band order matches what prep_rasters_for_R.py downloaded
names(r) <- c("B2", "B3", "B4", "B5", "B7", "B8A", "B11", "B12")

# Bounding box for axes
ext_r <- ext(r)
AOI <- list(west = ext_r[1], east = ext_r[2],
            south = ext_r[3], north = ext_r[4])

# ---- Compute SR indices + argmax classification ----

eps <- 1e-6
clamp_lo <- function(x, lo) clamp(x, lower = lo, upper = Inf, values = TRUE)
b2  <- r$B2;  b3 <- r$B3;  b4 <- r$B4;  b5 <- r$B5
b7  <- r$B7;  b8a <- r$B8A; b11 <- r$B11; b12 <- r$B12
sr1 <- b4 - 0.135                                              # Silicic
sr2 <- 0.83 - b2 / clamp_lo(b5, eps)                           # Adv. Argillic
sr3 <- 0.09 / clamp_lo(b5, eps)                                # Argillic-Phyllic
sr4 <- b3 - 0.48 * b11                                          # Propylitic
sr5 <- (sqrt(clamp(b12, lower = 0, upper = Inf,
                   values = TRUE)) - b11)^2                     # Iron Oxide
sr6 <- b3 * b12 / clamp_lo(b7^2, eps) - 0.45                    # Potassic-Skarn

# Per-band p2/p98 stretch so argmax is comparable across classes
percentile_stretch <- function(x, p_lo = 2, p_hi = 98) {
  v <- values(x)
  v <- v[is.finite(v)]
  q <- quantile(v, probs = c(p_lo, p_hi) / 100, na.rm = TRUE)
  (x - q[1]) / max(q[2] - q[1], eps)
}

sr_stack <- rast(list(
  Silicic          = percentile_stretch(sr1),
  AdvArgillic      = percentile_stretch(sr2),
  ArgillicPhyllic  = percentile_stretch(sr3),
  Propylitic       = percentile_stretch(sr4),
  IronOxide        = percentile_stretch(sr5),
  PotassicSkarn    = percentile_stretch(sr6)
))
sr_stack <- clamp(sr_stack, 0, 1)
classmap <- which.max(sr_stack)
levels(classmap) <- data.frame(
  ID = 1:6,
  class = c("Silicic", "Adv. Argillic", "Argillic-Phyllic",
            "Propylitic", "Iron Oxide", "Potassic-Skarn")
)

clay_ratio <- b11 / clamp_lo(b12, eps)

# ---- True / false colour RGB rasters (p2/p98 stretch) ----

stretch_rgb <- function(b1, b2, b3, gamma = 0.85) {
  do_one <- function(x) {
    s <- percentile_stretch(x)
    s <- clamp(s, 0, 1)
    s^gamma
  }
  rast(list(R = do_one(b1), G = do_one(b2), B = do_one(b3)))
}

rgb_true <- stretch_rgb(r$B4, r$B3, r$B2)
rgb_swir <- stretch_rgb(r$B12, r$B11, r$B4)

# ---- Ground-truth polygons ----

ALT_MAP <- c(
  "Alteración Silicea" = "Silicic", "vuggy silica" = "Silicic",
  "Alteracion Argilica y Argilica avanzada" = "Adv. Argillic",
  "Alteracion Solfatárica" = "Adv. Argillic",
  "Alteracion Argilica" = "Argillic-Phyllic",
  "Alteracion Sericitica" = "Argillic-Phyllic",
  "Alteración Cuarzo-Sericitica(Fílica)" = "Argillic-Phyllic",
  "Alteracion Propilitica" = "Propylitic",
  "Oxidos e Hidróxidos de Hierro" = "Iron Oxide",
  "Alteracion Potasica" = "Potassic-Skarn", "skarn" = "Potassic-Skarn"
)

gt <- st_read(ATLAS, quiet = TRUE)
if (is.na(st_crs(gt))) st_crs(gt) <- 32719
gt <- st_transform(gt, 4326)
attrs <- read_csv(ATLAS_CSV, show_col_types = FALSE)
gt <- gt |>
  left_join(attrs |> select(INT_ORIG, ALTERACION), by = "INT_ORIG") |>
  mutate(class = unname(ALT_MAP[ALTERACION]),
         class = ifelse(is.na(class), "Undifferentiated", class))

# Clip to AOI
aoi_box <- st_polygon(list(matrix(c(
  AOI$west, AOI$south, AOI$east, AOI$south,
  AOI$east, AOI$north, AOI$west, AOI$north,
  AOI$west, AOI$south
), ncol = 2, byrow = TRUE))) |> st_sfc(crs = 4326)
gt_aoi <- suppressWarnings(st_intersection(gt, aoi_box))
gt_class <- gt_aoi |> filter(class != "Undifferentiated") |>
  mutate(class = factor(class,
                        levels = c("Silicic", "Adv. Argillic", "Argillic-Phyllic",
                                   "Propylitic", "Iron Oxide", "Potassic-Skarn")))
gt_undiff <- gt_aoi |> filter(class == "Undifferentiated")

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
                       breaks = seq(round(AOI$west, 1), round(AOI$east, 1), 0.1),
                       expand = expansion(0)),
    scale_y_continuous(name = "Latitude (°S)",
                       limits = c(AOI$south, AOI$north),
                       breaks = seq(round(AOI$south, 1), round(AOI$north, 1), 0.1),
                       expand = expansion(0)),
    labs(subtitle = subtitle),
    coord_sf(crs = 4326, expand = FALSE)
  )
}

# ---- Build 6 panels ----

# Panel A: true colour
p_a <- ggplot() +
  geom_spatraster_rgb(data = rgb_true, max_col_value = 1) +
  annotate("segment", x = AOI$west + 0.02,
           xend = AOI$west + 0.02 + 0.1,
           y = AOI$south + 0.012,
           yend = AOI$south + 0.012,
           colour = "black", linewidth = 0.9) +
  annotate("text", x = AOI$west + 0.07,
           y = AOI$south + 0.018,
           label = "~10 km", colour = "white",
           fontface = "bold", size = 2.4,
           fill = "black") +
  style_axes("Sentinel-2 true colour (B4-B3-B2)") +
  map_theme()

# Panel B: SWIR false colour
p_b <- ggplot() +
  geom_spatraster_rgb(data = rgb_swir, max_col_value = 1) +
  style_axes("SWIR false colour (B12-B11-B4)") +
  map_theme()

# Panel C: Clay Ratio
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

# Panel D: SR per-pixel classification (no legend — palette is shared, see caption)
p_d <- ggplot() +
  geom_spatraster(data = classmap) +
  scale_fill_manual(values = pal_alteration, na.value = NA, guide = "none") +
  style_axes("SR per-pixel classification") +
  map_theme()

# Panel E: Ground truth on true colour (no legend — shared palette)
p_e <- ggplot() +
  geom_spatraster_rgb(data = rgb_true, max_col_value = 1, alpha = 0.55) +
  {if (nrow(gt_undiff))
    geom_sf(data = gt_undiff, fill = "#CCCCCC",
            colour = "grey40", linewidth = 0.15, alpha = 0.6)} +
  {if (nrow(gt_class))
    geom_sf(data = gt_class,
            aes(fill = class),
            colour = "black", linewidth = 0.2, alpha = 0.9)} +
  scale_fill_alteration() +
  guides(fill = "none") +
  style_axes("Ground truth polygons") +
  map_theme()

# Panel F: SR classification + ground truth outlines
p_f <- ggplot() +
  geom_spatraster(data = classmap, alpha = 0.85) +
  scale_fill_manual(values = pal_alteration, na.value = NA,
                    guide = "none") +
  {if (nrow(gt_undiff))
    geom_sf(data = gt_undiff, fill = NA,
            colour = "black", linewidth = 0.3, linetype = "dotted")} +
  {if (nrow(gt_class))
    geom_sf(data = gt_class, fill = NA,
            colour = "black", linewidth = 0.45)} +
  style_axes("SR classification + GT outlines") +
  map_theme()

fig <- (p_a | p_b | p_c) / (p_d | p_e | p_f) +
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

dir.create("figures", showWarnings = FALSE)
save_paper(fig, OUT_PDF, width_cm = 18.0, height_cm = 12.5)
