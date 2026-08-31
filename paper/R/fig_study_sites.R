# fig_study_sites.R --- 2-panel study sites figure (split from fig_study_area.R)
# (a) Western Hemisphere with three sites; legend OUTSIDE panel
# (b) Region III Atacama: Sentinel-2 true colour basemap + Atlas Metalífero polygons
# Class distribution moved to separate figure (fig_class_distribution.R).
# Run: Rscript paper/R/fig_study_sites.R  (cwd = project root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(sf)
  library(terra)
  library(tidyterra)
  library(patchwork)
  library(dplyr)
  library(readr)
  library(ggspatial)
  library(rnaturalearth)
  library(rnaturalearthdata)
})

source("paper/R/theme_paper.R")
setup_paper_theme("springer")

OUT_PDF <- "figures/study_sites.pdf"

# ============================================================
# Panel (a) — Western Hemisphere with three site markers
# ============================================================

world <- ne_countries(scale = 110, returnclass = "sf") |>
  filter(continent %in% c("North America", "South America"))
chile <- world |> filter(name_long == "Chile")
ocean <- ne_download(scale = 110, type = "ocean", category = "physical",
                     returnclass = "sf") |> suppressWarnings()

# Maricunga AOI box (Region III) for the S2 footprint indicator
maricunga_aoi <- st_polygon(list(matrix(c(
  -69.4239, -27.0501, -68.5963, -27.0501, -68.5963, -25.9999,
  -69.4239, -25.9999, -69.4239, -27.0501), ncol = 2, byrow = TRUE))) |>
  st_sfc(crs = 4326)

sites <- tibble(
  site = factor(c("Region III, Chile (Training, 26.85°S 69.10°W)",
                  "Region IV, Chile (Validation, 30.50°S 70.50°W)",
                  "Cuprite, NV, USA (External, 37.55°N 117.15°W)"),
                levels = c("Region III, Chile (Training, 26.85°S 69.10°W)",
                           "Region IV, Chile (Validation, 30.50°S 70.50°W)",
                           "Cuprite, NV, USA (External, 37.55°N 117.15°W)")),
  shape = c(22, 22, 23),    # squares for Chile, diamond for Cuprite
  fill  = c("red", "blue", "green"),
  lon   = c(-69.10, -70.50, -117.15),
  lat   = c(-26.85, -30.50, 37.55)
)

cities <- tibble(
  name = c("Copiapó", "Santiago"),
  lon  = c(-70.33, -70.67),
  lat  = c(-27.37, -33.45)
)

panel_a <- ggplot() +
  geom_sf(data = ocean, fill = "#D8E5F0", colour = NA) +
  geom_sf(data = world, fill = "#EBE0C6",
          colour = "#9C8B66", linewidth = 0.25) +
  geom_sf(data = chile, fill = "#C9B074",
          colour = "#5B4513", linewidth = 0.45) +
  geom_sf(data = maricunga_aoi, fill = NA, colour = "red",
          linewidth = 0.5, linetype = "dashed") +
  geom_point(data = sites,
             aes(lon, lat, shape = site, fill = site),
             size = 3.2, stroke = 0.45, colour = "black") +
  geom_point(data = cities, aes(lon, lat),
             shape = 19, size = 0.8, colour = "black") +
  geom_text(data = cities, aes(lon + 1.5, lat, label = name),
            size = 2.0, colour = "#333333", hjust = 0) +
  scale_shape_manual(values = setNames(sites$shape, sites$site), name = NULL) +
  scale_fill_manual(values = setNames(sites$fill, sites$site), name = NULL) +
  scale_x_continuous(limits = c(-130, -60),
                     breaks = c(-120, -90, -60),
                     name = NULL) +
  scale_y_continuous(limits = c(-55, 50),
                     breaks = c(-40, -20, 0, 20, 40),
                     name = NULL) +
  coord_sf(crs = 4326, expand = FALSE) +
  labs(title = "Study sites — Western Hemisphere") +
  guides(shape = guide_legend(ncol = 1, override.aes = list(size = 2.4),
                              keyheight = unit(0.32, "cm")),
         fill  = guide_legend(ncol = 1)) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.background = element_blank(),
    legend.text = element_text(size = 5.5),
    legend.key.size = unit(0.32, "cm"),
    legend.spacing.y = unit(0.5, "pt"),
    legend.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.margin = margin(-4, 0, 0, 0, "pt"),
    plot.title = element_text(face = "bold", size = 8,
                              hjust = 0.5, margin = margin(t = 8, b = 3)),
    axis.text = element_text(size = 6),
    plot.margin = margin(4, 4, 4, 4, "pt")
  )

# ============================================================
# Panel (b) — Region III Atacama: S2 true colour + Atlas polygons
# ============================================================

ATLAS <- "data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp"
ATLAS_CSV <- "data/external/atlas_metalifero_IIIR/alteracion.csv"
S2_TIF <- "data/maps/maricunga_composite.tif"

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

gdf <- st_read(ATLAS, quiet = TRUE)
if (is.na(st_crs(gdf))) st_crs(gdf) <- 32719
gdf <- st_transform(gdf, 4326)
attrs <- read_csv(ATLAS_CSV, show_col_types = FALSE)
gdf <- gdf |>
  left_join(attrs |> select(INT_ORIG, ALTERACION), by = "INT_ORIG") |>
  mutate(class = unname(ALT_MAP[ALTERACION]),
         class = ifelse(is.na(class), "Undifferentiated", class)) |>
  mutate(class = factor(class,
                        levels = c("Undifferentiated",
                                   "Silicic", "Adv. Argillic",
                                   "Argillic-Phyllic", "Propylitic",
                                   "Iron Oxide", "Potassic-Skarn")))

# Load S2 composite for the same AOI as Maricunga TIF
r <- rast(S2_TIF)
names(r) <- c("B2", "B3", "B4", "B5", "B7", "B8A", "B11", "B12")

# Percentile stretch for true-colour basemap
eps <- 1e-6
percentile_stretch <- function(x, p_lo = 2, p_hi = 98, gamma = 0.85) {
  v <- values(x); v <- v[is.finite(v)]
  q <- quantile(v, probs = c(p_lo, p_hi) / 100, na.rm = TRUE)
  s <- clamp((x - q[1]) / max(q[2] - q[1], eps), lower = 0, upper = 1,
             values = TRUE)
  s^gamma
}
rgb_true <- rast(list(R = percentile_stretch(r$B4),
                      G = percentile_stretch(r$B3),
                      B = percentile_stretch(r$B2)))

# Clip polygons to TIF extent for the panel
ext_vals <- as.vector(ext(r))
bbox_b <- c(xmin = ext_vals[1], xmax = ext_vals[2],
            ymin = ext_vals[3], ymax = ext_vals[4])
aoi_box <- st_polygon(list(matrix(c(
  ext_vals[1], ext_vals[3], ext_vals[2], ext_vals[3],
  ext_vals[2], ext_vals[4], ext_vals[1], ext_vals[4],
  ext_vals[1], ext_vals[3]
), ncol = 2, byrow = TRUE))) |> st_sfc(crs = 4326)
gdf_aoi <- suppressWarnings(st_intersection(gdf, aoi_box))

panel_b <- ggplot() +
  geom_spatraster_rgb(data = rgb_true, max_col_value = 1) +
  geom_sf(data = gdf_aoi,
          aes(fill = class),
          colour = "black", linewidth = 0.2, alpha = 0.7) +
  scale_fill_alteration() +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 4)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  coord_sf(xlim = c(bbox_b["xmin"], bbox_b["xmax"]),
           ylim = c(bbox_b["ymin"], bbox_b["ymax"]),
           expand = FALSE, crs = 4326) +
  labs(x = NULL, y = NULL,
       title = "Region III — Maricunga true colour + alteration polygons") +
  guides(fill = "none") +
  theme(
    plot.title = element_text(face = "bold", size = 8,
                              hjust = 0.5, margin = margin(t = 8, b = 3)),
    axis.text = element_text(size = 6),
    plot.margin = margin(4, 4, 4, 4, "pt")
  )

# ============================================================
# Compose with patchwork — 2 panels only
# ============================================================

fig <- (panel_a | panel_b) +
  plot_layout(widths = c(1, 1.5)) +
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 10),
        plot.tag.position = c(0.02, 0.985))

dir.create("figures", showWarnings = FALSE)
save_paper(fig, OUT_PDF, width_cm = 18.0, height_cm = 10.0)
