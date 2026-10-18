# fig_v2_study_area.R --- F1 (main_v2): study areas and the hard-negative design
# (a) locator: selection site, test site 1 and the distant distractor boxes (northern Chile / Bolivia)
# (b) Atacama, (c) Coquimbo, (d) Cuprite: mapped alteration, host-rock ring and distractor samples
# Run: Rscript paper/R/fig_v2_study_area.R  (cwd = project root)
suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(readr); library(sf)
  library(rnaturalearth); library(ggspatial)
})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")
sf_use_s2(FALSE)

cat_of <- function(g) dplyr::case_when(
  g == "ring" ~ "Host-rock ring",
  grepl("salar|playa", g) ~ "Salt flat / playa",
  grepl("litho|alluvium|sand", g) ~ "Unaltered rock, alluvium, sand",
  g == "shadow" ~ "Shadow",
  g %in% c("snow", "water", "vegetation") ~ "Snow, water, vegetation",
  TRUE ~ NA_character_)
LEV <- c("Mapped alteration", "Host-rock ring", "Salt flat / playa", "Unaltered rock, alluvium, sand",
         "Shadow", "Snow, water, vegetation")
COL <- setNames(c("#D55E00", "#56B4E9", "#0072B2", "#E69F00", "#000000", "#009E73"), LEV)

neg <- bind_rows(read_csv("data/ground_truth/pilot_hard_negatives.csv", show_col_types = FALSE),
                 read_csv("data/ground_truth/cuprite_hard_negatives.csv", show_col_types = FALSE),
                 read_csv("data/ground_truth/site6_hard_negatives.csv", show_col_types = FALSE)) |>
  mutate(cat = factor(cat_of(group), LEV))
pos3 <- read_csv("data/ground_truth/binary_training_s2_positives_geo.csv", show_col_types = FALSE)
pos5 <- read_csv("data/ground_truth/cuprite_positives_geo.csv", show_col_types = FALSE)

atlas <- st_read("data/external/atlas_metalifero_IIIR/Geometria/RMM_ALTERACI.shp", quiet = TRUE) |>
  st_set_crs(32719) |> st_make_valid() |> st_transform(4326)
aw <- read_csv("data/external/sernageomin_atacama_oeste/alteracion_atacama_oeste.csv", show_col_types = FALSE) |>
  mutate(geometry = st_as_sfc(wkt_utm19s, crs = 32719)) |> st_as_sf() |> st_make_valid() |> st_transform(4326)
coq <- lapply(Sys.glob("data/external/shapefiles_alteracion/*/GB_ALTERACION_P.shp"),
              function(f) st_read(f, quiet = TRUE) |> st_make_valid() |> st_transform(4326) |> select(geometry)) |>
  bind_rows()

pts <- function(d, size = 0.25, alpha = 0.6) geom_point(data = d, aes(lon, lat, colour = cat), size = size,
                                                         alpha = alpha, stroke = 0, show.legend = TRUE)
map_theme <- theme(axis.title = element_blank(), axis.text = element_text(size = 5.5),
                   panel.border = element_rect(fill = NA, colour = "grey30", linewidth = 0.3),
                   plot.margin = margin(2, 3, 2, 3))
leg_dummy <- tibble(lon = NA_real_, lat = NA_real_, cat = factor(LEV, LEV))
scale_cat <- scale_colour_manual(values = COL, limits = LEV, drop = FALSE, name = NULL,
                                 guide = guide_legend(override.aes = list(size = 2.2, alpha = 1, shape = 16), nrow = 2, byrow = TRUE))

# (a) locator
world <- ne_countries(scale = "medium", returnclass = "sf")
boxes <- tibble(name = c("Atacama\n(selection +\ntest 3)", "Coquimbo\n(test 1)"),
                xmin = c(-70.7, -71.4), xmax = c(-68.3, -70.4), ymin = c(-28.1, -31.1), ymax = c(-25.9, -29.9))
far <- neg |> filter(group %in% c("salar_uyuni", "salar_atacama", "sand_atacama")) |>
  group_by(group) |> summarise(lon = mean(lon), lat = mean(lat), cat = first(cat), .groups = "drop")
pa <- ggplot() +
  geom_sf(data = world, fill = "grey95", colour = "grey55", linewidth = 0.2) +
  geom_rect(data = boxes, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), fill = NA, colour = "black", linewidth = 0.4) +
  geom_text(data = boxes, aes(x = xmin - 0.25, y = (ymin + ymax) / 2, label = name), hjust = 1, size = 2, lineheight = 0.85) +
  geom_point(data = far, aes(lon, lat, colour = cat), size = 2.2) +
  annotate("text", x = -67.65, y = -20.2, label = "Uyuni", size = 1.9, hjust = 0, vjust = -1) +
  geom_point(data = leg_dummy, aes(lon, lat, colour = cat), na.rm = TRUE) +
  annotate("text", x = -66.9, y = -31.6, label = "Cuprite, Nevada:\n37.5°N, 117.2°W (d)", size = 1.9, hjust = 1, lineheight = 0.85) +
  coord_sf(xlim = c(-74.5, -66.5), ylim = c(-32, -19.5), expand = FALSE) +
  scale_x_continuous(breaks = c(-72, -68)) + scale_cat + map_theme + guides(colour = "none")
# (b) Atacama
pb <- ggplot() +
  geom_sf(data = atlas, fill = COL[["Mapped alteration"]], colour = NA) +
  geom_sf(data = aw, fill = COL[["Mapped alteration"]], colour = NA) +
  pts(filter(neg, site %in% c("R3", "R6"), lon > -70.75, lon < -68.2, lat > -28.2, lat < -25.8)) +
  annotate("label", x = c(-70.15, -68.95), y = -25.97, label = c("Atacama West (test 3)", "Maricunga (selection)"),
           size = 2, fill = "white", alpha = 0.85) +
  geom_point(data = leg_dummy, aes(lon, lat, colour = cat), na.rm = TRUE) +
  coord_sf(xlim = c(-70.75, -68.25), ylim = c(-28.15, -25.85), expand = FALSE) +
  scale_x_continuous(breaks = c(-70.5, -69.5, -68.5)) + scale_cat + map_theme +
  annotation_scale(location = "br", height = unit(0.08, "cm"), text_cex = 0.45, line_width = 0.3)
# (c) Coquimbo
bb <- st_bbox(coq)
pc <- ggplot() +
  geom_sf(data = coq, fill = COL[["Mapped alteration"]], colour = NA) +
  pts(filter(neg, site == "R4")) +
  coord_sf(xlim = c(bb$xmin - 0.05, bb$xmax + 0.05), ylim = c(bb$ymin - 0.05, bb$ymax + 0.05), expand = FALSE) +
  scale_x_continuous(breaks = c(-71, -70.6)) + scale_cat + map_theme + guides(colour = "none") +
  annotation_scale(location = "br", height = unit(0.08, "cm"), text_cex = 0.45, line_width = 0.3)
# (d) Cuprite
pd <- ggplot() +
  geom_point(data = pos5, aes(lon, lat), colour = COL[["Mapped alteration"]], size = 0.15, alpha = 0.6, stroke = 0) +
  pts(filter(neg, site == "R5", lon > -117.45, lon < -116.95, lat > 37.35, lat < 37.72)) +
  coord_sf(xlim = c(-117.45, -116.95), ylim = c(37.35, 37.72), expand = FALSE, crs = 4326) +
  scale_x_continuous(breaks = c(-117.4, -117.0)) + scale_cat + map_theme + guides(colour = "none") +
  annotation_scale(location = "br", height = unit(0.08, "cm"), text_cex = 0.45, line_width = 0.3)

fig <- (pa | pb | (pc / pd)) +
  plot_layout(widths = c(0.8, 1.55, 0.9), guides = "collect") +
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 8), plot.tag.location = "panel", plot.tag.position = c(0.03, 0.97), legend.position = "bottom",
        legend.text = element_text(size = 6.5), legend.margin = margin(0, 0, 0, 0))
save_paper(fig, "figures/v2/fig1_study_areas.pdf", width_cm = 18, height_cm = 12)
