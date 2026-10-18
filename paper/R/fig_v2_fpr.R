# fig_v2_fpr.R --- F5: false-positive rate per negative group, selected index vs clay ratio
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(forcats)})
source("paper/R/theme_paper.R"); setup_paper_theme("elsevier")
nice <- c(ring = "Host-rock ring", shadow = "Shadow", vegetation = "Vegetation*", water = "Water*", snow = "Snow",
          salar_uyuni = "Salar de Uyuni", salar_pedernales = "Salar de Pedernales", salar_maricunga = "Salar de Maricunga",
          salar_atacama = "Salar de Atacama", sand_atacama = "Aeolian sand", playa = "Playa", alluvium = "Alluvium",
          litho_alluvium = "Unaltered alluvium", litho_andesite = "Unaltered andesite", litho_basalt = "Unaltered basalt",
          litho_granitoid = "Unaltered granitoid", litho_greenschist = "Unaltered greenschist",
          litho_sedimentary = "Unaltered sedimentary")
nice <- c(nice, playa = "Playa", alluvium = "Alluvium")
d <- read_csv("paper/data/figs_v2/f5_fpr.csv", show_col_types = FALSE) |>
  mutate(group = nice[group], site = factor(site, c("Atacama (held-out)", "Coquimbo", "Cuprite", "Atacama West"), labels = c("Maricunga", "Coquimbo", "Cuprite", "Atacama W.")))
ord <- d |> filter(index == "Clay ratio") |> group_by(group) |> summarise(o = max(fpr))
d <- d |> left_join(ord, by = "group") |> mutate(group = fct_reorder(group, o))
wide <- tidyr::pivot_wider(d, id_cols = c(site, group), names_from = index, values_from = fpr)
p <- ggplot(d, aes(fpr, group)) +
  geom_segment(data = wide, aes(x = `Clay ratio`, xend = `Selected index`, y = group, yend = group),
               colour = "grey70", linewidth = 0.4) +
  geom_point(aes(colour = index, shape = index), size = 1.8) +
  facet_grid(site ~ ., scales = "free_y", space = "free_y") +
  scale_colour_manual(values = c("Clay ratio" = "#0072B2", "Selected index" = "#D55E00")) +
  scale_shape_manual(values = c("Clay ratio" = 17, "Selected index" = 16)) +
  scale_x_continuous(labels = scales::percent, breaks = c(0, .5, 1), limits = c(0, 1), expand = expansion(mult = c(0.02, 0.04))) +
  labs(x = "False-positive rate at the selection threshold", y = NULL, colour = NULL, shape = NULL) +
  theme(legend.position = "top", strip.text.y = element_text(face = "bold", angle = -90),
        panel.grid.major.y = element_line(colour = "grey94", linewidth = 0.25), plot.margin = margin(4, 4, 4, 4))
save_paper_single(p, "figures/v2/fig5_fpr_groups.pdf", height_cm = 18)
