# Cambios en figuras — 2026-06-25

Nota para el proyecto del paper: qué cambió en `figures/` y qué hacer al compilar.

## TL;DR — acción requerida

- **Recompilar el PDF** para reflejar la corrección de títulos de `pareto_fronts`.
- **No hay que tocar ningún `.tex`**: las referencias `\includegraphics` siguen
  apuntando a los nombres base correctos (`pareto_fronts.png`, `study_area.png`, …).
  El set `_handoff` es paralelo y **ningún `.tex` lo referencia** (verificado).

## 1. Único cambio de contenido: `pareto_fronts`

Fix en `paper/R/fig_pareto_fronts.R` (los subtítulos de panel desbordaban el ancho
de panel y el tag pisaba el título → aparecía un falso `3)`):

- Línea 59: `sprintf("%s (AUC = %.3f, F1 = %.3f)")` → `sprintf("%s (AUC %.2f, F1 %.2f)")`
  (2 decimales, sin `= ` — la causa del desborde).
- Línea 111: `plot.subtitle` size `8 → 7`.

Resultado: los 6 títulos caben en una línea, tags `a)`–`f)` correctos.
`figures/pareto_fronts.{pdf,png}` quedaron regenerados con la corrección
→ **al recompilar, el paper toma los títulos arreglados automáticamente.**

## 2. Resto de figuras base: regeneradas pero SIN cambio visual

Al generar el handoff se re-sourcearon todos los scripts, así que estos PNG/PDF
tienen mtime de hoy, pero el `.R` no cambió → re-renderizan idénticos:
`study_area`, `spectral_profiles`, `auc_heatmap`, `alteration_comparison`,
`cuprite_comparison`, `physical_interpretation`. No requieren atención.

## 3. Nuevo set editable `_handoff` (NO referenciado por el .tex)

Set paralelo de fuentes editables (svglite → SVG con texto editable, fuentes Arial
portables, fondo transparente, **paleta de R intacta, cero recoloreo**). Pensado para
retoque fino en Inkscape (reposicionar etiquetas, anotar, componer plates), no para
compilar. Cada uno trae `.svg` + `.pdf` + `.png` @600 dpi.

Aplicado a las **5 figuras vectoriales**:
`pareto_fronts_handoff`, `auc_heatmap_handoff`, `spectral_profiles_handoff`,
`physical_interpretation_handoff`, `study_area_handoff`.

**NO** aplicado a las 2 figuras de mapa (`alteration_comparison`, `cuprite_comparison`):
su handoff embebía el raster basemap como base64 (~3–4 MB c/u) sin valor editable.
Se mantienen como **R-directo** (`alteration_comparison.*`, `cuprite_comparison.*`).

> Regla del ecosistema: figura vector puro → handoff; mapa con raster embebido
> (el SVG del handoff trae `data:image`) → R-directo sin handoff.

## 4. Estado de referencias .tex (sin cambios necesarios)

`paper/main.tex`, `paper/em_upload/main.tex` y `paper/nrr_submission/main_nrr.tex`
referencian los nombres base — todos presentes y vigentes. Nada que actualizar.
Si en el futuro se quisiera publicar la versión retocada en Inkscape de alguna de las
5 vectoriales, bastaría reemplazar el `.png`/`.pdf` base por el `_handoff` (mismo
contenido, solo cambia la fuente de generación).
