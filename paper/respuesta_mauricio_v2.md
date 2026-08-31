# Respuesta al informe final de Mauricio Latorre

**Manuscrito:** *Symbolic Regression as Spectral Feature Engineering: Discovering Interpretable Indices for Hydrothermal Alteration Mapping from Sentinel-2*
**Target:** ISPRS Journal of Photogrammetry and Remote Sensing
**Versión revisada:** 2026-04-27 (37 pp + supplementary 6 pp)
**Documento previo:** `respuesta_mauricio.pdf` (22-04-2026, atiende primer informe del 20-04)

---

Mauricio, gracias por la segunda pasada. Tu informe final es mucho más corto que el primero (lo que es buena señal) y todos los puntos eran accionables sin re-correr experimentos. Los apliqué directamente al manuscrito; este documento es el registro punto por punto, paralelo al primero.

**Probabilidad estimada por ti:** 80–90 % pasar a revisión, 55–70 % aceptación final tras major revision favorable. Coincido con tu lectura.

---

## Resumen ejecutivo

| # | Punto | Categoría | Estado |
|---|---|---|---|
| 1 | Nested validation | Mayor | Diferida (Opción C — explico abajo) |
| 2 | Suavizar Abstract/Conclusion | Mayor | **Aplicada** |
| 3 | Reordenar Discussion (consolidar 6.4–6.6 en 6.7) | Mayor | **Aplicada** (8 → 6 subsecciones) |
| 4 | Cuidar el experimento land cover ("confirms" → "supports") | Mayor | **Aplicada** |
| 5 | Caracteres corruptos en Supplementary | Menor | **Aplicada** (causa raíz: faltaba `\usepackage[T1]{fontenc}`) |
| 6 | `Supplementary Table~??` faltante | Menor | **Aplicada** (cross-ref reemplazada por `Supplementary Table~S5` literal) |
| 7 | Inconsistencia de numeración Section 4 vs 4.5 | Menor | **Aplicada** |
| 8 | Suavizar "unbiased" → "independent" para Cuprite | Menor | **Aplicada** |
| 9 | Mejorar Data Availability / repo público con README | Menor | **Aplicada** (README mínimo creado en raíz del repo) |

**Adicionales no solicitados pero requeridos por ti vía correo:** reordenamiento de autoría (ahora Parra → Ríos → Latorre), 5 afiliaciones nuevas (UOH-Bioingeniería, SYSTEMIX, CGR-Millennium, CMM-UChile, INTA-UChile), agradecimientos completos (ANILLO ACT210004, ACE210010, ICN2021_044, FB210005, FONDECYT 1230194, Núcleo UOH).

---

## Respuesta punto por punto

### 1. Nested validation (mejora mayor)

**Recomendación:** "si hay tiempo, hacerlo. Aunque el 70/30 es suficiente para pasar a revisión, un nested protocol elevaría la probabilidad de aceptación hacia el rango alto."

**Decisión:** **Diferir a fase de major revision.** Tu propio informe dice que el 70/30 es suficiente para pasar a revisión y que sin nested validation el paper sigue siendo competitivo. La inversión es ~1–2 h de desarrollo del script más ~15 h de cómputo (5 outer folds × 6 clases × ~30 min/PySR run). Tres consideraciones nos llevaron a postergarlo:

1. **Riesgo asimétrico antes del submit.** Un resultado inesperado en la nested CV (e.g., divergencia con el 70/30) requeriría reabrir la discusión metodológica justo antes de enviar.
2. **Disponibilidad anticipada para revisión.** Si un reviewer metodológico lo pide en el primer round, podemos correr la nested CV durante el ciclo de respuesta (típicamente 8–12 semanas en ISPRS) y entregarlo como anexo de la replica. Eso convierte una crítica esperable en evidencia adicional sin retrasar el submit.
3. **Estado actual del blindaje metodológico.** El manuscrito ya tiene tres líneas de defensa contra optimismo metodológico: split 70/30 con hold-out bloqueado, polygon-disjoint CV via DBSCAN, y spatial block CV a 1/2/5/10 km. Para un primer round, esto se considera robusto en la literatura geoespacial reciente (Ploton et al. 2020, Meyer et al. 2019).

Si tras leer este resumen prefieres que revierta y corra la nested CV antes del submit, dime y la incluyo (agrega ~2–3 días al timeline).

---

### 2. Suavizar Abstract/Conclusion (mejora mayor)

**Recomendación:** la frase "applicable to any multispectral classification problem" es muy fuerte; bajarla a "applicable to a broad range of multispectral classification problems where labeled data are available."

**Qué se hizo:** aplicada literal en tres lugares del manuscrito:

- **Abstract (línea final):** "applicable to any multispectral classification problem where compact, interpretable features are preferred" → "applicable to a **broad range of multispectral classification problems where labeled pixels are available** and compact, interpretable features are preferred over black-box models."
- **Conclusions, item de land cover:** "confirming that the framework generalizes beyond geological applications" → "**supporting the generalization of the framework to non-geological multispectral classification problems.**"
- **Conclusions, párrafo final ("Why this matters"):** "directly applicable to other classification domains... and to any multispectral sensor" → "**broadly applicable across classification domains... and across multispectral sensors where labeled pixels are available**."

**Ubicación:** `main.tex` línea 41 (Abstract), línea 705 (Conclusions item), línea 714 (Why this matters).

---

### 3. Reordenar Discussion (mejora mayor)

**Recomendación:** consolidar parte de 6.4–6.6 dentro de 6.7, o dejar 6.7 como cierre más breve.

**Qué se hizo:** estructura nueva del Discussion, de 8 a 6 subsecciones:

| Antes | Después |
|---|---|
| 6.1 SR as Spectral Feature Engineering | 6.1 SR as Spectral Feature Engineering |
| 6.2 Standalone Index Performance | 6.2 Standalone Index Performance |
| 6.3 Cross-site Transferability | 6.3 Cross-site Transferability |
| 6.4 Limitations of the SWIR Configuration | *(eliminada — redundante con item 3 de la enumeración del 6.7)* |
| 6.5 Ground Truth Quality | *(condensada en bullet nuevo dentro de la enumeración de Limitations)* |
| 6.6 Methodological Considerations | 6.4 Methodological Considerations |
| 6.7 Limitations and scope | 6.5 Limitations and scope (con bullet de ground truth integrado) |
| 6.8 Cross-domain proof-of-concept | 6.6 Cross-domain proof-of-concept |

**Justificación de los movimientos:**

- **6.4 SWIR eliminada completa:** su contenido ("Sentinel-2's two SWIR bands cannot capture fine absorption features...") ya estaba en el item 3 de la enumeración de 6.7, en versión más detallada. Mantenerla duplicaba.
- **6.5 Ground Truth condensada:** los tres párrafos originales (Atlas Metalífero scale, Cuprite ASTER-derived ground truth, USGS argillic vs phyllic class mapping) los fundí en un bullet único dentro de la enumeración de Limitations, reteniendo todos los puntos sustantivos.
- **6.6 Methodological Considerations** quedó intacta como subsección independiente — no es estrictamente una limitación sino una explicación de las elecciones metodológicas (complejidad ≤ 8, OvR, comparación con DL).

**Resultado:** −1 página (de 38 a 37), Discussion más limpia, sin pérdida de contenido sustantivo.

**Ubicación:** `main.tex` §6.

---

### 4. Cuidar el experimento land cover (mejora mayor)

**Recomendación:** está bien declarado como prueba secundaria, pero la frase "confirms that the framework generalizes beyond geological applications" podría bajarse a "supports".

**Qué se hizo:** cambio aplicado en los tres lugares donde aparecía la frase:

- **Abstract:** "confirming that the framework generalizes beyond geological applications" → "**supporting that the framework generalizes beyond geological applications**".
- **Discussion §6.6 (cross-domain proof-of-concept):** "this cross-domain result is reported as supporting evidence for framework generality" — ya estaba en este tono, no requería cambio adicional.
- **Conclusions, item de bullets:** "confirming that the framework generalizes beyond geological applications" → "**supporting the generalization of the framework to non-geological multispectral classification problems**".

**Ubicación:** `main.tex` líneas 41, 688, 705.

---

### 5. Caracteres corruptos en Supplementary (mejora menor obligatoria)

**Recomendación:** corregir "Ingenier ΩΩimmediate´ıa" y "Matem ΩΩimmediate´atica" antes del envío.

**Qué se hizo:** identifiqué la causa raíz — el `supplementary.tex` no tenía `\usepackage[T1]{fontenc}` en el preámbulo. Sin T1 fontenc, los acentos en LaTeX (`\'i`, `\'a`) caen al encoding OT1 default, que carece de glyphs apropiados para muchos diacríticos en contextos ciertos (tablas, paréntesis con espacios), produciendo glyphs de fallback corruptos como los que viste.

**Fix:** agregué `\usepackage[T1]{fontenc}` tanto al `supplementary.tex` como al `main.tex` (este último también lo necesitaba como precaución). Recompilado y verificado: 0 caracteres corruptos restantes.

**Ubicación:** `supplementary.tex` línea 4, `main.tex` línea 7.

---

### 6. `Supplementary Table~??` faltante (mejora menor obligatoria)

**Recomendación:** "En el manuscrito aparece 'Supplementary Table ??' en la sección land cover. Eso es un error editorial visible."

**Qué se hizo:** la causa era una `\ref{tab:s_landcover}` en `main.tex` que apuntaba a un label definido en `supplementary.tex`. Como `pdflatex` compila los dos archivos por separado, esa cross-reference no resuelve y se imprime como `??`.

**Fix:** reemplacé la `\ref` por el número literal:

```latex
% Antes
Full results are reported in Supplementary Table~\ref{tab:s_landcover}.

% Después
Full results are reported in Supplementary Table~S5.
```

Esto es consistente con el resto del manuscrito, que ya usaba `Supplementary Table~S1` en formato literal en el §4.4 (línea 446).

**Bonus:** mientras inspeccionaba la tabla S5 detecté un bug pre-existente en su declaración — el spec de columnas era `llccccccc` (9 columnas) pero los datos tenían 10 columnas, lo que iba a producir errores `Extra alignment tab has been changed to \cr` en compilación. Lo corregí a `llcccccccc`.

**Ubicación:** `main.tex` línea 688, `supplementary.tex` línea 175.

---

### 7. Inconsistencia de numeración Section 4 vs Cuprite 4.5 (mejora menor obligatoria)

**Recomendación:** "El texto dice 'Section 4' para Cuprite, pero en el PDF Cuprite está dentro de Results como 4.5 y la discusión en sección 6. Conviene revisar todas las referencias internas."

**Qué se hizo:** revisé todas las cross-references a Cuprite. La gran mayoría usa `\ref{sec:cuprite_results}` que resuelve auto a 4.5. Encontré una sola inconsistencia: en §6.3 (Cross-site Transferability) había un "Section~\ref{sec:results}, Tables ensemble and cuprite" que apuntaba al §4 completo (Results) cuando el contexto se refería específicamente a Cuprite (§4.5). Lo cambié a `sec:cuprite_results`.

**Ubicación:** `main.tex` línea 616.

---

### 8. Suavizar "unbiased" → "independent" para Cuprite (mejora menor obligatoria)

**Recomendación:** "Cuprite es independiente, pero su ground truth viene de ASTER/USGS, no de campo directo. Yo usaría 'independent' más que 'unbiased', porque el propio paper reconoce limitaciones del ground truth de Cuprite."

**Qué se hizo:** revisé los dos usos de "unbiased" en el manuscrito:

- **Línea 199 (Methods):** "the unbiased evaluation of standalone index AUC reported in Section X" — este uso se refiere a la evaluación sobre el hold-out 30 % bloqueado, que es estadísticamente unbiased. Lo dejé.
- **Línea 451 (Cuprite, Results):** "The Cuprite site constitutes the primary unbiased test of method transferability" — este es exactamente el caso que mencionas. Cambiado a **"primary independent test"**.

**Ubicación:** `main.tex` línea 451.

---

### 9. Mejorar Data Availability / repo público (mejora menor obligatoria)

**Recomendación:** "si se menciona GitHub, asegurar que el repositorio esté público, limpio, con README, scripts reproducibles y fórmulas."

**Qué se hizo:**

- **Repo público:** verificado, retorna HTTP 200. Sincronizado con `master` local en commit `e3051c8`.
- **README mínimo creado** en la raíz (`README.md`, ~80 líneas). Contiene: cita del paper con la autoría correcta, estructura del repo, dependencias mínimas (PySR + scientific Python + GeoPandas + GEE), pipeline mínimo de 5 scripts ordenados para reproducir resultados principales, lista de scripts de robustez, ubicación de fórmulas en `data/results/equations_*.csv`, fuentes de datos, contacto.
- **Fórmulas:** ya estaban como CSV plano en `data/results/equations_atlas_*.csv` (Chile training), `equations_cuprite_*.csv` (validación externa), `equations_lc_*.csv` (land cover). Documentado en el README.
- **Scripts:** los 40+ scripts de la carpeta `scripts/` están con nombres descriptivos y son ejecutables; el README marca cuáles son del pipeline principal vs robustez.

**Pendiente futuro (no bloqueante para submit):** decidir LICENSE — sin definir aún. Sugerencia: MIT para código y CC-BY 4.0 para datos derivados. Te consulto antes del submit.

**Ubicación:** `/README.md` (nuevo).

---

## Cambios administrativos solicitados por correo (25-04)

Aplicados en bloque junto con los puntos del informe final:

- **Reordenamiento de autoría:** Parra (CITIAPS) → Ríos (DIM-UChile) → Latorre (5 afiliaciones). Aplicado en `main.tex`, `supplementary.tex` y `CRediT Author Statement`.
- **5 afiliaciones nuevas para ti:** Lab Bioingeniería UOH, SYSTEMIX UOH, Millennium CGR, CMM UChile, INTA UChile. Direcciones y postcodes incluidos según lo que mandaste.
- **Acknowledgements ampliados:** ANID Anillo Regular ACT210004, ACE210010, ICN2021_044, FB210005, FONDECYT 1230194, Fondo Interdisciplinario y Proyecto Núcleo UOH. Atribuidos explícitamente a M.L.

---

## Estado del manuscrito

- **Páginas:** 37 (main) + 6 (supplementary) = 43 total.
- **Compilación:** 0 errores, 0 referencias indefinidas, 0 caracteres corruptos.
- **Versión:** 2026-04-27.
- **Listo para submit:** sí, condicional al cierre de los 4 puntos abiertos con Gonzalo.

---

## Archivos entregables

| Archivo | Descripción |
|---|---|
| `paper/main.pdf` (37 pp, 7.3 MB) | Manuscrito revisado post-informe final |
| `paper/supplementary.pdf` (6 pp) | Material suplementario actualizado |
| `paper/respuesta_mauricio.pdf` (7 pp) | Respuesta al primer informe (20-04) |
| `paper/respuesta_mauricio_v2.pdf` (este doc) | Respuesta al informe final (25-04) |
| `paper/respuesta_gonzalo.pdf` (6 pp) | Respuesta a los 5 puntos de Gonzalo |
| `README.md` (nuevo) | Documentación del repositorio |

Gracias nuevamente — el doble ciclo dejó el paper más blindado de lo que estaba en abril. Quedo atento a cualquier comentario adicional.

Francisco
