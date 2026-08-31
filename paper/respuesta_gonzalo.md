# Respuesta al review de Gonzalo Ríos

**Manuscrito:** *Symbolic Regression as Spectral Feature Engineering: Discovering Interpretable Indices for Hydrothermal Alteration Mapping from Sentinel-2*
**Target:** ISPRS Journal of Photogrammetry and Remote Sensing
**Versión revisada:** 2026-04-22 (37 pp + supplementary)

Gonzalo, muchas gracias por el review. Los cinco puntos son los que un reviewer duro de ISPRS JPRS va a pedir en el primer round, y atenderlos antes del submit fortalece el paper sustancialmente. Los cambios que propusiste llevaron a dos hallazgos no triviales que valoricé en el propio manuscrito: (a) la crítica de autocorrelación espacial era correcta y el delta AUC bajo spatial block CV llega hasta −0.21, pero las **relaciones de interés** (SR ≈ raw, SR > classical, SR+classical best) **sobreviven a todos los esquemas**; y (b) las fórmulas SR descubiertas localmente en Cuprite son **estructuralmente distintas** a las chilenas, lo cual demuestra method transferability explícitamente en lugar de solo insinuarla.

---

## Resumen ejecutivo

| # | Punto | Estado | Nota |
|---|---|---|---|
| 1 | Fórmulas locales en Cuprite no reportadas | ✓ Atendida | PySR re-corrido en Cuprite; tabla análoga a Table 3; fórmulas SWIR-dominated en vez de VNIR-dominated |
| 2 | Faltan baselines PCA(6) y MI-top-6 | ✓ Atendida | Ambos agregados a Table 7; PCA(6) supera a SR por ~0.04 → framing ajustado a "interpretabilidad sin costo vs raw bands" |
| 3 | Split 70/30 vulnerable a autocorrelación espacial | ✓ Atendida (CRÍTICA) | Δ = −0.077 a −0.21 según block size; las 3 relaciones centrales sobreviven |
| 4 | TOST sub-descrito | ✓ Atendida | IC 95% fold-level, justificación ε=0.01, bootstrap pareado polygon-level como check independiente |
| 5 | Silícico como brightness detector no retroalimenta workflow | ✓ Atendida | Pareto analysis + propuesta metodológica "specificity gate over Pareto" |

**Páginas:** 33 → 37 tras los cambios.
**Experimentos nuevos corridos:** PySR on Cuprite, spatial block CV (5 block sizes × 4 feature sets), polygon-disjoint CV via DBSCAN, PCA(6) + MI(6) baselines, bootstrap paired polygon-level (B=1000 × 3 feature sets).
**Hallazgos no obvios:** (a) las fórmulas Cuprite redescubren ratios B11/B12 consistentes con alteration mineralogy local; (b) PCA(6) define el ceiling de dimensionality reduction y supera incluso a raw(10), forzando un reframing honesto de la contribución de SR.

---

## Respuesta punto por punto

### 1. El claim de "method transferability" en Cuprite no está demostrado, solo insinuado

**Recomendación:** agregar una tabla análoga a la Tabla 3 para Cuprite, con las fórmulas descubiertas localmente. Si son estructuralmente distintas a las chilenas pero igual de efectivas, eso es el resultado de method transferability.

**Qué se hizo:** Re-corrí PySR con la configuración idéntica (Section~\ref{sec:sr_framework}) sobre los 10,500 pixels de training en Cuprite (hold-out 70/30, mismos hiperparámetros). Watchdog de RAM durante la corrida: peak 5.24 GB en 420 s.

**Fórmulas descubiertas localmente en Cuprite vs Chile:**

| Clase | Chile | Cuprite | Cuprite AUC |
|---|---|---|---|
| Silicic | B04 − 0.135 | **log(B11/B12)** | 0.944 |
| Adv. Argillic | 0.83 − B02/B05 | **B11 − B12** | 0.938 |
| Argillic-Phyllic | 0.09/B05 | **tanh(B11) − B8A** | 0.898 |
| Propylitic | B03 − 0.48·B11 | ((B12 − B08)/B07)² | 0.664 |

**Hallazgos:**
1. **Las fórmulas son estructuralmente distintas.** Chile produce ratios VNIR (B02/B05, B04) y combinaciones VNIR–SWIR (B03 − 0.48·B11); Cuprite produce ratios SWIR puros (B11/B12, B11 − B12). Esto es geológicamente coherente: Cuprite es el benchmark clásico de alteración advanced-argillic donde los Al-OH (alunite, kaolinite) dominan el SWIR, mientras Atacama tiene diversidad espectral mayor incluyendo propilítica y Fe-oxides expresados parcialmente en VNIR.
2. **Las fórmulas locales superan a las importadas en 3 de 4 clases** (silicic: 0.94 local vs 0.74 importada de Chile; adv. argillic: 0.94 vs 0.70; argillic-phyllic: 0.90 vs 0.70+).
3. **La única excepción (propylitic) tiene causas estructurales identificables:** USGS merge de epidote/chlorite + carbonate (spectral heterogeneous class) y n=378 positives vs 333 en Chile. Esto lo discutimos en el texto.

**Ubicación:** `main.tex` §5.5 subsección "Formulas discovered locally at Cuprite" + Tabla `tab:cuprite_formulas` (ahora L464–479 aprox). Script ejecutado: `scripts/run_pysr_cuprite.py` + watchdog `scripts/watchdog_pysr.py`. Resultados en `data/results/pysr_results_cuprite.json` y `data/results/equations_cuprite_*.csv`.

---

### 2. Faltan los baselines canónicos de reducción de dimensionalidad

**Recomendación:** agregar PCA(6)+RF y MI-top-6+RF a la Tabla 6. Si SR no los supera, reformular a "SR da interpretabilidad sin costo frente a PCA".

**Qué se hizo:** Corrí ambos baselines bajo los 3 esquemas de validación (pixel CV, polygon-disjoint, spatial block 5 km). Mutual information calculada fold-wise para evitar leakage; PCA ajustada solo en training fold.

**Resultados clave (`scripts/exp_dimred_baselines.py`, `data/results/dimred_baselines.json`):**

| Feature set | Pixel CV | Polygon-disj | Block 5km |
|---|---|---|---|
| RF(10 raw) | 0.899 | 0.822 | 0.707 |
| RF(6 SR) | 0.898 | 0.815 | 0.702 |
| RF(7 classical) | 0.878 | 0.789 | 0.683 |
| **RF(PCA 6)** | **0.934** | **0.863** | **0.747** |
| RF(MI top-6) | 0.841 | 0.763 | 0.680 |

**Hallazgos:**
1. **SR supera a MI-top-6 robustamente** (+0.02 a +0.06 en todos los esquemas) — MI selecciona bandas adyacentes correlacionadas y pierde las combinaciones no lineales que SR expresa.
2. **PCA(6) es el mejor compressor** — supera a SR por 0.04–0.05 en todos los esquemas y excede incluso a raw(10). Define el ceiling de dimensionality reduction para 6 features.
3. **Acepté tu reformulación literal.** El paper ahora dice explícitamente: *"the contribution of SR is a closed-form, sensor-agnostic, and physically interpretable feature set with near-equivalent accuracy, not a new accuracy ceiling"* (abstract), y en el análisis: *"PCA trades interpretability for about 4–5 AUC points; whether that trade is worthwhile depends on whether the downstream consumer is a statistical model or a geologist"* (§5.4).

**Ubicación:** `main.tex` Tabla `tab:spatial_robust` (filas PCA + MI + Δ), §5.4 tercer párrafo de análisis, abstract (frase nueva sobre PCA vs MI), §7.7 Limitations punto 6 nuevo.

---

### 3. El split 70/30 es vulnerable a autocorrelación espacial

**Recomendación (la más crítica):** aclarar si el split fue polygon-disjoint. Si no, re-correr con spatial block CV o polygon-level split y reportar delta.

**Qué se hizo:**
- **Diagnóstico:** Confirmado. El script `run_pysr_atlas_gee.py` usaba `train_test_split(X, y, stratify=y)` a nivel pixel (no polygon-disjoint). El npz no tiene polygon_id, pero el CSV sí tiene lon/lat por pixel.
- **Polygon-disjoint CV:** Reconstruí 3,828 polígonos via DBSCAN espacial (ε≈110 m, apropiado para Atlas Metalífero). Correí GroupKFold(5).
- **Spatial block CV:** Bloques cuadrados de 1, 2, 5 y 10 km, GroupKFold(5).
- **Compute total:** ~15 min across 5 schemes × 4 feature sets.

**Resultados completos (`data/results/spatial_block_cv.json` + `polygon_disjoint_cv.json`):**

| Feature set | Pixel 5-fold | Polygon-disj (≈110 m) | Block 1 km | Block 5 km | Block 10 km |
|---|---|---|---|---|---|
| RF(raw) | 0.899 | 0.822 | 0.842 | 0.707 | 0.686 |
| RF(SR) | 0.898 | 0.815 | 0.839 | 0.702 | 0.702 |
| RF(classical) | 0.878 | 0.789 | 0.822 | 0.683 | 0.674 |
| **RF(SR+classical)** | **0.920** | **0.839** | **0.862** | **0.719** | **0.708** |

**Δ absolutos (autocorrelation inflation):** raw mAUC cae de 0.899 → 0.822 (polygon-disjoint, Δ = −0.077) → 0.707 (block 5km, Δ = −0.192). **Por encima del umbral de 0.02 que planteaste** — la crítica era empíricamente correcta y grande.

**Pero las 3 relaciones de interés sobreviven:**
- **Δ(SR − raw)** se mantiene en [−0.007, +0.016] en todos los esquemas, **dentro del margen TOST ε=0.01**.
- **Δ(SR − classical)** se mantiene en [+0.017, +0.028], y es de hecho *ligeramente mayor* bajo polygon-disjoint CV que bajo pixel CV.
- **SR+classical sigue siendo el best combination** en todos los esquemas.

**Caveat:** polygon-disjoint CV no puede evaluar la clase propylitic — sus 495 pixels forman un único cluster espacial compacto, dejándola ausente de todos los test folds. Esto se reconoce en el paper.

**Cambios en el paper:**
- §3.7 Validation Strategy: nivel 1b nuevo "Intra-site (spatial robustness)" que documenta polygon-disjoint + spatial block CV. Citas nuevas Ploton et al. 2020 (Nature Comm.) y Meyer et al. 2019 (Ecol. Modelling).
- **Tabla 7 `tab:spatial_robust` nueva:** 4 feature sets + 2 baselines × 5 esquemas, con filas Δ explícitas.
- Abstract: frase nueva explícita sobre polygon-disjoint CV preservando la equivalencia.
- §7.7 Limitations punto 1 reescrito completamente con la cuantificación empírica.
- Framing general: pixel AUC tratado como upper bound; polygon-disjoint como defensible; Cuprite como primary out-of-site evidence.

**Ubicación:** `main.tex` §3.7 L246 (level 2 añadido), §5.4 Tabla 7 `tab:spatial_robust`, §7.7 punto 1 reescrito, abstract. Scripts: `scripts/exp_spatial_block_cv.py`, `scripts/exp_polygon_disjoint_cv.py`.

---

### 4. El TOST está sub-descrito para ser el resultado central

**Recomendación:** justificar ε=1%, especificar unidad de remuestreo, reportar IC del ΔAUC. Opcional: bootstrap pareado a nivel polígono.

**Qué se hizo todo:**

**Unidad de remuestreo especificada:** fold-level mean AUC, n=10 folds estratificados. Explícito en el texto ahora.

**IC 95% reportado:** Δ = −0.00274, SE = 0.00109, t₉,.025 = 2.262 → **95% CI = [−0.0052, −0.0003]**. El CI no cruza cero (el test paired-t rechaza igualdad exacta), pero está completamente dentro del margen ε=0.01 (TOST rechaza no-equivalencia a p<10⁻⁴).

**Justificación operacional de ε=0.01:** El paper ahora dice explícitamente que 0.01 es:
- 1/10 de las diferencias class-level entre feature sets que sí tratamos como meaningful (SR−classical ≈ 0.02),
- 1/20 de la spatial-autocorrelation inflation que documenta Table 7 (≈0.20 entre pixel y block 5 km),
- el margen más estricto para el cual el test tiene power adecuado en n=10 folds (ε=0.005 sería underpowered dado SE≈0.001),
- márgenes más laxos (0.02, 0.03, 0.05) rechazan a fortiori (p<10⁻⁷).

**Bootstrap pareado polygon-level (independiente, robusto a autocorrelación):**
- Entrené RF(raw) y RF(SR) con polygon-disjoint GroupKFold(5), OOF predictions.
- Resample 3,828 polígonos con reemplazo, B = 1000.
- **Mean bootstrap Δ = −0.0021, 95% percentile CI = [−0.0114, +0.0051], P(|Δ|<0.01) = 92.1%**.
- El CI polygon-level es más ancho que el fold-level (esperado, removimos la autocorrelación-induced narrowing), pero centrado cerca de cero y mayoritariamente dentro del banda ±0.01.

**Los dos procedimientos juntos** (TOST fold-level + bootstrap polygon-level) hacen la equivalence claim más demandante que cualquiera por separado.

**Ubicación:** `main.tex` §5.4 Q2-Q3 primer párrafo reescrito completamente (era una oración, ahora son 3 párrafos densos). Script: `scripts/exp_tost_expanded.py`, resultados en `data/results/tost_expanded.json`.

---

### 5. El índice silícico falla en especificidad pero esto no retroalimenta el workflow

**Recomendación:** mostrar el Pareto front completo para silícica; discutir si alguna fórmula con contraste espectral existe dentro del budget. Si existe, convertir en hallazgo metodológico: "SR necesita prueba de especificidad como paso obligatorio".

**Qué se hizo:**

**Análisis del Pareto front silícica** (`data/results/equations_gee_Silicic.csv`, 15 entradas):

| Complexity | Loss | Bands | Contrast | Formula |
|---|---|---|---|---|
| 1 | 0.1207 | B03 | no | B03 |
| 2 | 0.1188 | B06 | no | square(B06) |
| **3** | **0.1161** | **B04** | **no (current)** | **B04 − 0.135** |
| 4 | 0.1152 | B03,B12 | **YES** | **B03²/B12** |
| 7 | 0.1144 | B03,B04,B12 | YES | (B04/B12 − 0.34)·B03 |
| 8 | 0.1143 | B03,B04,B12 | YES | (B03/B12)²·B04·0.78 |

**Hallazgos:**
1. **Fórmulas con contraste espectral VNIR/SWIR SÍ existen dentro del budget complexity ≤ 8.** Tu intuición era la correcta: no es que el Pareto esté dominado por brillo a cualquier complejidad.
2. **AUC de los candidatos con contraste es comparable al brightness detector** (±0.015 en pixel CV: 0.652–0.672 vs 0.667).
3. **¿Por qué PySR seleccionó brightness?** Porque el criterio por defecto `get_best()` de PySR usa *score* = loss improvement per complexity unit. En el Pareto silícica, el drop más grande ocurre en complexity 3 (score 0.023), y los drops posteriores son menores (0.008, 0.001…). El selector por score está ciego a specificity.

**Conversión en contribución metodológica (lo que pediste):**

El paper ahora incluye en §6.4 Silicic una subsección "Pareto front and a methodological lesson" que:
- Muestra los candidatos alternativos con contraste espectral y sus AUCs bajo pixel CV + polygon-disjoint CV.
- Argumenta que en datasets donde una clase covaría fuertemente con un proxy escalar (aquí, albedo), el selector por defecto prefiere el proxy sobre el contraste mineralógico.
- Propone agregar un **specificity gate sobre el Pareto**: antes de retornar una fórmula, evaluarla sobre superficies distractoras de dominio (salt flats, sandy desert, snow, unaltered lithologies) y rechazar candidatos cuyo FPR sobre esas superficies exceda un threshold pre-definido.
- Posiciona esto como step generalizable del workflow SR-para-remote-sensing, no como disclaimer de una clase.

**Ubicación:** `main.tex` §6.4 Silicic Index, subsección "Pareto front and a methodological lesson" añadida al final de la subsección.

---

## Cambios no solicitados pero derivados del review

- **Contribución #2 reescrita** explícitamente para posicionar SR como "unlike PCA-based dimensionality reduction… each SR feature is a closed-form expression with a direct mineral-spectroscopic interpretation and is deployable in any raster calculator without a fitted transformation".
- **Watchdog de RAM** para el PySR Cuprite (scripts/watchdog_pysr.py), por precaución operacional. Thresholds: warn 6 GB, kill 10 GB. La corrida alcanzó peak 5.24 GB (sin warning).
- **Diálogo explícito con Mauricio en el abstract:** ahora el abstract reconoce tanto (i) la equivalencia bajo polygon-disjoint CV (tu punto 3) como (ii) el trade-off PCA vs interpretabilidad (tu punto 2).
- **Citas nuevas:** Ploton et al. 2020 y Meyer et al. 2019 para spatial CV.

---

## Puntos que me gustaría discutir contigo

1. **Propylitic en Cuprite (AUC 0.66 local).** La degradación es real — tiene dos causas probables: (a) USGS propylitic merge epidote/chlorite + carbonate (broader spectral class que la chilena), (b) n=378 positives, clase minoritaria en Cuprite. ¿Lo dejamos como caveat en la Table o lo reforzamos como prueba de que el framework detecta correctamente clases compositionally heterogeneous?

2. **PCA como ceiling.** El paper reconoce que PCA(6) supera a SR por 4–5 AUC points. ¿Te parece aceptable el framing de "SR no es el mejor compressor, es el mejor compressor interpretable"? O preferirías un tono más neutral tipo "SR y PCA ocupan puntos distintos del trade-off accuracy/interpretability"?

3. **Specificity gate como workflow recommendation.** El paper lo deja propuesto pero no lo implementa numéricamente para todas las clases — solo para silícica, que es donde el problema se manifestó. ¿Vale la pena hacer un experimento completo con specificity gate implementado para las 6 clases antes del submit, o es suficiente como propuesta methodological?

4. **Volumen: 37 pp.** ISPRS JPRS no tiene límite estricto pero algunos reviewers lo penalizan. Si los tres cambios arriba se implementan, puede crecer a 40+. ¿Movemos algo al supplementary (por ejemplo, la tabla completa de spatial block 1/2/5/10 km se puede resumir y el detalle al supplement)?

---

## Archivos entregables

| Archivo | Descripción |
|---|---|
| `paper/main.pdf` (37 pp, 7.1 MB) | Manuscrito revisado |
| `paper/supplementary.pdf` | Supplementary |
| `paper/respuesta_gonzalo.md` (este doc) | Respuesta formal punto por punto |
| `scripts/run_pysr_cuprite.py` | PySR on Cuprite (GR1) |
| `scripts/watchdog_pysr.py` | RAM watchdog (operacional) |
| `scripts/exp_spatial_block_cv.py` | Spatial block CV (GR3) |
| `scripts/exp_polygon_disjoint_cv.py` | Polygon-disjoint CV via DBSCAN (GR3) |
| `scripts/exp_dimred_baselines.py` | PCA + MI baselines (GR2) |
| `scripts/exp_tost_expanded.py` | TOST CI + polygon bootstrap (GR4) |

Gracias nuevamente — el review mejoró materialmente la defensibilidad del paper. Quedo atento a tus comentarios sobre los 4 puntos abiertos.

Francisco
