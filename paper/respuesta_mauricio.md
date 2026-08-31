# Respuesta al informe de Mauricio Latorre

**Manuscrito:** *Symbolic Regression as Spectral Feature Engineering: Discovering Interpretable Indices for Hydrothermal Alteration Mapping from Sentinel-2*
**Target:** ISPRS Journal of Photogrammetry and Remote Sensing
**Versión revisada:** 2026-04-22 (33 pp + 5 pp supplementary)

Mauricio, muchas gracias por el informe — fue enormemente útil tanto para consolidar el reframing post-desk-reject de RSE como para cerrar los flancos metodológicos que quedaban abiertos. Abajo va la respuesta punto por punto (numeración del informe) indicando qué se cambió y dónde.

---

## Resumen ejecutivo

| # | Recomendación | Prioridad | Estado |
|---|---|---|---|
| 1 | Definir contribución principal con una sola idea central | P1 | ✓ Atendida |
| 2 | Reescribir lista de contributions (framework first) | P1 | ✓ Atendida (5 → 4) |
| 3 | Leakage: fortalecer metodológicamente o endurecer redacción | P1 | ✓ Atendida (opción 2: hold-out 70/30 documentado) |
| 4 | Reforzar distinción method vs formula transferability | P1 | ✓ Atendida |
| 5 | Expandir fundamento geológico/espectroscópico por fórmula | P2 | ✓ Atendida (template aplicado a las 5 fórmulas) |
| 6 | Justificar mejor las clases y el problema geológico | P2 | ✓ Parcial (ya estaba cubierto en Study Areas/Intro) |
| 7 | Reforzar comparación con baselines y framing experimental | P2 | ✓ Atendida (Results reestructurado por Q1–Q4) |
| 8 | Elevar equivalencia SR $\approx$ raw como mensaje central | P2 | ✓ Atendida |
| 9 | Manejo del experimento cross-domain en land cover | P3 | ✓ Atendida (opción b: reducido a evidencia secundaria) |
| 10 | Sección de limitaciones más fuerte y explícita | P3 | ✓ Atendida (nueva §7.7 Limitations and scope) |
| 11 | Ajustar conclusión con tres mensajes | P3 | ✓ Atendida |

**Nota clave sobre el punto 3 (leakage):** descubrimos al revisar el script `run_pysr_atlas_gee.py` que PySR **ya había sido corrido sobre un split 70/30**, pero la metodología del paper lo reportaba incorrectamente como "discovered on the full dataset". La diferencia empírica entre AUC en el 30% locked hold-out y el 100% dataset es <0.01 en todas las clases. El paper ahora refleja correctamente la opción 2 del informe (hold-out documentado), sin necesidad de correr nada nuevo.

**Cambios netos:** abstract, contributions, Results completa, Cuprite, Physical Interpretation completa, Limitations nueva, Conclusions. **Páginas:** 26 → 33.

---

## Respuesta punto por punto

### 1. Definición de la contribución principal y foco del manuscrito

**Recomendación:** instalar una sola frase central que resuma el aporte principal como "framework transferible para spectral feature engineering", no como "descubrimos nuevos índices".

**Qué se hizo:**
- Abstract reescrito; abre con: *"The central contribution of this work is not a single spectral index, but a transferable methodology for automated spectral feature discovery"* (opción B del informe, adaptada).
- Intro: ya contenía desde la revisión post-RSE la distinción *"we show that the framework---not any particular formula---is the transferable product"* (L71).
- Conclusión: primera frase reforzada con *"The central claim of this work concerns method transferability… rather than the universal optimality of any particular formula"* (L632).

**Ubicación:** `main.tex` L41–48 (abstract), L71 (intro), L632 (conclusion).

---

### 2. Reescritura de la lista de contribuciones

**Recomendación:** 4 contribuciones jerarquizadas con framework como #1 y demostraciones como evidencia secundaria.

**Qué se hizo:** la lista de 5 contributions se consolidó en 4 (fourfold), siguiendo la estructura sugerida:

1. Framework re-deployable de SR para spectral feature engineering
2. Equivalencia near-lossless (6 SR ≈ 10 bandas) + interpretabilidad física (fusionado, no separado)
3. Replicación metodológica en Cuprite (cross-continent) como evidencia de method transferability
4. Complementariedad SR + clásicos y best overall performance

Se eliminó "six novel spectral indices" como contribución separada (estaba en el nivel incorrecto de jerarquía).

**Ubicación:** `main.tex` L73–82.

---

### 3. Riesgo de leakage y validación del descubrimiento SR

**Recomendación:** agregar experimento (nested o hold-out 70/30) o endurecer redacción.

**Qué se hizo — corresponde a la opción 2 del informe, ejecutada sin costo computacional adicional:**

Al revisar `scripts/run_pysr_atlas_gee.py` se confirmó que PySR ya fue corrido sobre un split 70/30 estratificado (random_state=42), con AUC reportados sobre el 30% locked. Sin embargo, la versión anterior del paper reportaba incorrectamente que las fórmulas "were discovered on the full Region III training dataset… intra-site AUC values may be optimistically biased".

Se hizo verificación empírica comparando AUC en el 30% locked vs 100% dataset:

| Clase | AUC 30% locked | AUC 100% | Δ |
|---|---|---|---|
| Silicic | 0.662 | 0.667 | +0.005 |
| Adv. Argillic | 0.716 | 0.719 | +0.003 |
| Argillic-Phyllic | 0.628 | 0.623 | −0.005 |
| Propylitic | 0.910 | 0.907 | −0.003 |
| Iron Oxide | 0.744 | 0.754 | +0.010 |
| Potassic-Skarn | 0.770 | 0.763 | −0.007 |

Δ máxima 0.010, consistente con el hecho de que las fórmulas SR son funciones cerradas determinísticas (no hay parámetros adicionales que se ajusten al resto de los datos).

**Cambios en el paper:**
- **§3.3 SR Framework (L195):** reescrito para documentar el split 70/30 y la verificación robustness.
- **§3.7 Validation Strategy (L246):** nivel "Intra-site" renombrado a "Intra-site (locked hold-out)" con separación explícita: SR standalone en 30% locked; classifiers downstream (RF) en 5-fold CV.
- **Caption Table 3:** "5-fold CV" → "locked 30% held-out subset of Region III (unseen during PySR discovery)".
- **Caption Table 4:** nota aclaratoria de que AUC en 30% locked vs 5-fold CV difieren por <0.01.
- Eliminada la frase "optimistically biased" en toda la Discussion.
- El rol de Cuprite como "unbiased evaluation" sigue enfatizado, pero ahora como *primary* evidence de method transferability, no como parche del leakage.

**Ubicación:** `main.tex` L195–202 (§3.3), L245–253 (§3.7), L262 (Table 3 caption), L297 (Table 4 caption).

---

### 4. Distinción entre formula transferability y method transferability

**Recomendación:** convertir esa distinción en un eje argumental explícito a lo largo del paper.

**Qué se hizo:**
- **Intro L71:** ya contenía desde la versión previa *"We emphasize the distinction between formula transferability… and method transferability. The latter is the primary contribution"*. Se reforzó marcando la distinción con `\emph`.
- **Results opening:** párrafo nuevo de apertura introduce la distinción ANTES de cualquier subsección y la conecta explícitamente con Q4.
- **Cuprite section (§5.5):** renombrada *"Q4: Replication of the feature engineering pattern at Cuprite, Nevada"*; apertura: *"The Cuprite site constitutes the primary unbiased test of method transferability in this study"*.
- **Discussion §7.3:** la distinción abre la subsección y se repite al interpretar la heterogeneidad cross-site.
- **Conclusions:** primera frase del párrafo de cierre menciona explícitamente method transferability (y qué NO se reclama).

**Ubicación:** `main.tex` L71 (intro), L256–266 (Results roadmap), L395–399 (Cuprite §5.5), L553 (Discussion), L632 (Conclusions).

---

### 5. Reforzar el fundamento geológico y espectroscópico

**Recomendación:** expandir cada fórmula con formato "Fórmula → significado espectral → interpretación mineralógica → limitación Sentinel-2".

**Qué se hizo:** las 5 subsecciones de §6 Physical Interpretation se reescribieron siguiendo exactamente ese template, con 4 párrafos etiquetados (`\emph{Spectral mechanism.}`, `\emph{Mineralogical interpretation.}`, `\emph{Performance.}`, `\emph{Sentinel-2 limitation.}`):

- **Propylitic ($B_{03} - 0.48 B_{11}$):** absorciones Mg-OH de chlorite/epidote a 2.25–2.35 μm caen en B12 ancha → SR recupera el slope broadband como proxy.
- **Advanced Argillic ($0.83 - B_{02}/B_{05}$):** hallazgo no-obvio — SR descubre un discriminador VNIR, no SWIR (el Al-OH diagnóstico de kaolinite/alunite a 2.17/2.20 μm no se resuelve en S2).
- **SWIR Alteration ($(\sqrt{B_{12}} - B_{11})^2$):** explicación de por qué funciona como detector binario pero no class-specific.
- **Silicic ($B_{04} - 0.135$):** framework explica por qué la reststrahlen de silice (~9 μm, TIR) no es accesible a S2 ni ASTER SWIR, y SR correctamente elige brightness como mejor proxy.
- **Potassic-Skarn:** compositionally heterogeneous class → dos causas de poor transferability desglosadas.

Se añadió un preámbulo a §6 que explicita los 4 ejes de análisis.

**Ubicación:** `main.tex` L464–495 (§6 reescrita completa).

---

### 6. Justificación de las clases y del problema geológico

**Recomendación:** ampliar por qué las 6 clases son un test exigente y geológicamente relevante.

**Qué se hizo:** la justificación de las clases como test case difícil ya estaba cubierta:
- **Abstract:** *"a challenging test case with six spectrally similar classes"*.
- **Intro L71:** *"a challenging test case with six spectrally similar target classes, validated at three sites spanning two continents and two distinct geological provinces"*.
- **Study Areas §2.1 L91–92:** lista mineralogía diagnóstica por clase.
- **Physical Interpretation §6 (ahora expandida):** por qué cada clase es difícil en Sentinel-2.
- **Limitations §7.7:** punto 3 (hard upper bound SWIR) articula por qué el test es exigente.

No se agregó una subsección nueva porque habría duplicado contenido. La justificación es robusta, solo dispersa.

**Ubicación:** `main.tex` L41 (abstract), L71 (intro), L91 (study areas), L467 (interpretation preamble), L628–646 (Limitations).

---

### 7. Reforzar la comparación con baselines y el framing experimental

**Recomendación:** reorganizar Results para que cada experimento responda una sola pregunta (Q1–Q4).

**Qué se hizo:** la sección Results completa se reestructuró:

1. Párrafo de apertura nuevo con las 4 preguntas explícitas en un environment `description` y referencias cruzadas a cada subsección.
2. Subsecciones renombradas:
   - *Discovered Spectral Indices* → **Q1: Discovered formulas as standalone indices**
   - *Comparison with Classical Indices* → **Q1: Standalone performance against classical indices**
   - *Binary Alteration Detection* → **Q1: Binary alteration detection with a single SR formula**
   - *SR as Feature Engineering: Ensemble Results* → **Q2 and Q3: Feature engineering equivalence and complementarity**
   - *External Validation Cuprite* → **Q4: Replication of the feature engineering pattern at Cuprite**
   - *Sub-classification* → **Practical application: sub-classification of undifferentiated zones**
3. Transición Q1→Q2 añadida al final de §5.3: *"the strongest contribution of SR may lie not in the universality of any particular standalone formula, but in its value as a feature engineering mechanism, which is the subject of Q2"* (texto del informe aplicado casi literal).

**Ubicación:** `main.tex` L256–266 (roadmap Q1–Q4), L268–420 (subsecciones renombradas).

---

### 8. Reforzar el resultado de equivalencia SR features vs raw bands

**Recomendación:** elevar el hallazgo de equivalencia 6 SR ≈ 10 bandas (TOST) al nivel de mensaje central.

**Qué se hizo:**
- **Abstract:** frase nueva prominente *"The key empirical result supporting this claim is that a compact set of SR-derived features preserves nearly all the discriminative information contained in the original multispectral bands"*, seguida del TOST.
- **Contributions:** la equivalencia con TOST es la #2 de las 4 contribuciones (antes estaba diluida).
- **Results §5.4 (Q2 y Q3):** la tabla 5 se presenta con el lead *"Table~\ref{tab:ensemble} presents the central result of this study"*.
- **Discussion §7.1:** *"The central finding of this study"*.
- **Conclusion finding #1:** *"formally demonstrating lossless compression into interpretable features"*.

**Ubicación:** `main.tex` L41 (abstract), L76 (contribution #2), L369 (Results opening of Q2), L505 (Discussion), L635 (Conclusion finding #1).

---

### 9. Manejo del experimento cross-domain en land cover

**Recomendación:** decidir entre mantener como evidencia secundaria (b), reducir (b'), o mover a supplementary (c).

**Qué se hizo:** opción (b) — mantener como evidencia secundaria en el main, con el framing ajustado. Razones:

- El paper entero está reframed en torno a method transferability; el experimento land cover es el **único** test cross-domain (Cuprite es cross-site dentro del mismo dominio geológico). Moverlo al supplementary debilita ese eje.
- El experimento ocupa 2 párrafos — no distrae narrativamente.
- Los resultados detallados (Tabla completa de clases) ya están en Supplementary Table S1.

**Cambios:**
- Título: *"Applicability Beyond Alteration Mapping"* → **"Cross-domain proof-of-concept: land cover classification"**.
- Apertura conecta explícitamente con Cuprite (Q4) y aclara *"as a secondary test—not to shift the focus away from geology"*.
- Cierre con literal del informe: *"The main scientific contribution of this paper remains geological remote sensing; this cross-domain result is reported as supporting evidence for framework generality"*.

**Ubicación:** `main.tex` L620–625 (§7.8).

---

### 10. Crear una sección de limitaciones más fuerte

**Recomendación:** subsección explícita "Limitations and scope" con 5 puntos.

**Qué se hizo:** nueva subsección §7.7 **"Limitations and scope"** (`\label{sec:limitations}`), insertada antes de Applicability. Consolida los 5 puntos del informe con cross-references a las subsecciones detalladas (evita duplicación):

1. **Standalone-index optimism on the training site is small but not zero** — con la evidencia empírica Δ AUC <0.01 entre 30% locked y full dataset.
2. **Formula transferability varies by class** — rango cross-site 0.54–0.86, con el giro explícito "lo que transfiere robustamente es el framework".
3. **Sentinel-2 SWIR como hard upper bound** — enumera las absorciones diagnósticas específicas (kaolinite 2.17, alunite 2.20, chlorite/epidote 2.25–2.35 μm) que caen dentro de bandas anchas.
4. **Class imbalance** — propylitic 333 vs 10,167 negatives; RF downstream mitiga con balanced weights.
5. **No universalidad de fórmulas** — training context specific; el producto transferible es el framework.

Cierre: *"These limitations define the scope of our claims but do not weaken the central finding"* — limitations como honestidad controlada, no como debilidad.

**Ubicación:** `main.tex` L604–617 (§7.7 completa).

---

### 11. Ajustar la conclusión con tres mensajes

**Recomendación:** cerrar con (1) qué aporta, (2) qué NO se reclama, (3) por qué importa.

**Qué se hizo:** el párrafo final de Conclusions se reemplazó por 3 mensajes etiquetados con `\emph`:

1. ***What this paper contributes.*** Framework como outcome metodológico; equivalencia SR $\approx$ raw + replicación en Cuprite como soporte empírico principal.
2. ***What this paper does not claim.*** Fórmulas no universalmente óptimas; no superan clásicos en todo setting; transferibilidad standalone varía por clase. Producto transferible: la estrategia de descubrimiento, no una librería fija de índices.
3. ***Why this matters.*** Puente domain-driven ↔ data-driven; aplicable cross-domain (vegetación, suelo, agua, urbano); workflow operacional en un nuevo sitio.

Se conservan los 6 findings enumerados (ya estaban) entre la apertura y el cierre con 3 mensajes.

**Ubicación:** `main.tex` L657–663 (párrafo final reemplazado).

---

## Cambios no solicitados directamente pero derivados del informe

- **Labels LaTeX nuevos** para cross-references: `sec:q1_standalone_discovered`, `sec:q1_vs_classical`, `sec:q1_binary`, `sec:q2q3_feature_engineering`, `sec:cuprite_results`, `sec:crossdomain`, `sec:limitations`.
- **Verificación empírica del leakage** (Δ AUC 30% locked vs 100%) documentada en el propio paper como robustness check.
- **Consistencia editorial:** eliminadas todas las menciones de "external validation" (en el sentido de "add-on") — ahora Cuprite es "independent validation" y "primary evidence".

---

## Puntos abiertos y decisiones que me gustaría discutir

1. **Experimentos adicionales opcionales (puntos 3 y 7 del informe):** Mauricio propuso como opción 1 un nested discovery-validation protocol por folds. No lo implementamos porque (a) el hold-out 70/30 ya existía y la verificación empírica muestra Δ AUC <0.01, y (b) el coste computacional sería ~30 min × 5 folds × 6 clases = ~15 h en el cluster. Si consideras que un reviewer exigirá el protocol anidado, podemos agregarlo antes del submit.

2. **Sección §7.7 Limitations and scope duplica parcialmente §§7.4–7.6.** Mantuve las subsecciones detalladas (SWIR, Ground Truth, Methodological) como evidencia respaldatoria y §7.7 como resumen sintético. Si prefieres consolidar todo en §7.7 y eliminar las subsecciones previas, es una edición de 30 min.

3. **Reordenamiento de Discussion:** el orden actual es §7.1–7.3 (hallazgos) → §7.4–7.6 (limitaciones dispersas) → §7.7 (Limitations and scope) → §7.8 (Cross-domain). Podría ser más natural §7.1–7.3 → §7.8 (cross-domain) → §7.4–7.7 (todo lo de limitations junto). Lo dejé como está para no mover 4 subsecciones, pero es fácil de cambiar.

---

## Archivos entregables

| Archivo | Páginas | Tamaño | Notas |
|---|---|---|---|
| `paper/main.pdf` | 33 | 7.1 MB | Manuscrito principal revisado |
| `paper/supplementary.pdf` | 5 | 0.6 MB | Supplementary (sin cambios de contenido) |
| `paper/main.tex` | — | 67 KB | Fuente revisada |
| `paper/respuesta_mauricio.md` | — | — | Este documento |

---

Gracias otra vez por la revisión. El informe fue particularmente valioso para transformar la opción 3 (solo honesty) de la sección 3 del informe en la opción 2 (hold-out documentado) sin necesidad de rehacer los experimentos. Me quedo disponible para resolver cualquiera de los tres puntos abiertos antes de enviar a ISPRS.

Francisco
