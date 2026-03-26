# Revision Checklist — RSE-D-26-01234

**Status:** Pre-submission (simulación de peer review)
**Fecha simulación:** 2026-03-26
**Deadline revisión:** 60 días desde decisión

---

## Quick Wins (solo edición de texto, < 30 min total)

- [x] **M2** — Acknowledge SR formula overfitting risk. Agregar párrafo en Methods o Discussion: "SR formulas were discovered on the full Region III dataset; intra-site AUC is optimistically biased. Cuprite, where formulas were not trained, constitutes the unbiased evaluation."
- [x] **M6** — Expandir discusión de ground truth quality differences. Hacer más prominente que Chile = field-mapped vs Cuprite = ASTER-derived. Ya está en Discussion pero reviewer pide más énfasis.
- [x] **M7** — Agregar caveats explícitos al class mapping Chile↔Cuprite. USGS "argillic" ≠ advanced argillic sensu stricto. Discutir que "phyllic" se traslapa con argillic-phyllic.
- [x] **m2** — Verificar word count ≤ 15,000 (incluyendo refs y captions).
- [x] **m3** — Nota: `\linenumbers` es para review mode. Remover en versión final de submission. (RSE dice "do not number lines" pero elsarticle review mode lo pone por defecto — verificar si el editorial system lo acepta.)
- [x] **m4** — Cranmer (2023): verificar si PySR paper fue publicado formalmente. Si no, marcar como "preprint" en la referencia.
- [x] **m5** — Justificar `select_k_features = 6`. Agregar: "six was chosen to match the number of alteration classes, ensuring each formula uses at most as many bands as there are target classes."
- [x] **m6** — Agregar 2-3 oraciones de contexto geológico regional al Study Areas (host lithologies, edades).
- [x] **m7** — Mejorar mapa de Chile con Natural Earth boundaries o agregar disclaimer "map lines delineate study areas and do not necessarily depict accepted national boundaries" (requerido por RSE).
- [x] **m8** — Agregar link a repo GitHub/Zenodo en Data Availability. Hacer repo público antes de submission.
- [x] **m9** — Reforzar que constantes (0.48, 0.135, 0.83) son empíricas, no derivadas de física mineral. Ya se menciona pero ser más explícito.
- [x] **RSE-req** — Abstract ≤ 250 palabras (DONE: 197 palabras).
- [x] **RSE-req** — Highlights file separado (DONE: highlights.txt).
- [x] **RSE-req** — AI declaration (DONE: declaración de Claude en main.tex).
- [x] **RSE-req** — Funding statement (DONE: "no specific grant").
- [x] **m1** — Discutir impacto de class imbalance en SR discovery (MSE dominado por clases mayoritarias). Párrafo breve en Discussion o Methods.

---

## Experimentos Necesarios

### E-M1: TOST equivalence test (ALTO) — COMPLETADO
**Resultado:** TOST p<0.001 at ε=0.01 para Chile Y Cuprite. Equivalencia formal demostrada.
**Objetivo:** Probar formalmente que SR features ≡ raw bands (no solo "no different").
**Método:** Two One-Sided Tests (TOST) sobre diferencias de AUC per-fold. H0: |ΔAUC| > ε (ej. ε=0.02). Si se rechaza, la equivalencia es formal.
**Input:** Per-fold AUCs de RF(6SR) y RF(10raw) — ya los tenemos en `ci_per_fold.json` y `rf_ovr_auc.json`.
**Esfuerzo:** 1 hora (script + integrar al paper).
**Riesgo:** Bajo — las diferencias son <0.003, debería pasar con ε=0.02.

### E-M3: PySR sensitivity analysis (MEDIO)
**Objetivo:** Mostrar que las fórmulas descubiertas son estables ante cambios de hiperparámetros.
**Método:** Re-correr PySR con 3 configs: (a) parsimony=0.001, (b) parsimony=0.01, (c) maxsize=15. Comparar si las mismas bandas y operaciones aparecen.
**Input:** Training data existente.
**Esfuerzo:** 4-8 horas (PySR es lento, ~30 min por clase × 6 clases × 3 configs).
**Riesgo:** Medio — si las fórmulas cambian mucho, debilita el paper. Mitigación: reportar "top bands" en vez de "exact formulas" como estables.

### E-M4: Mejorar DL baseline (BAJO-MEDIO) — COMPLETADO
**Resultado:** Best MLP (128h, lr=0.01, 100ep) mAUC=0.933 — comparable a SVM-RBF. Integrado al paper.
**Objetivo:** Mostrar que el MLP fue razonablemente entrenado.
**Método:** Grid search sobre MLP: {32, 64, 128} hidden units × {1e-2, 1e-3, 1e-4} lr × {30, 60, 100} epochs. Reportar learning curves para el mejor config.
**Input:** Training data existente.
**Esfuerzo:** 2-3 horas.
**Riesgo:** Bajo — si el MLP mejora un poco, sigue sin superar RF. Si mejora mucho (>0.92), reframing necesario.

### E-M5: Silicic index specificity test (ALTO) — COMPLETADO
**Resultado:** FP rate 99-100% en salares, desierto, nieve, roca no alterada. Confirmado como proxy de brillo. Degradado a "brightness anomaly detector" en el paper.
**Objetivo:** Demostrar que B04−0.135 no es solo un threshold de brillo genérico.
**Método:**
  (a) Muestrear pixels S2 de superficies brillantes no-silícicas conocidas: salares, suelo seco, urban, snow.
  (b) Calcular B04−0.135 para estas superficies.
  (c) Si > 0.135 consistentemente → confirma que es un proxy de brillo → degradar a "screening tool" o remover.
**Input:** GEE sampling de zonas conocidas (salares de Atacama, áreas urbanas).
**Esfuerzo:** 2-3 horas.
**Riesgo:** Alto — es probable que salares den positivo. Mejor reconocerlo honestamente.
**Alternativa:** Remover el silicic index de la lista de "alteration indices" y presentarlo como "brightness anomaly detector useful for screening in arid terrains."

### E-M8: False positive analysis (MEDIO)
**Objetivo:** Cuantificar qué litologías no-alteradas dan falsos positivos con cada SR index.
**Método:**
  (a) Muestrear pixels de litologías conocidas no-alteradas (granitos, basaltos, sedimentos) en Region III usando mapa geológico SERNAGEOMIN.
  (b) Aplicar los 6 SR indices y reportar false positive rate por litología.
**Input:** Mapa geológico SERNAGEOMIN (si disponible como shapefile) + S2 GEE.
**Esfuerzo:** 3-4 horas.
**Riesgo:** Medio — si los FP son altos para ciertas litologías, hay que discutirlo. Pero la honestidad fortalece el paper.

---

## Orden sugerido de ejecución

1. **Quick wins** (todos, 30 min) — limpian issues menores inmediatamente
2. **E-M1 TOST** (1 hora) — cierra el issue más importante del reviewer 1
3. **E-M5 Silicic specificity** (2-3 horas) — cierra el issue más importante del reviewer 2
4. **E-M4 DL grid search** (2-3 horas) — fortalece la comparación
5. **E-M8 False positives** (3-4 horas) — agrega valor práctico
6. **E-M3 PySR sensitivity** (4-8 horas) — el más costoso, hacer al final

**Total estimado: 2-3 días de trabajo.**

---

## Notas

- Los issues M1, M2, M5, M6, M7 son los que determinan aceptación vs segundo round de revisión.
- M3 (PySR sensitivity) puede discutirse como "future work" si el tiempo apremia, pero es mejor tener datos.
- La respuesta punto-a-punto al editor debe ser concisa y mostrar exactamente dónde se hizo cada cambio (línea/página).
