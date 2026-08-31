# Diff: informe Mauricio vs manuscrito revisado (main.tex @ 2026-04-14)

Clasificación por esfuerzo real pendiente, tras el reframing post-desk-reject.

## Ya cubierto — solo pulir (low effort)

| # | Recomendación | Estado actual | Acción |
|---|---------------|---------------|--------|
| ML1 | Contribución principal como framework | Abstract L41, Intro L71: "framework---not any particular formula---is the transferable product" ya está explícito | Pulir frase de apertura del abstract al estilo opción B de Mauricio |
| ML2 | Jerarquizar contributions | 5 contributions ya con framework primero (L73–80); #5 "six novel indices" queda como highlight separado, no como peer de los otros | Consolidar de 5 → 4 eliminando/fusionando #5 |
| ML4 | Cuprite como validación principal | Ya reposicionado: Intro L71, Methods L248, Discussion L512 "substantially strengthens the generality claim" | Verificar que Results y Conclusion lo enfaticen como "primary evidence" |
| ML5 | Method vs formula transferability | Ya en Intro L71 y Discussion L504 como subsección explícita | Agregar la distinción al arranque de Results y a Conclusion |
| ML8 | Elevar TOST a mensaje central | Ya en Abstract L41, Contributions #2, Discussion L488, Conclusion #1 | Retoque: frase tipo "A key result is..." en abstract |
| ML9 | Land cover POC | Ya reducido a 1 párrafo en Discussion §7.7 (L557–564) + supplementary table | Decisión menor: mantener como está o mover todo a supplementary |

## Requiere trabajo de contenido (medium effort)

| # | Recomendación | Gap | Acción |
|---|---------------|-----|--------|
| ML6 | Interpretación espectroscópica por fórmula | Physical Interpretation §6 (L447–477) existe pero no sigue el template Mauricio "fórmula → significado espectral → mineralogía → limitación S2 vs hiperespectral" por subsección | Reescribir cada subsección con el template |
| ML7 | Results por Q1–Q4 | Results estructurado por experimentos (§5.1–5.6) no por preguntas | Reescribir subtítulos y agregar párrafo de apertura con las 4 preguntas |
| ML10 | Limitations consolidado | Limitations dispersos en §7.4 (SWIR), §7.5 (ground truth), §7.6 (metodología) — no hay una "Limitations and scope" unificada con los 5 puntos | Consolidar en una subsección al final de Discussion |
| ML11 | Conclusion con 3 mensajes | Tiene qué aporta y por qué importa; falta el "qué NO se reclama" (no universalidad de fórmulas) explícito | Agregar párrafo de cierre con los 3 mensajes |

## Requiere decisión estratégica + posible experimento (HIGH EFFORT)

| # | Recomendación | Estado | Decisión |
|---|---------------|--------|----------|
| **ML3** | Leakage discovery on full dataset | Paper actual usa **opción 3 de Mauricio** (solo reescribir scope, honesty en §3.3 L195). Mauricio recomienda ejecutar opción 1 (nested) o opción 2 (hold-out 70/30) | ¿Corremos hold-out 70/30? Cuesta ~3 h de PySR + reescritura. Opción 3 vigente es defendible por la honesty explícita y Cuprite como unbiased benchmark, pero el opción 2 blinda el paper ante reviewers duros |

## Prioridad recomendada

1. **#14 ML3 (decisión primero):** si se hace experimento nuevo, afecta Results y Methods. Decidir antes de todo lo demás.
2. Ronda quick-wins: ML1 polish + ML2 consolidar + ML5 refuerzo + ML8 polish (30–60 min total).
3. Trabajo de contenido: ML7 (Q1–Q4) → ML6 (template espectroscópico) → ML10 (Limitations) → ML11 (Conclusion).
4. Cierre: ML9 (land cover) — probablemente dejar como está.

## Cobertura global

De los 11 puntos, 6 están parcial/totalmente cubiertos por el reframing post-RSE. El único que requiere experimento nuevo (opcional) es **ML3**. El resto son reescrituras.
