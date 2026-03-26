# Web of Science — Queries para Estado del Arte

## Proyecto: Spectral Indices Discovery for Hydrothermal Alteration

Fecha de búsqueda: 2026-03-22

---

## Q1. Alteración hidrotermal + teledetección (panorama general)

```
TS=("hydrothermal alteration" AND ("remote sensing" OR "satellite imag*" OR "multispectral" OR "hyperspectral"))
```

**Objetivo**: Panorama amplio del campo. Identificar reviews, métodos dominantes, sensores más usados.
**Filtros sugeridos**: Ordenar por citas, limitar a 2015-2026 para estado del arte reciente.

---

## Q2. Índices espectrales para alteración / mapeo mineral

```
TS=(("spectral ind*" OR "band ratio*" OR "spectral ratio*" OR "mineral ind*") AND ("hydrothermal" OR "alteration" OR "mineral mapping" OR "litholog*"))
```

**Objetivo**: Inventariar los índices existentes (OHI, KLI, ALI, clay ratio, iron oxide ratio, etc.) y entender cómo se diseñaron.

---

## Q3. ASTER para alteración hidrotermal (baseline clásico)

```
TS=(ASTER AND ("hydrothermal alteration" OR "alteration mineral*" OR "alteration zone*" OR "alteration mapping") AND ("spectral" OR "band ratio" OR "index" OR "indices"))
```

**Objetivo**: Capturar los papers fundacionales de índices ASTER (Ninomiya, Rowan, Mars, Di Tommaso, etc.) que son el baseline a superar.

---

## Q4. Sentinel-2 para geología / alteración / litología

```
TS=("Sentinel-2" AND ("hydrothermal" OR "alteration" OR "mineral mapping" OR "litholog*" OR "geological mapping" OR "iron oxide" OR "clay mineral*"))
```

**Objetivo**: Papers que ya usan Sentinel-2 para mapeo geológico. Ver qué bandas/ratios usan y sus limitaciones reportadas frente a ASTER.

---

## Q5. Symbolic regression en teledetección / geociencias

```
TS=("symbolic regression" AND ("remote sensing" OR "satellite" OR "spectral" OR "geosci*" OR "Earth observation" OR "vegetation index" OR "NDVI"))
```

**Objetivo**: Precedentes de SR aplicado a teledetección. Papers clave: descubrimiento de índices de vegetación con SR (ej. Camps-Valls).

---

## Q6. Symbolic regression para descubrimiento de fórmulas / feature engineering

```
TS=("symbolic regression" AND ("discover*" OR "automat*" OR "feature engineer*" OR "formula*" OR "equation discover*" OR "interpretab*"))
```

**Objetivo**: Metodología de SR como herramienta de descubrimiento científico. Incluye PySR, gplearn, Eureqa. Capturar el framing de "AI-driven scientific discovery".

---

## Q7. Machine learning para mapeo de alteración hidrotermal

```
TS=(("machine learning" OR "deep learning" OR "random forest" OR "support vector" OR "neural network*" OR "classification") AND "hydrothermal alteration" AND ("remote sensing" OR "satellite" OR "multispectral"))
```

**Objetivo**: Competidores ML. Entender qué tan bien funcionan los métodos black-box para justificar la ventaja de interpretabilidad de SR.

---

## Q8. Zonas de estudio — Andes chilenos / sistemas geotérmicos

```
TS=(("El Tatio" OR "Puchuldiza" OR "Cerro Pabellon" OR "Copahue" OR "Maricunga") AND ("alteration" OR "hydrothermal" OR "geothermal" OR "epithermal" OR "volcanic"))
```

**Objetivo**: Papers con ground truth en nuestras zonas de estudio. Fuente de datos de validación y comparación directa.

---

## Q9. Mapeo mineral con Sentinel-2 SWIR (bandas B11/B12)

```
TS=("Sentinel-2" AND ("SWIR" OR "short-wave infrared" OR "band 11" OR "band 12" OR "B11" OR "B12") AND ("mineral*" OR "clay" OR "alteration" OR "litholog*" OR "geological"))
```

**Objetivo**: Papers que explotan específicamente las bandas SWIR de Sentinel-2 para geología. Evaluar si alguien ya propuso índices optimizados.

---

## Q10. Comparación ASTER vs Sentinel-2 para geología

```
TS=(ASTER AND "Sentinel-2" AND ("compar*" OR "evaluat*" OR "assess*") AND ("geological" OR "mineral" OR "alteration" OR "litholog*"))
```

**Objetivo**: Papers que comparan directamente ambos sensores. Entender trade-offs resolución espectral (ASTER) vs espacial/temporal (S2).

---

## Q11. Espectroscopía de minerales de alteración (fundamento físico)

```
TS=(("reflectance spectr*" OR "spectral signatur*" OR "absorption feature*") AND ("kaolinite" OR "alunite" OR "illite" OR "montmorillonite" OR "chlorite" OR "epidote" OR "goethite" OR "hematite" OR "jarosite") AND ("hydrothermal" OR "alteration"))
```

**Objetivo**: Base espectroscópica para interpretar físicamente las fórmulas descubiertas. Papers de referencia sobre firmas minerales (Clark, Hunt, Bishop).

---

## Q12. PySR / Genetic programming para índices espectrales

```
TS=(("PySR" OR "genetic programming" OR "evolutionary algorithm*" OR "symbolic" OR "gplearn" OR "Eureqa") AND ("spectral ind*" OR "vegetation ind*" OR "remote sensing ind*" OR "band combination*"))
```

**Objetivo**: Búsqueda directa de precedentes más cercanos. ¿Alguien ya usó GP/SR para descubrir índices espectrales de cualquier tipo?

---

## Q13. Alteración hidrotermal en contexto volcánico andino

```
TS=("hydrothermal alteration" AND ("Andes" OR "Andean" OR "Chile" OR "Central Volcanic Zone" OR "Southern Volcanic Zone") AND ("remote sensing" OR "satellite" OR "mapping"))
```

**Objetivo**: Contextualizar geográficamente. Papers que mapean alteración en los Andes con teledetección.

---

## Q14. Optimización de índices espectrales (métodos automáticos)

```
TS=(("optim*" OR "automat*" OR "search" OR "best" OR "novel") AND ("spectral ind*" OR "band ind*" OR "band ratio*" OR "band combination*") AND ("remote sensing" OR "satellite") AND ("mineral*" OR "geological" OR "soil" OR "vegetation"))
```

**Objetivo**: Cualquier método de optimización/búsqueda automática de índices, no solo SR. Incluye búsqueda exhaustiva, GA, bayesiana, etc.

---

## Notas de uso

- **Período recomendado**: 2000-2026 para Q1-Q4 (campo maduro), 2015-2026 para Q5-Q14 (SR es más reciente)
- **Ordenar por**: Times Cited (para seminales) y Date (para estado del arte)
- **Categorías WoS**: Remote Sensing, Geosciences Multidisciplinary, Geochemistry & Geophysics, Computer Science AI, Environmental Sciences
- **Operadores WoS**: `TS` = Topic (título + abstract + keywords), `*` = truncamiento, `NEAR/n` = proximidad
- Exportar como `.bib` (BibTeX) para integrar directamente

### Volumen esperado aproximado

| Query | Resultados estimados |
|-------|---------------------|
| Q1 | 800-1200 |
| Q2 | 400-700 |
| Q3 | 300-500 |
| Q4 | 80-150 |
| Q5 | 30-60 |
| Q6 | 200-400 |
| Q7 | 150-300 |
| Q8 | 50-100 |
| Q9 | 30-70 |
| Q10 | 10-30 |
| Q11 | 200-400 |
| Q12 | 20-50 |
| Q13 | 40-80 |
| Q14 | 50-120 |

**Total sin deduplicar**: ~2500-4000. Tras deduplicar y filtrar por relevancia: **150-300 papers** para lectura de abstracts, **40-60 para lectura completa**.
