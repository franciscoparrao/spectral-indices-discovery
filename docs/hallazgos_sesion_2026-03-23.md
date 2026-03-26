# Hallazgos — Sesión 2026-03-23

## Resumen Ejecutivo

Primera sesión de trabajo del proyecto Spectral Indices Discovery. Se completó el análisis bibliométrico, instalación del stack técnico, descarga de datos Sentinel-2 para El Tatio, y un primer run exploratorio de PySR. **Se identificó un problema metodológico crítico**: el ground truth actual genera un argumento circular que invalida los resultados de SR.

---

## 1. Estado del Arte (Análisis Bibliométrico)

### Datos
- **14 queries WoS** diseñadas cubriendo: alteración hidrotermal + RS, índices espectrales, ASTER, Sentinel-2, symbolic regression, ML, zonas de estudio, espectroscopía, y optimización de índices.
- **5,429 entradas brutas** → **4,354 papers únicos** tras deduplicación por DOI/título.
- Archivos: `docs/Q1.bib` a `docs/Q14.bib`, `docs/master_deduplicated.bib`
- Reporte: `docs/bibliometric_analysis.md`

### Hallazgos bibliométricos clave
- **Campo en crecimiento exponencial**: 267 papers/año (2020) → 487 (2025)
- **Gap confirmado**: La intersección SR + alteración hidrotermal + Sentinel-2 está **vacía**
  - Q5 (SR + remote sensing): solo 50 papers
  - Q12 (GP/SR para índices espectrales): 72 papers, mayormente vegetación
  - Q7 (ML para alteración): 0 papers exclusivos (siempre se solapa con otros temas)
- **Papers fundacionales identificados**:
  - van der Meer (2012) — 968 citas, review RS geológica
  - Sabins (1999) — 650 citas, RS para exploración mineral
  - Mars (2006) — 304 citas, mapeo argílica/fílica con ASTER
  - van der Meer (2014) — 230 citas, potencial de Sentinel-2 para geología
  - Koza (1994) — 4,785 citas, fundacional GP
  - Udrescu (2020) — 720 citas, AI Feynman (SR discovery)
- **Revistas dominantes**: Remote Sensing (287), Ore Geology Reviews (117), RSE (85), IJAEOG (83)

---

## 2. Stack Técnico

### Instalado
- **PySR v1.5.9** — Symbolic regression (Python wrapper para SymbolicRegression.jl)
- **Julia 1.11.9** — Backend computacional de PySR
- **SymbolicRegression.jl v1.11.3** — Motor de SR
- **Dependencias**: numpy, pandas, scipy, scikit-learn, rasterio, geopandas, matplotlib, seaborn
- **Entorno**: venv en `.venv/`

### Herramienta externa
- **SurtGis v0.2.0** (`~/proyectos/surtgis/target/release/surtgis`) — usado para descarga S2 via STAC. Comandos: `stac search`, `stac fetch`, `stac fetch-mosaic`, `stac composite`.

---

## 3. Datos Sentinel-2

### Escena descargada
- **Zona**: El Tatio, Chile (-22.33°S, 68.01°W)
- **Sensor**: Sentinel-2A L2A (reflectancia de superficie)
- **Fecha**: 2024-01-23 (0.1% nubes)
- **BBOX**: -68.10, -22.42, -67.92, -22.25
- **CRS**: EPSG:32719 (UTM 19S)
- **Tiles**: S2A_19KER + S2A_19KFR (mosaico de 2 tiles)

### Bandas descargadas
| Banda | Resolución | Grid | Píxeles válidos (1-10000) |
|-------|-----------|------|--------------------------|
| B02 (Blue) | 10m → 20m | 1867×1895 | 143,777 (16%) |
| B03 (Green) | 10m → 20m | 1867×1895 | 154,267 (17%) |
| B04 (Red) | 10m → 20m | 1867×1895 | 163,856 (19%) |
| B05 (RE1) | 20m | 934×948 | 444,049 (50%) |
| B06 (RE2) | 20m | 934×948 | 444,006 (50%) |
| B07 (RE3) | 20m | 934×948 | 443,539 (50%) |
| B08 (NIR) | 10m → 20m | 1867×1895 | 166,603 (19%) |
| B8A (NIR-n) | 20m | 934×948 | 443,246 (50%) |
| B11 (SWIR1) | 20m | 934×948 | 441,111 (50%) |
| B12 (SWIR2) | 20m | 934×948 | 442,755 (50%) |

**Nota**: Las bandas 10m tienen ~18% cobertura vs 50% para 20m. Esto se debe a que el mosaico de 2 tiles tiene un overlap parcial: el tile 19KER cubre la mitad este y 19KFR la mitad oeste, con overlap solo en la franja central. Las bandas 10m (B02-B04, B08) quedan con cobertura menor tras block-average a 20m porque los píxeles con valor 0 (fuera del tile) diluyen el promedio.

### Decisiones técnicas sobre descarga

**Problema con composite temporal**: El comando `surtgis stac composite` con SCL masking producía solo 3-7% de cobertura, incluso con `--scl-keep 2,4,5,6,7,11`. Causa: el SCL de Sentinel-2 está calibrado para paisajes europeos y clasifica mucho terreno árido/altiplánico como clases excluidas. Además, la mediana temporal sobre 20 fechas con baja coincidencia espacial entre tiles reduce drásticamente los píxeles válidos.

**Solución**: Usar `surtgis stac fetch-mosaic` para una sola fecha clara. En un desierto de alta montaña estable (sin cambios estacionales), una escena individual sin nubes es perfectamente válida.

---

## 4. Índices de Alteración Clásicos

Script: `scripts/explore_alteration.py`

### Índices calculados
| Índice | Fórmula | Propósito |
|--------|---------|-----------|
| Clay ratio | B11/B12 | Minerales OH (arcillas) |
| Iron oxide ratio | B04/B02 | Fe³⁺ (goethita, hematita) |
| Ferrous minerals | B12/B8A + B03/B04 | Fe²⁺ (clorita, epidota) |
| Alunite/kaolinite idx | (B11-B12)/(B11+B12) | Al-OH (alteración avanzada) |
| NDVI | (B8A-B04)/(B8A+B04) | Vegetación |
| Alteration intensity | B11/(B8A+B12) | Anomalía SWIR1 general |
| Silica proxy | B12/B11 | Silicificación (inverso clay) |

### Estadísticas de índices (píxeles válidos)
| Índice | p5 | p25 | p50 | p75 | p95 |
|--------|-----|------|------|------|------|
| clay_ratio | 0.333 | 0.805 | 1.079 | 1.509 | 3.800 |
| alunite_idx | -0.500 | -0.108 | 0.038 | 0.203 | 0.583 |
| iron_oxide | 0.916 | 1.346 | 1.680 | 2.062 | 3.043 |
| swir_bright | 0.001 | 0.003 | 0.007 | 0.013 | 0.032 |
| ndvi | -0.485 | -0.024 | 0.194 | 0.365 | 0.578 |

### Figuras generadas
- `figures/el_tatio_rgb.png` — True color (oscuro, bandas 10m con baja cobertura)
- `figures/el_tatio_swir.png` — SWIR composite (B12, B11, B4) — buena visualización geológica
- `figures/el_tatio_alteration.png` — Alteration composite (B11, B12, B4)
- `figures/el_tatio_all_indices.png` — Panel de 6 índices

---

## 5. Ground Truth

Script: `scripts/create_ground_truth.py`

### ⚠️ PROBLEMA METODOLÓGICO CRÍTICO: ARGUMENTO CIRCULAR

El ground truth actual fue creado **a partir de umbrales de los mismos índices espectrales** de la imagen Sentinel-2. Esto genera un **argumento circular**:

1. Se definen clases de alteración por umbrales de B11/B12 y otros ratios
2. PySR busca fórmulas que predigan esas clases
3. PySR "descubre" B11/B12 como la mejor fórmula → **no es un descubrimiento, es una tautología**

**Esto invalida las conclusiones actuales sobre las fórmulas descubiertas por PySR.** Los resultados son técnicamente correctos (PySR funcionó), pero no tienen valor científico porque el modelo está aprendiendo los mismos umbrales que se usaron para definir las clases.

### Solución requerida
El ground truth debe provenir de una fuente **independiente de los datos espectrales**:
1. **Mapas geológicos** (SERNAGEOMIN carta Cupo-Toconce — solicitada por transparencia)
2. **Datos de campo** (publicaciones con coordenadas de muestreo mineral)
3. **Mapas de alteración de otros sensores** (ASTER, WorldView-3) como referencia cruzada
4. **Datos de pozos** (Lahsen & Trujillo 1976 — 13 pozos con logs de alteración)

### Clasificación actual (solo para exploración, NO para entrenamiento final)

| Clase | Criterio (umbral) | Píxeles | % del válido |
|-------|-------------------|---------|-------------|
| 1 Silicic | swir_bright > p80, abs(alunite) < 0.05 | 139 | 0.04% |
| 2 Adv. Argillic | alunite_idx > 0.18 | 96,644 | 24.6% |
| 3 Argillic | alunite_idx 0.03-0.18 | 75,901 | 19.3% |
| 4 Propylitic | clay_ratio < 0.90 | 86,251 | 22.0% |
| 5 Iron Oxide | iron_oxide > 1.8 | 12,377 | 3.2% |
| 6 Unaltered | ninguno de los anteriores | 72,562 | 18.5% |
| 7 Vegetation | NDVI > 0.25 | 48,555 | 12.4% |

### Datos de training exportados
- `data/ground_truth/el_tatio_training_20m.npz` — 275,829 muestras × 6 bandas (B05-B12)
- `data/ground_truth/el_tatio_training_all.npz` — 53,099 muestras × 10 bandas (B02-B12)

---

## 6. Resultados PySR (Exploratorios — sujetos al problema del argumento circular)

### Run 1: 6 bandas 20m (B05, B06, B07, B8A, B11, B12)

Script: `scripts/run_pysr.py`
Configuración: 80 iteraciones, maxsize=15, select_k_features=4, parsimony=0.005

| Clase | Fórmula descubierta | Complejidad | AUC | F1 |
|-------|---------------------|-------------|-----|-----|
| Adv. Argillic | `tanh((B11/B12) - 1.54) / 1.56 + 0.39` | 10 | 0.9795 | 0.7607 |
| Argillic | `tanh(sq(B11/B12)) / (0.59 + B11/B12) - 0.20` | 13 | 0.9486 | 0.0000 |
| Propylitic | `tanh((B12/B11) - 0.69)` | 6 | 0.9699 | 0.7903 |
| Iron Oxide | `sqrt(B12) * 0.56 - B11` | 6 | 0.6749 | 0.0000 |
| Unaltered | `tanh(sq(B11/B12)) * sq(tanh(B12/B11)) - 0.21` | 13 | 0.9603 | 0.0000 |
| Vegetation | `sqrt(B8A)` | 2 | 0.7198 | 0.0076 |

### Run 2: 10 bandas (B02-B12)

Script: `scripts/run_pysr_10bands.py`

| Clase | Fórmula descubierta | Complejidad | AUC | F1 | Umbral |
|-------|---------------------|-------------|-----|-----|--------|
| Adv. Argillic | `0.86 - tanh(B12/B11)` | 6 | 0.9509 | 0.7667 | 0.25 |
| Argillic | `tanh(B02/B07) * 0.28` | 6 | 0.6873 | 0.2892 | 0.10 |
| Propylitic | `tanh(B02/B8A) * tanh(B12/B11 - 0.63)` | 11 | 0.9605 | 0.7144 | 0.30 |
| Iron Oxide | `tanh(B04/B8A) * tanh(sq(log(B04/B02)))` | 11 | 0.8860 | 0.5906 | 0.25 |
| Unaltered | `sqrt(B02) - B8A` | 4 | 0.6856 | 0.1427 | 0.10 |
| Vegetation | `tanh(B8A/B04 - 1.76) + 0.57` | 8 | 1.0000 | 0.9898 | 0.50 |

### Observaciones sobre las fórmulas (con la caveat del argumento circular)
- **B11/B12 domina** en todas las clases de alteración → esperado dado que el ground truth se definió con este ratio
- **Iron Oxide mejoró con 10 bandas** (AUC 0.67→0.89): usa B04/B02 (Red/Blue) que es el ratio clásico de Fe³⁺
- **Vegetation redescubrió el NDVI**: `tanh(B8A/B04)` es esencialmente NIR/Red normalizado
- **Las constantes optimizadas** (e.g., -1.54, -0.69) corresponden a los umbrales de decisión del ground truth

---

## 7. Solicitud de Transparencia

Archivo: `docs/solicitud_transparencia_sernageomin.md`

Solicitud formal preparada para obtener la **Carta Geológica Cupo-Toconce (Serie Básica 215-216, 2023)** via Ley 20.285. SERNAGEOMIN sufrió un ciberataque (ransomware) en diciembre 2025, con sus plataformas digitales fuera de servicio.

La carta cubre:
- 22°00'-22°30'S, 68°30'W hasta frontera Chile-Bolivia
- Incluye El Tatio explícitamente
- Formato: GDB + Shapefile + PDF (223 páginas)
- Autores: Álvarez, Tunik, Giambiagi, Rodríguez

---

## 8. Problemas Técnicos Resueltos

| Problema | Causa | Solución |
|----------|-------|----------|
| Composite temporal con 3% cobertura | SCL masking agresivo en terreno árido altiplánico | Escena individual con `fetch-mosaic` |
| Cobertura subió a 7% con SCL ampliado | Composite median pierde datos en edges de tiles | Abandonar composite, usar fecha individual |
| Bandas 10m/20m desalineadas | Block average produce 933×947 vs 934×948 nativo | trim/pad al shape del 20m |
| PySR `deterministic=True` error | Requiere `parallelism='serial'` | Remover flag `deterministic` |
| Stratified split falla | Clase Silicic con 1 muestra en set 10-band | Filtrar clases < MIN_SAMPLES antes del split |

---

## 9. Próximos Pasos Críticos

### Prioridad 1: Resolver el argumento circular
- Obtener ground truth independiente (SERNAGEOMIN, datos de campo, ASTER)
- Revisar proyecto GEE (~/proyectos/google_earth_engine/) para datos de Maricunga
- Re-entrenar PySR con ground truth válido

### Prioridad 2: Otras zonas de estudio
- Descargar S2 para Puchuldiza, Copahue, Maricunga
- Aplicar pipeline completo a cada zona
- Implementar leave-one-site-out cross-validation

### Prioridad 3: Validación y comparación
- Comparar con índices ASTER clásicos (OHI, KLI, ALI)
- Baseline Random Forest sobre todas las bandas
- Métricas: OA, Kappa, F1 por clase, ROC AUC

### Prioridad 4: Paper
- Estructura ya definida en CLAUDE.md
- Target: Remote Sensing of Environment (IF 13.5)

---

## Archivos del Proyecto

```
spectral-indices-discovery/
├── CLAUDE.md                                    # Planificación del proyecto
├── data/
│   ├── sentinel2/el_tatio/
│   │   ├── B02.tif ... B12.tif                  # 10 bandas S2 L2A (2024-01-23)
│   │   └── idx_*.tif                            # Índices de alteración
│   ├── ground_truth/
│   │   ├── el_tatio_ground_truth.tif            # Clasificación raster (uint8)
│   │   ├── el_tatio_class_centroids.geojson     # Centroides por clase
│   │   ├── el_tatio_training_20m.npz            # 275K × 6 bandas
│   │   └── el_tatio_training_all.npz            # 53K × 10 bandas
│   └── results/
│       ├── pysr_results_summary.json            # Resultados 6 bandas
│       ├── pysr_results_10bands.json            # Resultados 10 bandas
│       └── equations_*.csv                      # Ecuaciones por clase
├── docs/
│   ├── wos_queries.md                           # 14 queries Web of Science
│   ├── bibliometric_analysis.md                 # Análisis de 4,354 papers
│   ├── master_deduplicated.bib                  # BibTeX deduplicado
│   ├── Q1.bib ... Q14.bib                       # BibTeX originales
│   ├── solicitud_transparencia_sernageomin.md   # Solicitud Ley 20.285
│   └── hallazgos_sesion_2026-03-23.md           # ESTE ARCHIVO
├── figures/
│   ├── el_tatio_rgb.png                         # True color
│   ├── el_tatio_swir.png                        # SWIR composite
│   ├── el_tatio_alteration.png                  # Alteration composite
│   ├── el_tatio_all_indices.png                 # Panel de 6 índices
│   └── el_tatio_ground_truth.png                # Mapa de GT
├── scripts/
│   ├── analyze_bibliography.py                  # Parser bibliométrico
│   ├── explore_alteration.py                    # Índices clásicos
│   ├── create_ground_truth.py                   # GT por umbrales
│   ├── run_pysr.py                              # PySR 6 bandas
│   └── run_pysr_10bands.py                      # PySR 10 bandas
└── .venv/                                       # Entorno virtual Python
```
