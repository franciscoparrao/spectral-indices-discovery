# Spectral Indices Discovery — Alteración Hidrotermal

## Paper Target

**"Automated Discovery of Spectral Indices for Hydrothermal Alteration Mapping using Symbolic Regression on Sentinel-2: A Case Study in the Chilean Andes"**

Target: **Remote Sensing of Environment** (Elsevier hybrid, IF ~13.5, sin APC vía suscripción).
JAG descartado (full OA, APC ~$3000). Alternativas: ISPRS Journal (IF 12.7), Ore Geology Reviews (IF 6.3).

## Investigador

- **Nombre**: Francisco Parra O.
- **Grado**: Doctor en Informática
- **Afiliación**: CITIAPS
- **Expertise**: GIS, remote sensing, machine learning, susceptibilidad, embeddings geoespaciales
- **Stack**: Python (scikit-learn, PyTorch, GDAL, rasterio, geopandas)
- **Hardware**: Cluster doméstico (i7-1270P 38GB + Worker GPU MX150), burst Hetzner Cloud
- **Papers en revisión**: 5 + 1 en revisiones mayores
- **Herramientas**: MCP Gateway con 1500+ tools GIS (GDAL, QGIS, GRASS, OTB, Planetary Computer)

## El Problema

### Estado del arte — Índices de alteración hidrotermal

Los índices espectrales actuales para alteración hidrotermal son:
- Ratios simples diseñados para ASTER hace 20+ años (OHI, KLI, ALI)
- Diferencias normalizadas genéricas (no optimizadas para alteración)
- No aprovechan todas las bandas de Sentinel-2

| Índice | Fórmula | Sensor original | Limitación |
|--------|---------|-----------------|------------|
| OHI | B7/B6 | ASTER | No discrimina tipos de arcilla |
| KLI | B4/B6 × B8/B6 | ASTER | No existe equivalente Sentinel-2 |
| ALI | B5/B6 | ASTER | Específico para ASTER SWIR |
| Fe²⁺ | B5/B4 | ASTER | Muy simple, mucho falso positivo |
| Clay ratio | SWIR1/SWIR2 | Landsat/S2 | Genérico |

### El gap

Nadie ha usado **symbolic regression** para descubrir índices óptimos de alteración hidrotermal con Sentinel-2. ASTER tiene 5 bandas SWIR vs 2 de Sentinel-2, pero Sentinel-2 compensa con:
- Mejor resolución espacial (10-20m vs 30m)
- Mejor resolución temporal (5 días vs bajo demanda)
- Acceso gratuito y global
- 13 bandas totales (más combinaciones posibles si incluyes VNIR + Red Edge)

**Pregunta de investigación**: ¿Se pueden descubrir automáticamente fórmulas espectrales que discriminen tipos de alteración hidrotermal usando Sentinel-2, superando los índices clásicos diseñados para ASTER?

### Por qué symbolic regression y no deep learning

- **Interpretabilidad**: La fórmula descubierta se puede entender físicamente. Un geólogo puede validar si las bandas tienen sentido con la espectroscopia mineral.
- **Reproducibilidad**: Cualquiera puede calcular `log(B11/B4) * sqrt(B12)` en cualquier GIS. Un modelo CNN no.
- **Publicabilidad**: "Descubrimos un nuevo índice espectral" es más impactante y citado que "entrenamos un modelo".
- **Generalización**: Una fórmula simple generaliza mejor que un modelo complejo a nuevos sitios.

## Zonas de Estudio

Sistemas volcánico-hidrotermales conocidos en Chile con datos de alteración publicados:

| Zona | Latitud | Tipo | Alteración dominante | Ground truth |
|------|---------|------|---------------------|--------------|
| **El Tatio** | -22.33 | Geotérmico | Sílice, argílica | SERNAGEOMIN + papers |
| **Puchuldiza** | -19.41 | Geotérmico | Argílica avanzada, propilítica | Papers + mapeo SERNAGEOMIN |
| **Cerro Pabellón** | -22.28 | Geotérmico (planta activa) | Sílice, argílica | Datos de exploración publicados |
| **Copahue-Caviahue** | -37.85 | Volcánico activo | Argílica avanzada, propilítica | Bien mapeado (Argentina-Chile) |
| **Maricunga** | -26.85 | Epithermal Au | Argílica avanzada, sílice | Exploración minera publicada |

Criterio de selección: entrenar en 3-4 zonas, validar en la restante (leave-one-site-out).

## Clases de Alteración

| Clase | Minerales clave | Firma espectral | Bandas Sentinel-2 relevantes |
|-------|----------------|-----------------|------------------------------|
| **Sílice (silicic)** | Cuarzo, cristobalita, ópalo | Alta reflectancia general, absorción reducida | SWIR alto, VNIR alto |
| **Argílica avanzada** | Alunita, caolinita, pirofilita | Absorción fuerte ~2.17 µm y ~2.20 µm | B11, B12 (SWIR) |
| **Argílica** | Illita, montmorillonita, smectita | Absorción ~2.20 µm | B11, B12, Red Edge |
| **Propilítica** | Clorita, epidota, calcita | Absorción ~2.33 µm, reflectancia en NIR | B8, B11, B12 |
| **Óxidos de hierro** | Goethita, hematita, jarosita | Absorción ~0.9 µm, alta reflectancia rojo | B4, B8, B8A |
| **Sin alteración** | Roca fresca | Variable según litología | Baseline |

## Bandas Sentinel-2

| Banda | Longitud de onda | Resolución | Nombre | Relevancia para alteración |
|-------|-----------------|------------|--------|---------------------------|
| B1 | 443 nm | 60m | Coastal aerosol | Baja (corrección atmosférica) |
| B2 | 490 nm | 10m | Blue | Media (Fe³⁺) |
| B3 | 560 nm | 10m | Green | Media |
| B4 | 665 nm | 10m | Red | Alta (óxidos de Fe) |
| B5 | 705 nm | 20m | Red Edge 1 | Media |
| B6 | 740 nm | 20m | Red Edge 2 | Media |
| B7 | 783 nm | 20m | Red Edge 3 | Media |
| B8 | 842 nm | 10m | NIR | Alta (contraste con alteración) |
| B8A | 865 nm | 20m | NIR narrow | Alta (Fe²⁺/Fe³⁺) |
| B9 | 945 nm | 60m | Water vapour | Baja |
| B11 | 1610 nm | 20m | SWIR 1 | **Muy alta** (minerales OH) |
| B12 | 2190 nm | 20m | SWIR 2 | **Muy alta** (Al-OH, Mg-OH) |

Las bandas B11 y B12 son las más críticas. El ratio B11/B12 y sus variaciones son la base de la mayoría de los índices de alteración.

## Pipeline Técnico

### Fase 1: Datos (2-3 semanas)

```python
# 1. Descargar Sentinel-2 L2A (reflectancia de superficie) via Planetary Computer
# Usar el MCP server de Planetary Computer ya integrado
# Filtrar: cloud cover < 10%, fechas de verano (diciembre-marzo, menos nieve)

# 2. Para cada zona de estudio:
#    - Recortar al área de interés
#    - Resample todas las bandas a 20m (resolución de B11/B12)
#    - Calcular mediana temporal (reducir ruido, nubes residuales)

# 3. Ground truth:
#    - Digitalizar mapas de alteración de SERNAGEOMIN / papers
#    - Crear shapefile/GeoJSON con polígonos por clase de alteración
#    - Extraer valores de bandas por pixel dentro de cada polígono
```

### Fase 2: Symbolic Regression (2-3 semanas)

```python
from pysr import PySRRegressor
import numpy as np

# X = bandas espectrales [N_pixels, 10]
# (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 — omitir B1,B9 por baja resolución)
# y = clase de alteración (codificada) o intensidad

# Para clasificación: one-vs-rest, un modelo por clase
# Para regresión: intensidad de alteración (si hay datos continuos)

model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log", "exp", "tanh", "square"],
    maxsize=20,                    # máximo 20 nodos (interpretable)
    populations=30,                # poblaciones paralelas
    population_size=50,
    parsimony=0.003,               # penalizar complejidad
    weight_optimize=0.001,         # optimizar constantes
    constraints={
        "sqrt": 5, "log": 5,      # limitar profundidad de operadores
        "exp": 3, "tanh": 5,
    },
    select_k_features=6,           # máximo 6 bandas por fórmula
    progress=True,
    temp_equation_file=True,
    # Variables con nombres de bandas
    variable_names=["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"],
)

# Entrenar por clase de alteración
for clase in ["silicic", "adv_argillic", "argillic", "propylitic", "iron_oxide"]:
    y_binary = (labels == clase).astype(float)
    model.fit(X_train, y_binary)

    # Top 5 fórmulas por clase
    print(f"\n=== {clase} ===")
    print(model.latex_table())
    # Ejemplo output:
    # silicic:     0.92  →  tanh(B11 / (B4 + B12))
    # adv_argillic: 0.89  →  log(B11/B12) * (B8 - B4)/(B8 + B4)
    # propylitic:   0.87  →  sqrt(B12/B8A) - B11/(B7 + B11)
```

### Fase 3: Validación (2-3 semanas)

```python
# 1. Leave-one-site-out cross-validation
#    Entrenar en 3 zonas, evaluar en la 4ta. Repetir para cada zona.

# 2. Métricas:
#    - Overall Accuracy, Kappa, F1 por clase
#    - ROC AUC por clase
#    - Comparar con:
#      a) Índices ASTER clásicos adaptados a S2
#      b) NDVI, Clay Ratio, ratios SWIR simples
#      c) Random Forest sobre todas las bandas (baseline ML)
#      d) Índices publicados para alteración con Sentinel-2

# 3. Interpretación física:
#    - ¿Las bandas en la fórmula corresponden a absorciones minerales conocidas?
#    - ¿El comportamiento de la fórmula es consistente con la espectroscopia?
#    - Graficar respuesta espectral de cada clase vs fórmula descubierta

# 4. Mapas de alteración:
#    - Aplicar fórmulas descubiertas a las zonas de estudio
#    - Comparar visualmente con mapas de referencia
#    - Generar figuras para el paper
```

### Fase 4: Paper (4-6 semanas)

```latex
% Estructura sugerida
1. Introduction — gap en índices de alteración para S2, SR como método
2. Study Areas — 4 zonas chilenas, geología, alteración conocida
3. Data and Methods
   3.1 Sentinel-2 data acquisition and preprocessing
   3.2 Ground truth compilation
   3.3 Symbolic regression framework
   3.4 Validation strategy (leave-one-site-out)
   3.5 Comparison with existing indices
4. Results
   4.1 Discovered indices by alteration class
   4.2 Accuracy comparison
   4.3 Spatial validation (maps)
   4.4 Physical interpretation
5. Discussion
   5.1 Advantages of SR-discovered indices over classical
   5.2 Limitations (SWIR resolution, atmospheric effects)
   5.3 Applicability to other volcanic/geothermal settings
6. Conclusions
```

## Dependencias

```bash
# Symbolic Regression
pip install pysr          # Motor de SR (usa Julia bajo el capó)
pip install juliacall     # Bridge Python-Julia (instalado por pysr)

# Geoespacial (ya instalado en el sistema)
# GDAL, rasterio, geopandas, shapely

# ML baseline
pip install scikit-learn

# Visualización
pip install matplotlib seaborn

# Acceso a datos
# Planetary Computer via MCP Gateway (ya configurado)
# O directamente: pip install planetary-computer pystac-client
```

## Estructura del Proyecto

```
spectral-indices-discovery/
├── CLAUDE.md                        # Este archivo
├── data/
│   ├── sentinel2/                   # Imágenes S2 por zona
│   │   ├── el_tatio/
│   │   ├── puchuldiza/
│   │   ├── cerro_pabellon/
│   │   └── copahue/
│   ├── ground_truth/                # Shapefiles de alteración
│   └── results/                     # Outputs de SR
├── notebooks/
│   ├── 01_download_sentinel2.ipynb  # Descargar datos via PC
│   ├── 02_preprocess.ipynb          # Resample, clip, mediana temporal
│   ├── 03_extract_training.ipynb    # Extraer bandas por polígono
│   ├── 04_symbolic_regression.ipynb # Correr PySR
│   ├── 05_validation.ipynb          # LOSO-CV, métricas, comparación
│   ├── 06_interpretation.ipynb      # Análisis espectral, sentido físico
│   └── 07_maps.ipynb                # Generar mapas de alteración
├── src/
│   ├── models/                      # Wrappers de PySR, baselines
│   ├── data/                        # Loaders, extractores
│   └── utils/                       # Plotting, métricas, IO
├── figures/                         # Figuras para el paper
└── paper/                           # Draft LaTeX
```

## Cronograma Estimado

| Fase | Duración | Entregable |
|------|----------|-----------|
| Descarga y preprocesamiento S2 | 2 semanas | Stacks de bandas por zona |
| Compilación ground truth | 2 semanas | Shapefiles de alteración |
| Symbolic regression | 2 semanas | Fórmulas candidatas por clase |
| Validación y comparación | 2 semanas | Tablas de accuracy, mapas |
| Interpretación física | 1 semana | Gráficos espectrales |
| Escritura del paper | 4 semanas | Manuscrito completo |
| **Total** | **~13 semanas** | **Paper listo para enviar** |

## Contribución Esperada

1. **Nuevos índices espectrales** para cada tipo de alteración hidrotermal, optimizados para Sentinel-2
2. **Metodología reproducible** de descubrimiento automático de índices
3. **Validación multi-sitio** en el contexto volcánico-hidrotermal chileno
4. **Base de datos** de firmas espectrales de alteración en Sentinel-2 para los Andes centrales
5. **Comparación cuantitativa** con índices ASTER clásicos adaptados

## Notas de Contexto

- Proyecto hermano: `~/proyectos/geofisica-ml/` (vulcanología multi-sensor, en standby)
- Proyecto hermano: `~/proyectos/inversion-geofisica-ml/` (inversión gravimétrica con ML)
- Proyecto previo: `~/proyectos/volcanismo/` (trabajo anterior relacionado)
- Zenodo tools: `~/proyectos/zenodo-tools/` (búsqueda y descarga de datasets)
- Skills relevantes: `/gis`, `/performance`, `/testing`, `/review`

## Próximos Pasos

1. Instalar PySR: `pip install pysr`
2. Buscar mapas de alteración de SERNAGEOMIN para las zonas seleccionadas
3. Descargar Sentinel-2 de El Tatio como zona piloto
4. Notebook 01: descarga y visualización de bandas
5. Notebook 02: preprocesamiento y extracción de training data

## Enumeración exhaustiva en log-espacio (2026-10-09)

- `scripts/enum_logspace.py`: enumera TODOS los productos de ratios hasta m factores
  (vectores enteros suma 0 sobre 10 bandas, sign-canónicos, primitivos). Para AUC, una
  diferencia normalizada es la misma expresión que el ratio (tanh monótono), así que la
  familia {ratios, productos/cocientes, ND} queda cubierta completa. m≤4 = 87 000
  expresiones (~12 min), m≤5 = 452 706 (~1 h). Salida en `data/results/enum_logspace_m{M}_{design}.{csv.gz,json}`.
- **Criterio de selección vigente** (`--select-on all`, default): AUC ponderado de positivos R3
  contra TODOS los grupos negativos de R3 (anillo 50 %, distractores 50 % repartido por
  grupo; vegetación y agua con peso 0 porque se enmascaran). **Región IV nunca entra en la
  selección**: es el único test cross-site válido. El split interno de R3 es por píxel, así
  que las columnas R3_ho están infladas por autocorrelación; no usarlas como evidencia.
- Gotcha: con el criterio del piloto (`--select-on ring`, positivos vs anillo) la familia
  contiene índices buenos en R4 pero la selección no los encuentra (Spearman sel–R4 0.31) y
  la compuerta falla. Con `all`, el ganador m=4 `B02·B8A²·B12/(B03·B04·B11²)` pasa la
  compuerta R4 por +0.11 AUC e iguala la logística sobre log-ratios (0.756). El ganador es
  uno de decenas de empates a 0.004: el resultado es la estructura
  (B12/B11 · B8A/B11 · B8A/B04 · B02/B03; valores bajos = alterado), no la fórmula puntual.
  Corrección 2026-10-17: no hay término SWIR al cuadrado; con el diseño final solo 4 expresiones
  quedan a <0.005 del ganador (no decenas).
- Literatura y gaps: `docs/revision_literatura_enumeracion_indices_2026-10-09.md`.
- **m≤5 (10 oct 2026)**: ganador `B02·B8A³·B12/(B03·B04·B06·B11²)`, R4 ring 0.751 vs 0.738 de m=4 y 0.628 del
  clay ratio; techo de la familia en R4 ring 0.763. Rendimiento decreciente: m=4 es el índice a proponer,
  m=5 es evidencia de saturación. Empate con el segundo a 0.0001.
- **Positivos de R3 georreferenciados**: `data/ground_truth/binary_training_s2_positives_geo.csv` (remuestreo GEE
  del mismo compuesto con `scripts/resample_r3_positives_geo.py`; lon/lat + `poly_id` del Atlas, 460 polígonos).
  Los positivos legacy de `binary_training_s2.csv` NO se pueden georreferenciar (GEE devuelve otra muestra con
  `geometries=True`; misma distribución, cuantiles difieren <0.003). Usar `--positives geo` en adelante.
- **Split espacial** (`--split pixel|polygon|block`, `--block-km 10`, distractores siempre en bloques de 2 km):
  el ganador m=4 es el mismo bajo los cuatro diseños y la compuerta R4 es idéntica (+0.110 ring, +0.123 hard).
  Inflación por autocorrelación en R3 held-out con distractores: ganador 0.858 (píxel) → 0.854 (polígono) → 0.844
  (bloque 10 km); el clay ratio cae igual (0.803 → 0.800 → 0.788), así que la ventaja en R3 se mantiene en
  +0.054–0.058. El gap selección−held-out crece de 0.00 a 0.03 con el bloque. Para el paper, reportar R3 con
  split por bloque y R4 como test cross-site; `R3_ho` por píxel ya no se usa como evidencia.
- **Cuprite como segundo sitio cross-site (R5, 12 oct 2026)**: `scripts/pilot_sample_cuprite.py` (todo en ráster con
  transformada de distancia; la versión vectorial reventaba la RAM) genera `cuprite_positives_geo.csv` (3 760 px, clase
  USGS, blob) y `cuprite_hard_negatives.csv` (anillo 4 000, sombra 1 500, vegetación 800, playa 800, aluvio 800, agua
  520). El mapa USGS cubre mucho más que el bbox (2/3 de la alteración mapeada está fuera, Goldfield al norte); el
  anillo se traza solo sobre terreno mapeado. `enum_logspace.py --cuprite` agrega R5 y su compuerta; R5 nunca entra en
  la selección. Resultado (m≤4, geo, bloque): ganador 0.960 vs clay 0.903 en R5 ring (+0.057), 0.960 vs 0.867 en R5
  hard (+0.094), FPR distractores 4.7 % vs 28 %; techo logístico 0.961. **Compuerta pasa en los dos sitios
  independientes.** Gotcha: el ganador dispara en agua (FPR 0.69 vs 0.14 del clay); agua se enmascara con NDWI y está
  excluida de *_hard, pero hay que decirlo en el paper. Recall en umbral de Youden es menor que el del clay (0.78 vs
  0.89): la ganancia es de especificidad.
- **Baselines y ROC en el diseño final (13 oct 2026)**: `scripts/hardneg_design.py` (loader compartido, reproduce los
  conjuntos del enumerador) + `scripts/final_design_baselines.py` → `data/results/final_design_baselines.json`,
  `figures/roc_final_design.png` (exploratoria). ΔAUC ganador−clay con IC 95 % bootstrap por bloque (10 km R3/R4, 2 km
  R5): R4 ring +0.110 [+0.04, +0.17], R4 hard +0.123 [+0.07, +0.19], R5 ring +0.057 [+0.02, +0.11], R5 hard +0.094
  [+0.05, +0.16]. **RF entrenado en R3 no transfiere** (R4 0.67–0.69 vs 0.74–0.75 del índice); la logística sobre
  log-ratios sí (0.755/0.772). **Resultado incómodo a reportar**: en R3 vs anillo solo, el clay ratio gana por 0.036
  [−0.065, −0.004]; la ventaja in situ del ganador viene de apagar distractores.
- **Estabilidad temporal** (`scripts/temporal_stability_sample.py` + `_eval.py` → `data/results/temporal_stability.json`,
  CSVs en `data/ground_truth/temporal/`, no versionados): mismo índice, mismas etiquetas, otro compuesto.
  Misma estación del año anterior: estable en los tres sitios (R3 hard 0.843, R4 hard 0.774, R5 hard 0.957; Spearman
  0.94–0.97 con el base). **Estación opuesta degrada**: invierno R3 hard 0.726 (clay 0.764, lo supera), invierno R4
  hard 0.733 (clay 0.588, sigue ganando), primavera R5 hard 0.903 (clay 0.878, ventaja cae a +0.025). Los términos
  VNIR (B8A/B04, B02/B03) son sensibles a iluminación y nieve; el clay ratio es más robusto a la estación. El índice
  se declara para compuestos de verano. Chequeo: el base remuestreado reproduce los valores almacenados (dif. rel.
  mediana 0): la colección GEE no se movió.
- **Tercer sitio R6 "Atacama oeste" y P-A con yacimientos (18 oct 2026)**: `scripts/sample_site6_and_pa.py` +
  `scripts/evaluate_site6_and_pa.py` → `data/results/site6_and_pa.json`. R6 = 320 polígonos de 5 cartas SERNAGEOMIN
  (La Guardia, Carrera Pinto, Los Loros, Qda Salitrosa, El Salvador; `data/external/sernageomin_atacama_oeste/`), nunca
  usados; se descartan muestras a <5 km de la alteración del Atlas III. **La compuerta FALLA en R6**: ganador +0.016
  (anillo) y +0.013 (con distractores), IC incluyen 0; el ganador de cajas lejanas es el mejor ahí (+0.029/+0.024).
  Sombra casi ausente en R6 (10 px); distractores efectivos = playa + aluvio. P-A contra yacimientos (Yousefi &
  Carranza; máscara NDVI/NDWI; IC bootstrap por clusters de 2 km): índice > clay en Coquimbo (intersección +10.6 pts,
  success AUC +0.131) y en R6 (+11.4, +0.146), ambos IC excluyen 0; clay ≥ índice en Maricunga (n.s.) y Cuprite
  (intersección −4.8 [−11.2, −0.3]). El clay ratio queda bajo azar (success AUC 0.40 y 0.47) en los distritos
  cretácicos de Cu-Fe-Au (Coquimbo, R6): ahí el índice gana por sus términos férricos/VNIR, no por Al-OH; en
  distritos epitermales dominados por SWIR (Maricunga, Goldfield-Cuprite) gana el clay ratio. Yacimientos:
  `data/external/yacimientos/` (Coquimbo 609, MRDS Cuprite 308) y Atlas III `yacimientos.csv`.
- **Remuestreo de la selección (18 oct 2026)**: `scripts/selection_resampling.py` (AUC vectorizado con searchsorted,
  validado contra sklearn; 5 procesos) → `data/results/selection_resampling.csv`, 21 particiones × 3 criterios; la
  semilla 0 reproduce la Tabla de criterios. Anillo+distractores: 5 ganadores distintos (el del paper en 13/21), pasa
  Coquimbo y Cuprite 21/21, Atacama oeste 4/21. Cajas lejanas: Coquimbo 18/21, Cuprite y R6 0/21. Anillo solo: 4, 3 y
  0/21. El efecto del criterio es robusto; ninguno transfiere de forma fiable al tercer sitio.
- **Manuscrito vigente: `paper/main_v3.tex`** (reencuadre: índices específicos del tipo de depósito). Versión
  Springer/NRR generada con `scripts/build_springer_v3.py` → `paper/springer_v3/` (editar solo main_v3.tex).
- Corridas largas: checkpoint cada 10 lotes en `data/results/.ckpt_enum_logspace_*.npz` (reanuda solo), lanzar con
  `nice -n 10`, `OMP_NUM_THREADS=6` y `systemd-run --user --scope -p MemoryMax=12G` (la máquina se congeló el 9 oct).
