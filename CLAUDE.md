# Spectral Indices Discovery — Alteración Hidrotermal

## Paper Target

**"Automated Discovery of Spectral Indices for Hydrothermal Alteration Mapping using Symbolic Regression on Sentinel-2: A Case Study in the Chilean Andes"**

Target principal: Remote Sensing of Environment (IF 13.5) o ISPRS Journal (IF 12.7).
Alternativas: Ore Geology Reviews (IF 6.3), Remote Sensing MDPI (IF 5.0, rápido).

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
