# Revisión de literatura: descubrimiento automático de índices espectrales por enumeración exhaustiva y regresión simbólica

Fecha: 2026-10-09. Generada por agente de búsqueda web (~40 búsquedas, ~20 abstracts/textos completos) a pedido de F. Parra. Marcas: **[V]** verificado (abstract o texto completo), **[V-sec]** verificado por fuente secundaria, **[a verificar]** de memoria, no confirmado.

**BLUF.** La enumeración exhaustiva de índices de dos y tres bandas es rutina en agricultura hiperespectral desde 2000, y la programación genética (GP) para evolucionar índices existe desde 2005-2008; en 2026 ya hay un paper con PySR sobre Sentinel-2 validado en 11 regiones independientes. Pero no se encontró ningún trabajo que combine (a) enumeración con garantía de óptimo global, (b) criterio AUC con corrección de sesgo de selección, (c) frente de Pareto exacto complejidad-desempeño y (d) aplicación a alteración hidrotermal en Sentinel-2/ASTER. La regresión simbólica exhaustiva (ESR, Bartlett et al. 2024) da el marco teórico pero sólo con verosimilitud gaussiana y en cosmología. En geología, la selección de ratios sigue siendo por conocimiento experto; lo más cercano a una búsqueda sistemática es GA-PCA (estocástico) y BRMT (todos los ratios ASTER a PCA, sin criterio supervisado).

---

## 1. Referencias clave por tema

### Tema 1 — Enumeración exhaustiva de combinaciones de bandas

1. **Thenkabail, Smith & De Pauw (2000)**, *Remote Sens. Environ.* 71(2):158-182, DOI 10.1016/S0034-4257(99)00067-X. **[V]** NDVI de banda angosta para *todas* las parejas de 490 canales (350-1050 nm) en cinco cultivos. Precedente más antiguo que trata el sobreajuste de la búsqueda masiva: compara el R² obtenido contra el R² de datos aleatorios (baseline nulo informal).
2. **Hansen & Schjoerring (2003)**, *RSE* 86(4):542-553. **[V-sec]** NDVI de todas las parejas + PLS para biomasa y N en trigo. Origen de los "mapas de contorno" de R² sobre la matriz (λi, λj).
3. **Inoue et al. (2008)**, *RSE* 112, DOI 10.1016/j.rse.2007.04.011. **[V]** NDSI sobre todos los pares para eficiencia fotosintética en arroz.
4. **Li, Mistele, Hu, Chen & Schmidhalter (2013)**, *Agric. For. Meteorol.* 180:44-57, DOI 10.1016/j.agrformet.2013.05.003. **[V]** Compara tres algoritmos de optimización de índices (todas las combinaciones de dos bandas, posición de red-edge, área de tres bandas) para N en trigo; los índices optimizados fueron más robustos que 15 índices publicados. **Li et al. (2014)**, *ISPRS JPRS* 92:112-123 **[V-sec]** extiende a tres bandas.
5. **Mahlein et al. (2013)**, *RSE* 128:21-30, DOI 10.1016/j.rse.2012.09.019. **[V-sec]** Índices de enfermedad específicos (RELIEF-F + NDSI de dos bandas).
6. **Gerstmann, Möller & Gläßer (2016)**, *Int. J. Appl. Earth Obs. Geoinf.* 52:115-125. **[V-sec]** Optimización automatizada de índices para separabilidad de cereales en RapidEye; búsqueda exhaustiva sobre pares de bandas y constantes.
7. **Koh, Banerjee, Spangenberg & Kant (2022)**, *New Phytologist*, DOI 10.1111/nph.17947. **[V]** AutoVI: optimización bayesiana (TPE/Optuna) sobre 33 formas de índice (2-6 bandas). Argumenta que la fuerza bruta "escala mal", pero no la compara. Sobreajuste al pasar de 20 000 a 40 000 iteraciones (n = 80).
8. **Liu, Cui, Wang, Han & Kou (2025)**, *Molecules* 30(15):3215, DOI 10.3390/molecules30153215. **[V]** Único trabajo encontrado que mide exhaustivo vs heurístico: 4 657 296 tripletas en 8 h vs CARS en 2 s; R² 0.77 vs 0.76. Criterio: correlación con Cu en suelo.
9. **Philpot (2021)**, "ND-Space: Normalized Difference Spectral Mapping", *IGARSS 2021*: 4015-4018. **[V-sec]** Mapeo del espacio completo de diferencias normalizadas.
10. **Sensors 2017, 17(12):2711** (perfil vertical de clorofila en trigo). **[V]** Todas las combinaciones de dos y tres bandas con formas NDVI/SR/CI.
11. Software: paquete R **hsdar**, `nri(recursive=TRUE)`. **[V]**

**Geología / minerales:**

12. **Yang, Jia, Dong, Zhao & Zhao (2024)**, *Remote Sensing* 16(2):392, DOI 10.3390/rs16020392. **[V]** GA-PCA: algoritmo genético embebido en PCA para combinaciones de bandas Landsat-8 OLI para óxidos de hierro e hidroxilos. Estocástico, no exhaustivo.
13. **Askari, Beiranvand Pour, Pradhan, Sarfi & Nazemnejad (2018)**, *Sensors* 18(10):3213, DOI 10.3390/s18103213. **[V]** BRMT: matriz no repetitiva de todos los ratios VNIR-SWIR de ASTER, seguida de PCA. Enumeración completa de ratios, sin criterio supervisado ni ranking.
14. **van der Meer, van der Werff & van Ruitenbeek (2014)**, *RSE* 148:124-133, DOI 10.1016/j.rse.2014.03.022. **[V]** Ratios Sentinel-2 para Fe³⁺, Fe²⁺, laterita, gossan, por analogía con ASTER. **van der Werff & van der Meer (2015)**, *Remote Sensing* 7(10):12635 **[V]**.
15. Estudios de caso WorldView-3 (Chadormalu 2019, Mountain Pass 2018), PRISMA (Aktogay 2025, Kuhpanj 2024) **[V-sec]**: ratios elegidos a mano. No se encontró ningún estudio EnMAP ni PRISMA con búsqueda sistemática de ratios.

### Tema 2 — Regresión simbólica / GP en teledetección

16. **Ross, Gualtieri, Fueten & Budkewitsch (2005)**, *Applied Soft Computing* 5(2):147-156, DOI 10.1016/j.asoc.2004.06.003. **[V]** GP evoluciona identificadores de alunita, caolinita y buddingtonita en 168 bandas AVIRIS de Cuprite; salida presente/ausente. Único trabajo de GP para minerales encontrado; clasificador, no índice continuo.
17. **Chion, Landry & Da Costa (2008)**, *IEEE TGRS* 46(8):2446-2457, DOI 10.1109/TGRS.2008.922061. **[V]** GP-SVI: regresión GP para N en maíz desde CASI.
18. **Momm, Easson & Kuszmaul (2008)**, *Proc. SPIE* 6812, DOI 10.1117/12.766367; **Momm & Easson (2011)**, *J. Appl. Remote Sens.*, DOI 10.1117/1.3662089. **[V]** Análisis de incertidumbre del GP para índices (Monte Carlo, 800 realizaciones): la variabilidad por semilla es sustancial. Evidencia directa del no-determinismo del GP.
19. **Puente, Olague et al. (2009)**, EvoWorkshops LNCS 5484:375-384; **Puente et al. (2019)**, *Remote Sensing* 11(2):156, DOI 10.3390/rs11020156. **[V]** GP sintetiza índices para el factor C de RUSLE.
20. **Almeida et al. (2015)**, *Ecological Informatics* 26:61-69, DOI 10.1016/j.ecoinf.2015.01.003. **[V]** GP para índices de fenología desde cámaras digitales.
21. **Albarracín, Oliveira, Hirota, dos Santos & Torres (2020)**, *Remote Sensing* 12(14):2267, DOI 10.3390/rs12142267. **[V]** GPVI: un índice GP por cada par de clases, maximizando separabilidad.
22. **Batista & Silva (2020)**, arXiv 2002.00053. **[V]** M3GP evoluciona "hiper-rasgos" para área quemada.
23. **Stein, Vaske & Mielke (2026)**, *GECCO '26*, pp. 1220-1229, DOI 10.1145/3795095.3805155. **[V-sec]** Variantes de GP para índices a medida de patógeno-cultivo-sensor.
24. **Chrysostomou, Neophytides, Mavrovouniotis & Hadjimitsis (2026)**, *Sci. Rep.* 16:4491, DOI 10.1038/s41598-025-34720-x. **[V, texto completo]** PySR sobre Sentinel-2 L2A con etiquetas WorldCover. Operadores: +, −, ×, división protegida, min, max, abs, √; excluyen exp, log, trig y potencias. Fitness = Jeffries-Matusita − λ·complejidad − λ·término fenológico; población 1000, hasta 800 generaciones; selección desde el frente de Pareto. Entrenan sólo en Limassol con muestreo por bloques de 1 km, validan en 11 regiones independientes. SRVI = (2·NIR − 3·Red)/(NIR + Red + 0.5·(Green + SWIR1)); SRWI = ((G+B) − (NIR+SWIR1))/((G+B) + (NIR+SWIR1)). No usan AUC ni discuten sesgo de selección.
25. **Lotfi, Carter, Ha, Meysami, Nketia & Shirtliffe (2025, v2 2026)**, arXiv 2512.21948. **[V, texto completo]** "Spectral Feature Polynomial": biblioteca enumerada de rasgos invariantes a escala (ND de 2 bandas, ND3, curvatura normalizada; 372 términos base → 69 750 rasgos de grado 2 para S2), prefiltro ANOVA-F + SelectKBest/RFE con SVM lineal; 46 folds (aleatorio, año-fuera, bloques espaciales 3×3) para kochia en S2. Se autodescribe como "enumeración exhaustiva", pero el prefiltro lo hace no exhaustivo; la elección de método, C y pesos usa los mismos folds que reportan accuracy. Sin log ni potencias. Trabajo hermano: Normalized Difference Layer (arXiv 2601.06777) **[V]**.
26. **Hammoud, Sun, Champenois & Brewin (2026)**, arXiv 2609.18531. **[V]** PySR para log10(Chl-a) desde reflectancias hiperespectrales tipo PACE; recupera la estructura de ratios OC3-OC6.
27. **Yu et al. (2025)**, SatelliteFormula, arXiv 2506.06176. **[V]** ViT + optimizador simbólico; afirma que la SR "no se ha aplicado sistemáticamente a teledetección".
28. **Camps-Valls et al. (2021)**, *Science Advances* 7:eabc7447, kNDVI. **[V-sec]**

### Tema 3 — Regresión simbólica exhaustiva / determinista fuera de teledetección

29. **Bartlett, Desmond & Ferreira (2024)**, "Exhaustive Symbolic Regression", *IEEE TEVC* 28(4):950-964, DOI 10.1109/TEVC.2023.3280250 (arXiv 2211.11461). **[V, texto completo]** Enumera todos los árboles con base {x, a, inv, +, −, ×, ÷, pow} hasta complejidad 10: ~168 M nominales → 5.2 M árboles válidos → 134 234 ecuaciones únicas; parámetros por BFGS; selección por MDL. Garantía: "óptimo verdadero si los parámetros se optimizan perfectamente". Sólo verosimilitud gaussiana; no trata clasificación ni AUC.
30. **Kammerer, Kronberger, Burlacu, Winkler, Kommenda & Affenzeller (2020)**, "Symbolic Regression by Exhaustive Search", *GPTP XVII*, pp. 79-99 (arXiv 2109.13895). **[V]** Enumeración desde gramática libre de contexto, NLS para constantes, caché de duplicados semánticos. Antecedentes: **Worm & Chiu (2013)** PGE, *GECCO '13* **[V]**; **Korns (2013)** GPTP X **[V]**; **Kronberger et al. (2018)** GPTP XVI **[V]**.
31. **Kronberger, de França, Desmond, Bartlett & Kammerer (2024)**, "The Inefficiency of Genetic Programming for Symbolic Regression", *PPSN XVIII*, pp. 273-289 (arXiv 2404.17292). **[V]** Con el espacio enumerado miden la probabilidad de que GP encuentre la mejor expresión: GP explora una fracción pequeña de las expresiones únicas. Comparación formal GP vs enumeración, en dos datasets univariados.
32. **McConaghy (2011)**, FFX, *GPTP IX*, pp. 235-260, DOI 10.1007/978-1-4614-1770-5_13. **[V]** Enumera bases y las combina con elastic-net pathwise; determinista, frente de Pareto exactitud-complejidad.
33. **Udrescu & Tegmark (2020)**, AI Feynman, *Sci. Adv.* 6(16) **[V venue; DOI a verificar]**; **Brunton, Proctor & Kutz (2016)**, SINDy, *PNAS* 113(15):3932 **[a verificar]**; **Guimerà et al. (2020)**, Bayesian Machine Scientist, *Sci. Adv.* 6(5) **[V-sec]**.
34. **La Cava et al. (2021)**, SRBench, *NeurIPS Datasets & Benchmarks* **[V]**; **Burlacu, Kronberger & Kommenda (2020)**, Operon, *GECCO Companion* pp. 1562-1570 **[V]**.
35. **GP con AUC como fitness:** **Bhowan, Zhang & Johnston (2010)**, *GECCO '10* pp. 845-852 **[V]**; **Bhowan, Johnston & Zhang (2012)** *IEEE SMC-B* 42(2):406-421 **[V]**; **Bhowan et al. (2013)** *IEEE TEVC* 17(3):368-386 **[V-sec]**. Existe la maquinaria, nunca aplicada a índices espectrales.

### Tema 4 — Evaluación: AUC, Pareto, sesgo de selección, validación espacial

36. **Ambroise & McLachlan (2002)**, *PNAS* 99(10):6562-6566, DOI 10.1073/pnas.102102699. **[V]** Sesgo de selección cuando la selección de rasgos no se repite dentro de cada fold.
37. **Roberts et al. (2017)**, *Ecography* 40:913-929 **[V]**; **Ploton et al. (2020)**, *Nat. Commun.* 11:4540 **[V]**; **Meyer & Pebesma (2022)**, *Nat. Commun.* 13 **[V]**. CV por bloques, validación espacial, área de aplicabilidad.
38. **Smits & Kotanchek (2004)**, GPTP II **[V-sec]**; **Kommenda et al. (2021)**, arXiv 2109.00238 **[V]**: medidas de complejidad para SR multiobjetivo.
39. ROC-AUC en geología: estudio 2026 Macizo de Sirwa, *J. African Earth Sci.* **[V-sec]**, valida un mapa de prospectividad con ROC-AUC; no usa AUC para elegir el ratio. Ningún estudio encontrado que rankee índices por AUC.

### Tema 5 — Invariancia a escala / espacio log

40. **Green & Craig (1985)**, log residuals, JPL Publ. 85-41 **[V-sec]**: resta en espacio log; elimina factores multiplicativos de iluminación y topografía.
41. **Stumpf, Holderied & Sinclair (2003)**, *Limnol. Oceanogr.* 48:547-556 **[V-sec]**: batimetría con ratio de logaritmos.
42. NLI (Goel & Qin 1994), RDVI (Roujean & Bréon 1995) **[a verificar]**: formas potencia ad hoc, nunca como espacio de búsqueda.
43. Análisis composicional (Aitchison; ilr de Egozcue et al. 2003) **[V genérico]**: sin aplicación encontrada a bandas de reflectancia.
44. **Montero et al. (2023)**, catálogo de índices, *Sci. Data*, DOI 10.1038/s41597-023-02096-0 **[V]**.

---

## 2. Qué está resuelto

**Verificado:**
- Enumerar todas las parejas (B_i − B_j)/(B_i + B_j) y correlacionarlas con una variable es estándar en agricultura hiperespectral desde Thenkabail (2000); hay software (hsdar). Tripletas exhaustivas factibles hasta ~10⁶-10⁷ combinaciones en horas (Liu 2025).
- GP para evolucionar índices está maduro: Ross 2005, Chion 2008, Puente 2009/2019, Albarracín 2020, Stein 2026. El no-determinismo del GP está documentado (Momm 2008).
- PySR ya produjo índices Sentinel-2 compactos con validación en regiones independientes (Chrysostomou 2026); "biblioteca enumerada + filtro" con validación por bloques existe para vegetación (Lotfi 2025).
- La teoría de SR exhaustiva existe: enumeración con deduplicación semántica (Kammerer 2020, ESR 2024), selección por MDL, garantía de óptimo condicionada a parámetros, evidencia de que GP explora una fracción pequeña del espacio (Kronberger 2024).
- Sesgo de selección (Ambroise & McLachlan 2002) y validación espacial (Roberts 2017, Ploton 2020, Meyer & Pebesma 2022) resueltos en teoría; Thenkabail (2000) ya usaba un baseline aleatorio.
- Fitness AUC en GP multiobjetivo existe (Bhowan 2010-2013).

**Inferido (probable):** los ratios geológicos en S2/ASTER/Landsat se siguen eligiendo por física de absorción + inspección visual; en ~10 estudios de caso revisados ninguno reporta una búsqueda sobre el espacio completo con criterio supervisado.

---

## 3. Gaps detectados

| # | Gap | Evidencia | Confianza |
|---|---|---|---|
| G1 | Enumeración exhaustiva con garantía de óptimo global para índices geológicos/de alteración en Sentinel-2, ASTER o Landsat. | 5 búsquedas. Lo más cercano: BRMT (no supervisado), GA-PCA 2024 (estocástico), Koh 2022 (TPE, agricultura). | Muy probable (80-95 %) |
| G2 | Regresión simbólica aplicada a alteración hidrotermal o minerales con sensores multiespectrales. | Único hallazgo: Ross 2005 (AVIRIS, clasificador GP, 20 años). | Muy probable (80-95 %) |
| G3 | AUC (o pérdida de ranking) como criterio de selección de índices, con corrección del sesgo de selección de una búsqueda masiva. | Chrysostomou usa JM; Lotfi usa accuracy y no trata comparaciones múltiples; Thenkabail (2000) único con baseline nulo, informal. | Muy probable (80-95 %) |
| G4 | Frente de Pareto exacto complejidad-desempeño para índices espectrales. | ESR lo hace en cosmología con MDL gaussiano; en teledetección sólo aproximaciones de GP. | Muy probable (80-95 %) |
| G5 | Comparación formal GP estocástica vs enumeración exhaustiva en índices espectrales. | Kronberger 2024 en SR univariada; Liu 2025 exhaustivo vs CARS; Koh 2022 afirma sin medir; Momm 2008 mide varianza sin referencia exhaustiva. | Muy probable (80-95 %) |
| G6 | Índices como productos de potencias de bandas (espacio log-lineal) con invariancia a escala, como espacio de búsqueda sistemático. | Precedentes aislados: log residuals (1985), Stumpf (2003), NLI/RDVI ad hoc, CoDA sin aplicación a bandas. Chrysostomou excluye log y potencias; Lotfi no las usa. | Probable (55-80 %): puede haber trabajos en espectroscopía de suelos con log(1/R). |
| G7 | Validación cross-site / CV espacial de índices geológicos descubiertos automáticamente. | Existe para vegetación (Chrysostomou, Lotfi). En geología la validación es por coincidencia con mapa geológico. | Muy probable (80-95 %) |
| G8 | Extensión de SR exhaustiva con MDL a verosimilitudes de clasificación/ranking. | ESR sólo gaussiana; Kammerer/PGE/FFX son de regresión. No se buscaron extensiones post-2024. | Probable (55-80 %) |

**Riesgo de novedad a citar de frente:** Lotfi et al. (2025, arXiv 2512.21948) es el competidor más cercano a "biblioteca enumerada de índices invariantes + selección con CV espacial". Una propuesta nueva debe diferenciarse en (i) exhaustividad real sin prefiltro, (ii) criterio AUC con corrección de sesgo, (iii) espacio log-lineal con potencias, (iv) dominio geológico, (v) frente de Pareto exacto.

---

## 4. Supuestos y límites

- Las búsquedas web no cubren Google Scholar ni bases chinas, donde la literatura de "optimización de índices de tres bandas" es abundante (afecta G1 y G6).
- No se leyó texto completo de: Stein 2026, Albarracín 2020, Koh 2022 (se usó bioRxiv), BRMT 2018 (abstract), Gerstmann 2016 (cita secundaria).
- Las fechas 2026 de varios arXiv/Sci. Rep. son las que entregan los índices; no validadas contra PDF.
