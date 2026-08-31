# Correo final a coautores (cierre + envío de versión final)

**Para:** gonzalo.rios@dim.uchile.cl
**CC:** mauricio.latorre@uoh.cl
**Asunto:** Re: Spectral Indices Discovery — versión final aplicando tu input, lista para submit ISPRS

---

Hola Gonzalo, hola Mauricio,

Gonzalo, gracias por las respuestas. Apliqué los 4 puntos a esta versión final del manuscrito (`main.pdf`, 37 pp + `supplementary.pdf`, 6 pp adjuntos).

**Cierre de los 4 puntos abiertos:**

1. **Propylitic en Cuprite — reforzado como evidencia.** §4.5 reescrita: el AUC 0.66 ya no se presenta como underperformance, sino como una señal honesta del framework sobre la heterogeneidad y el tamaño limitado de la clase USGS-propylitic en Cuprite. La frase de cierre del párrafo dice ahora: *"the framework adapts to local class quality, surfacing rather than concealing differences in ground-truth coherence between sites."* Lo enmarca como propiedad deseable de un método de descubrimiento automático (fallar informativamente en clases mal definidas) y refuerza el caso de method transferability.

2. **PCA — tono más neutral en tres lugares.**
   - Abstract: *"...not a new accuracy ceiling"* → *"SR and PCA therefore occupy distinct points on the accuracy–interpretability trade-off"*.
   - Results §4.4: el párrafo entero reescrito. PCA ya no se llama "ceiling against which SR cannot compete"; ahora *"PCA and SR occupy distinct positions on the trade-off rather than a strict ranking"* y se cierra con *"the two are complementary rather than substitutable"*.
   - Limitations (item 6): *"SR is not the most accurate dimensionality reduction"* → *"SR and PCA address different requirements within the accuracy–interpretability trade-off"*.

3. **Specificity gate — sin cambios al manuscrito.** El texto actual de §5.4 (Pareto front and a methodological lesson) ya lo presenta explícitamente como recomendación metodológica para futuros estudios SR-on-remote-sensing, no como disclaimer de la clase silícica. Coincido con tu lectura de que es suficiente como propuesta; el experimento completo de specificity gate sobre las 6 clases queda disponible para una versión posterior si un reviewer lo pide.

4. **Volumen — decidí dejarlo en 37 pp.** Razones:
   - ISPRS JPRS no tiene límite estricto y 37 pp está dentro del rango habitual de papers metodológicos del journal.
   - Mover `tab:spatial_robust` al supplementary rompe el §4.4: la tabla soporta simultáneamente la robustez espacial y la comparación con baselines de dim red, y separarla obliga a mantener resúmenes redundantes en el main.
   - Si el AE pide condensar en major revision, hay ~1.5 pp obvios para mover sin tocar el flujo: la Tabla 11 (DL comparison) al supplementary y compactación de la enumeración de Limitations. Lo dejo como contingencia.

**Estado del manuscrito tras estos cambios:**

- 37 pp main + 6 pp supplementary, compila limpio (0 errores, 0 referencias indefinidas).
- Todos los puntos de Mauricio (11 + 9 = 20 entre los dos informes) y todos los de Gonzalo (5 + 4 = 9) atendidos.
- Estimación de Mauricio en su informe final: 80–90 % pasar a revisión, 55–70 % aceptación tras major revision. Coincido.

**Dos cambios menores aplicados de mi lado:**

1. **Afiliación actualizada.** Mi afiliación en el manuscrito y supplementary pasó de `CITIAPS, Santiago, Chile` a `Universidad de Santiago de Chile (USACH), Santiago, Chile`. Ya no estoy en CITIAPS y la portada queda alineada con mi convenio actual. Email corresponding actualizado a `francisco.parra.o@usach.cl`.
2. **Agradecimiento postdoc agregado.** Bloque nuevo en Acknowledgements: *"F.P.\\ acknowledges funding support from the Postdoctorate 2026 Project, Code 062619MC\_POSTDOC, Vicerrectoría de Investigación, Innovación y Creación, Universidad de Santiago de Chile."*

Con esos dos cambios la versión adjunta crece a 38 pp (era 37 antes del párrafo nuevo en Acknowledgements).

Si después de revisar el PDF no tienen más comentarios, envío a ISPRS Journal of Photogrammetry and Remote Sensing en los próximos días. Cualquier observación adicional, bienvenida.

Gracias a los dos por el doble ciclo de review — el paper quedó materialmente más sólido que en la versión inicial.

Saludos,
Francisco

---

**Adjuntos:**
- `main.pdf` — Manuscrito final, 38 pp
- `supplementary.pdf` — Material suplementario, 6 pp
