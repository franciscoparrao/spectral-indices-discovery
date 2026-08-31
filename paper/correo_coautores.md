# Correo a coautores (Gonzalo + Mauricio)

**Para:** gonzalo.rios@dim.uchile.cl
**CC:** mauricio.latorre@uoh.cl
**Asunto:** Spectral Indices Discovery — versión post-revisiones, listo para submit ISPRS (pendiente input de Gonzalo)

---

Hola Gonzalo, hola Mauricio,

Les mando la versión consolidada del manuscrito (`main.pdf`, 37 pp) y el supplementary actualizado (`supplementary.pdf`, 6 pp), ya con todas las revisiones de ambos integradas. Adjunto también las respuestas formales: una para Gonzalo (`respuesta_gonzalo.pdf`) y dos para Mauricio (`respuesta_mauricio.pdf` para el primer informe del 20-04, y `respuesta_mauricio_v2.pdf` para el informe final del 25-04).

**Mauricio**, lo tuyo aplicado en esta versión:

- **Reordenamiento de autoría** (Parra → Ríos → Latorre) y tus 5 afiliaciones agregadas (UOH-Bioingeniería, SYSTEMIX, CGR-Millennium, CMM-UChile, INTA-UChile).
- **Agradecimientos** completos: ANILLO ACT210004, ACE210010, ICN2021_044, FB210005, FONDECYT 1230194, Núcleo UOH.
- **Caracteres corruptos** en el supplementary corregidos (faltaba `\usepackage[T1]{fontenc}`).
- **Cross-ref roto** `Supplementary Table~??` reparado (ahora apunta a S5 explícito).
- **Numeración** de Cuprite ajustada (referencia a 4.5, no a 4).
- **"unbiased" → "independent"** para Cuprite (tu observación de que el ground truth viene de ASTER/USGS, no de campo).
- **Abstract y Conclusion suavizados:** "applicable to any multispectral classification problem" → "applicable to a broad range of multispectral classification problems where labeled pixels are available". Land cover: "confirms" → "supports".
- **Discussion consolidada** de 8 a 6 subsecciones: eliminé el 6.4 "Limitations of SWIR Configuration" (era redundante con el item 3 de la enumeración del 6.7) y condensé el 6.5 "Ground Truth Quality" en un bullet nuevo dentro de la enumeración. Resultado más limpio sin perder contenido.

Con eso quedan atendidos los 11/11 puntos de tu informe final. **Sobre el nested validation:** seguí tu recomendación de no implementarlo antes del submit (15h de cómputo, opcional según tu propio informe) y dejarlo como respuesta a un eventual reviewer metodológico en major revision. Si crees que conviene revertir y correrlo ahora, me dices.

**Gonzalo**, te resumo lo que respondimos a tus 5 puntos (detalle completo en `respuesta_gonzalo.pdf`):

- **Cuprite formulas (GR1):** PySR re-corrido localmente; las fórmulas chilenas y las de Cuprite son **estructuralmente distintas** (VNIR vs SWIR-dominated), evidencia explícita de method transferability.
- **PCA(6) y MI-top-6 (GR2):** ambos baselines agregados. PCA(6) supera a SR por ~4–5 AUC points y define el ceiling. Acepté tu reformulación literal — el contribution de SR es interpretabilidad sin costo frente a raw bands, no un nuevo accuracy ceiling.
- **Autocorrelación espacial (GR3, la crítica más fuerte):** confirmada empíricamente. Bajo polygon-disjoint y spatial block CV el ΔAUC absoluto cae hasta −0.21, pero las tres relaciones de interés (SR ≈ raw, SR > classical, SR+classical best) sobreviven a todos los esquemas. Tabla nueva `tab:spatial_robust` con 5 esquemas × 4 feature sets + 2 baselines.
- **TOST expandido (GR4):** CI 95% fold-level, justificación operacional de ε=0.01, y bootstrap pareado polygon-level (B=1000) como check independiente.
- **Specificity gate (GR5):** análisis del Pareto front silícico mostró que sí existen fórmulas con contraste VNIR/SWIR dentro del budget complexity ≤ 8. Convertido en recomendación metodológica, no en disclaimer.

**Lo que quedó pendiente esperando tu input — los 4 puntos abiertos:**

1. **Propylitic en Cuprite (AUC 0.66 local).** ¿Lo dejamos como caveat en la tabla o lo reforzamos como evidencia de que el framework detecta correctamente clases compositionally heterogeneous? Causas probables: (a) USGS merge epidote/chlorite + carbonate, (b) n=378 positives.

2. **PCA como ceiling.** ¿Aceptable el framing de "SR no es el mejor compressor, es el mejor compressor interpretable"? O preferirías un tono más neutral tipo "SR y PCA ocupan puntos distintos del trade-off accuracy/interpretability".

3. **Specificity gate como workflow recommendation.** Está propuesto pero no implementado numéricamente para las 6 clases — solo para silícica, donde se manifestó el problema. ¿Vale la pena hacer el experimento completo antes del submit, o suficiente como propuesta metodológica?

4. **Volumen (37 pp).** ISPRS JPRS no tiene límite estricto pero algunos reviewers lo penalizan. ¿Movemos algo al supplementary (e.g., la tabla de spatial block 1/2/5/10 km resumida en main, detalle al supp)?

**Plan:** apenas tengamos el OK de Gonzalo sobre estos 4 puntos, enviamos a ISPRS Journal of Photogrammetry and Remote Sensing. Si les acomoda una llamada corta (30 min) esta semana o la próxima para cerrar todo de una, organizo. Si prefieren responder por aquí, también.

Gracias a los dos — el paper quedó materialmente más sólido con el doble ciclo de review. Cualquier comentario adicional, bienvenido.

Saludos,
Francisco

---

**Adjuntos:**
- `main.pdf` — Manuscrito revisado, 37 pp
- `supplementary.pdf` — Material suplementario, 6 pp
- `respuesta_mauricio.pdf` — Respuesta formal a los 11 puntos del primer informe de Mauricio (20 abr)
- `respuesta_mauricio_v2.pdf` — Respuesta formal a los 9 puntos del informe final de Mauricio (25 abr)
- `respuesta_gonzalo.pdf` — Respuesta formal a los 5 puntos de Gonzalo (22 abr)
