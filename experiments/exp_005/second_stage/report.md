# Segunda etapa anti-falsas-alarmas — NO le gana a subir el umbral

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../../LEGAL.md).


Idea: un clasificador que filtre los eventos candidatos del CNN para bajar las falsas
alarmas sin perder crisis. Evaluado en LOSO honesto (la 2da etapa nunca ve al paciente
de test). Figura: [segunda_etapa.png](segunda_etapa.png).

## El resultado, contra el baseline correcto

La pregunta no es "¿baja las FA/h?" (sí lo hace) sino **"¿le gana a simplemente subir el
umbral del CNN?"** — que también baja las FA/h y es gratis. Comparación a FA/h similar:

| Operación | Cobertura | FA/h |
|---|---|---|
| **CNN más estricto** (percentil 0.99) | **77 %** | 0.70 |
| CNN + 2da etapa (umbral 0.5) | 65 % | 0.52 |
| **CNN más estricto** (percentil 0.995) | **73 %** | 0.22 |
| CNN + 2da etapa (umbral 0.7) | 58 % | 0.22 |
| **CNN más estricto** (percentil 0.98) | **77 %** | 1.97 |
| CNN + 2da etapa (umbral 0.3) | 67 % | 1.56 |

**El baseline trivial domina en todos los puntos** (10–15 pts más de cobertura a la misma
tasa de falsas alarmas). La segunda etapa aprendida agrega complejidad y rinde **peor**.

## Por qué

El **score del propio CNN ya es el mejor discriminador**. Cuando la 2da etapa rechaza
eventos, también tira algunos verdaderos — peor que quedarse con los que el CNN puntúa
más alto. Subir el umbral hace exactamente eso, sin features hechas a mano ni un modelo extra.

## Conclusión

- **No se adopta la segunda etapa.** Dominada por una línea de config.
- **Hallazgo útil (y gratis):** el eje sensibilidad ↔ falsas alarmas se controla muy bien
  con el umbral de 1ra etapa. Con **percentil 0.99 se obtiene 77 % de cobertura a < 1
  FA/h** — que es justo la calibración por defecto del modelo de despliegue.
- Con esto van **dos intentos de mejora rechazados con evidencia** (híbrido y 2da etapa).
  Señal fuerte: **CNN + calibración por paciente está cerca de su techo dado el dato.**
  El margen que queda es chb06 (limitación de datos), no complejidad de modelo.
