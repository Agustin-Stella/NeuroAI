# Híbrido CNN + baseline — resultado NEGATIVO (y por qué importa)

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../../LEGAL.md).


Hipótesis (de exp_004): la CNN y el baseline de bandpower fallan en pacientes distintos,
así que combinarlos debería cubrir más crisis. **La evidencia dice que no.**
Evaluación por evento (LOSO, chb01–09), figura: [hibrido.png](hibrido.png).

## Resultado

| Modelo | Cobertura (crisis) | FA/h (mediana) |
|---|---|---|
| **CNN sola** | **81 %** (39/48) | **7.2** |
| Baseline | 73 % (35/48) | 7.4 |
| Híbrido (rank-average) | 79 % (38/48) | 7.4 |

El híbrido quedó **peor que la CNN sola** (−2 pts). No se adopta.

## Por qué NO ayuda (el hallazgo real)

Las **9 crisis que la CNN no detecta son todas de chb06** (detecta 1 de 10). Y el
**baseline también falla chb06 por completo** (0 de 10). Es decir: los dos modelos
fallan en el **mismo** paciente.

Por eso *ningún* esquema de combinación puede romper el techo del 81 %:
- Promedio / rank-average → diluye la señal fuerte de la CNN con el baseline más débil
  (por eso bajó a 79 %).
- Unión ("OR", disparar si cualquiera se activa) → tampoco, porque ambos fallan chb06,
  así que no hay crisis nueva que rescatar.

El techo del 81 % **no es un problema de diversidad de modelos: es chb06**, un paciente
con crisis cortas y sin contraste de amplitud (ver el error analysis). Tercera vez que
la evidencia apunta a lo mismo: es una limitación de **datos/señal**, no de modelo.

## Conclusión

- **No se adopta el híbrido.** La CNN sola sigue siendo el mejor modelo.
- La complementariedad que se veía en exp_004 (a nivel de AUPRC por ventana) **no se
  traslada a la cobertura por evento con calibración por paciente**: la CNN ya captura
  lo que el baseline aporta, y más.
- Es la mentalidad del proyecto funcionando: se probó, no hubo evidencia de mejora, **no
  se agrega la complejidad**.

## Qué SÍ mejoraría las estadísticas (para la próxima)

1. **Cobertura**: solo sube atacando chb06 → datos/features para crisis cortas y sutiles
   (ventanas más cortas, detección espectral del onset). No más modelos.
2. **Falsas alarmas** (el otro eje, ~7 FA/h es alto): una **segunda etapa** que filtre
   eventos-artefacto, o post-procesado más estricto. Más tractable que subir cobertura.
