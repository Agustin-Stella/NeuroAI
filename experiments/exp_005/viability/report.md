# Test de viabilidad por evento — exp_005

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../../LEGAL.md).


¿Funcionaría como herramienta de asistencia? La pregunta correcta no es el AUPRC por
ventana, sino: **¿cuántas crisis detecta y a cuántas falsas alarmas/hora?** Evaluado
sobre los 9 folds LOSO (pacientes no vistos), con calibración por percentil label-free
y suavizado (exigir ≥2 ventanas seguidas). Motor: [`neuropilot/evaluation/events.py`](../../../neuropilot/evaluation/events.py).
Figura: [curva_viabilidad.png](curva_viabilidad.png).

## Resultado principal (percentil 95, min 2 ventanas)

| Paciente | crisis | detectadas | sens. evento | FA/h |
|---|---|---|---|---|
| chb01 | 7 | 7 | **100%** | 7.0 |
| chb02 | 3 | 3 | **100%** | 7.4 |
| chb03 | 7 | 7 | **100%** | 5.7 |
| chb04 | 4 | 4 | **100%** | 7.9 |
| chb05 | 5 | 5 | **100%** | 4.7 |
| chb06 | 10 | 1 | 10% | 7.4 |
| chb07 | 3 | 3 | **100%** | 8.0 |
| chb08 | 5 | 5 | **100%** | 5.3 |
| chb09 | 4 | 4 | **100%** | 7.2 |

**En 8 de 9 pacientes detecta el 100% de las crisis.** El único que falla es chb06 (el
problema de datos ya diagnosticado: crisis cortas y sin contraste de amplitud). Global:
**39/48 crisis (81%)**, arrastrado casi enteramente por chb06.

## El compromiso es ajustable (curva de viabilidad)

| Percentil calib. | sens. evento | FA/h (mediana) |
|---|---|---|
| 0.90 | 85% | 15.2 |
| 0.95 | 81% | 7.2 |
| 0.98 | 77% | 2.0 |
| **0.99** | **77%** | **0.7** |

Se puede bajar a **<1 falsa alarma/hora** manteniendo ~77% de sensibilidad por evento
(y ese 77% sigue siendo ~"todas menos chb06"). La perilla sensibilidad↔FA/h es
exactamente lo que una herramienta clínica necesita exponer.

## Veredicto: **es viable como asistencia** (con reservas honestas)

Detecta todas las crisis en 8/9 pacientes no vistos, con FA/h ajustable a <1/hora. Para
un detector cross-paciente sobre datos crudos, es un resultado real y defendible.

**Reservas:**
- Muestra chica: 48 crisis / 9 pacientes → incertidumbre alta.
- chb06 sin resolver (limitación de datos, no del modelo).
- CHB-MIT es pediátrico y relativamente controlado; el mundo real es más ruidoso.
- La calibración por percentil usa toda la grabación (la variante temporal ya mostró
  que calibrar sobre un tramo basal da casi lo mismo → deployable).
- FA/h en el punto sensible (~7/h) es alto para producción; el punto <1/h es más usable.

**Conclusión:** hay señal de viabilidad suficiente para justificar construir el MVP
ejecutable (inferencia end-to-end sobre un EDF + demo visual) y, después, la capa de
producto (API/explicabilidad).
