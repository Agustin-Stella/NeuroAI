# Calibración y reporte robusto — exp_005

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../../LEGAL.md).


Análisis sobre los checkpoints de exp_005 (sin reentrenar).
Figura: [umbral_por_paciente.png](umbral_por_paciente.png).

## #1 — Resumen LOSO robusto (no dejar que chb06 defina el éxito)

| Métrica | Media | **Mediana** | Media sin chb06 | Baseline |
|---|---|---|---|---|
| AUPRC | 0.517 | **0.498** | 0.581 | 0.377 |
| sens@95spec | 0.772 | **0.890** | — | 0.553 |

La media (0.517) está deprimida por el outlier chb06 (0.002). La **mediana de AUPRC
(0.498)** y la media sin chb06 (0.581) describen mejor al modelo típico: en el
paciente mediano detecta el **89% de las crisis a 95% de especificidad**. El modelo
no es "mediocre con un fold roto": es bueno en la mayoría, con 1–2 sujetos difíciles.

## #3 — El modelo rankea bien pero está MAL CALIBRADO

El umbral que da 95% de especificidad **cambia radicalmente por paciente**:

| | rango de umbrales@95spec | conclusión |
|---|---|---|
| chb01, chb03 | ~0.01–0.05 | scores comprimidos abajo |
| chb09 | 1.000 | scores saturados arriba |
| **Todos** | **0.011 → 1.000 (rango 0.99)** | **no existe un umbral global útil** |

Contraste cuantificado (sensibilidad @95% spec):

| Estrategia de umbral | sens media |
|---|---|
| **Por paciente** | **0.772** |
| Global único (0.965) | 0.579 |

Un umbral fijo global pierde **~19 puntos de sensibilidad** y además la especificidad
real se descontrola por paciente (p. ej. chb09 cae a spec 0.48). El ranking del modelo
(ROC-AUC ~0.91) es bueno; el problema es de **punto de operación**, y se resuelve
eligiendo el umbral por paciente (o calibrando, p. ej. Platt/temperature scaling sobre
datos de validación del sujeto).

## Implicancia práctica

Cualquier reporte de sensibilidad/especificidad del sistema **debe fijar el umbral por
paciente** (o por un tramo de calibración del sujeto), nunca uno global. El desbalance
extremo + el shift de dominio LOSO + `pos_weight=15` dejan al modelo sobre-confiado y
con escalas de score incomparables entre sujetos.

## ✅ Solución: umbral por percentil (label-free, deployable)

En LOSO el paciente de test es desconocido → no se puede elegir el umbral con sus
etiquetas de crisis (leakage). Como las ventanas ictales son ~0.1–0.5 %, el **percentil
95 de los scores del propio paciente** aproxima el umbral de 95 % especificidad **sin
usar etiquetas**. Implementado en [`neuropilot/evaluation/calibration.py`](../../../neuropilot/evaluation/calibration.py);
evaluación en [`calibration_eval.py`](../calibration_eval.py). Figura:
[calibracion_metodos.png](calibracion_metodos.png).

| Estrategia (@95 % spec objetivo) | sens media | spec media |
|---|---|---|
| GLOBAL (umbral único, roto) | 0.573 | 0.94 (descontrolada: chb09 → 0.50) |
| **PERCENTIL por paciente (label-free)** | **0.761** | 0.96 |
| TEMPORAL (calibra en 1er 50 %, label-free) | 0.767 | 0.93 |
| ORÁCULO (usa etiquetas del test) | 0.772 | 0.96 |

**El percentil label-free recupera el 99 % de la sensibilidad del oráculo** (0.761 vs
0.772) manteniendo la especificidad clavada en ~0.95 por paciente, **sin mirar las
crisis**. Recupera +19 puntos de sensibilidad sobre el umbral global y elimina el
descontrol de especificidad (chb09: 0.50 → 0.95).

La variante TEMPORAL (calibra sobre el tramo basal inicial y evalúa en el resto) da casi
lo mismo → el método es robusto y fiel a despliegue: **calibrar el umbral sobre una
ventana basal del paciente, sin anotar crisis.**

### Qué queda para producción
- Elegir el objetivo de especificidad según el caso de uso (asistencia → priorizar
  sensibilidad; puede convenir 90 % spec).
- Calibrar sobre un tramo basal real del sujeto (no toda la grabación) — la variante
  TEMPORAL ya lo prueba.
- chb06 sigue en 0.10: la calibración NO lo arregla (es el problema de datos ya
  diagnosticado), pero tampoco lo empeora.
