# Experimento exp_004 — CNN 1D NORMALIZADA con LOSO real

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../LEGAL.md).


## Setup

- Igual que exp_003 pero: entrada z-score por ventana, 15 epochs, weight_decay=0.0001, dropout=0.5

## Resultados por fold

| Test | AUPRC (CNN v2) | AUPRC (CNN v1) | AUPRC (baseline) | sens@95spec | ictales |
|---|---|---|---|---|---|
| chb01 | 0.069 | 0.253 | 0.721 | 0.375 | 112 |
| chb02 | 0.05 | 0.005 | 0.239 | 0.341 | 44 |
| chb03 | 0.182 | 0.011 | 0.565 | 0.495 | 103 |
| chb04 | 0.035 | 0.005 | 0.016 | 0.358 | 95 |
| chb05 | 0.058 | 0.019 | 0.721 | 0.114 | 140 |
| chb06 | 0.002 | 0.001 | 0.002 | 0.1 | 40 |

## Resumen LOSO (media ± desvío)

| Métrica | CNN v2 | CNN v1 | Baseline |
|---|---|---|---|
| AUPRC | 0.066 ± 0.056 | 0.049 ± 0.091 | 0.377 ± 0.306 |
| sens@95spec | 0.297 ± 0.143 | 0.073 ± 0.163 | 0.553 ± 0.281 |
| ROC-AUC | 0.726 ± 0.102 | 0.467 ± 0.115 | 0.755 ± 0.117 |

## ⚖️ Veredicto: avance grande, pero el baseline sigue ganando

**La normalización arregló el problema central.** El ROC-AUC pasó de **0.467**
(peor que el azar) a **0.726** (competitivo con el baseline, 0.755). El diagnóstico
de exp_003 era correcto: el modelo fallaba por la escala de la entrada, no por la
arquitectura. Ahora la CNN **sí aprende a distinguir crisis entre pacientes**.

Pero en la métrica que más importa clínicamente **el baseline sigue arriba**:
- AUPRC: CNN 0.066 vs baseline **0.377**.
- sens@95spec: CNN 0.297 vs baseline **0.553**.

Interpretación: con desbalance extremo (~1% ictal), el ROC-AUC puede verse bien
mientras el AUPRC sigue bajo. La CNN **ordena** razonablemente (ranking decente) pero
tiene demasiados falsos positivos entre los pocos positivos reales → precisión baja.

**Un matiz interesante:** la CNN es mucho más **estable** entre sujetos
(AUPRC ±0.056) que el baseline (±0.306). El baseline es genial en algunos pacientes
y pésimo en otros; la CNN es consistentemente mediocre. Ninguno es "bueno" todavía.

## Conclusión

- exp_003 → exp_004: la CNN dejó de ser peor que el azar. Gran paso, diagnóstico
  confirmado.
- Pero **no supera al baseline** en AUPRC ni sensibilidad. **El baseline (exp_002,
  AUPRC 0.377) sigue siendo el mejor modelo honesto del proyecto.**

## Próximos pasos (para que la CNN supere al baseline)

1. **Más pacientes** (chb07–chb12+). Es la palanca más fuerte en EEG; 5 de train es
   muy poco para una red.
2. **Atacar la precisión:** el problema es AUPRC, no ranking. Probar umbral óptimo,
   más regularización, o una arquitectura que capture mejor el patrón temporal
   (CNN+LSTM de la Etapa 3 avanzada).
3. Tuning de hiperparámetros (lr, dropout, tamaño de la red) con un paciente de
   validación, no de test.
