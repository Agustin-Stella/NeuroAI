# Experimento exp_003 — CNN 1D con LOSO real (6 pacientes)

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../LEGAL.md).


## Identificación

- **ID:** exp_003
- **Fecha:** 2026-08-07
- **Modelo:** CNN 1D (aprende features de la señal cruda)
- **Hipótesis:** superar el baseline (exp_002, AUPRC 0.377 ± 0.306) con el mismo LOSO.

## Setup

- Datos: chb01–chb06, montaje canónico de 23 canales
- Señal resampleada a 128 Hz; ventana 4 s; overlap 0
- Train: negativos submuestreados 15:1; CNN 6 epochs, batch 128, lr 0.001, pos_weight, seed 42
- Eval: paciente de test COMPLETO (sin submuestrear)

## Resultados por fold

| Test | AUPRC (CNN) | AUPRC (baseline) | sens@95spec | F1 | ictales |
|---|---|---|---|---|---|
| chb01 | 0.253 | 0.721 | 0.438 | 0.035 | 112 |
| chb02 | 0.005 | 0.239 | 0.0 | 0.0 | 44 |
| chb03 | 0.011 | 0.565 | 0.0 | 0.023 | 103 |
| chb04 | 0.005 | 0.016 | 0.0 | 0.009 | 95 |
| chb05 | 0.019 | 0.721 | 0.0 | 0.038 | 140 |
| chb06 | 0.001 | 0.002 | 0.0 | 0.002 | 40 |

## Resumen LOSO (media ± desvío entre sujetos)

| Métrica | CNN | Baseline (exp_002) |
|---|---|---|
| AUPRC | 0.049 ± 0.091 | 0.377 ± 0.306 |
| sens@95spec | 0.073 ± 0.163 | 0.553 ± 0.281 |
| ROC-AUC | 0.467 ± 0.115 | 0.755 ± 0.117 |
| F1 | 0.018 ± 0.015 | 0.218 ± 0.218 |

## ⚖️ Veredicto: la CNN PERDIÓ contra el baseline

En TODOS los pacientes la CNN dio AUPRC igual o peor que el baseline. El ROC-AUC
promedio (0.467) es **peor que el azar** (0.5). No hay ninguna métrica en la que
gane. Conclusión honesta: **esta versión de la CNN es peor que la regresión
logística.** No se adopta.

Esto NO es un fracaso del proyecto — es la mentalidad científica funcionando: se
midió con el mismo LOSO y la evidencia dice que el modelo "más sofisticado" no
mejora nada. Justamente el principio del proyecto: no afirmar que un modelo es mejor
sin evidencia.

## Diagnóstico (por qué falló)

1. **Falta de normalización de la entrada (causa más probable).** La CNN recibió la
   señal cruda en voltios (~1e-4), sin normalizar por canal. El baseline sí
   normalizaba sus features (StandardScaler). La amplitud del EEG varía mucho entre
   pacientes; sin normalizar, lo aprendido en train no traslada al test.
   Ya tenemos `ChannelNormalizer` para esto — **no se usó en esta corrida**.

2. **Sobreajuste.** `train_loss` cayó a ~0.05 en 6 epochs: la red memorizó a los 5
   pacientes de train. Con tan pocos sujetos, memoriza rasgos del paciente en vez de
   la patología, y falla en uno nuevo. ROC-AUC < 0.5 es la firma de ese colapso.

3. **Pocos datos / pocos pacientes.** En EEG, la generalización depende muchísimo de
   la cantidad de sujetos. 5 pacientes de train es muy poco para una CNN.

## Próximos pasos (en orden de prioridad)

1. **Normalizar la entrada por canal** (z-score con `ChannelNormalizer`, ajustado en
   train) antes de la CNN. Es el arreglo más barato y probablemente el de mayor impacto.
2. **Regularizar más:** más dropout, weight decay, y early stopping sobre un paciente
   de validación (no de test).
3. **Más pacientes.** Bajar chb07–chb12 y repetir. Más datos es la palanca más fuerte.
4. Si con eso no supera al baseline: reconsiderar arquitectura (o volver al baseline
   como modelo de referencia, que sigue siendo el mejor honesto hasta ahora).

**Estado:** el baseline (exp_002, AUPRC 0.377 ± 0.306) sigue siendo el mejor modelo
honesto del proyecto.
