# Experimento exp_001 — Baseline sobre chb01

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../LEGAL.md).


## Identificación

- **ID:** exp_001
- **Fecha:** 2026-08-07
- **Autor:** Agustín Stella
- **Hipótesis:** las features de banda + regresión logística detectan actividad
  ictal en señal EEG real (smoke test del baseline sobre chb01).

## ⚠️ Validez metodológica

**Esto NO es una evaluación de generalización.** Solo se descargó **un paciente
(chb01)**, así que la división train/test es **por archivo dentro del mismo
paciente**. Eso viola el invariante de split por paciente y **sobreestima** el
rendimiento (el modelo ve el mismo cerebro en train y test). Es un test de
integración del baseline sobre datos reales, no un resultado científico.

La evaluación honesta (LOSO entre pacientes) requiere descargar más sujetos.

## Reproducibilidad

- **Config:** filtros 0.5–40 Hz + notch 60 Hz; ventana 4 s; overlap 0; sin normalizador
- **Seed:** 42
- **Modelo:** `StandardScaler` + `LogisticRegression(class_weight="balanced")`
- **Features:** potencia de banda δ/θ/α/β/γ por canal (`bandpower_features`, log1p)
- **Datos:** chb01 (23 canales, 256 Hz)
  - Train: chb01_01,02,03,04,05,15 → 5400 ventanas, 27 ictales (0.50%)
  - Test:  chb01_06,07,08,16,18 → 4500 ventanas, 36 ictales (0.80%)

## Resultados

| Métrica | Valor |
|---|---|
| **AUPRC** | **0.806** |
| ROC-AUC | 0.991 |
| **Sensibilidad @ 95% spec** | **0.944** |
| Sensibilidad @ 0.5 | 0.667 |
| Especificidad @ 0.5 | 0.999 |
| Precision @ 0.5 | 0.857 |
| F1 @ 0.5 | 0.750 |
| Confusión @ 0.5 | tp=24 fp=4 tn=4460 fn=12 |

## Observaciones

- El baseline **claramente aprende**: con desbalance de ~0.8%, un AUPRC de 0.806
  está muy por encima del azar (que sería ≈ la prevalencia, ~0.008).
- El umbral 0.5 es conservador (muchos falsos negativos): 0.667 de sensibilidad.
  Moviendo el umbral se llega a 0.944 de sensibilidad manteniendo 95% de
  especificidad — coherente con priorizar no perder crisis.
- **Estos números son un techo optimista** por el leakage de paciente. Sirven como
  cota superior y como confirmación de que el pipeline completo corre sobre datos
  reales.

## Próximo paso

- Descargar 3–5 pacientes más y repetir con **split por paciente / LOSO** usando
  `splits/v1.json`. Ese será el número honesto contra el cual comparar la CNN 1D.
