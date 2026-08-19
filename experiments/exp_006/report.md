# Experimento exp_006 — CNN 1D LOSO (6 pacientes: chb01–chb06)

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../LEGAL.md).


## Setup

- Pool: chb01, chb02, chb03, chb04, chb05, chb06 (LOSO, 6 folds).
- Ventana 4.0s, overlap 0.0, band-pass 0.5-40.0 Hz + notch 60.0 Hz.
- Z-score por canal (fit solo en train). Negativos submuestreados 15:1 + pos_weight.
- CNN1D (32, 64, 128) k=7, dropout 0.5, 15 epochs, lr 0.001, wd 0.0001, seed 42, cpu.

## Resultados por fold

| Test | AUPRC | ROC-AUC | sens@95spec | F1 | ictales | ventanas |
|---|---|---|---|---|---|---|
| chb01 | 0.214 | 0.9598 | 0.8393 | 0.2427 | 112 | 36496 |
| chb02 | 0.5588 | 0.9653 | 0.8864 | 0.3676 | 44 | 9138 |
| chb03 | 0.3408 | 0.901 | 0.7767 | 0.194 | 103 | 9000 |
| chb04 | 0.737 | 0.9503 | 0.8632 | 0.5393 | 95 | 20388 |
| chb05 | 0.9167 | 0.9853 | 0.9357 | 0.0927 | 140 | 7202 |
| chb06 | 0.0032 | 0.7395 | 0.2 | 0.0048 | 40 | 34103 |

## Resumen LOSO (media ± desvío entre sujetos)

| Métrica | CNN (exp_006, 6 pac.) | CNN v2 (exp_004) | Baseline (exp_002) |
|---|---|---|---|
| AUPRC | 0.462 ± 0.310 | 0.066 ± 0.056 | 0.377 ± 0.306 |
| sens@95spec | 0.750 ± 0.251 | 0.297 ± 0.143 | 0.553 ± 0.281 |
| ROC-AUC | 0.917 ± 0.083 | 0.726 ± 0.102 | 0.755 ± 0.117 |
| F1 | 0.240 ± 0.176 | — | 0.218 ± 0.218 |

> Nota: la columna exp_004/baseline se midió sobre 6 pacientes (chb01–06). Si este run usa otro pool, la comparación **no es estrictamente pareja** fold a fold.

## ⚠️ Leer con cabeza científica

- **Varianza entre sujetos alta** (AUPRC ±0.31): el promedio LOSO esconde folds muy dispares.
- **Folds que colapsan:** chb06 con AUPRC < 0.05 (sujetos sistemáticamente difíciles). No es un modelo uniformemente confiable.
- **Atribución.** Si esta config difiere de otra en más de un factor (p. ej. normalización *y* nº de pacientes), la diferencia no se puede atribuir a una sola variable: hace falta cambiar un factor por vez.

## Cómo se corrió (reproducible)

```bash
nohup .venv/bin/python -m neuropilot.training.run_loso \
    --data-root data/chb-mit --out experiments/exp_006 \
    > experiments/exp_006/train.log 2>&1 &
```

Checkpoints por fold en `experiments/exp_006/checkpoints/`. Guardado incremental en `results.json` (reanudable ante cortes).
