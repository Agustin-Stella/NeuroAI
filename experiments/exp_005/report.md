# Experimento exp_005 — CNN 1D LOSO (9 pacientes: chb01–chb09)

> ⚠️ **Descargo:** Proyecto **educativo y de investigación**. NeuroPilot AI **no** es un dispositivo médico, no está aprobado para uso clínico y **no debe usarse para diagnóstico ni decisiones sobre pacientes reales**. Ver [LEGAL.md](../../LEGAL.md).


## Setup

- Pool: chb01, chb02, chb03, chb04, chb05, chb06, chb07, chb08, chb09 (LOSO, 9 folds).
- Ventana 4.0s, overlap 0.0, band-pass 0.5-40.0 Hz + notch 60.0 Hz.
- Z-score por canal (fit solo en train). Negativos submuestreados 15:1 + pos_weight.
- CNN1D (32, 64, 128) k=7, dropout 0.5, 15 epochs, lr 0.001, wd 0.0001, seed 42, cpu.

## Resultados por fold

| Test | AUPRC | ROC-AUC | sens@95spec | F1 | ictales | ventanas |
|---|---|---|---|---|---|---|
| chb01 | 0.5944 | 0.9903 | 0.9911 | 0.3967 | 112 | 36496 |
| chb02 | 0.4922 | 0.988 | 0.9545 | 0.2828 | 44 | 9138 |
| chb03 | 0.229 | 0.8404 | 0.6505 | 0.2909 | 103 | 9000 |
| chb04 | 0.6305 | 0.9432 | 0.8737 | 0.3319 | 95 | 20388 |
| chb05 | 0.8096 | 0.9803 | 0.8929 | 0.1797 | 140 | 7202 |
| chb06 | 0.0019 | 0.6434 | 0.1 | 0.0036 | 40 | 34103 |
| chb07 | 0.4809 | 0.9494 | 0.8902 | 0.1272 | 82 | 22535 |
| chb08 | 0.4981 | 0.9115 | 0.6364 | 0.4194 | 231 | 8100 |
| chb09 | 0.9119 | 0.9786 | 0.9565 | 0.0123 | 69 | 13311 |

## Resumen LOSO (media ± desvío entre sujetos)

| Métrica | CNN (exp_005, 9 pac.) | CNN v2 (exp_004) | Baseline (exp_002) |
|---|---|---|---|
| AUPRC | 0.517 ± 0.261 | 0.066 ± 0.056 | 0.377 ± 0.306 |
| sens@95spec | 0.772 ± 0.266 | 0.297 ± 0.143 | 0.553 ± 0.281 |
| ROC-AUC | 0.914 ± 0.106 | 0.726 ± 0.102 | 0.755 ± 0.117 |
| F1 | 0.227 ± 0.146 | — | 0.218 ± 0.218 |

> Nota: la columna exp_004/baseline se midió sobre 6 pacientes (chb01–06). Si este run usa otro pool, la comparación **no es estrictamente pareja** fold a fold.

## ⚠️ Leer con cabeza científica

- **Varianza entre sujetos alta** (AUPRC ±0.26): el promedio LOSO esconde folds muy dispares.
- **Folds que colapsan:** chb06 con AUPRC < 0.05 (sujetos sistemáticamente difíciles). No es un modelo uniformemente confiable.
- **Atribución.** Si esta config difiere de otra en más de un factor (p. ej. normalización *y* nº de pacientes), la diferencia no se puede atribuir a una sola variable: hace falta cambiar un factor por vez.

## Cómo se corrió (reproducible)

```bash
nohup .venv/bin/python -m neuropilot.training.run_loso \
    --data-root data/chb-mit --out experiments/exp_005 \
    > experiments/exp_005/train.log 2>&1 &
```

Checkpoints por fold en `experiments/exp_005/checkpoints/`. Guardado incremental en `results.json` (reanudable ante cortes).
