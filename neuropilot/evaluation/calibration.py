"""Calibración del punto de operación por paciente — sin leakage, deployable.

Problema (diagnosticado en el error analysis de exp_005): el modelo rankea bien
(ROC-AUC ~0.91) pero los scores NO son comparables entre sujetos — el umbral que
da 95% de especificidad va de ~0.01 a 1.0 según el paciente. Un umbral global fijo
pierde ~19 puntos de sensibilidad.

En LOSO el paciente de test es desconocido: NO se puede elegir el umbral con sus
etiquetas de crisis (sería leakage). Solución realista y sin etiquetas:

  **Umbral por percentil sobre los scores del propio paciente.** Como las ventanas
  ictales son ~0.1–0.5 % del total, el percentil ``target_specificity`` de TODOS los
  scores del paciente aproxima el umbral que da esa especificidad. Es *label-free*:
  se calibra sobre la grabación del sujeto sin saber dónde están las crisis, tal como
  se haría en despliegue (calibrar sobre un tramo basal del paciente).

Este módulo no entrena ni infiere: opera sobre arrays de scores ya calculados.
"""
from __future__ import annotations

import numpy as np


def percentile_threshold(scores: np.ndarray, target_specificity: float = 0.95) -> float:
    """Umbral **label-free** = cuantil ``target_specificity`` de los scores del paciente.

    Fundamento: con desbalance extremo (positivos ~1 %), el cuantil q de todos los
    scores ≈ el cuantil q de los negativos, que es exactamente el umbral cuya
    especificidad es q. No usa etiquetas de crisis → sin leakage y deployable.

    Parameters
    ----------
    scores : probabilidades de crisis del paciente (1D).
    target_specificity : especificidad objetivo en [0, 1] (ej. 0.95).
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return float("inf")
    if not 0.0 <= target_specificity <= 1.0:
        raise ValueError("target_specificity debe estar en [0, 1]")
    return float(np.quantile(s, target_specificity))


def temporal_holdout_threshold(
    scores: np.ndarray, cal_frac: float = 0.5, target_specificity: float = 0.95
) -> float:
    """Variante fiel a despliegue: calibra el percentil sobre el **tramo inicial**.

    Usa el primer ``cal_frac`` de la grabación (en orden temporal) como segmento de
    calibración basal y devuelve el umbral por percentil de ESE tramo. Evita mirar
    las ventanas que después se evaluarán. Sigue siendo label-free.

    Nota: asume que ``scores`` viene en orden temporal (así lo produce el cache, que
    materializa archivo por archivo en orden).
    """
    s = np.asarray(scores, dtype=float)
    n_cal = max(1, int(round(len(s) * cal_frac)))
    return percentile_threshold(s[:n_cal], target_specificity)
