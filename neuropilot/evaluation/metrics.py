"""Métricas médicas para clasificación de crisis (numpy puro).

Con clases muy desbalanceadas (en chb01 medimos ~1.1% de ventanas ictales),
*accuracy* miente y AUC-ROC es optimista. Por eso el proyecto reporta:

  - **Sensibilidad / Recall** (prioritaria: el falso negativo es el costo clínico),
  - **Especificidad**, **Precision**, **F1**,
  - **AUPRC** (average precision) como agregado principal,
  - **Sensibilidad a especificidad fija** (ej. sens@95%spec) como número comparable
    entre modelos,
  - **FPR** y **FNR**.

Todo se implementa con numpy (sin sklearn) para tener control exacto, tests
deterministas y cero dependencias pesadas en la evaluación.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Métricas a un umbral fijo
# --------------------------------------------------------------------------- #
def confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> tuple[int, int, int, int]:
    """Devuelve (tp, fp, tn, fn) prediciendo positivo si ``score >= threshold``."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


@dataclass
class BinaryMetrics:
    """Conjunto de métricas de un clasificador binario."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float  # recall / TPR
    specificity: float  # TNR
    precision: float
    f1: float
    fpr: float
    fnr: float
    auprc: float  # independiente del umbral
    n_pos: int
    n_neg: int

    def as_dict(self) -> dict:
        return asdict(self)


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> BinaryMetrics:
    """Calcula todas las métricas a un umbral dado, más el AUPRC global."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, threshold)

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    return BinaryMetrics(
        threshold=threshold,
        tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        fpr=_safe_div(fp, fp + tn),
        fnr=_safe_div(fn, fn + tp),
        auprc=average_precision(y_true, y_score),
        n_pos=int(np.sum(y_true == 1)),
        n_neg=int(np.sum(y_true == 0)),
    )


# --------------------------------------------------------------------------- #
# Métricas basadas en el ranking de scores (independientes del umbral)
# --------------------------------------------------------------------------- #
def _pr_points(y_true: np.ndarray, y_score: np.ndarray):
    """Precision y recall en cada umbral distinto (scores descendentes)."""
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    # último índice de cada grupo de score igual (umbrales distintos)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp_t = tp[last].astype(float)
    fp_t = fp[last].astype(float)
    total_pos = float(tp[-1])
    precision = tp_t / np.maximum(tp_t + fp_t, 1)
    recall = tp_t / total_pos if total_pos > 0 else np.zeros_like(tp_t)
    return precision, recall, s[last]


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUPRC como *average precision*: ``sum_k (R_k - R_{k-1}) * P_k``.

    Devuelve ``nan`` si no hay positivos.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if np.sum(y_true == 1) == 0:
        return float("nan")
    precision, recall, _ = _pr_points(y_true, y_score)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_specificity: float = 0.95
) -> tuple[float, float]:
    """Máxima sensibilidad alcanzable con especificidad >= objetivo.

    Returns
    -------
    (sensitivity, threshold). El punto trivial "predecir todo negativo"
    (spec=1, sens=0) siempre es factible, así que nunca queda vacío.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    sens = tp[last] / n_pos
    spec = (n_neg - fp[last]) / n_neg
    thr = s[last]
    # punto trivial: umbral por encima del máximo -> nada positivo
    sens = np.r_[0.0, sens]
    spec = np.r_[1.0, spec]
    thr = np.r_[np.inf, thr]

    feasible = spec >= target_specificity
    masked_sens = np.where(feasible, sens, -1.0)
    best = int(np.argmax(masked_sens))
    return float(sens[best]), float(thr[best])


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC vía el estadístico de Mann-Whitney (rango medio ante empates).

    Se reporta como referencia, pero AUPRC es más informativa con desbalance.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    # rango medio para empates
    i = 0
    n = len(y_score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = len(pos)
    n_neg = len(neg)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# --------------------------------------------------------------------------- #
# Agregación LOSO (media ± desvío entre sujetos)
# --------------------------------------------------------------------------- #
def summarize_loso(fold_metrics: list[dict]) -> dict[str, dict[str, float]]:
    """Resume métricas por fold en media y desvío entre sujetos.

    Parameters
    ----------
    fold_metrics : lista de dicts (ej. ``BinaryMetrics.as_dict()`` por fold).

    Returns
    -------
    dict ``{metrica: {"mean": ..., "std": ...}}`` para las claves numéricas.
    """
    if not fold_metrics:
        return {}
    keys = [
        k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))
    ]
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        values = np.array([f[k] for f in fold_metrics], dtype=float)
        summary[k] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return summary
