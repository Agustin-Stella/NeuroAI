"""Tests de neuropilot.evaluation.metrics (numpy puro, corre siempre)."""

from __future__ import annotations

import numpy as np
import pytest

from neuropilot.evaluation import metrics


# --------------------------------------------------------------------------- #
# confusion / binary_metrics
# --------------------------------------------------------------------------- #
def test_confusion_at_threshold():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    tp, fp, tn, fn = metrics.confusion_at_threshold(y, s, threshold=0.5)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics.binary_metrics(y, s, threshold=0.5)
    assert m.sensitivity == 1.0
    assert m.specificity == 1.0
    assert m.precision == 1.0
    assert m.f1 == 1.0
    assert m.auprc == 1.0
    assert m.fpr == 0.0 and m.fnr == 0.0


def test_fpr_fnr_are_complements():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.3, 0.8, 0.2, 0.1])
    m = metrics.binary_metrics(y, s, threshold=0.5)
    assert m.fnr == pytest.approx(1 - m.sensitivity)
    assert m.fpr == pytest.approx(1 - m.specificity)


def test_counts_reported():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    m = metrics.binary_metrics(y, s)
    assert m.n_pos == 2 and m.n_neg == 3


# --------------------------------------------------------------------------- #
# average_precision (AUPRC)
# --------------------------------------------------------------------------- #
def test_average_precision_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.average_precision(y, s) == pytest.approx(1.0)


def test_average_precision_known_value():
    # Caso a mano: y=[1,0,1,0], scores desc [0.9,0.8,0.7,0.6]
    # AP = 0.5*1 + 0.5*0.6667 = 0.8333...
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert metrics.average_precision(y, s) == pytest.approx(0.8333333, abs=1e-6)


def test_average_precision_no_positives_is_nan():
    y = np.array([0, 0, 0])
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(metrics.average_precision(y, s))


# --------------------------------------------------------------------------- #
# sensitivity_at_specificity
# --------------------------------------------------------------------------- #
def test_sensitivity_at_specificity_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    sens, thr = metrics.sensitivity_at_specificity(y, s, target_specificity=0.95)
    assert sens == 1.0


def test_sensitivity_at_specificity_tradeoff():
    # Un negativo con score alto obliga a subir el umbral para spec=1.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.95, 0.85, 0.75, 0.45, 0.55, 0.30, 0.20, 0.10])
    # spec=1.0 requiere umbral > 0.55 -> sólo 3 positivos capturados -> sens=0.75
    sens, thr = metrics.sensitivity_at_specificity(y, s, target_specificity=1.0)
    assert sens == pytest.approx(0.75)


def test_sensitivity_at_specificity_degenerate_returns_nan():
    y = np.array([0, 0, 0])  # sin positivos
    s = np.array([0.1, 0.2, 0.3])
    sens, thr = metrics.sensitivity_at_specificity(y, s)
    assert np.isnan(sens)


# --------------------------------------------------------------------------- #
# roc_auc
# --------------------------------------------------------------------------- #
def test_roc_auc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.roc_auc(y, s) == pytest.approx(1.0)


def test_roc_auc_random_is_half_with_ties():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])  # todos empatados -> AUC 0.5
    assert metrics.roc_auc(y, s) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# summarize_loso
# --------------------------------------------------------------------------- #
def test_summarize_loso_mean_std():
    folds = [
        {"sensitivity": 0.8, "auprc": 0.6},
        {"sensitivity": 1.0, "auprc": 0.8},
    ]
    summary = metrics.summarize_loso(folds)
    assert summary["sensitivity"]["mean"] == pytest.approx(0.9)
    assert summary["auprc"]["mean"] == pytest.approx(0.7)
    assert summary["sensitivity"]["std"] == pytest.approx(0.1)


def test_summarize_loso_ignores_nan():
    folds = [{"auprc": 0.5}, {"auprc": float("nan")}, {"auprc": 0.7}]
    summary = metrics.summarize_loso(folds)
    assert summary["auprc"]["mean"] == pytest.approx(0.6)


def test_summarize_loso_empty():
    assert metrics.summarize_loso([]) == {}
