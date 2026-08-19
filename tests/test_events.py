"""Tests del motor de eventos (agrupación de ventanas + métricas por evento)."""
import numpy as np

from neuropilot.evaluation.events import Event, _runs, aggregate_events, event_metrics


def test_runs_finds_maximal_true_runs():
    assert _runs(np.array([0, 1, 1, 0, 1, 0], dtype=bool)) == [Event(1, 3), Event(4, 5)]


def test_runs_empty_and_all():
    assert _runs(np.zeros(4, dtype=bool)) == []
    assert _runs(np.ones(3, dtype=bool)) == [Event(0, 3)]


def test_aggregate_min_consecutive_filters_short_runs():
    # positivos en ventanas 1 (aislada) y 3,4 (racha de 2)
    scores = np.array([0, 1, 0, 1, 1, 0.0])
    assert aggregate_events(scores, 0.5, min_consecutive=2, max_gap=0) == [Event(3, 5)]
    # con min_consecutive=1 entran las dos rachas
    assert aggregate_events(scores, 0.5, min_consecutive=1, max_gap=0) == [Event(1, 2), Event(3, 5)]


def test_aggregate_max_gap_merges_across_hole():
    scores = np.array([1, 0, 1, 0, 0.0])  # positivos en 0 y 2, hueco de 1
    assert aggregate_events(scores, 0.5, min_consecutive=1, max_gap=1) == [Event(0, 3)]
    assert aggregate_events(scores, 0.5, min_consecutive=1, max_gap=0) == [Event(0, 1), Event(2, 3)]


def test_event_metrics_hit_and_false_alarm():
    labels = np.zeros(10, dtype=int); labels[5:7] = 1        # una crisis real (ventanas 5-6)
    scores = np.zeros(10, dtype=float)
    scores[5:7] = 0.9                                         # detecta la crisis
    scores[0:2] = 0.9                                         # + una falsa alarma (ventanas 0-1)
    m = event_metrics(labels, scores, threshold=0.5, window_seconds=4.0, min_consecutive=1)
    assert m.n_true_events == 1
    assert m.n_detected_true == 1
    assert m.event_sensitivity == 1.0
    assert m.n_false_alarms == 1
    assert m.false_alarms_per_hour > 0


def test_event_metrics_missed_seizure():
    labels = np.zeros(8, dtype=int); labels[4:6] = 1
    scores = np.zeros(8, dtype=float)                        # el modelo no dispara nunca
    m = event_metrics(labels, scores, threshold=0.5)
    assert m.n_true_events == 1 and m.n_detected_true == 0
    assert m.event_sensitivity == 0.0 and m.n_false_alarms == 0
