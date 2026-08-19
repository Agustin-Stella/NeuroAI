"""Tests de neuropilot.windowing.segment (numpy puro, corre siempre)."""

from __future__ import annotations

import numpy as np
import pytest

from neuropilot.data.loaders import SeizureAnnotation
from neuropilot.windowing.segment import segment_signal


def _ramp(n_channels: int, n_times: int) -> np.ndarray:
    """Señal donde data[c, i] = i, para verificar el slicing exacto."""
    row = np.arange(n_times, dtype=float)
    return np.stack([row] * n_channels)


# --------------------------------------------------------------------------- #
# Forma y conteo de ventanas
# --------------------------------------------------------------------------- #
def test_shapes_no_overlap():
    data = _ramp(2, 100)  # sfreq=10 -> 10 s
    ws = segment_signal(data, sfreq=10, window_seconds=1.0, overlap=0.0)
    assert ws.windows.shape == (10, 2, 10)  # 10 ventanas de 1 s (10 muestras)
    assert ws.labels.shape == (10,)
    assert ws.times.tolist() == [i * 1.0 for i in range(10)]


def test_window_content_matches_slice():
    data = _ramp(1, 100)
    ws = segment_signal(data, sfreq=10, window_seconds=1.0, overlap=0.0)
    assert ws.windows[0, 0].tolist() == list(range(0, 10))
    assert ws.windows[3, 0].tolist() == list(range(30, 40))


def test_overlap_increases_window_count():
    data = _ramp(1, 100)
    no_ov = segment_signal(data, sfreq=10, window_seconds=1.0, overlap=0.0)
    ov = segment_signal(data, sfreq=10, window_seconds=1.0, overlap=0.5)
    # step = 5 muestras -> starts 0,5,...,90 -> 19 ventanas
    assert len(ov) == 19
    assert len(no_ov) == 10
    assert ov.times[1] == pytest.approx(0.5)


def test_window_larger_than_signal_is_empty():
    data = _ramp(2, 5)
    ws = segment_signal(data, sfreq=10, window_seconds=1.0)  # win=10 > 5
    assert len(ws) == 0
    assert ws.windows.shape == (0, 2, 10)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        segment_signal(_ramp(1, 100), sfreq=10, overlap=1.0)


def test_invalid_ndim_raises():
    with pytest.raises(ValueError):
        segment_signal(np.zeros(100), sfreq=10)


# --------------------------------------------------------------------------- #
# Etiquetado
# --------------------------------------------------------------------------- #
def test_no_seizures_all_normal():
    ws = segment_signal(_ramp(1, 100), sfreq=10, window_seconds=1.0)
    assert ws.labels.tolist() == [0] * 10
    assert ws.positive_rate == 0.0


def test_seizure_labels_correct_windows():
    # Crisis 4-6 s. Ventanas de 1 s sin overlap: solo las que arrancan en 4 y 5 s.
    seizures = [SeizureAnnotation(4.0, 6.0)]
    ws = segment_signal(_ramp(1, 100), sfreq=10, window_seconds=1.0, seizures=seizures)
    assert ws.labels.tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]


def test_partial_overlap_below_threshold_is_normal():
    # Crisis de 0.3 s dentro de la ventana [4,5): fracción 0.3 < 0.5 -> normal.
    seizures = [SeizureAnnotation(4.0, 4.3)]
    ws = segment_signal(
        _ramp(1, 100), sfreq=10, window_seconds=1.0, seizures=seizures, ictal_overlap=0.5
    )
    assert ws.labels[4] == 0


def test_partial_overlap_above_threshold_is_ictal():
    seizures = [SeizureAnnotation(4.0, 4.3)]
    ws = segment_signal(
        _ramp(1, 100), sfreq=10, window_seconds=1.0, seizures=seizures, ictal_overlap=0.2
    )
    assert ws.labels[4] == 1


def test_ictal_overlap_zero_means_any_overlap():
    seizures = [SeizureAnnotation(4.0, 4.05)]  # 0.05 s
    ws = segment_signal(
        _ramp(1, 100), sfreq=10, window_seconds=1.0, seizures=seizures, ictal_overlap=0.0
    )
    assert ws.labels[4] == 1


def test_multiple_seizures():
    seizures = [SeizureAnnotation(1.0, 2.0), SeizureAnnotation(7.0, 9.0)]
    ws = segment_signal(_ramp(1, 100), sfreq=10, window_seconds=1.0, seizures=seizures)
    assert ws.labels.tolist() == [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]


def test_positive_rate():
    seizures = [SeizureAnnotation(0.0, 5.0)]  # primeras 5 ventanas ictales
    ws = segment_signal(_ramp(1, 100), sfreq=10, window_seconds=1.0, seizures=seizures)
    assert ws.positive_rate == pytest.approx(0.5)
