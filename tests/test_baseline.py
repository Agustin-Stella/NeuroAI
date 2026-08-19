"""Tests de neuropilot.models.baseline.

bandpower_features usa scipy (en deps). El clasificador usa sklearn (importorskip).
"""

from __future__ import annotations

import numpy as np
import pytest

from neuropilot.models.baseline import DEFAULT_BANDS, BandPowerBaseline, bandpower_features


def _make_windows(n, n_ch, sfreq, freq_hz, seconds=2.0, noise=0.1, seed=0):
    """Ventanas con un tono sinusoidal a ``freq_hz`` + ruido."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sfreq * seconds)) / sfreq
    base = np.sin(2 * np.pi * freq_hz * t)
    data = np.broadcast_to(base, (n, n_ch, len(t))).copy()
    data += rng.normal(0, noise, size=data.shape)
    return data


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_feature_shape():
    w = _make_windows(5, 3, sfreq=256, freq_hz=10)
    feats = bandpower_features(w, sfreq=256)
    assert feats.shape == (5, 3 * len(DEFAULT_BANDS))


def test_features_reflect_dominant_band():
    # Tono en banda alpha (10 Hz) -> más potencia en alpha que en gamma.
    w = _make_windows(4, 1, sfreq=256, freq_hz=10, noise=0.01)
    feats = bandpower_features(w, sfreq=256, log=False)
    band_names = list(DEFAULT_BANDS)
    alpha_col = band_names.index("alpha")  # 1 canal -> columna = índice de banda
    gamma_col = band_names.index("gamma")
    assert feats[:, alpha_col].mean() > feats[:, gamma_col].mean()


def test_features_invalid_ndim():
    with pytest.raises(ValueError):
        bandpower_features(np.zeros((10, 20)), sfreq=256)


# --------------------------------------------------------------------------- #
# Clasificador (sklearn)
# --------------------------------------------------------------------------- #
def test_baseline_separates_two_frequency_classes():
    pytest.importorskip("sklearn")
    from neuropilot.evaluation.metrics import average_precision

    sfreq = 256
    # Clase 0: actividad en 6 Hz (theta). Clase 1: actividad en 20 Hz (beta).
    neg = _make_windows(60, 2, sfreq, freq_hz=6, seed=1)
    pos = _make_windows(60, 2, sfreq, freq_hz=20, seed=2)
    X = np.concatenate([neg, pos])
    y = np.array([0] * 60 + [1] * 60)

    clf = BandPowerBaseline(sfreq=sfreq).fit(X, y)
    scores = clf.predict_proba(X)
    # El baseline debe separar clases claramente distintas (sanity check).
    assert average_precision(y, scores) > 0.95


def test_predict_proba_and_predict_shapes():
    pytest.importorskip("sklearn")
    sfreq = 256
    X = np.concatenate([
        _make_windows(20, 2, sfreq, 6, seed=1),
        _make_windows(20, 2, sfreq, 20, seed=2),
    ])
    y = np.array([0] * 20 + [1] * 20)
    clf = BandPowerBaseline(sfreq=sfreq).fit(X, y)
    proba = clf.predict_proba(X)
    pred = clf.predict(X, threshold=0.5)
    assert proba.shape == (40,)
    assert set(np.unique(pred)).issubset({0, 1})
