"""Tests de neuropilot.preprocessing.

Normalización: numpy puro (corre siempre).
Filtros: requieren MNE (se saltean con importorskip si no está).
"""

from __future__ import annotations

import numpy as np
import pytest

from neuropilot.preprocessing.normalization import ChannelNormalizer


# --------------------------------------------------------------------------- #
# Normalización (numpy puro)
# --------------------------------------------------------------------------- #
def test_fit_transform_2d_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(4, 1000))  # (C, T)
    out = ChannelNormalizer().fit_transform(X)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-3)


def test_per_channel_statistics_independent():
    # Canal 0 con media alta, canal 1 con media baja: cada uno se normaliza solo.
    X = np.stack([np.full(500, 100.0), np.full(500, -100.0)]) + \
        np.random.default_rng(1).normal(0, 1, size=(2, 500))
    norm = ChannelNormalizer().fit(X)
    assert norm.mean_[0] > 50
    assert norm.mean_[1] < -50
    out = norm.transform(X)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-6)


def test_3d_batch_shape_preserved():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 3, 256))  # (N, C, T)
    out = ChannelNormalizer().fit_transform(X)
    assert out.shape == X.shape
    # media por canal sobre todo el lote ~ 0
    per_channel_mean = out.mean(axis=(0, 2))
    assert np.allclose(per_channel_mean, 0.0, atol=1e-6)


def test_fit_on_train_transform_on_test_no_leakage():
    # El test se normaliza con estadísticos de train (no los suyos propios).
    train = np.random.default_rng(3).normal(0, 1, size=(2, 1000))
    test = np.random.default_rng(4).normal(10, 5, size=(2, 1000))  # otra distribución
    norm = ChannelNormalizer().fit(train)
    out = norm.transform(test)
    # Como usa media/desvío de train, el test NO queda con media 0.
    assert not np.allclose(out.mean(axis=1), 0.0, atol=0.5)


def test_constant_channel_does_not_divide_by_zero():
    X = np.ones((2, 100))  # desvío 0
    out = ChannelNormalizer().fit_transform(X)
    assert np.all(np.isfinite(out))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ChannelNormalizer().transform(np.zeros((2, 10)))


def test_transform_channel_mismatch_raises():
    norm = ChannelNormalizer().fit(np.zeros((3, 100)))
    with pytest.raises(ValueError):
        norm.transform(np.zeros((5, 100)))


def test_invalid_ndim_raises():
    with pytest.raises(ValueError):
        ChannelNormalizer().fit(np.zeros(10))  # 1D


def test_save_load_roundtrip(tmp_path):
    X = np.random.default_rng(5).normal(2, 4, size=(3, 500))
    norm = ChannelNormalizer().fit(X)
    path = norm.save(tmp_path / "norm.json")
    loaded = ChannelNormalizer.load(path)
    assert np.allclose(loaded.mean_, norm.mean_)
    assert np.allclose(loaded.std_, norm.std_)
    # transform idéntico tras el round-trip
    assert np.allclose(loaded.transform(X), norm.transform(X))


# --------------------------------------------------------------------------- #
# Filtros (requieren MNE)
# --------------------------------------------------------------------------- #
def _make_raw(sfreq=256.0, seconds=30.0, freqs=(5.0,)):
    mne = pytest.importorskip("mne")
    t = np.arange(int(sfreq * seconds)) / sfreq
    signal = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
    data = np.stack([signal, signal])  # 2 canales iguales
    info = mne.create_info(["ch1", "ch2"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="error")


def _band_power(raw, fmin, fmax):
    """Potencia aproximada en una banda vía FFT del primer canal."""
    data = raw.get_data()[0]
    sfreq = raw.info["sfreq"]
    fft = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(len(data), d=1 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return fft[mask].sum()


def test_notch_reduces_line_frequency():
    pytest.importorskip("mne")
    from neuropilot.preprocessing import filters

    raw = _make_raw(freqs=(10.0, 60.0))  # señal + línea de 60 Hz
    before = _band_power(raw, 59, 61)
    filtered = filters.notch(raw, freqs=60.0)
    after = _band_power(filtered, 59, 61)
    assert after < 0.1 * before  # el pico de 60 Hz cae fuerte


def test_bandpass_attenuates_out_of_band():
    pytest.importorskip("mne")
    from neuropilot.preprocessing import filters

    raw = _make_raw(freqs=(5.0, 80.0))  # una en banda, otra fuera
    filtered = filters.bandpass(raw, l_freq=1.0, h_freq=40.0)
    in_band = _band_power(filtered, 3, 7)
    out_band = _band_power(filtered, 75, 85)
    assert out_band < 0.05 * in_band  # 80 Hz muy atenuada respecto a 5 Hz


def test_filters_do_not_mutate_input_when_copy():
    pytest.importorskip("mne")
    from neuropilot.preprocessing import filters

    raw = _make_raw(freqs=(5.0, 80.0))
    original = raw.get_data().copy()
    _ = filters.bandpass(raw, l_freq=1.0, h_freq=40.0, copy=True)
    assert np.allclose(raw.get_data(), original)  # el original queda intacto


def test_preprocess_raw_pipeline_runs():
    pytest.importorskip("mne")
    from neuropilot.preprocessing import filters

    raw = _make_raw(freqs=(5.0, 60.0, 80.0))
    out = filters.preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freq=60.0)
    # 5 Hz sobrevive; 60 y 80 caen
    assert _band_power(out, 3, 7) > _band_power(out, 59, 61)
    assert _band_power(out, 3, 7) > _band_power(out, 75, 85)
