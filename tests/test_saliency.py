"""Tests de explicabilidad (saliencia por gradiente)."""
import numpy as np

from neuropilot.explain.saliency import explain_event, window_saliency
from neuropilot.models.cnn1d import CNN1D


def _tiny_model():
    return CNN1D(n_channels=4, n_filters=(8,), kernel_size=3, pool=2)


def test_window_saliency_shape_and_sign():
    w = np.random.randn(4, 64).astype(np.float32)
    sal = window_saliency(_tiny_model(), w)
    assert sal.shape == (4, 64)
    assert (sal >= 0).all()          # es |gradiente|


def test_explain_event_aggregates_and_normalizes():
    windows = np.random.randn(6, 4, 64).astype(np.float32)
    exp = explain_event(_tiny_model(), windows, (2, 4), ["a", "b", "c", "d"],
                        window_seconds=4.0)
    assert exp.channel_importance.shape == (4,)
    assert abs(exp.channel_importance.sum() - 1.0) < 1e-5   # normalizada a fracción
    assert exp.saliency.shape == (4, 2 * 64)                # 2 ventanas concatenadas
    assert exp.start_sec == 8.0 and exp.end_sec == 16.0
    assert len(exp.top_channels(2)) == 2
