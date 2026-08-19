"""Tests de neuropilot.models.cnn1d (requieren torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuropilot.models.cnn1d import CNN1D


def test_forward_output_shape():
    model = CNN1D(n_channels=23)
    x = torch.randn(4, 23, 1024)  # batch de 4 ventanas de 4s a 256Hz
    out = model(x)
    assert out.shape == (4,)  # un logit por ventana


def test_forward_different_channels():
    model = CNN1D(n_channels=18)
    out = model(torch.randn(2, 18, 512))
    assert out.shape == (2,)


def test_gradients_flow():
    model = CNN1D(n_channels=8)
    x = torch.randn(6, 8, 256)
    y = torch.randint(0, 2, (6,)).float()
    loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_predict_proba_in_unit_range():
    model = CNN1D(n_channels=4)
    p = model.predict_proba(torch.randn(5, 4, 256))
    assert p.shape == (5,)
    assert torch.all((p >= 0) & (p <= 1))


def test_even_kernel_raises():
    with pytest.raises(ValueError):
        CNN1D(kernel_size=8)


def test_bad_input_dim_raises():
    model = CNN1D(n_channels=4)
    with pytest.raises(ValueError):
        model(torch.randn(4, 256))  # falta la dimensión de canal


def test_has_parameters():
    assert CNN1D().count_parameters() > 0
