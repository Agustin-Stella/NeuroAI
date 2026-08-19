"""Tests de neuropilot.training.trainer (requieren torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset

from neuropilot.models.cnn1d import CNN1D
from neuropilot.training import trainer


def test_pos_weight_from_labels():
    labels = np.array([0, 0, 0, 0, 1])  # 4 neg, 1 pos
    assert trainer.pos_weight_from_labels(labels) == pytest.approx(4.0)


def test_pos_weight_no_positives():
    assert trainer.pos_weight_from_labels(np.zeros(10)) == 1.0


def _toy_dataset(n=64, n_ch=4, w=128, seed=0):
    """Dos clases separables: la clase 1 tiene mayor amplitud."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n_ch, w, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()
    x = x + y.view(-1, 1, 1) * 3.0  # desplaza la clase positiva
    return TensorDataset(x, y), y.numpy()


def test_train_reduces_loss():
    ds, labels = _toy_dataset()
    model = CNN1D(n_channels=4, n_filters=(8, 16))
    hist = trainer.train_model(
        model, ds, epochs=5, batch_size=16,
        pos_weight=trainer.pos_weight_from_labels(labels), verbose=False,
    )
    assert len(hist.train_loss) == 5
    assert hist.train_loss[-1] < hist.train_loss[0]  # aprende algo


def test_train_with_validation_records_val_loss():
    ds, labels = _toy_dataset()
    val, _ = _toy_dataset(n=32, seed=1)
    model = CNN1D(n_channels=4, n_filters=(8,))
    hist = trainer.train_model(model, ds, val_ds=val, epochs=2, batch_size=16, verbose=False)
    assert len(hist.val_loss) == 2


def test_predict_proba_shape_and_range():
    ds, labels = _toy_dataset(n=40)
    model = CNN1D(n_channels=4, n_filters=(8,))
    trainer.train_model(model, ds, epochs=1, batch_size=16, verbose=False)
    proba = trainer.predict_proba(model, ds)
    assert proba.shape == (40,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_can_separate_toy_classes():
    # Sanity end-to-end: en datos claramente separables, debe aprender bien.
    from neuropilot.evaluation.metrics import roc_auc

    ds, labels = _toy_dataset(n=128)
    model = CNN1D(n_channels=4, n_filters=(8, 16))
    trainer.train_model(
        model, ds, epochs=8, batch_size=16,
        pos_weight=trainer.pos_weight_from_labels(labels), verbose=False,
    )
    proba = trainer.predict_proba(model, ds)
    assert roc_auc(labels, proba) > 0.85
