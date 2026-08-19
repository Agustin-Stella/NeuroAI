"""Bucle de entrenamiento para modelos de PyTorch (CNN 1D y siguientes).

Maneja el fuerte desbalance de clases con ``BCEWithLogitsLoss(pos_weight=...)``:
el peso de la clase positiva compensa que las ventanas ictales son ~1% del total.

Reproducibilidad: fija semillas y (opcionalmente) algoritmos deterministas. El
entrenador es agnóstico al Dataset: funciona con cualquier ``torch`` Dataset que
devuelva ``(x, y)`` con ``x`` de forma ``(C, W)`` e ``y`` escalar 0/1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Fija semillas de numpy y torch para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def pos_weight_from_labels(labels: np.ndarray) -> float:
    """``n_neg / n_pos`` — peso de la clase positiva para compensar el desbalance."""
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train_model(
    model: torch.nn.Module,
    train_ds: Dataset,
    *,
    val_ds: Dataset | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    pos_weight: float | None = None,
    device: str = "cpu",
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool = True,
    verbose: bool = True,
) -> TrainHistory:
    """Entrena ``model`` sobre ``train_ds``. Devuelve el historial de pérdidas.

    Parameters
    ----------
    pos_weight : peso de la clase positiva; si None, no se pondera. Usar
        :func:`pos_weight_from_labels` sobre las etiquetas de train.
    """
    set_seed(seed)
    device_t = torch.device(device)
    model.to(device_t)

    pw = torch.tensor([pos_weight], device=device_t) if pos_weight else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=True,  # evita un último batch de tamaño 1 (rompe BatchNorm)
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds is not None
        else None
    )

    history = TrainHistory()
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device_t).float()
            y = y.to(device_t).float()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.train_loss.append(float(np.mean(losses)))

        if val_loader is not None:
            history.val_loss.append(_eval_loss(model, val_loader, criterion, device_t))

        if verbose:
            msg = f"epoch {epoch + 1}/{epochs}  train_loss={history.train_loss[-1]:.4f}"
            if history.val_loss:
                msg += f"  val_loss={history.val_loss[-1]:.4f}"
            print(msg)

    return history


@torch.no_grad()
def _eval_loss(model, loader, criterion, device_t) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device_t).float()
        y = y.to(device_t).float()
        losses.append(criterion(model(x), y).item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Probabilidades de crisis (sigmoide) para todas las muestras del dataset."""
    device_t = torch.device(device)
    model.to(device_t).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device_t).float()
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])
