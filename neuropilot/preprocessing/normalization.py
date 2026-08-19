"""Normalización por canal — fit en train, transform en todo.

Invariante anti-leakage: los estadísticos (media/desvío por canal) se calculan
**solo con datos de train** y se aplican sin cambios a val y test. Calcular la
media sobre todo el dataset filtra información del test al train de forma
silenciosa y es un error clásico que infla métricas.

El patrón fit/transform hace explícito ese contrato: se hace ``fit`` una vez con
train, se guarda el normalizador, y se hace ``transform`` en val/test con los
mismos estadísticos.

Trabaja con arrays de numpy (ventanas). No filtra ni segmenta.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Eje de canal por convención: penúltimo. Soporta (C, T) y (N, C, T).
_CHANNEL_AXIS = -2


class ChannelNormalizer:
    """Z-score por canal: ``(x - media) / desvío``, media/desvío por canal.

    Shapes soportados en fit/transform:
      - ``(C, T)``     una época/registro,
      - ``(N, C, T)``  un lote de N ventanas.

    Los estadísticos son por canal (vector de tamaño C).
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None  # (C,)
        self.std_: np.ndarray | None = None  # (C,)

    # -- fit / transform ---------------------------------------------------- #
    def fit(self, X: np.ndarray) -> "ChannelNormalizer":
        """Calcula media y desvío por canal. Usar SOLO con datos de train."""
        X = np.asarray(X, dtype=float)
        if X.ndim not in (2, 3):
            raise ValueError(f"Esperaba array (C, T) o (N, C, T); recibí ndim={X.ndim}")
        n_channels = X.shape[_CHANNEL_AXIS]
        # Colapsa todo menos el eje de canal.
        flat = np.moveaxis(X, _CHANNEL_AXIS, 0).reshape(n_channels, -1)
        self.mean_ = flat.mean(axis=1)
        self.std_ = flat.std(axis=1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la normalización con los estadísticos ya ajustados."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.shape[_CHANNEL_AXIS] != self.mean_.shape[0]:
            raise ValueError(
                f"Cantidad de canales {X.shape[_CHANNEL_AXIS]} no coincide con "
                f"la del fit ({self.mean_.shape[0]})"
            )
        shape = [1] * X.ndim
        shape[_CHANNEL_AXIS] = self.mean_.shape[0]
        mean = self.mean_.reshape(shape)
        std = self.std_.reshape(shape)
        return (X - mean) / (std + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # -- persistencia ------------------------------------------------------- #
    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.std_ is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("ChannelNormalizer no está ajustado; llamá a fit() primero")

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "eps": self.eps,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChannelNormalizer":
        obj = cls(eps=data.get("eps", 1e-8))
        obj.mean_ = np.asarray(data["mean"], dtype=float)
        obj.std_ = np.asarray(data["std"], dtype=float)
        return obj

    def save(self, path: str | Path) -> Path:
        """Guarda los estadísticos en JSON para reusarlos en val/test/inferencia."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ChannelNormalizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
