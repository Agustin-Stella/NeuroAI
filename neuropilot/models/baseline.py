"""Baseline honesto: features de banda + regresión logística.

Es la **línea de honestidad** del proyecto: un modelo clásico, interpretable y
barato de entrenar. Si el deep learning no le gana de forma medible a esto, algo
anda mal (en los datos, en la evaluación o en el modelo). Define el número contra
el cual se compara todo lo demás.

Features: potencia por banda fisiológica (δ/θ/α/β/γ) por canal, estimada con Welch
(PSD) e integrada en cada banda. Para una ventana de C canales salen ``C * n_bandas``
features. Se aplica ``log1p`` para comprimir el rango dinámico.

Modelo: ``StandardScaler`` + ``LogisticRegression(class_weight="balanced")`` para
compensar el fuerte desbalance de clases.

Este módulo NO evalúa: las métricas viven en ``neuropilot.evaluation.metrics``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

# Bandas EEG estándar (Hz). Deberían poder venir de la config del experimento.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpower_features(
    windows: np.ndarray,
    sfreq: float,
    *,
    bands: dict[str, tuple[float, float]] | None = None,
    log: bool = True,
) -> np.ndarray:
    """Extrae potencia por banda de un lote de ventanas.

    Parameters
    ----------
    windows : (N, C, W) ventanas.
    sfreq   : frecuencia de muestreo (Hz).
    bands   : mapa banda -> (fmin, fmax); por defecto :data:`DEFAULT_BANDS`.
    log     : aplica ``log1p`` a la potencia (recomendado).

    Returns
    -------
    (N, C * n_bandas) matriz de features. El orden es banda-mayor:
    ``[c0_delta, c1_delta, ..., c0_theta, ...]``.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3:
        raise ValueError(f"Esperaba windows (N, C, W); recibí ndim={windows.ndim}")
    bands = bands or DEFAULT_BANDS

    n_win, n_ch, n_samp = windows.shape
    nperseg = min(n_samp, int(sfreq))  # ~1 s de resolución si la ventana lo permite
    freqs, psd = welch(windows, fs=sfreq, nperseg=nperseg, axis=-1)  # psd: (N, C, F)

    feats = []
    for fmin, fmax in bands.values():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            power = np.zeros((n_win, n_ch))
        else:
            power = trapezoid(psd[:, :, mask], freqs[mask], axis=-1)  # (N, C)
        feats.append(power)
    features = np.concatenate(feats, axis=1)  # (N, C * n_bandas)
    return np.log1p(features) if log else features


class BandPowerBaseline:
    """Clasificador baseline: features de banda + regresión logística.

    Ejemplo
    -------
    >>> clf = BandPowerBaseline(sfreq=256).fit(train_windows, train_labels)
    >>> scores = clf.predict_proba(test_windows)   # prob. de clase ictal
    """

    def __init__(
        self,
        sfreq: float,
        *,
        bands: dict[str, tuple[float, float]] | None = None,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        # import perezoso para no exigir sklearn si solo se usan las features
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.sfreq = sfreq
        self.bands = bands or DEFAULT_BANDS
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
            ),
        )

    def features(self, windows: np.ndarray) -> np.ndarray:
        return bandpower_features(windows, self.sfreq, bands=self.bands)

    def fit(self, windows: np.ndarray, labels: np.ndarray) -> "BandPowerBaseline":
        self.model.fit(self.features(windows), np.asarray(labels).astype(int))
        return self

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Probabilidad de la clase ictal (1) para cada ventana."""
        return self.model.predict_proba(self.features(windows))[:, 1]

    def predict(self, windows: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(windows) >= threshold).astype(int)
