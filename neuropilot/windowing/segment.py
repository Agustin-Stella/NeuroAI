"""Ventaneo y etiquetado de la señal EEG.

Convierte una señal **continua ya filtrada** en ventanas de tamaño fijo, cada una
etiquetada como ictal (1) o normal (0) según las crisis anotadas en el summary.
Es el paso que produce las muestras que consumirá el ``Dataset`` de PyTorch.

Reglas del proyecto codificadas acá:
  - **Overlap solo en train.** El overlap infla las métricas si se usa en eval; por
    eso es un parámetro explícito que en eval debe ser 0.
  - **Etiqueta por fracción de solapamiento.** Una ventana es ictal si la fracción
    de su duración que cae dentro de una crisis supera ``ictal_overlap``.

No filtra ni normaliza: opera sobre el array de señal (numpy). El orden del
pipeline es: filtrar (Raw) -> ventanear (acá) -> normalizar (stats de train).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neuropilot.data.loaders import SeizureAnnotation


@dataclass
class WindowedSignal:
    """Resultado del ventaneo de un registro.

    Attributes
    ----------
    windows : (N, C, W)  ventanas apiladas (N ventanas, C canales, W muestras).
    labels  : (N,)       0 = normal, 1 = ictal.
    times   : (N,)       inicio de cada ventana en segundos (relativo al registro).
    sfreq   : frecuencia de muestreo (Hz).
    window_seconds : duración de la ventana.
    """

    windows: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    sfreq: float
    window_seconds: float

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    @property
    def positive_rate(self) -> float:
        """Proporción de ventanas ictales (útil para ver el desbalance)."""
        return float(self.labels.mean()) if len(self) else 0.0


def num_windows(
    n_times: int, sfreq: float, window_seconds: float = 4.0, overlap: float = 0.0
) -> int:
    """Cantidad de ventanas completas que produce una señal de ``n_times`` muestras.

    Fuente única de verdad para el conteo (la usa también el Dataset para armar su
    índice sin cargar la señal). Coincide con la cantidad que devuelve
    :func:`segment_signal` para los mismos parámetros.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap debe estar en [0, 1); recibí {overlap}")
    win = int(round(window_seconds * sfreq))
    if win <= 0:
        raise ValueError("window_seconds * sfreq debe ser >= 1 muestra")
    if n_times < win:
        return 0
    step = max(1, int(round(win * (1.0 - overlap))))
    return (n_times - win) // step + 1


def _overlap_seconds(
    w_start: float, w_end: float, seizures: Sequence[SeizureAnnotation]
) -> float:
    """Segundos de la ventana [w_start, w_end) que caen dentro de alguna crisis."""
    total = 0.0
    for s in seizures:
        lo = max(w_start, s.start_sec)
        hi = min(w_end, s.end_sec)
        if hi > lo:
            total += hi - lo
    return total


def segment_signal(
    data: np.ndarray,
    sfreq: float,
    seizures: Sequence[SeizureAnnotation] = (),
    *,
    window_seconds: float = 4.0,
    overlap: float = 0.0,
    ictal_overlap: float = 0.5,
) -> WindowedSignal:
    """Corta ``data`` (C, T) en ventanas y las etiqueta.

    Parameters
    ----------
    data          : señal (n_channels, n_times).
    sfreq         : frecuencia de muestreo en Hz.
    seizures      : crisis del registro (segundos relativos al inicio).
    window_seconds: duración de cada ventana.
    overlap       : fracción de solapamiento entre ventanas consecutivas, en
                    ``[0, 1)``. **Usar > 0 solo en train**; en eval debe ser 0.
    ictal_overlap : fracción mínima de la ventana dentro de una crisis para
                    etiquetarla como ictal. ``0`` = cualquier solapamiento.

    Returns
    -------
    WindowedSignal con las ventanas, etiquetas y tiempos de inicio.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Esperaba data (C, T); recibí ndim={data.ndim}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap debe estar en [0, 1); recibí {overlap}")

    n_channels, n_times = data.shape
    win = int(round(window_seconds * sfreq))
    if win <= 0:
        raise ValueError("window_seconds * sfreq debe ser >= 1 muestra")
    step = max(1, int(round(win * (1.0 - overlap))))

    if n_times < win:
        empty = np.empty((0, n_channels, win), dtype=float)
        return WindowedSignal(empty, np.empty(0, dtype=int), np.empty(0, dtype=float),
                              sfreq, window_seconds)

    starts = np.arange(0, n_times - win + 1, step, dtype=int)
    windows = np.stack([data[:, s : s + win] for s in starts], axis=0)

    times = starts / sfreq
    labels = np.zeros(len(starts), dtype=int)
    if len(seizures):
        win_dur = win / sfreq
        for i, start_sec in enumerate(times):
            frac = _overlap_seconds(start_sec, start_sec + win_dur, seizures) / win_dur
            if (frac > 0 if ictal_overlap <= 0 else frac >= ictal_overlap):
                labels[i] = 1

    return WindowedSignal(windows, labels, times, sfreq, window_seconds)


def segment_raw(
    raw,
    seizures: Sequence[SeizureAnnotation] = (),
    *,
    window_seconds: float = 4.0,
    overlap: float = 0.0,
    ictal_overlap: float = 0.5,
) -> WindowedSignal:
    """Igual que :func:`segment_signal` pero tomando un ``mne.io.Raw`` filtrado.

    Extrae la señal y la frecuencia del Raw; no importa MNE (usa la API pública).
    """
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    return segment_signal(
        data,
        sfreq,
        seizures,
        window_seconds=window_seconds,
        overlap=overlap,
        ictal_overlap=ictal_overlap,
    )
