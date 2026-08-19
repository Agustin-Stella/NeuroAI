"""Filtrado de la señal EEG continua (band-pass + notch).

Los filtros se aplican sobre la señal **continua** (``mne.io.Raw``), ANTES del
ventaneo. Filtrar ventanas cortas por separado introduce artefactos de borde;
por eso este paso va sobre el registro completo y el ventaneo viene después.

Este módulo NO normaliza ni segmenta: solo filtra. La normalización (que depende
de estadísticos de train) vive en ``neuropilot.preprocessing.normalization``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import mne


def _ensure_loaded(raw: "mne.io.BaseRaw", copy: bool) -> "mne.io.BaseRaw":
    """Devuelve un Raw con datos en memoria (los filtros de MNE lo requieren)."""
    raw = raw.copy() if copy else raw
    if not raw.preload:
        raw.load_data(verbose="error")
    return raw


def bandpass(
    raw: "mne.io.BaseRaw",
    *,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    copy: bool = True,
    verbose: bool | str | None = "error",
) -> "mne.io.BaseRaw":
    """Filtro pasa-banda ``[l_freq, h_freq]`` Hz.

    Rango por defecto 0.5–40 Hz: descarta deriva de muy baja frecuencia y ruido
    de alta frecuencia, conservando las bandas fisiológicas relevantes (δ/θ/α/β).
    """
    raw = _ensure_loaded(raw, copy)
    return raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)


def notch(
    raw: "mne.io.BaseRaw",
    *,
    freqs: float | Sequence[float] = 60.0,
    copy: bool = True,
    verbose: bool | str | None = "error",
) -> "mne.io.BaseRaw":
    """Filtro notch para la línea eléctrica.

    Usar 60 Hz en América, 50 Hz en Europa. Se pueden pasar varios (ej. la
    fundamental y sus armónicos): ``freqs=[60, 120]``.
    """
    raw = _ensure_loaded(raw, copy)
    freq_list = [freqs] if isinstance(freqs, (int, float)) else list(freqs)
    return raw.notch_filter(freqs=freq_list, verbose=verbose)


def preprocess_raw(
    raw: "mne.io.BaseRaw",
    *,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: float | Sequence[float] | None = 60.0,
    copy: bool = True,
    verbose: bool | str | None = "error",
) -> "mne.io.BaseRaw":
    """Pipeline de filtrado estándar: band-pass y luego notch.

    Es el punto de entrada de conveniencia. Los parámetros deberían venir de la
    config del experimento (``configs/*.yaml``), no hardcodeados.

    Parameters
    ----------
    notch_freq : frecuencia(s) de línea; ``None`` desactiva el notch.
    """
    out = bandpass(raw, l_freq=l_freq, h_freq=h_freq, copy=copy, verbose=verbose)
    if notch_freq is not None:
        # copy=False: ya trabajamos sobre la copia devuelta por bandpass.
        out = notch(out, freqs=notch_freq, copy=False, verbose=verbose)
    return out
