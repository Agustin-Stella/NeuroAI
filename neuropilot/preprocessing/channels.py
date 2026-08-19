"""Harmonización de canales entre pacientes.

En CHB-MIT distintos pacientes (y hasta distintos archivos del mismo paciente)
traen distinta cantidad de canales (23 vs 24). Para entrenar un modelo entre
pacientes hace falta un **montaje canónico**: el mismo conjunto de canales, en el
mismo orden, en todos lados. Descubierto en exp_002.

Este módulo no filtra ni normaliza: solo selecciona/ordena canales.
"""

from __future__ import annotations

from typing import Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import mne


def common_channels(channel_lists: Iterable[Sequence[str]]) -> list[str]:
    """Intersección de canales presente en todas las listas, ordenada.

    Parameters
    ----------
    channel_lists : iterable de listas de nombres de canal (ej. ``raw.ch_names``
        de cada archivo).
    """
    common: set[str] | None = None
    for names in channel_lists:
        s = set(names)
        common = s if common is None else (common & s)
    return sorted(common) if common else []


def pick_canonical(
    raw: "mne.io.BaseRaw", channels: Sequence[str], *, copy: bool = True
) -> "mne.io.BaseRaw":
    """Selecciona y reordena ``raw`` para quedarse con ``channels`` en ese orden.

    Lanza ``ValueError`` si al Raw le falta alguno de los canales pedidos.
    """
    raw = raw.copy() if copy else raw
    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Al Raw le faltan canales del montaje canónico: {missing}")
    raw.pick(list(channels))
    raw.reorder_channels(list(channels))
    return raw
