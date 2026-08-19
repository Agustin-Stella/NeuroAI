"""Carga de datos crudos EEG — CHB-MIT.

Responsabilidad ÚNICA de este módulo: leer datos tal como están en disco y
exponerlos (señal + canales + anotaciones de crisis del summary).

NO hace preprocessing. Nada de filtrado, resampleo, normalización ni ventaneo:
eso vive en ``neuropilot.preprocessing`` y ``neuropilot.windowing``. Mantener
esta separación es lo que permite testear y razonar cada etapa por separado.

El dataset CHB-MIT organiza cada paciente en una carpeta ``chbXX/`` con:
  - archivos EDF (``chbXX_YY.edf``),
  - un archivo de resumen ``chbXX-summary.txt`` que describe canales, frecuencia
    de muestreo y, por archivo, los tiempos de inicio/fin de cada crisis.

Este módulo parsea ese summary y lee los EDF con MNE-Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # evita importar MNE si solo se usa el parser del summary
    import mne


# --------------------------------------------------------------------------- #
# Estructuras de datos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeizureAnnotation:
    """Una crisis dentro de un archivo, en segundos relativos al inicio del EDF."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class EdfFileInfo:
    """Metadata de un EDF según el summary de CHB-MIT."""

    file_name: str
    start_time: str | None  # "HH:MM:SS" tal cual (puede pasar de 24h) o None
    end_time: str | None
    n_seizures: int
    seizures: list[SeizureAnnotation] = field(default_factory=list)

    @property
    def has_seizures(self) -> bool:
        return self.n_seizures > 0


@dataclass
class SummaryInfo:
    """Resumen completo de un paciente CHB-MIT."""

    sampling_rate_hz: float | None
    channels: list[str]
    files: dict[str, EdfFileInfo]  # keyed por file_name (ej. "chb01_03.edf")

    def seizures_for_file(self, file_name: str) -> list[SeizureAnnotation]:
        """Crisis de un EDF. Devuelve lista vacía si no hay o el archivo no está."""
        info = self.files.get(file_name)
        return list(info.seizures) if info else []

    @property
    def files_with_seizures(self) -> list[str]:
        return [name for name, info in self.files.items() if info.has_seizures]


# --------------------------------------------------------------------------- #
# Parser del summary (Python puro, sin dependencias)
# --------------------------------------------------------------------------- #
_RE_SAMPLING = re.compile(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", re.IGNORECASE)
_RE_CHANNEL = re.compile(r"^\s*Channel\s+\d+:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_FILE_NAME = re.compile(r"File Name:\s*(\S+)", re.IGNORECASE)
_RE_FILE_START = re.compile(r"File Start Time:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_FILE_END = re.compile(r"File End Time:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_N_SEIZURES = re.compile(r"Number of Seizures in File:\s*(\d+)", re.IGNORECASE)
# Soporta tanto "Seizure Start Time:" (un evento) como "Seizure 1 Start Time:".
_RE_SEIZ_START = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)\s*seconds", re.IGNORECASE)
_RE_SEIZ_END = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)\s*seconds", re.IGNORECASE)


def parse_summary(summary_path: str | Path) -> SummaryInfo:
    """Parsea un archivo ``chbXX-summary.txt`` de CHB-MIT.

    Parameters
    ----------
    summary_path : ruta al archivo de resumen.

    Returns
    -------
    SummaryInfo con frecuencia de muestreo, canales (en orden, deduplicados) y
    un dict de EdfFileInfo por nombre de archivo.
    """
    text = Path(summary_path).read_text(encoding="utf-8", errors="replace")

    # Frecuencia de muestreo (puede no estar presente).
    sr_match = _RE_SAMPLING.search(text)
    sampling_rate = float(sr_match.group(1)) if sr_match else None

    # Canales: aparecen una o varias veces (cambios de montaje). Dedup en orden.
    channels: list[str] = []
    seen: set[str] = set()
    for name in _RE_CHANNEL.findall(text):
        if name not in seen:
            seen.add(name)
            channels.append(name)

    # Bloques por archivo: cada uno arranca en "File Name:".
    files: dict[str, EdfFileInfo] = {}
    blocks = re.split(r"(?=File Name:)", text)
    for block in blocks:
        name_match = _RE_FILE_NAME.search(block)
        if not name_match:
            continue  # el primer bloque (header/canales) no tiene File Name
        file_name = name_match.group(1)

        start_match = _RE_FILE_START.search(block)
        end_match = _RE_FILE_END.search(block)
        n_match = _RE_N_SEIZURES.search(block)
        n_seizures = int(n_match.group(1)) if n_match else 0

        starts = [float(s) for s in _RE_SEIZ_START.findall(block)]
        ends = [float(e) for e in _RE_SEIZ_END.findall(block)]
        seizures = [
            SeizureAnnotation(start_sec=s, end_sec=e) for s, e in zip(starts, ends)
        ]

        files[file_name] = EdfFileInfo(
            file_name=file_name,
            start_time=start_match.group(1) if start_match else None,
            end_time=end_match.group(1) if end_match else None,
            n_seizures=n_seizures,
            seizures=seizures,
        )

    return SummaryInfo(
        sampling_rate_hz=sampling_rate, channels=channels, files=files
    )


# --------------------------------------------------------------------------- #
# Lectura de EDF con MNE (sin preprocessing)
# --------------------------------------------------------------------------- #
def read_edf(
    edf_path: str | Path,
    *,
    preload: bool = False,
    verbose: bool | str | None = "error",
) -> "mne.io.BaseRaw":
    """Lee un archivo EDF con MNE. NO aplica ningún preprocessing.

    Parameters
    ----------
    edf_path : ruta al ``.edf``.
    preload  : si True, carga la señal a memoria (necesario para graficar/segmentar).
    verbose  : nivel de verbosidad de MNE; por defecto silencia todo salvo errores.

    Returns
    -------
    mne.io.BaseRaw con la señal cruda.
    """
    import mne  #  el parser del summary no necesita MNE

    path = Path(edf_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el EDF: {path}")
    return mne.io.read_raw_edf(path, preload=preload, verbose=verbose)


def get_channels(raw: "mne.io.BaseRaw") -> list[str]:
    """Devuelve la lista de nombres de canales de un Raw de MNE."""
    return list(raw.ch_names)


def to_mne_annotations(
    seizures: list[SeizureAnnotation], *, description: str = "seizure"
) -> "mne.Annotations":
    """Convierte crisis (segundos relativos al inicio del EDF) a mne.Annotations.

    Útil para superponer las crisis sobre la señal (``raw.set_annotations(...)``)
    en el notebook de exploración.
    """
    import mne

    onsets = [s.start_sec for s in seizures]
    durations = [s.duration_sec for s in seizures]
    descriptions = [description] * len(seizures)
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)
