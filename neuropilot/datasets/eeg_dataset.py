"""Dataset de PyTorch que sirve ventanas EEG etiquetadas.

Pieza integradora del pipeline. Junta:
  loaders (EDF + summary) -> preprocessing (filtros) -> windowing -> normalización

y expone ``(ventana, label)`` para entrenar, **respetando el split por paciente**.

Decisiones de diseño:
  - **Memory-bounded.** Un registro CHB-MIT es ~1 h de señal; no se carga todo a
    RAM. El índice global se arma leyendo solo el *header* de cada EDF (cantidad de
    muestras) y las ventanas se materializan por archivo con un caché LRU. Como el
    acceso durante una época suele ser contiguo por archivo, un caché chico alcanza.
  - **Inyección de dependencias.** La carga y el conteo por registro son funciones
    reemplazables (``load_fn`` / ``count_fn``), lo que permite testear sin EDFs
    reales ni MNE.
  - **Overlap solo en train.** La factory :meth:`from_split` fija ``overlap=0`` en
    val/test automáticamente.
  - **Normalización sin leakage.** El normalizador se ajusta con train (ver
    :func:`fit_channel_normalizer`) y se pasa a los datasets de val/test.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from neuropilot.data.loaders import SeizureAnnotation, parse_summary, read_edf
from neuropilot.data.splits import PatientSplit
from neuropilot.preprocessing.filters import preprocess_raw
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.windowing.segment import WindowedSignal, num_windows, segment_raw

try:  # torch es opcional en import: solo hace falta al iterar el dataset
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - torch no instalado
    _TorchDataset = object  # type: ignore


# --------------------------------------------------------------------------- #
# Registro (un archivo EDF con sus crisis)
# --------------------------------------------------------------------------- #
@dataclass
class EegRecord:
    patient_id: str
    edf_path: Path
    seizures: list[SeizureAnnotation] = field(default_factory=list)


def build_records_for_split(
    data_root: str | Path,
    split: PatientSplit,
    subset: str,
    *,
    require_edf: bool = True,
) -> list[EegRecord]:
    """Arma la lista de registros de un subset ('train'|'val'|'test') del split.

    Espera la organización estándar de CHB-MIT:
    ``data_root/<pid>/<pid>-summary.txt`` + ``data_root/<pid>/<pid>_NN.edf``.

    Parameters
    ----------
    require_edf : si True, omite entradas del summary cuyo EDF no existe en disco.
    """
    data_root = Path(data_root)
    if subset not in {"train", "val", "test"}:
        raise ValueError(f"subset inválido: {subset!r}")
    patients: list[str] = getattr(split, subset)

    records: list[EegRecord] = []
    for pid in patients:
        summary_path = data_root / pid / f"{pid}-summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"Falta el summary del paciente: {summary_path}")
        summary = parse_summary(summary_path)
        for file_name, info in summary.files.items():
            edf_path = data_root / pid / file_name
            if require_edf and not edf_path.exists():
                continue
            records.append(
                EegRecord(patient_id=pid, edf_path=edf_path, seizures=list(info.seizures))
            )
    return records


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EegWindowDataset(_TorchDataset):
    """Sirve ``(ventana, label)`` a partir de una lista de :class:`EegRecord`.

    ``ventana`` es un tensor float32 de forma ``(C, W)`` y ``label`` un entero
    (0 normal / 1 ictal).
    """

    def __init__(
        self,
        records: Sequence[EegRecord],
        *,
        window_seconds: float = 4.0,
        overlap: float = 0.0,
        ictal_overlap: float = 0.5,
        filter_kwargs: dict | None = None,
        normalizer: ChannelNormalizer | None = None,
        load_fn: Callable[[EegRecord], WindowedSignal] | None = None,
        count_fn: Callable[[EegRecord], int] | None = None,
        cache_size: int = 2,
    ) -> None:
        self.records = list(records)
        self.window_seconds = window_seconds
        self.overlap = overlap
        self.ictal_overlap = ictal_overlap
        self.filter_kwargs = filter_kwargs or {}
        self.normalizer = normalizer
        self._load_fn = load_fn or self._default_load
        self._count_fn = count_fn or self._default_count
        self._cache: "OrderedDict[int, WindowedSignal]" = OrderedDict()
        self._cache_size = max(1, cache_size)

        # Índice global: cantidad de ventanas por registro y offsets acumulados.
        self._counts = np.array([self._count_fn(r) for r in self.records], dtype=int)
        self._offsets = np.concatenate([[0], np.cumsum(self._counts)])

    # -- defaults de producción (leen del EDF con MNE) ---------------------- #
    def _default_count(self, record: EegRecord) -> int:
        raw = read_edf(record.edf_path, preload=False)
        return num_windows(
            raw.n_times, float(raw.info["sfreq"]), self.window_seconds, self.overlap
        )

    def _default_load(self, record: EegRecord) -> WindowedSignal:
        raw = read_edf(record.edf_path, preload=True)
        raw = preprocess_raw(raw, copy=False, **self.filter_kwargs)
        return segment_raw(
            raw,
            record.seizures,
            window_seconds=self.window_seconds,
            overlap=self.overlap,
            ictal_overlap=self.ictal_overlap,
        )

    # -- API de Dataset ----------------------------------------------------- #
    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def labels(self) -> np.ndarray:
        """Etiquetas de todas las ventanas (materializa todos los registros)."""
        return np.concatenate([self._windowed(i).labels for i in range(len(self.records))])

    @property
    def positive_rate(self) -> float:
        lbls = self.labels
        return float(lbls.mean()) if len(lbls) else 0.0

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        import torch  # import perezoso

        rec_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local = idx - int(self._offsets[rec_idx])
        ws = self._windowed(rec_idx)
        x = torch.from_numpy(np.ascontiguousarray(ws.windows[local])).float()
        y = int(ws.labels[local])
        return x, y

    # -- caché LRU de ventanas por registro --------------------------------- #
    def _windowed(self, rec_idx: int) -> WindowedSignal:
        cached = self._cache.get(rec_idx)
        if cached is not None:
            self._cache.move_to_end(rec_idx)
            return cached
        ws = self._load_fn(self.records[rec_idx])
        if self.normalizer is not None and len(ws):
            ws = WindowedSignal(
                windows=self.normalizer.transform(ws.windows),
                labels=ws.labels,
                times=ws.times,
                sfreq=ws.sfreq,
                window_seconds=ws.window_seconds,
            )
        self._cache[rec_idx] = ws
        self._cache.move_to_end(rec_idx)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ws

    # -- factory que respeta el split --------------------------------------- #
    @classmethod
    def from_split(
        cls,
        data_root: str | Path,
        split: PatientSplit,
        subset: str,
        *,
        window_seconds: float = 4.0,
        train_overlap: float = 0.5,
        ictal_overlap: float = 0.5,
        filter_kwargs: dict | None = None,
        normalizer: ChannelNormalizer | None = None,
        **kwargs,
    ) -> "EegWindowDataset":
        """Construye el dataset de un subset. Fija ``overlap=0`` fuera de train."""
        records = build_records_for_split(data_root, split, subset)
        overlap = train_overlap if subset == "train" else 0.0
        return cls(
            records,
            window_seconds=window_seconds,
            overlap=overlap,
            ictal_overlap=ictal_overlap,
            filter_kwargs=filter_kwargs,
            normalizer=normalizer,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Ajuste del normalizador con datos de train (sin leakage, memory-bounded)
# --------------------------------------------------------------------------- #
def fit_channel_normalizer(
    records: Sequence[EegRecord],
    *,
    window_seconds: float = 4.0,
    ictal_overlap: float = 0.5,
    filter_kwargs: dict | None = None,
    load_fn: Callable[[EegRecord], WindowedSignal] | None = None,
    eps: float = 1e-8,
) -> ChannelNormalizer:
    """Ajusta un :class:`ChannelNormalizer` acumulando estadísticos por canal.

    Recorre los registros (típicamente los de **train**) acumulando suma y suma de
    cuadrados por canal, sin cargar todo a la vez. Usa ``overlap=0`` para no pesar
    de más las ventanas solapadas.
    """
    dataset = EegWindowDataset(
        records,
        window_seconds=window_seconds,
        overlap=0.0,
        ictal_overlap=ictal_overlap,
        filter_kwargs=filter_kwargs,
        load_fn=load_fn,
        # El ajuste solo usa _windowed (load_fn), no el índice: evitamos que el
        # count_fn por defecto lea los headers de todos los EDF.
        count_fn=lambda r: 0,
    )

    total = None
    total_sq = None
    count = 0
    for i in range(len(dataset.records)):
        w = dataset._windowed(i).windows  # (N, C, W)
        if w.shape[0] == 0:
            continue
        s = w.sum(axis=(0, 2))
        sq = (w**2).sum(axis=(0, 2))
        total = s if total is None else total + s
        total_sq = sq if total_sq is None else total_sq + sq
        count += w.shape[0] * w.shape[2]

    if count == 0:
        raise ValueError("No hay ventanas para ajustar el normalizador")

    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    norm = ChannelNormalizer(eps=eps)
    norm.mean_ = mean
    norm.std_ = np.sqrt(var)
    return norm
