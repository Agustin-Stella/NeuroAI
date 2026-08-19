"""Tests de neuropilot.datasets.eeg_dataset.

Estrategia: la carga/conteo por registro se inyectan con datos sintéticos, así el
grueso de la lógica (índice global, LRU, etiquetas, normalizador) se testea sin
EDFs reales ni MNE. Lo que necesita torch se saltea con importorskip.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from neuropilot.data.loaders import SeizureAnnotation
from neuropilot.data.splits import PatientSplit
from neuropilot.datasets.eeg_dataset import (
    EegRecord,
    EegWindowDataset,
    build_records_for_split,
    fit_channel_normalizer,
)
from neuropilot.windowing.segment import WindowedSignal


# --------------------------------------------------------------------------- #
# Helpers: WindowedSignal sintético y loaders inyectables
# --------------------------------------------------------------------------- #
def _fake_ws(n_windows: int, n_channels: int = 2, win: int = 4, base: float = 0.0):
    windows = np.arange(n_windows * n_channels * win, dtype=float).reshape(
        n_windows, n_channels, win
    ) + base
    labels = np.array([i % 2 for i in range(n_windows)], dtype=int)
    times = np.arange(n_windows, dtype=float)
    return WindowedSignal(windows, labels, times, sfreq=1.0, window_seconds=float(win))


def _records(n: int):
    return [EegRecord(patient_id="chb01", edf_path=Path(f"f{i}.edf")) for i in range(n)]


# --------------------------------------------------------------------------- #
# build_records_for_split (sin torch, sin MNE)
# --------------------------------------------------------------------------- #
SUMMARY = textwrap.dedent(
    """\
    Data Sampling Rate: 256 Hz
    Channel 1: FP1-F7

    File Name: {pid}_01.edf
    Number of Seizures in File: 0

    File Name: {pid}_03.edf
    Number of Seizures in File: 1
    Seizure Start Time: 100 seconds
    Seizure End Time: 130 seconds
    """
)


def _make_data_root(tmp_path, patients):
    for pid in patients:
        pdir = tmp_path / pid
        pdir.mkdir()
        (pdir / f"{pid}-summary.txt").write_text(SUMMARY.format(pid=pid))
        (pdir / f"{pid}_01.edf").write_bytes(b"")  # EDF vacío, solo para existencia
        (pdir / f"{pid}_03.edf").write_bytes(b"")
    return tmp_path


def test_build_records_for_split(tmp_path):
    split = PatientSplit(train=["chb01"], val=["chb02"], test=["chb03"])
    root = _make_data_root(tmp_path, ["chb01", "chb02", "chb03"])
    records = build_records_for_split(root, split, "train")
    assert [r.edf_path.name for r in records] == ["chb01_01.edf", "chb01_03.edf"]
    assert all(r.patient_id == "chb01" for r in records)
    # Las crisis del summary llegan al registro correcto.
    seiz = records[1].seizures
    assert seiz == [SeizureAnnotation(100.0, 130.0)]


def test_build_records_only_selected_subset(tmp_path):
    split = PatientSplit(train=["chb01"], val=["chb02"], test=["chb03"])
    root = _make_data_root(tmp_path, ["chb01", "chb02", "chb03"])
    val = build_records_for_split(root, split, "val")
    assert {r.patient_id for r in val} == {"chb02"}


def test_build_records_missing_edf_skipped(tmp_path):
    split = PatientSplit(train=["chb01"], val=["chb02"], test=["chb03"])
    root = _make_data_root(tmp_path, ["chb01", "chb02", "chb03"])
    (root / "chb01" / "chb01_03.edf").unlink()  # borro un EDF
    records = build_records_for_split(root, split, "train", require_edf=True)
    assert [r.edf_path.name for r in records] == ["chb01_01.edf"]


def test_build_records_missing_summary_raises(tmp_path):
    split = PatientSplit(train=["chb99"], val=["chb02"], test=["chb03"])
    root = _make_data_root(tmp_path, ["chb02", "chb03"])
    with pytest.raises(FileNotFoundError):
        build_records_for_split(root, split, "train")


# --------------------------------------------------------------------------- #
# Índice global (no requiere torch)
# --------------------------------------------------------------------------- #
def test_len_is_sum_of_window_counts():
    counts = {0: 3, 1: 5, 2: 0, 3: 2}
    ds = EegWindowDataset(
        _records(4),
        load_fn=lambda r: _fake_ws(counts[int(r.edf_path.name[1])]),
        count_fn=lambda r: counts[int(r.edf_path.name[1])],
    )
    assert len(ds) == 10


def test_count_fn_matches_load_fn():
    # Si count_fn miente, el índice se corrompe; acá deben coincidir.
    ds = EegWindowDataset(
        _records(2),
        load_fn=lambda r: _fake_ws(4),
        count_fn=lambda r: 4,
    )
    assert len(ds) == 8


# --------------------------------------------------------------------------- #
# __getitem__ (requiere torch)
# --------------------------------------------------------------------------- #
def test_getitem_returns_tensor_and_label():
    pytest.importorskip("torch")
    ds = EegWindowDataset(_records(1), load_fn=lambda r: _fake_ws(3), count_fn=lambda r: 3)
    x, y = ds[0]
    assert x.shape == (2, 4)
    assert x.dtype.__str__() == "torch.float32"
    assert y in (0, 1)


def test_getitem_crosses_record_boundary():
    pytest.importorskip("torch")
    # Registro 0 con 2 ventanas (base 0), registro 1 con 2 ventanas (base 1000).
    def load(r):
        return _fake_ws(2, base=0.0 if r.edf_path.name == "f0.edf" else 1000.0)

    ds = EegWindowDataset(_records(2), load_fn=load, count_fn=lambda r: 2)
    assert len(ds) == 4
    x0, _ = ds[0]
    x2, _ = ds[2]  # primera ventana del segundo registro
    assert float(x0[0, 0]) == 0.0
    assert float(x2[0, 0]) == 1000.0


def test_getitem_out_of_range_raises():
    pytest.importorskip("torch")
    ds = EegWindowDataset(_records(1), load_fn=lambda r: _fake_ws(2), count_fn=lambda r: 2)
    with pytest.raises(IndexError):
        _ = ds[5]


def test_negative_index():
    pytest.importorskip("torch")
    ds = EegWindowDataset(_records(1), load_fn=lambda r: _fake_ws(3), count_fn=lambda r: 3)
    x_last, _ = ds[-1]
    x_2, _ = ds[2]
    assert np.allclose(x_last.numpy(), x_2.numpy())


# --------------------------------------------------------------------------- #
# Caché LRU: load_fn no debe llamarse de más
# --------------------------------------------------------------------------- #
def test_cache_avoids_reloading_same_record():
    pytest.importorskip("torch")
    calls = {"n": 0}

    def load(r):
        calls["n"] += 1
        return _fake_ws(3)

    ds = EegWindowDataset(_records(1), load_fn=load, count_fn=lambda r: 3, cache_size=1)
    base = calls["n"]  # el índice ya invocó count_fn, no load_fn
    _ = ds[0]
    _ = ds[1]
    _ = ds[2]
    assert calls["n"] - base == 1  # un solo load para las 3 ventanas del registro


# --------------------------------------------------------------------------- #
# Normalizador (sin torch)
# --------------------------------------------------------------------------- #
def test_fit_channel_normalizer_matches_manual():
    windows = np.random.default_rng(0).normal(3.0, 2.0, size=(10, 2, 5))
    recs = _records(1)
    norm = fit_channel_normalizer(recs, load_fn=lambda r: WindowedSignal(
        windows, np.zeros(10, int), np.arange(10.0), 1.0, 5.0))
    # media/desvío por canal calculados sobre (N*W) muestras
    flat = np.moveaxis(windows, 1, 0).reshape(2, -1)
    assert np.allclose(norm.mean_, flat.mean(axis=1))
    assert np.allclose(norm.std_, flat.std(axis=1))


def test_normalizer_applied_in_getitem():
    pytest.importorskip("torch")
    from neuropilot.preprocessing.normalization import ChannelNormalizer

    ws = _fake_ws(4)
    norm = ChannelNormalizer().fit(ws.windows)
    ds = EegWindowDataset(
        _records(1), load_fn=lambda r: _fake_ws(4), count_fn=lambda r: 4, normalizer=norm
    )
    # Con el normalizador ajustado sobre estos mismos datos, la media global ~0.
    xs = np.stack([ds[i][0].numpy() for i in range(4)])
    assert abs(float(xs.mean())) < 1e-6


def test_positive_rate():
    ds = EegWindowDataset(_records(2), load_fn=lambda r: _fake_ws(4), count_fn=lambda r: 4)
    # _fake_ws alterna labels 0,1,0,1 -> rate 0.5
    assert ds.positive_rate == pytest.approx(0.5)
