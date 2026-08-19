"""Tests básicos de neuropilot.data.loaders.

El parser del summary es Python puro y se testea sin dependencias.
Lo que necesita MNE/numpy se saltea automáticamente si no están instalados
(``pytest.importorskip``), para no romper la suite en entornos mínimos.
"""

from __future__ import annotations

import textwrap

import pytest

from neuropilot.data import loaders
from neuropilot.data.loaders import SeizureAnnotation


# --------------------------------------------------------------------------- #
# Fixture: un summary de CHB-MIT reducido pero fiel al formato real.
#   - chb01_01.edf : sin crisis
#   - chb01_03.edf : una crisis (formato "Seizure Start Time:")
#   - chb01_15.edf : dos crisis (formato "Seizure N Start Time:")
#   - chb01_18.edf : sin tiempos de inicio/fin (deben quedar None)
# --------------------------------------------------------------------------- #
SAMPLE_SUMMARY = textwrap.dedent(
    """\
    Data Sampling Rate: 256 Hz
    *************************

    Channels in EDF Files:
    **********************
    Channel 1: FP1-F7
    Channel 2: F7-T7
    Channel 3: T7-P7
    Channel 4: P7-O1

    File Name: chb01_01.edf
    File Start Time: 11:42:54
    File End Time: 12:42:54
    Number of Seizures in File: 0

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

    File Name: chb01_15.edf
    File Start Time: 01:44:44
    File End Time: 02:44:44
    Number of Seizures in File: 2
    Seizure 1 Start Time: 1732 seconds
    Seizure 1 End Time: 1772 seconds
    Seizure 2 Start Time: 3000 seconds
    Seizure 2 End Time: 3050 seconds

    File Name: chb01_18.edf
    Number of Seizures in File: 0
    """
)


@pytest.fixture()
def summary(tmp_path):
    path = tmp_path / "chb01-summary.txt"
    path.write_text(SAMPLE_SUMMARY, encoding="utf-8")
    return loaders.parse_summary(path)


# --------------------------------------------------------------------------- #
# Parser del summary
# --------------------------------------------------------------------------- #
def test_sampling_rate(summary):
    assert summary.sampling_rate_hz == 256.0


def test_channels_in_order_and_deduplicated(summary):
    assert summary.channels == ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]


def test_all_files_parsed(summary):
    assert set(summary.files) == {
        "chb01_01.edf",
        "chb01_03.edf",
        "chb01_15.edf",
        "chb01_18.edf",
    }


def test_file_without_seizures(summary):
    info = summary.files["chb01_01.edf"]
    assert info.n_seizures == 0
    assert info.seizures == []
    assert info.has_seizures is False
    assert info.start_time == "11:42:54"
    assert info.end_time == "12:42:54"


def test_single_seizure(summary):
    seizures = summary.seizures_for_file("chb01_03.edf")
    assert seizures == [SeizureAnnotation(start_sec=2996.0, end_sec=3036.0)]
    assert seizures[0].duration_sec == 40.0


def test_multiple_seizures(summary):
    seizures = summary.seizures_for_file("chb01_15.edf")
    assert seizures == [
        SeizureAnnotation(1732.0, 1772.0),
        SeizureAnnotation(3000.0, 3050.0),
    ]
    assert summary.files["chb01_15.edf"].n_seizures == 2


def test_missing_times_are_none(summary):
    info = summary.files["chb01_18.edf"]
    assert info.start_time is None
    assert info.end_time is None


def test_files_with_seizures_helper(summary):
    assert summary.files_with_seizures == ["chb01_03.edf", "chb01_15.edf"]


def test_seizures_for_unknown_file_is_empty(summary):
    assert summary.seizures_for_file("nope.edf") == []


# --------------------------------------------------------------------------- #
# Piezas que dependen de MNE (se saltean si no está instalado)
# --------------------------------------------------------------------------- #
def _make_raw():
    mne = pytest.importorskip("mne")
    np = pytest.importorskip("numpy")
    info = mne.create_info(ch_names=["FP1-F7", "F7-T7"], sfreq=256.0, ch_types="eeg")
    data = np.zeros((2, 256 * 5))  # 5 s de señal en cero
    return mne.io.RawArray(data, info, verbose="error")


def test_get_channels():
    raw = _make_raw()
    assert loaders.get_channels(raw) == ["FP1-F7", "F7-T7"]


def test_to_mne_annotations():
    pytest.importorskip("mne")
    seizures = [SeizureAnnotation(10.0, 25.0), SeizureAnnotation(100.0, 130.0)]
    ann = loaders.to_mne_annotations(seizures, description="seizure")
    assert list(ann.onset) == [10.0, 100.0]
    assert list(ann.duration) == [15.0, 30.0]
    assert list(ann.description) == ["seizure", "seizure"]


def test_read_edf_missing_file_raises(tmp_path):
    pytest.importorskip("mne")
    with pytest.raises(FileNotFoundError):
        loaders.read_edf(tmp_path / "no_existe.edf")
