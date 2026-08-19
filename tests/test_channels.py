"""Tests de neuropilot.preprocessing.channels (Python puro para common_channels)."""

from __future__ import annotations

import pytest

from neuropilot.preprocessing import channels


def test_common_channels_intersection():
    lists = [
        ["A", "B", "C", "D"],
        ["B", "C", "D", "E"],
        ["C", "D", "B"],
    ]
    assert channels.common_channels(lists) == ["B", "C", "D"]


def test_common_channels_sorted():
    assert channels.common_channels([["Z", "A"], ["A", "Z", "M"]]) == ["A", "Z"]


def test_common_channels_empty_when_no_overlap():
    assert channels.common_channels([["A"], ["B"]]) == []


def test_common_channels_single_list():
    assert channels.common_channels([["C", "A", "B"]]) == ["A", "B", "C"]


# --- pick_canonical requiere MNE ---
def test_pick_canonical_selects_and_reorders():
    mne = pytest.importorskip("mne")
    import numpy as np

    info = mne.create_info(["A", "B", "C", "D"], sfreq=256.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((4, 256)), info, verbose="error")
    out = channels.pick_canonical(raw, ["C", "A"])
    assert out.ch_names == ["C", "A"]  # seleccionados y en ese orden


def test_pick_canonical_missing_raises():
    mne = pytest.importorskip("mne")
    import numpy as np

    info = mne.create_info(["A", "B"], sfreq=256.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 256)), info, verbose="error")
    with pytest.raises(ValueError):
        channels.pick_canonical(raw, ["A", "Z"])
