"""Tests de neuropilot.data.splits.

Todo es Python puro (sin MNE), así que corre siempre. El foco está en el
invariante: ninguna filtración de paciente entre train/val/test.
"""

from __future__ import annotations

import pytest

from neuropilot.data import splits
from neuropilot.data.splits import LosoFold, PatientSplit


PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]  # chb01..chb24


# --------------------------------------------------------------------------- #
# patient_id_from_filename
# --------------------------------------------------------------------------- #
def test_patient_id_from_filename():
    assert splits.patient_id_from_filename("chb01_03.edf") == "chb01"
    assert splits.patient_id_from_filename("chb24_15.edf") == "chb24"
    assert splits.patient_id_from_filename("/data/chb-mit/chb07/chb07_01.edf") == "chb07"


def test_patient_id_from_filename_invalid():
    with pytest.raises(ValueError):
        splits.patient_id_from_filename("random_file.edf")


# --------------------------------------------------------------------------- #
# make_patient_split
# --------------------------------------------------------------------------- #
def test_split_is_deterministic():
    a = splits.make_patient_split(PATIENTS, seed=42)
    b = splits.make_patient_split(PATIENTS, seed=42)
    assert a == b


def test_split_independent_of_input_order():
    a = splits.make_patient_split(PATIENTS, seed=7)
    b = splits.make_patient_split(list(reversed(PATIENTS)), seed=7)
    assert a == b


def test_different_seed_gives_different_split():
    a = splits.make_patient_split(PATIENTS, seed=1)
    b = splits.make_patient_split(PATIENTS, seed=2)
    assert a != b


def test_no_patient_leakage():
    s = splits.make_patient_split(PATIENTS, seed=42)
    assert set(s.train) & set(s.val) == set()
    assert set(s.train) & set(s.test) == set()
    assert set(s.val) & set(s.test) == set()


def test_all_patients_accounted_for():
    s = splits.make_patient_split(PATIENTS, seed=42)
    assert sorted(s.all_patients) == sorted(p.lower() for p in PATIENTS)


def test_split_sizes_respect_fractions():
    s = splits.make_patient_split(PATIENTS, val_frac=0.25, test_frac=0.25, seed=42)
    n = len(PATIENTS)
    assert len(s.test) == round(0.25 * n)
    assert len(s.val) == round(0.25 * n)
    assert len(s.train) == n - len(s.test) - len(s.val)


def test_zero_val_fraction():
    s = splits.make_patient_split(PATIENTS, val_frac=0.0, test_frac=0.2, seed=3)
    assert s.val == []
    assert len(s.test) == round(0.2 * len(PATIENTS))


def test_too_few_patients_raises():
    with pytest.raises(ValueError):
        splits.make_patient_split(["chb01", "chb02"], seed=1)


def test_fractions_too_large_raises():
    with pytest.raises(ValueError):
        splits.make_patient_split(PATIENTS, val_frac=0.6, test_frac=0.5, seed=1)


# --------------------------------------------------------------------------- #
# PatientSplit validación
# --------------------------------------------------------------------------- #
def test_patientsplit_rejects_overlap():
    with pytest.raises(ValueError, match="LEAKAGE"):
        PatientSplit(train=["chb01", "chb02"], val=["chb02"], test=["chb03"])


def test_patientsplit_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicad"):
        PatientSplit(train=["chb01", "chb01"], val=["chb02"], test=["chb03"])


def test_split_of():
    s = PatientSplit(train=["chb01"], val=["chb02"], test=["chb03"])
    assert s.split_of("chb01") == "train"
    assert s.split_of("chb02") == "val"
    assert s.split_of("CHB03") == "test"  # case-insensitive
    with pytest.raises(KeyError):
        s.split_of("chb99")


# --------------------------------------------------------------------------- #
# save / load (round-trip)
# --------------------------------------------------------------------------- #
def test_save_and_load_roundtrip(tmp_path):
    s = splits.make_patient_split(PATIENTS, seed=42)
    path = tmp_path / "v1.json"
    splits.save_split(s, path, seed=42, val_frac=0.15, test_frac=0.15)
    loaded = splits.load_split(path)
    assert loaded == s


def test_saved_json_has_metadata(tmp_path):
    import json

    s = splits.make_patient_split(PATIENTS, seed=42)
    path = tmp_path / "v1.json"
    splits.save_split(s, path, seed=42, val_frac=0.15, test_frac=0.15)
    data = json.loads(path.read_text())
    assert data["schema"] == splits.SCHEMA
    assert data["seed"] == 42
    assert data["n_patients"] == len(PATIENTS)
    assert data["counts"]["train"] == len(s.train)


def test_load_rejects_bad_schema(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "otro/v9", "train": [], "val": [], "test": []}))
    with pytest.raises(ValueError, match="Schema"):
        splits.load_split(path)


# --------------------------------------------------------------------------- #
# LOSO
# --------------------------------------------------------------------------- #
def test_loso_fold_count_equals_patients():
    folds = splits.loso_folds(PATIENTS)
    assert len(folds) == len(PATIENTS)


def test_loso_each_test_is_single_unique_patient():
    folds = splits.loso_folds(PATIENTS)
    test_patients = [f.test_patient for f in folds]
    assert sorted(test_patients) == sorted(p.lower() for p in PATIENTS)
    assert len(set(test_patients)) == len(PATIENTS)


def test_loso_train_excludes_test():
    for fold in splits.loso_folds(PATIENTS):
        assert fold.test_patient not in fold.train_patients
        assert len(fold.train_patients) == len(PATIENTS) - 1


def test_loso_needs_two_patients():
    with pytest.raises(ValueError):
        splits.loso_folds(["chb01"])
