"""Confirmar que la receta de adaptación generaliza a OTRO paciente difícil: chb06.

Base = checkpoint LOSO de chb06 (modelo que NO vio a chb06 → zero-shot honesto).
Adaptar con 2 archivos de chb06, testear en el resto (held-out, sin leakage).
Comparar cobertura por evento zero-shot vs adaptado (percentil 0.99, min 1 ventana).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuropilot.data.loaders import read_edf
from neuropilot.evaluation.calibration import percentile_threshold
from neuropilot.evaluation.events import _runs, aggregate_events
from neuropilot.inference.adapt import adapt_to_patient
from neuropilot.inference.detector import _build_model
from neuropilot.preprocessing.channels import pick_canonical
from neuropilot.preprocessing.filters import preprocess_raw
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.run_loso import WindowArrayDataset, records_for_patient
from neuropilot.training.trainer import predict_proba, set_seed
from neuropilot.windowing.segment import segment_signal

DR = Path("data/chb-mit/chb06")
BASE = Path("experiments/exp_005/checkpoints/chb06.pt")   # LOSO: no vio a chb06
OUT = Path("experiments/exp_008_patient_adapt")
ADAPT = ["chb06_01.edf", "chb06_04.edf"]
TEST = ["chb06_09.edf", "chb06_10.edf", "chb06_13.edf", "chb06_18.edf", "chb06_24.edf"]
SPEC, MINC = 0.99, 1


def materialize(channels, fnames):
    out = {}; recs = {r.edf_path.name: r for r in records_for_patient(DR.parent, "chb06")}
    for fn in fnames:
        rec = recs[fn]
        raw = pick_canonical(read_edf(rec.edf_path, preload=True), channels, copy=False)
        raw = preprocess_raw(raw, copy=False)
        seg = segment_signal(raw.get_data(), float(raw.info["sfreq"]), rec.seizures,
                             window_seconds=4.0, overlap=0.0)
        out[fn] = (seg.windows.astype(np.float32), seg.labels.astype(int))
        print(f"  [{fn}] {len(seg)} vent, {int(seg.labels.sum())} ictales", flush=True)
    return out


def coverage(model, norm, test_data):
    det = tot = 0
    for fn, (w, y) in test_data.items():
        s = predict_proba(model, WindowArrayDataset(w, y, norm), batch_size=256)
        preds = aggregate_events(s, percentile_threshold(s, SPEC), min_consecutive=MINC, max_gap=1)
        true_ev = _runs(y == 1)
        det += sum(any(t.start < p.end and p.start < t.end for p in preds) for t in true_ev)
        tot += len(true_ev)
    return det, tot


def main():
    set_seed(42)
    channels = json.loads(Path("experiments/exp_005/results.json").read_text())["canonical_channels"]
    print("Materializando test de chb06…", flush=True)
    test_data = materialize(channels, TEST)

    base = torch.load(BASE, weights_only=False)
    zs_model = _build_model(base)
    zs_norm = ChannelNormalizer.from_dict(base["normalizer"])
    zs = coverage(zs_model, zs_norm, test_data)

    print("\nAdaptando chb06 (base = su checkpoint LOSO)…", flush=True)
    adapt_to_patient(ADAPT, DR, OUT / "chb06_adapted.pt", base_checkpoint=BASE, epochs=10)
    ad = torch.load(OUT / "chb06_adapted.pt", weights_only=False)
    ad_model = _build_model(ad)
    ad_norm = ChannelNormalizer.from_dict(ad["normalizer"])
    adp = coverage(ad_model, ad_norm, test_data)

    print("\n" + "=" * 52)
    print(f"chb06 (paciente difícil) — cobertura por evento @ {SPEC}/min{MINC}")
    print("=" * 52)
    print(f"  zero-shot : {zs[0]}/{zs[1]} = {zs[0]/zs[1]:.0%}")
    print(f"  adaptado  : {adp[0]}/{adp[1]} = {adp[0]/adp[1]:.0%}")
    print(f"\n>>> chb06: {zs[0]/zs[1]:.0%} -> {adp[0]/adp[1]:.0%}")


if __name__ == "__main__":
    main()
