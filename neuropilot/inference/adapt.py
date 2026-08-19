"""Adaptación al paciente: fine-tuning corto del modelo sobre dato del propio sujeto.

Motivación (exp_008): la generalización zero-shot a un paciente atípico es dura (~6 %),
pero adaptando el modelo con unas pocas crisis etiquetadas del paciente sube a ~90 %.
Es como se despliegan los detectores clínicos reales (patient-specific).

Produce un checkpoint **autocontenido** (pesos adaptados + normalizador del paciente +
montaje + preprocess + calibración sugerida) que el detector usa con ``--model``.

Uso:
    .venv/bin/python -m neuropilot.inference.adapt --patient-dir /ruta/chbNN \
        --files chbNN_01.edf chbNN_03.edf --out models/chbNN_adapted.pt
    # después:
    .venv/bin/python -m neuropilot.inference.detector otro.edf \
        --model models/chbNN_adapted.pt --min-consecutive 1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from neuropilot.data.loaders import parse_summary, read_edf
from neuropilot.datasets.eeg_dataset import EegRecord
from neuropilot.inference.detector import DEFAULT_MODEL, _build_model, _meta_from_ckpt
from neuropilot.preprocessing.channels import pick_canonical
from neuropilot.preprocessing.filters import preprocess_raw
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.run_loso import WindowArrayDataset, fit_normalizer
from neuropilot.training.trainer import set_seed, train_model
from neuropilot.windowing.segment import segment_signal


def _records(patient_dir: Path, files: list[str]) -> list[EegRecord]:
    """Registros (EDF + crisis del summary) para los archivos de adaptación pedidos."""
    pid = patient_dir.name
    summary = parse_summary(patient_dir / f"{pid}-summary.txt")
    recs = []
    for fn in files:
        info = summary.files.get(fn)
        if info is None:
            raise ValueError(f"{fn} no está en el summary de {pid}")
        recs.append(EegRecord(pid, patient_dir / fn, list(info.seizures)))
    return recs


def _materialize(records, channels, meta) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas etiquetadas de los archivos de adaptación (montaje válido requerido)."""
    chset = set(channels)
    ws_all, y_all = [], []
    for rec in records:
        if not chset <= set(read_edf(rec.edf_path, preload=False).ch_names):
            print(f"  [skip] {rec.edf_path.name}: montaje incompatible", flush=True)
            continue
        raw = pick_canonical(read_edf(rec.edf_path, preload=True), channels, copy=False)
        raw = preprocess_raw(raw, copy=False, **meta["preprocess"])
        seg = segment_signal(raw.get_data(), float(raw.info["sfreq"]), rec.seizures,
                             window_seconds=meta["window_seconds"], overlap=0.0)
        ws_all.append(seg.windows.astype(np.float32)); y_all.append(seg.labels.astype(int))
        print(f"  [{rec.edf_path.name}] {len(seg)} ventanas, {int(seg.labels.sum())} ictales", flush=True)
    if not ws_all:
        raise ValueError("Ningún archivo de adaptación con montaje válido")
    return np.concatenate(ws_all), np.concatenate(y_all)


def adapt_to_patient(
    edf_files: list[str], patient_dir: str | Path, out_path: str | Path, *,
    base_checkpoint: str | Path = DEFAULT_MODEL, epochs: int = 10, lr: float = 5e-4,
    neg_per_pos: int = 15, seed: int = 42,
) -> dict:
    """Adapta ``base_checkpoint`` al paciente usando ``edf_files`` (etiquetados vía summary)."""
    patient_dir = Path(patient_dir)
    set_seed(seed); rng = np.random.default_rng(seed)
    base = torch.load(base_checkpoint, weights_only=False)
    meta = _meta_from_ckpt(base, exp=Path("experiments/exp_005"))
    channels = meta["channels"]

    w, y = _materialize(_records(patient_dir, edf_files), channels, meta)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if len(pos) == 0:
        raise ValueError("Los archivos de adaptación no contienen crisis etiquetadas")
    keep = np.sort(np.concatenate(
        [pos, rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)]))
    w, y = w[keep], y[keep]

    normalizer = fit_normalizer(w)                       # z-score del propio paciente
    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    model = _build_model(base)                            # arranca de los pesos base
    print(f"Adaptando {patient_dir.name}: {len(y)} ventanas "
          f"({int((y == 1).sum())} crisis), pos_weight={pos_weight:.1f}", flush=True)
    train_model(model, WindowArrayDataset(w, y, normalizer), epochs=epochs, batch_size=64,
                lr=lr, weight_decay=1e-4, pos_weight=pos_weight, seed=seed, verbose=True)

    payload = {
        "model_state": model.state_dict(), "normalizer": normalizer.to_dict(),
        "config": base["config"], "canonical_channels": channels,
        "preprocess": meta["preprocess"], "window_seconds": meta["window_seconds"],
        # crisis cortas → min_consecutive=1 suele ser mejor tras adaptar
        "calibration": {"target_specificity": 0.99, "min_consecutive": 1, "max_gap": 1},
        "kind": "patient-adapted", "patient": patient_dir.name,
        "adapt_files": list(edf_files), "base": str(base_checkpoint),
        "adapted_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nModelo adaptado -> {out_path} (autocontenido, min_consecutive=1)", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapta el detector a un paciente (few-shot).")
    ap.add_argument("--patient-dir", required=True, help="Carpeta del paciente (con su -summary.txt)")
    ap.add_argument("--files", nargs="+", required=True, help="EDFs con crisis para adaptar")
    ap.add_argument("--out", required=True, help="Checkpoint adaptado de salida (.pt)")
    ap.add_argument("--base", default=str(DEFAULT_MODEL), help="Checkpoint base (default: despliegue)")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    adapt_to_patient(args.files, args.patient_dir, args.out,
                     base_checkpoint=args.base, epochs=args.epochs)


if __name__ == "__main__":
    main()
