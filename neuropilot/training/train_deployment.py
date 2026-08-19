"""Entrena el **modelo de despliegue**: uno solo, con TODOS los pacientes (sin LOSO).

A diferencia de ``run_loso`` (que entrena N modelos para *evaluar* generalización),
esto produce **un único modelo aplicable a un EEG nuevo**. La generalización a un
paciente no visto ya la estimó el LOSO (viability ~81% de crisis, 8/9 pacientes al
100%); acá exprimimos todos los datos para el modelo que se despliega.

El checkpoint es **autocontenido**: incluye el montaje canónico, los params de
preprocesamiento/ventaneo y los defaults de calibración, para que el detector no
dependa de ningún otro archivo.

Uso:
    .venv/bin/python -m neuropilot.training.train_deployment \
        --cache-dir experiments/exp_005/cache --out models/deployment_v1.pt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from neuropilot.models.cnn1d import CNN1D
from neuropilot.training.run_loso import (
    Config, WindowArrayDataset, build_train_windows, fit_normalizer,
)
from neuropilot.training.trainer import set_seed, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena el modelo de despliegue (todos los pacientes).")
    ap.add_argument("--data-root", default="data/chb-mit")
    ap.add_argument("--cache-dir", default="experiments/exp_005/cache")
    ap.add_argument("--channels-from", default="experiments/exp_005/results.json",
                    help="results.json de donde leer el montaje canónico.")
    ap.add_argument("--patients", nargs="+",
                    default=[f"chb{i:02d}" for i in range(1, 12)])
    ap.add_argument("--out", default="models/deployment_v1.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--neg-per-pos", type=int, default=15)
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    channels = json.loads(Path(args.channels_from).read_text())["canonical_channels"]

    cfg = Config(
        data_root=args.data_root, out=str(out_path), cache_dir=args.cache_dir,
        patients=list(args.patients), epochs=args.epochs, neg_per_pos=args.neg_per_pos,
    )
    print("=" * 70, flush=True)
    print(f"MODELO DE DESPLIEGUE | pool={cfg.patients} | canales={len(channels)}", flush=True)
    print("=" * 70, flush=True)

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # train = TODOS los pacientes (todos los positivos + negativos submuestreados)
    train_w, train_y = build_train_windows(cfg, cfg.patients, rng)
    normalizer = fit_normalizer(train_w)
    n_pos = int((train_y == 1).sum()); n_neg = int((train_y == 0).sum())
    pos_weight = (n_neg / n_pos) if n_pos else 1.0
    print(f"Train: {len(train_y)} ventanas ({n_pos} pos / {n_neg} neg), "
          f"pos_weight={pos_weight:.1f}", flush=True)

    model = CNN1D(n_channels=train_w.shape[1], n_filters=cfg.n_filters,
                  kernel_size=cfg.kernel_size, pool=cfg.pool, dropout=cfg.dropout)
    train_model(model, WindowArrayDataset(train_w, train_y, normalizer),
                epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                weight_decay=cfg.weight_decay, pos_weight=pos_weight,
                device=cfg.device, seed=cfg.seed, verbose=True)

    # checkpoint AUTOCONTENIDO
    payload = {
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "config": asdict(cfg),
        "canonical_channels": channels,
        "preprocess": {"l_freq": cfg.l_freq, "h_freq": cfg.h_freq, "notch_freq": cfg.notch_freq},
        "window_seconds": cfg.window_seconds,
        "calibration": {"target_specificity": 0.99, "min_consecutive": 2, "max_gap": 1},
        "pool": cfg.patients,
        "n_train_windows": int(len(train_y)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "kind": "deployment",
    }
    torch.save(payload, out_path)
    print(f"\nModelo de despliegue guardado -> {out_path}", flush=True)
    print(f"Autocontenido: {len(channels)} canales + calibración + preprocess incluidos.", flush=True)


if __name__ == "__main__":
    main()
