"""Adaptación al paciente (few-shot): ¿un poco de dato de chb12 lo rescata?

chb12 quedó fuera del entrenamiento y en zero-shot el modelo detectó ~1/14 crisis.
Este experimento prueba el enfoque realista y sin leakage:

  1. Materializar los archivos de chb12 con montaje válido (23 canales).
  2. ADAPTAR: fine-tuning del modelo de despliegue sobre un subconjunto de archivos de
     chb12 + normalizador ajustado al propio paciente.
  3. TESTEAR en el RESTO de los archivos de chb12 (held-out por archivo, sin leakage).
  4. Comparar cobertura por evento: zero-shot vs adaptado.

Es cómo se despliegan los detectores clínicos reales (patient-specific).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuropilot.data.loaders import read_edf
from neuropilot.evaluation.calibration import percentile_threshold
from neuropilot.evaluation.events import _runs, aggregate_events
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.channels import pick_canonical
from neuropilot.preprocessing.filters import preprocess_raw
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.run_loso import WindowArrayDataset, fit_normalizer, records_for_patient
from neuropilot.training.trainer import predict_proba, set_seed, train_model
from neuropilot.windowing.segment import segment_signal

DR = Path("data/chb-mit")
DEPLOY = Path("models/deployment_v1.pt")
OUT = Path("experiments/exp_008_patient_adapt")
WIN_S, SFREQ = 4.0, 256.0
TARGET_SPEC, MIN_CONS, MAX_GAP = 0.99, 2, 1

# split por archivo (los primeros para adaptar, el resto para testear) — sin leakage
ADAPT_FILES = ["chb12_06.edf", "chb12_08.edf", "chb12_09.edf", "chb12_10.edf"]
TEST_FILES = ["chb12_11.edf", "chb12_23.edf", "chb12_33.edf", "chb12_36.edf",
              "chb12_38.edf", "chb12_42.edf"]


def materialize(channels, fnames):
    """Devuelve {fname: (windows f32, labels)} para los archivos pedidos (montaje válido)."""
    out = {}
    recs = {r.edf_path.name: r for r in records_for_patient(DR, "chb12")}
    chset = set(channels)
    for fn in fnames:
        rec = recs[fn]
        if not chset <= set(read_edf(rec.edf_path, preload=False).ch_names):
            print(f"  [skip] {fn}: montaje incompatible"); continue
        raw = read_edf(rec.edf_path, preload=True)
        raw = pick_canonical(raw, channels, copy=False)
        raw = preprocess_raw(raw, copy=False)
        ws = segment_signal(raw.get_data(), float(raw.info["sfreq"]), rec.seizures,
                            window_seconds=WIN_S, overlap=0.0)
        out[fn] = (ws.windows.astype(np.float32), ws.labels.astype(int))
        print(f"  [{fn}] {len(ws)} ventanas, {int(ws.labels.sum())} ictales", flush=True)
    return out


def build_model(ckpt):
    cfg = ckpt["config"]; n_ch = ckpt["model_state"]["features.0.weight"].shape[1]
    m = CNN1D(n_channels=n_ch, n_filters=tuple(cfg["n_filters"]),
              kernel_size=cfg["kernel_size"], pool=cfg["pool"], dropout=cfg["dropout"])
    m.load_state_dict(ckpt["model_state"]); return m


def coverage_on(model, normalizer, files_data):
    """Cobertura por evento (crisis detectadas / totales) sobre un set de archivos."""
    det = tot = 0
    per = {}
    for fn, (w, y) in files_data.items():
        scores = predict_proba(model, WindowArrayDataset(w, y, normalizer), batch_size=256)
        thr = percentile_threshold(scores, TARGET_SPEC)
        preds = aggregate_events(scores, thr, min_consecutive=MIN_CONS, max_gap=MAX_GAP)
        true_ev = _runs(y == 1)
        c = sum(any(t.start < p.end and p.start < t.end for p in preds) for t in true_ev)
        det += c; tot += len(true_ev); per[fn] = (c, len(true_ev))
    return det, tot, per


def main():
    set_seed(42)
    channels = json.loads(Path("experiments/exp_005/results.json").read_text())["canonical_channels"]
    ckpt = torch.load(DEPLOY, weights_only=False)
    deploy_norm = ChannelNormalizer.from_dict(ckpt["normalizer"])

    print("Materializando chb12 (adapt)…", flush=True)
    adapt_data = materialize(channels, ADAPT_FILES)
    print("Materializando chb12 (test)…", flush=True)
    test_data = materialize(channels, TEST_FILES)

    # ----- ZERO-SHOT: modelo de despliegue, sin tocar nada -----
    zs_model = build_model(ckpt)
    zs_det, zs_tot, zs_per = coverage_on(zs_model, deploy_norm, test_data)

    # ----- ADAPTADO: normalizador del paciente + fine-tuning sobre archivos de adapt -----
    aw = np.concatenate([w for w, _ in adapt_data.values()])
    ay = np.concatenate([y for _, y in adapt_data.values()])
    # undersampling de negativos (todos los positivos + 15:1)
    rng = np.random.default_rng(42)
    pos = np.flatnonzero(ay == 1); neg = np.flatnonzero(ay == 0)
    keep_neg = rng.choice(neg, size=min(len(neg), 15 * max(1, len(pos))), replace=False)
    sel = np.sort(np.concatenate([pos, keep_neg]))
    aw_s, ay_s = aw[sel], ay[sel]
    patient_norm = fit_normalizer(aw_s)                       # z-score del propio paciente
    pw = (ay_s == 0).sum() / max(1, (ay_s == 1).sum())

    adapt_model = build_model(ckpt)                            # arranca de los pesos de despliegue
    print(f"\nAdaptando: {len(ay_s)} ventanas ({int((ay_s==1).sum())} pos), pos_weight={pw:.1f}", flush=True)
    train_model(adapt_model, WindowArrayDataset(aw_s, ay_s, patient_norm),
                epochs=8, batch_size=64, lr=5e-4, weight_decay=1e-4, pos_weight=pw,
                seed=42, verbose=True)
    ad_det, ad_tot, ad_per = coverage_on(adapt_model, patient_norm, test_data)

    # ----- reporte -----
    print("\n" + "=" * 60)
    print("COBERTURA POR EVENTO en archivos de TEST de chb12 (held-out)")
    print("=" * 60)
    print(f"{'archivo':14} {'zero-shot':>12} {'adaptado':>12}")
    for fn in test_data:
        print(f"{fn:14} {f'{zs_per[fn][0]}/{zs_per[fn][1]}':>12} {f'{ad_per[fn][0]}/{ad_per[fn][1]}':>12}")
    print("-" * 60)
    print(f"{'TOTAL':14} {f'{zs_det}/{zs_tot} = {zs_det/zs_tot:.0%}':>12} "
          f"{f'{ad_det}/{ad_tot} = {ad_det/ad_tot:.0%}':>12}")
    print(f"\n>>> chb12 (paciente no visto): zero-shot {zs_det/zs_tot:.0%} -> "
          f"adaptado {ad_det/ad_tot:.0%}")

    torch.save({"model_state": adapt_model.state_dict(), "normalizer": patient_norm.to_dict(),
                "config": ckpt["config"], "adapt_files": ADAPT_FILES, "patient": "chb12"},
               OUT / "chb12_adapted.pt")
    (OUT / "results.json").write_text(json.dumps({
        "patient": "chb12", "adapt_files": ADAPT_FILES, "test_files": TEST_FILES,
        "zero_shot": {"det": zs_det, "tot": zs_tot, "per_file": zs_per},
        "adapted": {"det": ad_det, "tot": ad_tot, "per_file": ad_per},
    }, indent=2, default=str))
    print(f"\nGuardado -> {OUT}/")


if __name__ == "__main__":
    main()
