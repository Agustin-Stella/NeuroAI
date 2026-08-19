"""¿Se puede subir el 65 %? Dos palancas: más dato de adaptación y punto de operación.

Test FIJO held-out (3 archivos de chb12). Se prueba:
  - cuánto dato de adaptación (few-shot: 2, 4, 7 archivos),
  - y para el mejor, un barrido del umbral (cobertura vs falsas alarmas/hora).
Todo honesto: se reporta cobertura Y FA/h; el test nunca se toca en la adaptación.
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

DR = Path("data/chb-mit"); DEPLOY = Path("models/deployment_v1.pt")
OUT = Path("experiments/exp_008_patient_adapt"); WIN_S = 4.0

TEST_FILES = ["chb12_36.edf", "chb12_38.edf", "chb12_42.edf"]        # fijo, held-out
ADAPT_POOL = ["chb12_06.edf", "chb12_08.edf", "chb12_09.edf", "chb12_10.edf",
              "chb12_11.edf", "chb12_23.edf", "chb12_33.edf"]


def materialize(channels, fnames):
    out = {}; recs = {r.edf_path.name: r for r in records_for_patient(DR, "chb12")}
    chset = set(channels)
    for fn in fnames:
        rec = recs[fn]
        if not chset <= set(read_edf(rec.edf_path, preload=False).ch_names):
            continue
        raw = pick_canonical(read_edf(rec.edf_path, preload=True), channels, copy=False)
        raw = preprocess_raw(raw, copy=False)
        ws = segment_signal(raw.get_data(), float(raw.info["sfreq"]), rec.seizures,
                            window_seconds=WIN_S, overlap=0.0)
        out[fn] = (ws.windows.astype(np.float32), ws.labels.astype(int))
    return out


def build_model(ckpt):
    cfg = ckpt["config"]; n = ckpt["model_state"]["features.0.weight"].shape[1]
    m = CNN1D(n_channels=n, n_filters=tuple(cfg["n_filters"]), kernel_size=cfg["kernel_size"],
              pool=cfg["pool"], dropout=cfg["dropout"]); m.load_state_dict(ckpt["model_state"]); return m


def adapt(ckpt, files_data, rng):
    aw = np.concatenate([w for w, _ in files_data.values()])
    ay = np.concatenate([y for _, y in files_data.values()])
    pos = np.flatnonzero(ay == 1); neg = np.flatnonzero(ay == 0)
    sel = np.sort(np.concatenate([pos, rng.choice(neg, size=min(len(neg), 15*max(1, len(pos))), replace=False)]))
    aw, ay = aw[sel], ay[sel]
    norm = fit_normalizer(aw); pw = (ay == 0).sum() / max(1, (ay == 1).sum())
    model = build_model(ckpt)
    train_model(model, WindowArrayDataset(aw, ay, norm), epochs=10, batch_size=64,
                lr=5e-4, weight_decay=1e-4, pos_weight=pw, seed=42, verbose=False)
    return model, norm, int((ay == 1).sum())


def evaluate(model, norm, test_data, spec, mincons):
    det = tot = fa = 0; hours = 0.0
    for fn, (w, y) in test_data.items():
        s = predict_proba(model, WindowArrayDataset(w, y, norm), batch_size=256)
        preds = aggregate_events(s, percentile_threshold(s, spec), min_consecutive=mincons, max_gap=1)
        true_ev = _runs(y == 1)
        det += sum(any(t.start < p.end and p.start < t.end for p in preds) for t in true_ev)
        fa += sum(not any(p.start < t.end and t.start < p.end for t in true_ev) for p in preds)
        tot += len(true_ev); hours += len(y) * WIN_S / 3600.0
    return det, tot, fa, fa / hours if hours else float("nan")


def main():
    set_seed(42); rng = np.random.default_rng(42)
    channels = json.loads(Path("experiments/exp_005/results.json").read_text())["canonical_channels"]
    ckpt = torch.load(DEPLOY, weights_only=False)
    deploy_norm = ChannelNormalizer.from_dict(ckpt["normalizer"])

    print("Materializando chb12…", flush=True)
    test_data = materialize(channels, TEST_FILES)
    pool_data = materialize(channels, ADAPT_POOL)
    ntest = sum(len(_runs(y == 1)) for _, y in test_data.values())
    print(f"Test fijo: {TEST_FILES} → {ntest} crisis\n", flush=True)

    # zero-shot en el test fijo
    zs = evaluate(build_model(ckpt), deploy_norm, test_data, 0.99, 2)
    print(f"ZERO-SHOT (0.99/2): {zs[0]}/{zs[1]} = {zs[0]/zs[1]:.0%}  |  {zs[3]:.1f} FA/h\n")

    # (1) few-shot: cuánto dato de adaptación
    print("=== (1) Cantidad de dato de adaptación (op. estricta 0.99/2) ===")
    print(f"{'archivos':>10} {'vent. crisis':>13} {'cobertura':>12} {'FA/h':>7}")
    best = None
    for n in (2, 4, 7):
        adapt_files = {f: pool_data[f] for f in ADAPT_POOL[:n] if f in pool_data}
        model, norm, npos = adapt(ckpt, adapt_files, np.random.default_rng(42))
        det, tot, fa, fah = evaluate(model, norm, test_data, 0.99, 2)
        print(f"{n:10} {npos:13} {f'{det}/{tot} = {det/tot:.0%}':>12} {fah:7.1f}")
        if n == 7: best = (model, norm)

    # (2) punto de operación con el mejor modelo adaptado (7 archivos)
    print("\n=== (2) Punto de operación (modelo adaptado con 7 archivos) ===")
    print(f"{'percentil':>10} {'min_cons':>9} {'cobertura':>12} {'FA/h':>7}")
    sweep = []
    for spec in (0.90, 0.95, 0.98, 0.99):
        for mc in (1, 2):
            det, tot, fa, fah = evaluate(best[0], best[1], test_data, spec, mc)
            sweep.append((spec, mc, det/tot, fah))
            if mc == 1:
                print(f"{spec:10.2f} {mc:9} {f'{det}/{tot} = {det/tot:.0%}':>12} {fah:7.1f}")

    # figura
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter([zs[3]], [zs[0]/zs[1]], s=170, color="#E45756", zorder=4,
               edgecolor="white", label="zero-shot")
    for mc, c in ((1, "#0E8C86"), (2, "#4C78A8")):
        pts = sorted([(fah, cov) for spec, m, cov, fah in sweep if m == mc])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, label=f"adaptado · min {mc} vent.")
    ax.set_xlabel("falsas alarmas por hora"); ax.set_ylabel("cobertura por evento")
    ax.set_title("chb12 adaptado: cobertura vs falsas alarmas/hora")
    ax.grid(alpha=.3); ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "push.png", dpi=120)
    print(f"\nFigura -> {OUT/'push.png'}")


if __name__ == "__main__":
    main()
