"""Test de VIABILIDAD por evento: ¿cuántas crisis detecta y a cuántas FA/hora?

Reformula las métricas a nivel clínico (evento), no de ventana. Usa los checkpoints
de exp_005 + calibración por percentil label-free + suavizado anti-falsas-alarmas.
NO depende de descargas ni de exp_007.

Salida: consola + figura ``viability/curva_viabilidad.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuropilot.evaluation.calibration import percentile_threshold
from neuropilot.evaluation.events import event_metrics
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.trainer import predict_proba

EXP = Path("experiments/exp_005")
CACHE = EXP / "cache"
OUT = EXP / "viability"; OUT.mkdir(exist_ok=True)
PATIENTS = [f"chb{i:02d}" for i in range(1, 10)]
WIN_S = 4.0


class ArrayDS(torch.utils.data.Dataset):
    def __init__(self, w, norm):
        self.w = w
        self.mean = norm.mean_.astype(np.float32).reshape(-1, 1)
        self.std = (norm.std_.astype(np.float32) + norm.eps).reshape(-1, 1)
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        x = np.asarray(self.w[i], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std), 0


def scores_for(pid):
    ck = torch.load(EXP / "checkpoints" / f"{pid}.pt", weights_only=False)
    cfg = ck["config"]; n_ch = ck["model_state"]["features.0.weight"].shape[1]
    m = CNN1D(n_channels=n_ch, n_filters=tuple(cfg["n_filters"]),
              kernel_size=cfg["kernel_size"], pool=cfg["pool"], dropout=cfg["dropout"])
    m.load_state_dict(ck["model_state"])
    norm = ChannelNormalizer.from_dict(ck["normalizer"])
    w = np.load(CACHE / f"{pid}_windows.npy", mmap_mode="r")
    y = np.load(CACHE / f"{pid}_labels.npy").astype(int)
    return predict_proba(m, ArrayDS(w, norm), batch_size=256), y


def main():
    data = {pid: scores_for(pid) for pid in PATIENTS}

    # --- punto de operación principal: percentil 95, exigir 2 ventanas seguidas --- #
    TARGET_SPEC, MIN_CONS, MAX_GAP = 0.95, 2, 1
    print("=" * 82)
    print(f"VIABILIDAD por evento | calibración percentil {TARGET_SPEC:.0%}, "
          f"min {MIN_CONS} ventanas seguidas, gap {MAX_GAP}")
    print("=" * 82)
    print(f"{'pac':6} {'crisis':>7} {'detect.':>8} {'sens_ev':>8} {'FA/h':>7} {'horas':>7}")
    print("-" * 82)
    tot_true = tot_det = 0
    fah_list, rows = [], []
    for pid in PATIENTS:
        s, y = data[pid]
        thr = percentile_threshold(s, TARGET_SPEC)
        em = event_metrics(y, s, threshold=thr, window_seconds=WIN_S,
                           min_consecutive=MIN_CONS, max_gap=MAX_GAP)
        tot_true += em.n_true_events; tot_det += em.n_detected_true
        fah_list.append(em.false_alarms_per_hour); rows.append((pid, em))
        print(f"{pid:6} {em.n_true_events:7d} {em.n_detected_true:8d} "
              f"{em.event_sensitivity:8.2f} {em.false_alarms_per_hour:7.2f} {em.hours:7.1f}")
    print("-" * 82)
    overall_sens = tot_det / tot_true
    print(f"\n>>> DETECTA {tot_det}/{tot_true} crisis ({overall_sens:.0%}) "
          f"con MEDIANA {np.median(fah_list):.1f} falsas alarmas/hora "
          f"(media {np.mean(fah_list):.1f})")

    # --- curva de viabilidad: barrer el umbral (percentil) --- #
    print("\nCurva sensibilidad-vs-FA/h (barriendo el percentil de calibración):")
    print(f"{'percentil':>10} {'min_cons':>9} {'sens_evento':>12} {'FA/h (mediana)':>15}")
    sweep = []
    for spec in (0.90, 0.95, 0.98, 0.99):
        for mc in (1, 2, 3):
            td = tt = 0; fahs = []
            for pid in PATIENTS:
                s, y = data[pid]
                thr = percentile_threshold(s, spec)
                em = event_metrics(y, s, threshold=thr, window_seconds=WIN_S,
                                   min_consecutive=mc, max_gap=1)
                td += em.n_detected_true; tt += em.n_true_events; fahs.append(em.false_alarms_per_hour)
            sens = td / tt; medfah = float(np.median(fahs))
            sweep.append((spec, mc, sens, medfah))
            if mc == 2:
                print(f"{spec:10.2f} {mc:9d} {sens:12.0%} {medfah:15.2f}")

    # --- figura --- #
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    for mc, c in ((1, "#E45756"), (2, "#4C78A8"), (3, "#54A24B")):
        pts = sorted([(fah, sens) for spec, m, sens, fah in sweep if m == mc])
        axL.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c,
                 label=f"min {mc} ventana(s)")
    axL.set_xlabel("falsas alarmas por hora (mediana)")
    axL.set_ylabel("sensibilidad por evento")
    axL.set_title("Curva de viabilidad: sensibilidad vs falsas alarmas/hora")
    axL.grid(alpha=0.3); axL.legend(); axL.set_ylim(0, 1.02)

    x = np.arange(len(PATIENTS))
    axR.bar(x, [em.event_sensitivity for _, em in rows], color="#4C78A8")
    for i, (_, em) in enumerate(rows):
        axR.annotate(f"{em.false_alarms_per_hour:.1f}", (i, em.event_sensitivity),
                     ha="center", va="bottom", fontsize=8)
    axR.set_xticks(x); axR.set_xticklabels(PATIENTS, rotation=45)
    axR.set_ylabel("sensibilidad por evento"); axR.set_ylim(0, 1.08)
    axR.set_title(f"Por paciente @ percentil {TARGET_SPEC:.0%}, min {MIN_CONS} vent.\n"
                  "(número arriba de cada barra = FA/h)")
    fig.suptitle("Viabilidad exp_005 — detección de crisis a nivel de evento", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "curva_viabilidad.png", dpi=120)
    print(f"\nFigura -> {OUT/'curva_viabilidad.png'}")


if __name__ == "__main__":
    main()
