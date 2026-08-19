"""Evalúa la calibración por percentil (label-free) vs umbral global vs oráculo.

Responde: ¿un umbral por paciente SIN usar etiquetas de crisis recupera la
sensibilidad del umbral oráculo (que sí usa etiquetas)? Si sí, es deployable.

Estrategias comparadas (objetivo: 95% especificidad):
  - GLOBAL   : un único umbral (percentil 95 de los scores agrupados) para todos.
  - PERCENTIL: percentil 95 de los scores del propio paciente (label-free).
  - TEMPORAL : percentil 95 del primer 50% de la grabación del paciente (label-free,
               fiel a despliegue: calibra en tramo basal, evalúa en el resto).
  - ORÁCULO  : umbral que da 95% spec usando las etiquetas del test (cota superior).

Usa solo checkpoints + cache de exp_005 (chb01–09). NO depende de descargas.
Salida: consola + figura ``calibration/calibracion_metodos.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuropilot.evaluation.calibration import percentile_threshold, temporal_holdout_threshold
from neuropilot.evaluation.metrics import confusion_at_threshold, sensitivity_at_specificity
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.trainer import predict_proba

EXP = Path("experiments/exp_005")
CACHE = EXP / "cache"
OUT = EXP / "calibration"
OUT.mkdir(exist_ok=True)
PATIENTS = [f"chb{i:02d}" for i in range(1, 10)]
TARGET_SPEC = 0.95


class ArrayDS(torch.utils.data.Dataset):
    def __init__(self, windows, norm):
        self.w = windows
        self.mean = norm.mean_.astype(np.float32).reshape(-1, 1)
        self.std = (norm.std_.astype(np.float32) + norm.eps).reshape(-1, 1)
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        x = np.asarray(self.w[i], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std), 0


def scores_for(pid):
    ckpt = torch.load(EXP / "checkpoints" / f"{pid}.pt", weights_only=False)
    cfg = ckpt["config"]
    n_ch = ckpt["model_state"]["features.0.weight"].shape[1]
    model = CNN1D(n_channels=n_ch, n_filters=tuple(cfg["n_filters"]),
                  kernel_size=cfg["kernel_size"], pool=cfg["pool"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    norm = ChannelNormalizer.from_dict(ckpt["normalizer"])
    windows = np.load(CACHE / f"{pid}_windows.npy", mmap_mode="r")
    labels = np.load(CACHE / f"{pid}_labels.npy").astype(int)
    return predict_proba(model, ArrayDS(windows, norm), batch_size=256), labels


def sens_spec(labels, scores, thr):
    tp, fp, tn, fn = confusion_at_threshold(labels, scores, thr)
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    return (tp / npos if npos else np.nan), (tn / nneg if nneg else np.nan)


def main():
    per, all_s, all_y = {}, [], []
    for pid in PATIENTS:
        s, y = scores_for(pid)
        per[pid] = (s, y); all_s.append(s); all_y.append(y)
    all_s = np.concatenate(all_s); all_y = np.concatenate(all_y)
    thr_global = percentile_threshold(all_s, TARGET_SPEC)

    print("=" * 90)
    print(f"CALIBRACIÓN — sensibilidad @ {TARGET_SPEC:.0%} especificidad objetivo | exp_005")
    print("=" * 90)
    print(f"{'pac':6} | {'GLOBAL':>13} | {'PERCENTIL(lf)':>18} | {'TEMPORAL(lf)':>16} | {'ORÁCULO':>13}")
    print(f"{'':6} | {'sens  spec':>13} | {'sens  spec':>18} | {'sens  spec':>16} | {'sens  spec':>13}")
    print("-" * 90)

    acc = {k: {"sens": [], "spec": []} for k in ("global", "pct", "temp", "oracle")}
    for pid in PATIENTS:
        s, y = per[pid]
        thr_pct = percentile_threshold(s, TARGET_SPEC)
        thr_temp = temporal_holdout_threshold(s, 0.5, TARGET_SPEC)
        _, thr_or = sensitivity_at_specificity(y, s, TARGET_SPEC)
        res = {}
        for k, thr in (("global", thr_global), ("pct", thr_pct), ("temp", thr_temp), ("oracle", thr_or)):
            se, sp = sens_spec(y, s, thr)
            res[k] = (se, sp); acc[k]["sens"].append(se); acc[k]["spec"].append(sp)
        print(f"{pid:6} | {res['global'][0]:5.2f} {res['global'][1]:5.2f}  | "
              f"{res['pct'][0]:6.2f} {res['pct'][1]:6.2f}    | "
              f"{res['temp'][0]:5.2f} {res['temp'][1]:5.2f}    | "
              f"{res['oracle'][0]:5.2f} {res['oracle'][1]:5.2f}")

    print("-" * 90)
    names = {"global": "GLOBAL (roto)", "pct": "PERCENTIL label-free",
             "temp": "TEMPORAL label-free", "oracle": "ORÁCULO (usa etiquetas)"}
    print(f"\n{'estrategia':26} {'sens media':>11} {'spec media':>11}")
    for k in ("global", "pct", "temp", "oracle"):
        se = np.nanmean(acc[k]["sens"]); sp = np.nanmean(acc[k]["spec"])
        print(f"{names[k]:26} {se:11.3f} {sp:11.3f}")
    pct_recovery = np.nanmean(acc["pct"]["sens"]) / np.nanmean(acc["oracle"]["sens"]) * 100
    print(f"\n=> PERCENTIL label-free recupera el {pct_recovery:.0f}% de la sensibilidad "
          f"del oráculo, SIN usar etiquetas de crisis (deployable).")

    # ---- figura ----------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(PATIENTS)); w = 0.2
    fig, ax = plt.subplots(figsize=(13, 5.5))
    order = [("global", "#E45756", "GLOBAL (roto)"),
             ("pct", "#4C78A8", "PERCENTIL label-free"),
             ("temp", "#72B7B2", "TEMPORAL label-free"),
             ("oracle", "#54A24B", "ORÁCULO (cota sup.)")]
    for i, (k, c, lbl) in enumerate(order):
        ax.bar(x + (i - 1.5) * w, acc[k]["sens"], w, color=c, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(PATIENTS)
    ax.set_ylabel(f"sensibilidad @ {TARGET_SPEC:.0%} spec")
    ax.set_title("Calibración por paciente: label-free recupera casi todo el oráculo, "
                 "y arregla el umbral global")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "calibracion_metodos.png", dpi=120)
    print(f"\nFigura -> {OUT/'calibracion_metodos.png'}")


if __name__ == "__main__":
    main()
