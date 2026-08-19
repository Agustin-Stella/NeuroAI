"""Calibración / punto de operación por paciente (modelo exp_005, sin reentrenar).

Motivado por el error analysis: el modelo rankea razonable (ROC-AUC ~0.91) pero
está mal calibrado — el score que separa crisis de interictal es MUY distinto por
paciente. Este script cuantifica eso y contrasta:

  - **Umbral por paciente** (elegido para 95% especificidad en ese sujeto).
  - **Umbral global único** (95% especificidad sobre los scores agrupados),
    aplicado a todos — como haría un despliegue con umbral fijo.

Además reporta el resumen LOSO con **mediana** (robusta a outliers como chb06),
no solo media.

Salida: consola + figura ``calibration/umbral_por_paciente.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuropilot.evaluation.metrics import (
    average_precision, confusion_at_threshold, roc_auc, sensitivity_at_specificity,
)
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.trainer import predict_proba

EXP = Path("experiments/exp_005")
CACHE = EXP / "cache"
OUT = EXP / "calibration"
OUT.mkdir(exist_ok=True)
PATIENTS = [f"chb{i:02d}" for i in range(1, 10)]


class ArrayDS(torch.utils.data.Dataset):
    def __init__(self, windows, norm: ChannelNormalizer):
        self.w = windows
        self.mean = norm.mean_.astype(np.float32).reshape(-1, 1)
        self.std = (norm.std_.astype(np.float32) + norm.eps).reshape(-1, 1)
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        x = np.asarray(self.w[i], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std), 0


def scores_for(pid: str):
    ckpt = torch.load(EXP / "checkpoints" / f"{pid}.pt", weights_only=False)
    cfg = ckpt["config"]
    n_ch = ckpt["model_state"]["features.0.weight"].shape[1]
    model = CNN1D(n_channels=n_ch, n_filters=tuple(cfg["n_filters"]),
                  kernel_size=cfg["kernel_size"], pool=cfg["pool"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    norm = ChannelNormalizer.from_dict(ckpt["normalizer"])
    windows = np.load(CACHE / f"{pid}_windows.npy", mmap_mode="r")
    labels = np.load(CACHE / f"{pid}_labels.npy").astype(int)
    scores = predict_proba(model, ArrayDS(windows, norm), batch_size=256)
    return scores, labels


def sens_spec_at(labels, scores, thr):
    tp, fp, tn, fn = confusion_at_threshold(labels, scores, thr)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    return (tp / n_pos if n_pos else np.nan), (tn / n_neg if n_neg else np.nan)


def main():
    per = {}
    all_scores, all_labels = [], []
    for pid in PATIENTS:
        s, y = scores_for(pid)
        per[pid] = (s, y)
        all_scores.append(s); all_labels.append(y)
    all_scores = np.concatenate(all_scores); all_labels = np.concatenate(all_labels)

    # umbral global único que da 95% spec sobre los scores agrupados
    _, thr_global = sensitivity_at_specificity(all_labels, all_scores, 0.95)

    print("=" * 84)
    print("CALIBRACIÓN — umbral por paciente vs umbral global (95% spec) | exp_005")
    print("=" * 84)
    print(f"Umbral GLOBAL para 95% spec agrupado: {thr_global:.4f}\n")
    print(f"{'pac':6} {'AUPRC':>7} {'ROC':>6} | {'thr@95(propio)':>14} "
          f"{'sens':>6} | {'sens@global':>11} {'spec@global':>11}")
    print("-" * 84)

    rows = []
    for pid in PATIENTS:
        s, y = per[pid]
        auprc = average_precision(y, s); auc = roc_auc(y, s)
        sens_own, thr_own = sensitivity_at_specificity(y, s, 0.95)
        sens_g, spec_g = sens_spec_at(y, s, thr_global)
        rows.append(dict(pid=pid, auprc=auprc, auc=auc, thr_own=thr_own,
                         sens_own=sens_own, sens_g=sens_g, spec_g=spec_g))
        print(f"{pid:6} {auprc:7.3f} {auc:6.3f} | {thr_own:14.4f} {sens_own:6.2f} | "
              f"{sens_g:11.2f} {spec_g:11.3f}")

    auprcs = np.array([r["auprc"] for r in rows])
    sens_own = np.array([r["sens_own"] for r in rows])
    sens_g = np.array([r["sens_g"] for r in rows])
    thr_own = np.array([r["thr_own"] for r in rows])

    print("-" * 84)
    print(f"\n#1  RESUMEN LOSO (robusto a outliers):")
    print(f"    AUPRC   media={np.nanmean(auprcs):.3f}  MEDIANA={np.nanmedian(auprcs):.3f}  "
          f"media sin chb06={np.nanmean(auprcs[[r['pid']!='chb06' for r in rows]]):.3f}")
    print(f"    sens@95 media={np.nanmean(sens_own):.3f}  MEDIANA={np.nanmedian(sens_own):.3f}")

    print(f"\n#3  CALIBRACIÓN:")
    print(f"    Umbrales@95spec por paciente: van de {thr_own.min():.3f} a {thr_own.max():.3f} "
          f"(rango {thr_own.max()-thr_own.min():.3f}) -> NO existe umbral global bueno.")
    print(f"    sens@95 con umbral PROPIO:  media={np.nanmean(sens_own):.3f}")
    print(f"    sens@95 con umbral GLOBAL:  media={np.nanmean(sens_g):.3f}  "
          f"(y la spec real se dispara lejos de 0.95 por paciente)")

    # ---- figura ----------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(PATIENTS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(x, thr_own, color="#54A24B")
    ax1.axhline(thr_global, ls="--", color="k", label=f"umbral global {thr_global:.3f}")
    ax1.set_xticks(x); ax1.set_xticklabels(PATIENTS, rotation=45)
    ax1.set_ylabel("umbral para 95% spec"); ax1.set_title(
        "El umbral óptimo cambia por paciente\n(recorre casi todo [0,1] -> no hay uno global)")
    ax1.legend()

    w = 0.38
    ax2.bar(x - w/2, sens_own, w, label="umbral por paciente", color="#4C78A8")
    ax2.bar(x + w/2, sens_g, w, label="umbral global", color="#E45756")
    ax2.set_xticks(x); ax2.set_xticklabels(PATIENTS, rotation=45)
    ax2.set_ylabel("sensibilidad @95% spec")
    ax2.set_title("Sensibilidad recuperada: por-paciente vs global")
    ax2.legend()
    fig.suptitle("Calibración exp_005 — el modelo rankea bien, pero el umbral debe ser por paciente",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "umbral_por_paciente.png", dpi=120)
    print(f"\nFigura -> {OUT/'umbral_por_paciente.png'}")


if __name__ == "__main__":
    main()
