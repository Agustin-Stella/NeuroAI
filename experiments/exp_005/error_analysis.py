"""Error analysis dirigido: por qué chb06 colapsa (AUPRC 0.002) y chb09 no (0.91).

Reproduce los scores desde los checkpoints guardados (sin reentrenar) y compara
las dos poblaciones en tres ejes:

  1. Naturaleza del error: ¿FN en crisis o FP en interictal? (a umbral 0.5 y al
     umbral que da 95% de especificidad).
  2. Distribución de scores predichos (ictal vs interictal), por paciente.
  3. Perfil del paciente: montaje/canales originales, nº y duración de crisis,
     desbalance, y amplitud de la señal (proxy de artefactos).

Salida: consola + figura ``error_analysis/scores_chb06_vs_chb09.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuropilot.data.loaders import parse_summary, read_edf
from neuropilot.evaluation.metrics import confusion_at_threshold, sensitivity_at_specificity
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.trainer import predict_proba

EXP = Path("experiments/exp_005")
CACHE = EXP / "cache"
DATA_ROOT = Path("data/chb-mit")
OUT = EXP / "error_analysis"
OUT.mkdir(exist_ok=True)


class ArrayDS(torch.utils.data.Dataset):
    """Sirve ventanas normalizadas (C,W) desde memmap, para inferencia."""
    def __init__(self, windows, normalizer: ChannelNormalizer):
        self.w = windows
        self.mean = normalizer.mean_.astype(np.float32).reshape(-1, 1)
        self.std = (normalizer.std_.astype(np.float32) + normalizer.eps).reshape(-1, 1)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        x = np.asarray(self.w[i], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std), 0


def scores_for(pid: str):
    """Recarga el modelo del fold `pid` y devuelve (scores, labels, windows_memmap)."""
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
    return scores, labels, windows


def seizure_profile(pid: str) -> dict:
    """Nº de crisis, duraciones y segundos ictales totales desde el summary."""
    summary = parse_summary(DATA_ROOT / pid / f"{pid}-summary.txt")
    durs = []
    for info in summary.files.values():
        durs += [s.duration_sec for s in info.seizures]
    # canales originales (antes del montaje canónico): del primer EDF
    first_edf = next(iter(summary.files))
    raw = read_edf(DATA_ROOT / pid / first_edf, preload=False)
    return {
        "sfreq": summary.sampling_rate_hz,
        "n_orig_channels": len(raw.ch_names),
        "n_seizures": len(durs),
        "seizure_durs": durs,
        "total_ictal_sec": float(sum(durs)),
    }


def error_breakdown(scores, labels, thr, name):
    tp, fp, tn, fn = confusion_at_threshold(labels, scores, thr)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    print(f"    {name} (thr={thr:.4f}): "
          f"sens={sens:.2f} spec={spec:.3f} prec={prec:.3f} | "
          f"TP={tp} FN={fn} (crisis perdidas) | FP={fp} (falsas alarmas)")


def amp_stats(windows, labels):
    """RMS por ventana (proxy de amplitud/artefactos), separado ictal/interictal."""
    # submuestreo para no cargar todo si es grande
    idx_pos = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(0)
    idx_neg = rng.choice(np.flatnonzero(labels == 0),
                         size=min(5000, int((labels == 0).sum())), replace=False)
    def rms(idx):
        w = np.asarray(windows[np.sort(idx)], dtype=np.float64)  # (n,C,W) en voltios
        return np.sqrt((w ** 2).mean(axis=(1, 2))) * 1e6  # a microvoltios
    return rms(idx_pos), rms(idx_neg)


def main():
    print("=" * 72)
    print("ERROR ANALYSIS — chb06 (colapsa) vs chb09 (funciona) | modelo exp_005")
    print("=" * 72)

    data = {}
    for pid in ("chb06", "chb09"):
        scores, labels, windows = scores_for(pid)
        prof = seizure_profile(pid)
        pos, neg = scores[labels == 1], scores[labels == 0]
        rms_pos, rms_neg = amp_stats(windows, labels)
        data[pid] = dict(scores=scores, labels=labels, pos=pos, neg=neg,
                         prof=prof, rms_pos=rms_pos, rms_neg=rms_neg)

        print(f"\n### {pid}")
        print(f"  Perfil: sfreq={prof['sfreq']}Hz, {prof['n_orig_channels']} canales orig, "
              f"{prof['n_seizures']} crisis, ictal total={prof['total_ictal_sec']:.0f}s")
        if prof["seizure_durs"]:
            d = np.array(prof["seizure_durs"])
            print(f"  Duración crisis: min={d.min():.0f}s mediana={np.median(d):.0f}s "
                  f"max={d.max():.0f}s")
        pr = labels.mean()
        print(f"  Ventanas: {len(labels)} | ictales={int(labels.sum())} "
              f"| positive_rate={pr*100:.3f}%  (1 cada {int(1/pr)})")
        print(f"  Scores ictal:     mediana={np.median(pos):.3f}  "
              f"p90={np.percentile(pos,90):.3f}  max={pos.max():.3f}")
        print(f"  Scores interictal:mediana={np.median(neg):.3f}  "
              f"p90={np.percentile(neg,90):.3f}  max={neg.max():.3f}")
        # ¿el modelo rankea las crisis por encima del interictal?
        above = (pos > np.median(neg)).mean()
        print(f"  Fracción de crisis con score > mediana interictal: {above*100:.0f}% "
              f"(si ~50%, el modelo NO separa)")
        print(f"  Amplitud RMS (µV): ictal med={np.median(data[pid]['rms_pos']):.0f} | "
              f"interictal med={np.median(data[pid]['rms_neg']):.0f}")
        # desglose de errores a dos umbrales
        _, thr95 = sensitivity_at_specificity(labels, scores, 0.95)
        error_breakdown(scores, labels, 0.5, "umbral 0.5   ")
        error_breakdown(scores, labels, thr95, "umbral@95spec")

    # ---- figura ----------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.linspace(0, 1, 41)
    for ax, pid in zip(axes[0], ("chb06", "chb09")):
        d = data[pid]
        ax.hist(d["neg"], bins=bins, density=True, alpha=0.6, label="interictal", color="#4C78A8")
        ax.hist(d["pos"], bins=bins, density=True, alpha=0.6, label="ictal (crisis)", color="#E45756")
        ax.set_title(f"{pid} — scores predichos (AUPRC "
                     f"{'0.002' if pid=='chb06' else '0.91'})")
        ax.set_xlabel("score de crisis"); ax.set_ylabel("densidad")
        ax.set_yscale("log"); ax.legend()
    for ax, pid in zip(axes[1], ("chb06", "chb09")):
        d = data[pid]
        ax.hist(d["rms_neg"], bins=40, density=True, alpha=0.6, label="interictal", color="#4C78A8")
        ax.hist(d["rms_pos"], bins=40, density=True, alpha=0.6, label="ictal", color="#E45756")
        ax.set_title(f"{pid} — amplitud RMS por ventana")
        ax.set_xlabel("RMS (µV)"); ax.set_ylabel("densidad"); ax.legend()
    fig.suptitle("Error analysis: chb06 (colapsa) vs chb09 (funciona) — modelo exp_005",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "scores_chb06_vs_chb09.png", dpi=120)
    print(f"\nFigura -> {OUT/'scores_chb06_vs_chb09.png'}")


if __name__ == "__main__":
    main()
