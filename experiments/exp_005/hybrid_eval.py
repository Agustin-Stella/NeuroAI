"""Híbrido CNN + baseline: ¿combinarlos mejora la cobertura por evento?

Hipótesis (de exp_004): la CNN y el baseline de bandpower fallan en pacientes
DISTINTOS, así que combinarlos debería cubrir más crisis que cualquiera solo.

Método, por cada paciente (LOSO sobre chb01–09):
  - CNN: scores desde el checkpoint de exp_005 (modelo que no vio a ese paciente).
  - Baseline: se entrena con los otros 8 pacientes y predice sobre el de test.
  - Híbrido: promedio de los rangos por-paciente de ambos (rank-average, robusto a
    que los scores estén en escalas distintas).
Se evalúan los tres a nivel de evento (cobertura + falsas alarmas/hora) con la misma
calibración por percentil. NO depende de descargas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuropilot.evaluation.calibration import percentile_threshold
from neuropilot.evaluation.events import event_metrics
from neuropilot.models.baseline import BandPowerBaseline
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.run_loso import Config, build_train_windows
from neuropilot.training.trainer import predict_proba, set_seed

EXP = Path("experiments/exp_005")
CACHE = EXP / "cache"
OUT = EXP / "hybrid"; OUT.mkdir(exist_ok=True)
PATIENTS = [f"chb{i:02d}" for i in range(1, 10)]
SFREQ = 256.0
TARGET_SPEC, MIN_CONS, MAX_GAP = 0.95, 2, 1

CFG = Config(data_root="data/chb-mit", out=str(EXP),
             cache_dir=str(CACHE), patients=PATIENTS, neg_per_pos=15)


class ArrayDS(torch.utils.data.Dataset):
    def __init__(self, w, norm):
        self.w = w
        self.mean = norm.mean_.astype(np.float32).reshape(-1, 1)
        self.std = (norm.std_.astype(np.float32) + norm.eps).reshape(-1, 1)
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        x = np.asarray(self.w[i], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std), 0


def cnn_scores(pid, windows):
    ck = torch.load(EXP / "checkpoints" / f"{pid}.pt", weights_only=False)
    cfg = ck["config"]; n_ch = ck["model_state"]["features.0.weight"].shape[1]
    m = CNN1D(n_channels=n_ch, n_filters=tuple(cfg["n_filters"]),
              kernel_size=cfg["kernel_size"], pool=cfg["pool"], dropout=cfg["dropout"])
    m.load_state_dict(ck["model_state"])
    norm = ChannelNormalizer.from_dict(ck["normalizer"])
    return predict_proba(m, ArrayDS(windows, norm), batch_size=256)


def baseline_scores(clf, windows, batch=1500):
    out = []
    for i in range(0, len(windows), batch):
        out.append(clf.predict_proba(np.asarray(windows[i:i + batch], dtype=float)))
    return np.concatenate(out)


def rank01(x):
    """Rango percentil en [0,1] (robusto a la escala de los scores)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def event_row(y, s):
    thr = percentile_threshold(s, TARGET_SPEC)
    return event_metrics(y, s, threshold=thr, window_seconds=4.0,
                         min_consecutive=MIN_CONS, max_gap=MAX_GAP)


def main():
    set_seed(42)
    rng = np.random.default_rng(42)
    agg = {k: {"det": 0, "tot": 0, "fah": []} for k in ("cnn", "base", "hybrid")}

    print("=" * 92)
    print(f"HÍBRIDO CNN + baseline | evento @ percentil {TARGET_SPEC:.0%}, min {MIN_CONS} vent.")
    print("=" * 92)
    print(f"{'pac':6} | {'CNN sens/FAh':>16} | {'BASE sens/FAh':>16} | {'HÍBRIDO sens/FAh':>18}")
    print("-" * 92)

    for pid in PATIENTS:
        y = np.load(CACHE / f"{pid}_labels.npy").astype(int)
        w = np.load(CACHE / f"{pid}_windows.npy", mmap_mode="r")

        s_cnn = cnn_scores(pid, w)

        # baseline entrenado con los otros 8 (mismo undersampling que el CNN)
        train_patients = [p for p in PATIENTS if p != pid]
        tw, ty = build_train_windows(CFG, train_patients, rng)
        clf = BandPowerBaseline(sfreq=SFREQ).fit(tw, ty)
        del tw
        s_base = baseline_scores(clf, w)

        s_hyb = 0.5 * rank01(s_cnn) + 0.5 * rank01(s_base)

        rows = {"cnn": event_row(y, s_cnn), "base": event_row(y, s_base),
                "hybrid": event_row(y, s_hyb)}
        for k, m in rows.items():
            agg[k]["det"] += m.n_detected_true; agg[k]["tot"] += m.n_true_events
            agg[k]["fah"].append(m.false_alarms_per_hour)

        def cell(m): return f"{m.event_sensitivity:.2f} / {m.false_alarms_per_hour:.1f}"
        print(f"{pid:6} | {cell(rows['cnn']):>16} | {cell(rows['base']):>16} | "
              f"{cell(rows['hybrid']):>18}")

    print("-" * 92)
    print(f"\n{'modelo':10} {'cobertura (crisis)':>20} {'FA/h (mediana)':>16}")
    labels = {"cnn": "CNN sola", "base": "Baseline", "hybrid": "HÍBRIDO"}
    summary = {}
    for k in ("cnn", "base", "hybrid"):
        cov = agg[k]["det"] / agg[k]["tot"]; medfah = float(np.median(agg[k]["fah"]))
        summary[k] = (cov, medfah, agg[k]["det"], agg[k]["tot"])
        print(f"{labels[k]:10} {agg[k]['det']}/{agg[k]['tot']} = {cov:5.0%}      "
              f"{medfah:12.2f}")

    dc = summary["hybrid"][0] - summary["cnn"][0]
    print(f"\n>>> Híbrido vs CNN: {summary['cnn'][0]:.0%} -> {summary['hybrid'][0]:.0%} "
          f"cobertura ({'+' if dc>=0 else ''}{dc*100:.0f} pts)")

    # figura
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    order = ["cnn", "base", "hybrid"]; colors = ["#4C78A8", "#B8712E", "#0E8C86"]
    for k, c in zip(order, colors):
        cov, fah, det, tot = summary[k]
        ax.scatter(fah, cov, s=180, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{labels[k]}\n{cov:.0%} · {fah:.1f} FA/h", (fah, cov),
                    textcoords="offset points", xytext=(10, 8), fontsize=9)
    ax.set_xlabel("falsas alarmas por hora (mediana)")
    ax.set_ylabel("cobertura por evento (crisis detectadas)")
    ax.set_title("Híbrido CNN + baseline vs cada uno solo")
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "hibrido.png", dpi=120)
    print(f"\nFigura -> {OUT/'hibrido.png'}")


if __name__ == "__main__":
    main()
