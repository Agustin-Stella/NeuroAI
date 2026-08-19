"""Segunda etapa anti-falsas-alarmas: filtrar eventos candidatos del CNN.

El CNN (1ra etapa) marca eventos; muchos son falsas alarmas (~7 FA/h). La 2da etapa
es un clasificador que mira cada evento candidato y decide **crisis real vs falsa
alarma**, para bajar las FA/h SIN perder cobertura.

Honesto (sin leakage): los scores del CNN de cada paciente son out-of-sample (su
checkpoint LOSO), y la 2da etapa se entrena con los OTROS 8 pacientes y se aplica al
de test. Nunca ve al paciente que evalúa.

Features por evento candidato (compactas, para no sobreajustar con pocas crisis):
  duración · max/mean/std del score del CNN · bandpower por banda (media y máx entre
  canales). Una crisis es sostenida, de alto score y con firma espectral; un artefacto
  suele ser breve o espectralmente distinto.

Salida: consola + figura ``second_stage/segunda_etapa.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuropilot.evaluation.calibration import percentile_threshold
from neuropilot.evaluation.events import Event, _runs, aggregate_events
from neuropilot.models.baseline import bandpower_features
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.trainer import predict_proba

EXP = Path("experiments/exp_005"); CACHE = EXP / "cache"
OUT = EXP / "second_stage"; OUT.mkdir(exist_ok=True)
PATIENTS = [f"chb{i:02d}" for i in range(1, 10)]
SFREQ, WIN_S = 256.0, 4.0
TARGET_SPEC, MIN_CONS, MAX_GAP = 0.95, 2, 1


class ArrayDS(torch.utils.data.Dataset):
    def __init__(self, w, norm):
        self.w = w; self.mean = norm.mean_.astype(np.float32).reshape(-1, 1)
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
    return predict_proba(m, ArrayDS(windows, ChannelNormalizer.from_dict(ck["normalizer"])), batch_size=256)


def event_features(scores, windows, ev: Event) -> np.ndarray:
    seg_s = scores[ev.start:ev.end]
    seg_w = np.asarray(windows[ev.start:ev.end], dtype=float)          # (n, C, W)
    bp = bandpower_features(seg_w, SFREQ).mean(axis=0).reshape(5, -1)  # (5 bandas, C)
    return np.array([ev.end - ev.start, seg_s.max(), seg_s.mean(), seg_s.std(),
                     *bp.mean(axis=1), *bp.max(axis=1)])               # 4 + 5 + 5 = 14


def overlaps_any(ev, events):
    return any(ev.overlaps(e) for e in events)


def coverage_fa(true_events, pred_events, hours):
    cov = sum(overlaps_any(t, pred_events) for t in true_events)
    fa = sum(not overlaps_any(p, true_events) for p in pred_events)
    return cov, len(true_events), fa, fa / hours if hours else float("nan")


def main():
    # 1) candidatos + features + etiqueta, por paciente (todo out-of-sample)
    data = {}
    for pid in PATIENTS:
        y = np.load(CACHE / f"{pid}_labels.npy").astype(int)
        w = np.load(CACHE / f"{pid}_windows.npy", mmap_mode="r")
        s = cnn_scores(pid, w)
        thr = percentile_threshold(s, TARGET_SPEC)
        cand = aggregate_events(s, thr, min_consecutive=MIN_CONS, max_gap=MAX_GAP)
        true_ev = _runs(y == 1)
        feats = np.array([event_features(s, w, ev) for ev in cand]) if cand else np.zeros((0, 14))
        is_true = np.array([overlaps_any(ev, true_ev) for ev in cand], dtype=int)
        data[pid] = dict(cand=cand, feats=feats, is_true=is_true, true_ev=true_ev,
                         scores=s, labels=y, hours=len(y) * WIN_S / 3600.0)
        print(f"[{pid}] {len(cand)} eventos candidatos ({int(is_true.sum())} reales / "
              f"{len(cand)-int(is_true.sum())} falsas alarmas)", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # 2) LOSO de la 2da etapa: predecir prob(real) para los eventos de cada paciente
    prob_real = {}
    for pid in PATIENTS:
        Xtr = np.vstack([data[p]["feats"] for p in PATIENTS if p != pid and len(data[p]["feats"])])
        ytr = np.concatenate([data[p]["is_true"] for p in PATIENTS if p != pid and len(data[p]["feats"])])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5))
        clf.fit(Xtr, ytr)
        prob_real[pid] = clf.predict_proba(data[pid]["feats"])[:, 1] if len(data[pid]["feats"]) else np.array([])

    # 3) CNN sola (todos los candidatos) vs CNN + 2da etapa (barriendo el umbral de aceptación)
    def totals(accept_fn):
        det = tot = fa = 0; fahs = []
        for pid in PATIENTS:
            d = data[pid]
            keep = [ev for ev, pr in zip(d["cand"], prob_real[pid]) if accept_fn(pr)]
            c, t, f, fah = coverage_fa(d["true_ev"], keep, d["hours"])
            det += c; tot += t; fa += f; fahs.append(fah)
        return det / tot, float(np.median(fahs)), det, tot

    base_cov, base_fah, base_det, tot = totals(lambda pr: True)  # CNN sola @ percentil 0.95
    print("\n" + "=" * 68)
    print(f"CNN sola @0.95:  cobertura {base_det}/{tot} = {base_cov:.0%}  |  {base_fah:.2f} FA/h")
    print("=" * 68)
    print(f"{'umbral 2da etapa':>18} {'cobertura':>12} {'FA/h (med)':>12}")
    sweep = []
    for a in (0.3, 0.5, 0.7, 0.8, 0.9):
        cov, fah, det, _ = totals(lambda pr, a=a: pr >= a)
        sweep.append((a, cov, fah, det))
        print(f"{a:18.2f} {det}/{tot} = {cov:3.0%}   {fah:12.2f}")

    # BASELINE TRIVIAL a comparar: subir el umbral del CNN (1ra etapa), sin 2da etapa
    def first_stage_totals(pct):
        det = tot = 0; fahs = []
        for pid in PATIENTS:
            d = data[pid]; s = d["scores"]
            thr = percentile_threshold(s, pct)
            preds = aggregate_events(s, thr, min_consecutive=MIN_CONS, max_gap=MAX_GAP)
            c, t, f, fah = coverage_fa(d["true_ev"], preds, d["hours"])
            det += c; tot += t; fahs.append(fah)
        return det / tot, float(np.median(fahs)), det
    print(f"\n{'CNN más estricto':>18} {'cobertura':>12} {'FA/h (med)':>12}")
    fs_sweep = []
    for pct in (0.95, 0.98, 0.99, 0.995):
        cov, fah, det = first_stage_totals(pct)
        fs_sweep.append((pct, cov, fah, det))
        print(f"{('percentil '+format(pct,'.3f')):>18} {det}/{tot} = {cov:3.0%}   {fah:12.2f}")

    # figura: mover el punto de operación abajo-izquierda sin perder cobertura
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # baseline trivial: subir el umbral del CNN
    ax.plot([s[2] for s in fs_sweep], [s[1] for s in fs_sweep], "s--", color="#4C78A8",
            zorder=3, label="CNN más estricto (subir umbral)")
    # 2da etapa aprendida
    ax.plot([s[2] for s in sweep], [s[1] for s in sweep], "o-", color="#0E8C86",
            zorder=3, label="CNN + 2da etapa (aprendida)")
    for a, cov, fah, det in sweep:
        ax.annotate(f"{a:.1f}", (fah, cov), textcoords="offset points", xytext=(6, -11),
                    fontsize=8, color="#0E8C86")
    ax.set_xlabel("falsas alarmas por hora (mediana)")
    ax.set_ylabel("cobertura por evento")
    ax.set_title("¿La 2da etapa aprendida le gana a solo subir el umbral?")
    ax.grid(alpha=.3); ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "segunda_etapa.png", dpi=120)
    print(f"\nFigura -> {OUT/'segunda_etapa.png'}")


if __name__ == "__main__":
    main()
