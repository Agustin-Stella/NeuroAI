"""Entrenamiento LOSO reproducible y **reanudable** de la CNN 1D sobre CHB-MIT.

Este script reemplaza el flujo interactivo (que se perdía al cerrar la terminal)
por un runner robusto:

  1. **Cache de ventanas.** Materializa una sola vez las ventanas de cada paciente a
     ``.npy`` (memmap). Re-filtrar ~100 EDFs con MNE en cada epoch sería intratable
     en CPU; el cache se calcula una vez y se reusa entre folds y entre corridas.
  2. **LOSO real.** Para cada paciente del pool entrena con el resto y evalúa sobre
     ese paciente completo (Leave-One-Subject-Out).
  3. **Guardado incremental.** Tras cada fold escribe el checkpoint (``.pt``) y
     appendea el resultado a ``results.json``. Si el proceso muere, al relanzarlo
     retoma desde el primer fold sin checkpoint. **Un corte no borra el progreso.**
  4. **Reporte automático.** Al terminar genera ``report.md`` y ``curva_loss.png``.

Anti-leakage: el split es por sujeto (LOSO) y el ``ChannelNormalizer`` se ajusta
SOLO con el train de cada fold. El montaje de canales es la intersección común a
todos los pacientes del pool (n_channels fijo, requisito de la CNN).

Uso típico (en background, sobrevive a la terminal):

    nohup .venv/bin/python -m neuropilot.training.run_loso \
        --data-root data/chb-mit \
        --out experiments/exp_005 \
        > experiments/exp_005/train.log 2>&1 &

Decisiones (hiperparámetros = exp_004; ver docs/research.md, reproducibilidad):
    ventana 4 s, overlap 0, band-pass 0.5-40 + notch 60, z-score por canal (fit
    en train), negativos submuestreados 15:1, CNN1D 32/64/128 k=7, 15 epochs,
    lr 1e-3, weight_decay 1e-4, dropout 0.5, pos_weight, seed 42, CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from neuropilot.data.loaders import parse_summary, read_edf
from neuropilot.datasets.eeg_dataset import EegRecord
from neuropilot.evaluation.metrics import (
    average_precision,
    binary_metrics,
    roc_auc,
    sensitivity_at_specificity,
    summarize_loso,
)
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.channels import common_channels, pick_canonical
from neuropilot.preprocessing.filters import preprocess_raw
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.training.trainer import predict_proba, set_seed, train_model
from neuropilot.windowing.segment import num_windows, segment_signal


# --------------------------------------------------------------------------- #
# Configuración del experimento (todo hiperparámetro vive acá / en la CLI)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_root: str
    out: str
    cache_dir: str
    patients: list[str] = field(default_factory=lambda: [f"chb{i:02d}" for i in range(1, 10)])
    # señal / ventaneo
    window_seconds: float = 4.0
    overlap: float = 0.0
    ictal_overlap: float = 0.5
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freq: float = 60.0
    # balance de train
    neg_per_pos: int = 15
    # modelo / entrenamiento
    n_filters: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 7
    pool: int = 4
    dropout: float = 0.5
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# 1) Montaje canónico: intersección de canales sobre todo el pool
# --------------------------------------------------------------------------- #
def records_for_patient(data_root: Path, pid: str) -> list[EegRecord]:
    """Lista de registros (EDF + crisis) de un paciente, según su summary."""
    summary_path = data_root / pid / f"{pid}-summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Falta el summary: {summary_path}")
    summary = parse_summary(summary_path)
    records = []
    for file_name, info in summary.files.items():
        edf_path = data_root / pid / file_name
        if edf_path.exists():
            records.append(EegRecord(pid, edf_path, list(info.seizures)))
    return records


def compute_canonical_channels(data_root: Path, records: dict[str, list[EegRecord]]) -> list[str]:
    """Intersección de canales presente en TODOS los EDF del pool (montaje común)."""
    channel_lists = []
    for pid, recs in records.items():
        for rec in recs:
            raw = read_edf(rec.edf_path, preload=False)
            channel_lists.append(list(raw.ch_names))
    return common_channels(channel_lists)


# --------------------------------------------------------------------------- #
# 2) Cache de ventanas por paciente (memmap en disco, se calcula una sola vez)
# --------------------------------------------------------------------------- #
def _cache_paths(cache_dir: Path, pid: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{pid}_windows.npy",
        cache_dir / f"{pid}_labels.npy",
        cache_dir / f"{pid}_meta.json",
    )


def materialize_patient(
    cfg: Config, pid: str, records: list[EegRecord], channels: list[str]
) -> dict:
    """Materializa (o reusa) el cache de ventanas de un paciente.

    Rellena un memmap ``(N, C, W)`` archivo por archivo para acotar la RAM a un
    EDF por vez. Idempotente: si el meta coincide con la config, no recomputa.
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    win_path, lbl_path, meta_path = _cache_paths(cache_dir, pid)

    signature = {
        "channels": channels,
        "window_seconds": cfg.window_seconds,
        "overlap": cfg.overlap,
        "ictal_overlap": cfg.ictal_overlap,
        "l_freq": cfg.l_freq,
        "h_freq": cfg.h_freq,
        "notch_freq": cfg.notch_freq,
    }
    if meta_path.exists() and win_path.exists() and lbl_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") == signature:
            print(f"  [{pid}] cache OK ({meta['n_windows']} ventanas, "
                  f"{meta['n_ictal']} ictales)", flush=True)
            return meta

    # Conteo total de ventanas (headers, sin cargar señal) para dimensionar el memmap.
    per_file_counts = []
    sfreq = None
    for rec in records:
        raw = read_edf(rec.edf_path, preload=False)
        sf = float(raw.info["sfreq"])
        sfreq = sf if sfreq is None else sfreq
        per_file_counts.append(
            num_windows(raw.n_times, sf, cfg.window_seconds, cfg.overlap)
        )
    total = int(sum(per_file_counts))
    win_samples = int(round(cfg.window_seconds * sfreq))
    n_ch = len(channels)

    print(f"  [{pid}] materializando {total} ventanas "
          f"({n_ch} ch x {win_samples} muestras)...", flush=True)

    windows = np.lib.format.open_memmap(
        win_path, mode="w+", dtype=np.float32, shape=(total, n_ch, win_samples)
    )
    labels = np.zeros(total, dtype=np.int8)

    pos = 0
    for rec, expected in zip(records, per_file_counts):
        raw = read_edf(rec.edf_path, preload=True)
        raw = pick_canonical(raw, channels, copy=False)
        raw = preprocess_raw(
            raw, l_freq=cfg.l_freq, h_freq=cfg.h_freq,
            notch_freq=cfg.notch_freq, copy=False,
        )
        ws = segment_signal(
            raw.get_data(), float(raw.info["sfreq"]), rec.seizures,
            window_seconds=cfg.window_seconds, overlap=cfg.overlap,
            ictal_overlap=cfg.ictal_overlap,
        )
        n = len(ws)
        if n:
            windows[pos : pos + n] = ws.windows.astype(np.float32)
            labels[pos : pos + n] = ws.labels.astype(np.int8)
            pos += n
        del raw

    windows.flush()
    np.save(lbl_path, labels)

    meta = {
        "patient": pid,
        "n_windows": int(total),
        "n_ictal": int(labels.sum()),
        "n_channels": n_ch,
        "win_samples": win_samples,
        "sfreq": float(sfreq),
        "signature": signature,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [{pid}] listo: {total} ventanas, {int(labels.sum())} ictales", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# 3) Ensamblado de datasets del fold
# --------------------------------------------------------------------------- #
class WindowArrayDataset(torch.utils.data.Dataset):
    """Sirve ``(ventana normalizada, label)`` desde un array ``(N, C, W)``.

    Aplica el z-score por canal en ``__getitem__`` para no duplicar en RAM el
    array normalizado (importante en eval, donde el paciente entero puede ser
    grande y se lee vía memmap).
    """

    def __init__(self, windows, labels, normalizer: ChannelNormalizer):
        self.windows = windows
        self.labels = np.asarray(labels).astype(np.int64)
        self.mean = normalizer.mean_.astype(np.float32).reshape(-1, 1)
        self.std = (normalizer.std_.astype(np.float32) + normalizer.eps).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = np.asarray(self.windows[idx], dtype=np.float32)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), int(self.labels[idx])


def build_train_windows(cfg: Config, train_patients: list[str], rng: np.random.Generator):
    """Arma el train del fold: TODOS los positivos + negativos submuestreados 15:1.

    Carga por paciente solo las filas elegidas desde el memmap (no todo el array),
    y las concatena en RAM. Devuelve ``(windows, labels)``.
    """
    cache_dir = Path(cfg.cache_dir)
    chunks_w, chunks_y = [], []
    for pid in train_patients:
        win_path, lbl_path, _ = _cache_paths(cache_dir, pid)
        labels = np.load(lbl_path)
        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)
        n_neg_keep = min(len(neg_idx), cfg.neg_per_pos * max(1, len(pos_idx)))
        neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        sel = np.sort(np.concatenate([pos_idx, neg_keep]))

        wmm = np.load(win_path, mmap_mode="r")
        chunks_w.append(np.asarray(wmm[sel], dtype=np.float32))  # materializa solo `sel`
        chunks_y.append(labels[sel].astype(np.int64))
        del wmm
    return np.concatenate(chunks_w, axis=0), np.concatenate(chunks_y, axis=0)


def fit_normalizer(windows: np.ndarray, eps: float = 1e-8) -> ChannelNormalizer:
    """Z-score por canal ajustado sobre el train del fold (sin leakage)."""
    norm = ChannelNormalizer(eps=eps)
    norm.mean_ = windows.mean(axis=(0, 2))
    norm.std_ = windows.std(axis=(0, 2))
    return norm


# --------------------------------------------------------------------------- #
# 4) Un fold LOSO
# --------------------------------------------------------------------------- #
def run_fold(cfg: Config, test_pid: str, train_patients: list[str], out_dir: Path) -> dict:
    t0 = time.time()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # -- train: undersampling + normalizador (fit solo en train) ------------- #
    train_w, train_y = build_train_windows(cfg, train_patients, rng)
    normalizer = fit_normalizer(train_w)
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    pos_weight = (n_neg / n_pos) if n_pos else 1.0
    print(f"  [{test_pid}] train: {len(train_y)} ventanas "
          f"({n_pos} pos / {n_neg} neg), pos_weight={pos_weight:.1f}", flush=True)

    train_ds = WindowArrayDataset(train_w, train_y, normalizer)

    model = CNN1D(
        n_channels=train_w.shape[1], n_filters=cfg.n_filters,
        kernel_size=cfg.kernel_size, pool=cfg.pool, dropout=cfg.dropout,
    )
    history = train_model(
        model, train_ds, epochs=cfg.epochs, batch_size=cfg.batch_size,
        lr=cfg.lr, weight_decay=cfg.weight_decay, pos_weight=pos_weight,
        device=cfg.device, seed=cfg.seed, verbose=True,
    )
    del train_w, train_ds

    # -- eval: paciente de test COMPLETO (memmap, sin submuestrear) ---------- #
    win_path, lbl_path, _ = _cache_paths(Path(cfg.cache_dir), test_pid)
    test_w = np.load(win_path, mmap_mode="r")
    test_y = np.load(lbl_path).astype(int)
    test_ds = WindowArrayDataset(test_w, test_y, normalizer)
    scores = predict_proba(model, test_ds, batch_size=256, device=cfg.device)

    m = binary_metrics(test_y, scores, threshold=0.5)
    sens95, _ = sensitivity_at_specificity(test_y, scores, 0.95)
    fold = {
        "patient": test_pid,
        "auprc": round(float(average_precision(test_y, scores)), 4),
        "roc_auc": round(float(roc_auc(test_y, scores)), 4),
        "sens95": round(float(sens95), 4),
        "f1": round(float(m.f1), 4),
        "n_ictal": int((test_y == 1).sum()),
        "n_test_windows": int(len(test_y)),
        "train_loss": [round(x, 4) for x in history.train_loss],
        "seconds": round(time.time() - t0, 1),
    }

    # checkpoint (permite reanudar y reusar el modelo)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "normalizer": normalizer.to_dict(),
         "config": asdict(cfg), "fold": fold},
        ckpt_dir / f"{test_pid}.pt",
    )
    print(f"  [{test_pid}] AUPRC={fold['auprc']}  ROC-AUC={fold['roc_auc']}  "
          f"sens@95={fold['sens95']}  ({fold['seconds']}s)", flush=True)
    return fold


# --------------------------------------------------------------------------- #
# 5) Persistencia incremental de resultados (reanudable)
# --------------------------------------------------------------------------- #
def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"folds": []}


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# 6) Reporte + curva de loss
# --------------------------------------------------------------------------- #
def write_report(cfg: Config, results: dict, out_dir: Path) -> None:
    folds = sorted(results["folds"], key=lambda f: f["patient"])
    summary = results["summary"]

    def ms(key):
        return f"{summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}"

    exp_name = Path(cfg.out).name
    n_pac = len(folds)
    pool_str = (f"{cfg.patients[0]}–{cfg.patients[-1]}"
                if len(cfg.patients) > 1 else cfg.patients[0])
    collapsed = [f["patient"] for f in folds if f["auprc"] < 0.05]

    lines = [
        f"# Experimento {exp_name} — CNN 1D LOSO ({n_pac} pacientes: {pool_str})",
        "",
        "## Setup",
        "",
        f"- Pool: {', '.join(cfg.patients)} (LOSO, {len(folds)} folds).",
        f"- Ventana {cfg.window_seconds}s, overlap {cfg.overlap}, "
        f"band-pass {cfg.l_freq}-{cfg.h_freq} Hz + notch {cfg.notch_freq} Hz.",
        f"- Z-score por canal (fit solo en train). Negativos submuestreados "
        f"{cfg.neg_per_pos}:1 + pos_weight.",
        f"- CNN1D {cfg.n_filters} k={cfg.kernel_size}, dropout {cfg.dropout}, "
        f"{cfg.epochs} epochs, lr {cfg.lr}, wd {cfg.weight_decay}, seed {cfg.seed}, "
        f"{cfg.device}.",
        "",
        "## Resultados por fold",
        "",
        "| Test | AUPRC | ROC-AUC | sens@95spec | F1 | ictales | ventanas |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        lines.append(
            f"| {f['patient']} | {f['auprc']} | {f['roc_auc']} | {f['sens95']} | "
            f"{f['f1']} | {f['n_ictal']} | {f['n_test_windows']} |"
        )
    lines += [
        "",
        "## Resumen LOSO (media ± desvío entre sujetos)",
        "",
        f"| Métrica | CNN ({exp_name}, {n_pac} pac.) | CNN v2 (exp_004) | Baseline (exp_002) |",
        "|---|---|---|---|",
        f"| AUPRC | {ms('auprc')} | 0.066 ± 0.056 | 0.377 ± 0.306 |",
        f"| sens@95spec | {ms('sens95')} | 0.297 ± 0.143 | 0.553 ± 0.281 |",
        f"| ROC-AUC | {ms('roc_auc')} | 0.726 ± 0.102 | 0.755 ± 0.117 |",
        f"| F1 | {ms('f1')} | — | 0.218 ± 0.218 |",
        "",
        "> Nota: la columna exp_004/baseline se midió sobre 6 pacientes (chb01–06). "
        "Si este run usa otro pool, la comparación **no es estrictamente pareja** "
        "fold a fold.",
        "",
        "## ⚠️ Leer con cabeza científica",
        "",
        f"- **Varianza entre sujetos alta** (AUPRC ±{summary['auprc']['std']:.2f}): el "
        "promedio LOSO esconde folds muy dispares.",
    ]
    if collapsed:
        lines.append(
            f"- **Folds que colapsan:** {', '.join(collapsed)} con AUPRC < 0.05 "
            "(sujetos sistemáticamente difíciles). No es un modelo uniformemente confiable."
        )
    lines += [
        "- **Atribución.** Si esta config difiere de otra en más de un factor "
        "(p. ej. normalización *y* nº de pacientes), la diferencia no se puede "
        "atribuir a una sola variable: hace falta cambiar un factor por vez.",
        "",
        "## Cómo se corrió (reproducible)",
        "",
        "```bash",
        "nohup .venv/bin/python -m neuropilot.training.run_loso \\",
        f"    --data-root {cfg.data_root} --out {cfg.out} \\",
        f"    > {cfg.out}/train.log 2>&1 &",
        "```",
        "",
        f"Checkpoints por fold en `{cfg.out}/checkpoints/`. Guardado incremental en "
        "`results.json` (reanudable ante cortes).",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))

    # curva de loss (una línea por fold)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for f in folds:
            ax.plot(range(1, len(f["train_loss"]) + 1), f["train_loss"],
                    marker="o", ms=3, label=f["patient"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss (BCE ponderada)")
        ax.set_title("exp_005 — train loss por fold LOSO (chb01–chb09)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "curva_loss.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # el reporte no debe fallar por el gráfico
        print(f"  [warn] no pude generar curva_loss.png: {e}", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenamiento LOSO reanudable (CHB-MIT).")
    ap.add_argument("--data-root", default="data/chb-mit")
    ap.add_argument("--out", default="experiments/exp_005")
    ap.add_argument("--cache-dir", default=None,
                    help="Dir del cache de ventanas (default: <out>/cache).")
    ap.add_argument("--patients", nargs="+", default=None,
                    help="Pool de pacientes (default: chb01..chb09).")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--neg-per-pos", type=int, default=15)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(out_dir / "cache")

    cfg = Config(
        data_root=args.data_root, out=args.out, cache_dir=cache_dir,
        epochs=args.epochs, neg_per_pos=args.neg_per_pos,
    )
    if args.patients:
        cfg.patients = args.patients

    data_root = Path(cfg.data_root)
    print("=" * 70, flush=True)
    print(f"exp LOSO | pool={cfg.patients} | out={cfg.out}", flush=True)
    print("=" * 70, flush=True)

    # -- 1) registros + montaje canónico ------------------------------------- #
    records = {pid: records_for_patient(data_root, pid) for pid in cfg.patients}
    for pid, recs in records.items():
        if not recs:
            raise FileNotFoundError(f"Sin EDFs para {pid} en {data_root}")
    channels = compute_canonical_channels(data_root, records)
    print(f"Montaje canónico: {len(channels)} canales -> {channels}", flush=True)

    # -- 2) cache de ventanas (una vez) -------------------------------------- #
    print("Materializando cache de ventanas...", flush=True)
    for pid in cfg.patients:
        materialize_patient(cfg, pid, records[pid], channels)

    # -- 3) LOSO con guardado incremental / resume --------------------------- #
    results_path = out_dir / "results.json"
    results = load_results(results_path)
    results["config"] = asdict(cfg)
    results["canonical_channels"] = channels
    done = {f["patient"] for f in results["folds"]
            if (out_dir / "checkpoints" / f"{f['patient']}.pt").exists()}

    for test_pid in cfg.patients:
        if test_pid in done:
            print(f"[{test_pid}] ya completo, se saltea.", flush=True)
            continue
        train_patients = [p for p in cfg.patients if p != test_pid]
        fold = run_fold(cfg, test_pid, train_patients, out_dir)
        results["folds"] = [f for f in results["folds"] if f["patient"] != test_pid]
        results["folds"].append(fold)
        results["summary"] = summarize_loso(
            [{k: f[k] for k in ("auprc", "roc_auc", "sens95", "f1")}
             for f in results["folds"]]
        )
        save_results(results_path, results)  # <- persiste tras CADA fold

    # -- 4) reporte final ---------------------------------------------------- #
    results["folds"] = sorted(results["folds"], key=lambda f: f["patient"])
    save_results(results_path, results)
    write_report(cfg, results, out_dir)
    print("=" * 70, flush=True)
    print(f"LISTO. Reporte en {out_dir}/report.md", flush=True)
    print(f"Resumen: {json.dumps(results['summary'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
