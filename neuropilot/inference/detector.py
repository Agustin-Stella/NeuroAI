"""MVP de inferencia end-to-end: un EDF entra, eventos de crisis salen.

Ata todo el pipeline en un solo punto ejecutable:

    EDF → montaje canónico → filtrado → ventaneo → normalización (del checkpoint)
        → CNN → calibración por percentil (label-free) → agrupación en eventos

y devuelve los **rangos temporales** donde el modelo detecta actividad compatible con
crisis, más un gráfico. Es una **herramienta de asistencia**: señala patrones, no
emite diagnóstico.

Uso:
    .venv/bin/python -m neuropilot.inference.detector RUTA.edf [--patient chbNN]
        [--target-spec 0.99] [--min-consecutive 2] [--out figura.png]

Por defecto usa el checkpoint LOSO del paciente (modelo que NUNCA vio ese sujeto:
predicción honesta sobre paciente no visto) y el montaje canónico de exp_005.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neuropilot.data.loaders import parse_summary, read_edf
from neuropilot.evaluation.calibration import percentile_threshold
from neuropilot.evaluation.events import Event, aggregate_events
from neuropilot.models.cnn1d import CNN1D
from neuropilot.preprocessing.channels import pick_canonical
from neuropilot.preprocessing.filters import preprocess_raw
from neuropilot.preprocessing.normalization import ChannelNormalizer
from neuropilot.windowing.segment import segment_signal

DEFAULT_EXP = Path("experiments/exp_005")


@dataclass
class Detection:
    """Resultado de correr el detector sobre un registro."""
    edf_path: str
    events_sec: list[tuple[float, float]]   # crisis detectadas [inicio, fin) en segundos
    scores: np.ndarray                       # score de crisis por ventana
    window_seconds: float
    threshold: float
    total_seconds: float
    events_idx: list[tuple[int, int]] = None   # eventos en índices de ventana [start, end)
    windows: np.ndarray = None                 # ventanas normalizadas (N, C, W), para explicar

    def summary(self) -> str:
        n = len(self.events_sec)
        head = (f"{n} evento(s) compatibles con crisis en "
                f"{self.total_seconds/60:.1f} min de registro "
                f"(umbral {self.threshold:.3f}):")
        lines = [head]
        for i, (a, b) in enumerate(self.events_sec, 1):
            lines.append(f"  #{i}: {a/60:.2f}–{b/60:.2f} min  "
                         f"({a:.0f}s–{b:.0f}s, dura {b-a:.0f}s)")
        if n == 0:
            lines.append("  (sin detecciones)")
        return "\n".join(lines)


def _patient_from_path(edf_path: Path) -> str:
    m = re.match(r"(chb\d+)", edf_path.name, re.IGNORECASE)
    if not m:
        raise ValueError(f"No pude derivar el paciente de {edf_path.name}")
    return m.group(1).lower()


def _build_model(ckpt: dict) -> CNN1D:
    cfg = ckpt["config"]
    n_ch = ckpt["model_state"]["features.0.weight"].shape[1]
    model = CNN1D(n_channels=n_ch, n_filters=tuple(cfg["n_filters"]),
                  kernel_size=cfg["kernel_size"], pool=cfg["pool"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _predict(model: CNN1D, windows: np.ndarray, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(windows), batch):
        xb = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch])).float()
        out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out) if out else np.array([])


def detect_from_edf(
    edf_path: str | Path,
    *,
    checkpoint_path: str | Path,
    channels: list[str],
    target_specificity: float = 0.99,
    min_consecutive: int = 2,
    max_gap: int = 1,
    window_seconds: float = 4.0,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: float = 60.0,
) -> Detection:
    """Corre el pipeline completo sobre un EDF y devuelve los eventos detectados."""
    edf_path = Path(edf_path)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = _build_model(ckpt)
    normalizer = ChannelNormalizer.from_dict(ckpt["normalizer"])

    raw = read_edf(edf_path, preload=True)
    raw = pick_canonical(raw, channels, copy=False)
    raw = preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, copy=False)
    ws = segment_signal(raw.get_data(), float(raw.info["sfreq"]),
                        window_seconds=window_seconds, overlap=0.0)

    windows = normalizer.transform(ws.windows).astype(np.float32)
    scores = _predict(model, windows)

    threshold = percentile_threshold(scores, target_specificity)
    events: list[Event] = aggregate_events(
        scores, threshold, min_consecutive=min_consecutive, max_gap=max_gap
    )
    events_sec = [ev.to_seconds(window_seconds) for ev in events]
    total_seconds = len(scores) * window_seconds
    return Detection(str(edf_path), events_sec, scores, window_seconds, threshold,
                     total_seconds, events_idx=[(ev.start, ev.end) for ev in events],
                     windows=windows)


def true_seizures(edf_path: str | Path) -> list[tuple[float, float]]:
    """Crisis reales del summary (si existe), para superponer en el gráfico/eval."""
    edf_path = Path(edf_path)
    pid = _patient_from_path(edf_path)
    summary_path = edf_path.parent / f"{pid}-summary.txt"
    if not summary_path.exists():
        return []
    info = parse_summary(summary_path).files.get(edf_path.name)
    return [(s.start_sec, s.end_sec) for s in info.seizures] if info else []


def plot_detection(det: Detection, out_path: str | Path, *, truth: list | None = None) -> Path:
    """Grafica score por tiempo, umbral, eventos detectados y (si hay) crisis reales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(det.scores)) * det.window_seconds / 60.0  # minutos
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, det.scores, lw=0.8, color="#4C78A8", label="score de crisis")
    ax.axhline(det.threshold, ls="--", color="k", lw=1, label=f"umbral {det.threshold:.3f}")
    for i, (a, b) in enumerate(det.events_sec):
        ax.axvspan(a/60, b/60, color="#E45756", alpha=0.35,
                   label="detectado" if i == 0 else None)
    if truth:
        for i, (a, b) in enumerate(truth):
            ax.axvspan(a/60, b/60, color="#54A24B", alpha=0.25,
                       label="crisis real" if i == 0 else None)
    ax.set_xlabel("tiempo (min)"); ax.set_ylabel("score"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Detección de crisis — {Path(det.edf_path).name}  "
                 f"({len(det.events_sec)} evento(s))")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return Path(out_path)


DEFAULT_MODEL = Path("models/deployment_v1.pt")

_DEFAULT_PREPROCESS = {"l_freq": 0.5, "h_freq": 40.0, "notch_freq": 60.0}
_DEFAULT_CALIB = {"target_specificity": 0.99, "min_consecutive": 2, "max_gap": 1}


def _meta_from_ckpt(ckpt: dict, *, exp: Path) -> dict:
    """Extrae montaje/preprocess/calibración del checkpoint (autocontenido).

    Un checkpoint de despliegue trae todo adentro. Uno LOSO (de ``run_loso``) no
    trae ``canonical_channels``: se completan desde ``results.json`` + defaults.
    """
    channels = ckpt.get("canonical_channels")
    if channels is None:
        channels = json.loads((exp / "results.json").read_text())["canonical_channels"]
    return {
        "channels": channels,
        "preprocess": ckpt.get("preprocess", _DEFAULT_PREPROCESS),
        "calibration": ckpt.get("calibration", _DEFAULT_CALIB),
        "window_seconds": ckpt.get("window_seconds", 4.0),
        "kind": ckpt.get("kind", "loso"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Detector de crisis EEG (MVP, asistencia).")
    ap.add_argument("edf", help="Ruta al archivo .edf")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="Checkpoint de despliegue autocontenido (corre sobre cualquier EDF).")
    ap.add_argument("--patient", default=None,
                    help="Solo si usás un checkpoint LOSO: chbNN (default: del nombre del archivo).")
    ap.add_argument("--exp", default=str(DEFAULT_EXP), help="Dir con checkpoints LOSO + results.json")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint .pt explícito (override).")
    ap.add_argument("--target-spec", type=float, default=None, help="Override de calibración.")
    ap.add_argument("--min-consecutive", type=int, default=None, help="Override de suavizado.")
    ap.add_argument("--out", default=None, help="PNG de salida (default: <edf>_deteccion.png)")
    ap.add_argument("--explain", action="store_true",
                    help="Genera además una figura de explicabilidad del evento más fuerte.")
    args = ap.parse_args()

    edf_path = Path(args.edf)
    exp = Path(args.exp)

    # elegir checkpoint: explícito > modelo de despliegue > LOSO del paciente
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif Path(args.model).exists():
        ckpt_path = Path(args.model)
    else:
        pid = (args.patient or _patient_from_path(edf_path)).lower()
        ckpt_path = exp / "checkpoints" / f"{pid}.pt"

    ckpt = torch.load(ckpt_path, weights_only=False)
    meta = _meta_from_ckpt(ckpt, exp=exp)
    cal = meta["calibration"]
    target_spec = args.target_spec if args.target_spec is not None else cal["target_specificity"]
    min_cons = args.min_consecutive if args.min_consecutive is not None else cal["min_consecutive"]

    if meta["kind"] == "deployment":
        origen = "modelo de despliegue (todos los pacientes)"
    elif meta["kind"] == "patient-adapted":
        origen = f"modelo adaptado al paciente ({ckpt.get('patient', '?')})"
    else:
        origen = f"checkpoint LOSO {ckpt_path.stem}"
    print(f"Detectando en {edf_path.name} | {origen} | target_spec={target_spec}")
    try:
        det = detect_from_edf(edf_path, checkpoint_path=ckpt_path, channels=meta["channels"],
                              target_specificity=target_spec, min_consecutive=min_cons,
                              max_gap=cal["max_gap"], window_seconds=meta["window_seconds"],
                              **meta["preprocess"])
    except ValueError as e:
        # montaje incompatible (ej. chb12): mensaje claro en vez de traceback
        print(f"\n❌ No puedo procesar este EDF: {e}")
        print(f"   El modelo requiere el montaje canónico de {len(meta['channels'])} "
              f"canales de CHB-MIT. Verificá que el registro tenga esos canales.")
        raise SystemExit(2)
    print("\n" + det.summary())

    truth = true_seizures(edf_path)
    if truth:
        # match simple para el demo: ¿cada crisis real cae dentro de algún evento?
        hits = sum(any(a < eb and ea < b for ea, eb in det.events_sec) for a, b in truth)
        print(f"\n[validación] crisis reales en este archivo: {len(truth)} | "
              f"detectadas: {hits}/{len(truth)}")

    out = args.out or str(edf_path.with_name(edf_path.stem + "_deteccion.png"))
    plot_detection(det, out, truth=truth)
    print(f"\nGráfico -> {out}")

    if args.explain and det.events_idx:
        from neuropilot.explain.saliency import explain_event, plot_explanation
        # evento más fuerte = mayor score máximo dentro del rango
        strongest = max(det.events_idx, key=lambda e: det.scores[e[0]:e[1]].max())
        model = _build_model(ckpt)
        exp = explain_event(model, det.windows, strongest, meta["channels"],
                            window_seconds=meta["window_seconds"])
        exp_out = str(Path(out).with_name(Path(out).stem + "_explicacion.png"))
        plot_explanation(exp, exp_out)
        top = ", ".join(f"{n} ({v:.0%})" for n, v in exp.top_channels(5))
        print(f"\n[explicabilidad] canales que más activaron la detección: {top}")
        print(f"Explicación -> {exp_out}")
    elif args.explain:
        print("\n[explicabilidad] sin detecciones que explicar.")

    print("\n⚠️  Herramienta de asistencia: señala patrones compatibles con crisis; "
          "no es diagnóstico médico.")


if __name__ == "__main__":
    main()
