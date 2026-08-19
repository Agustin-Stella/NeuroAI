"""Web local del MVP NeuroPilot: subís un EDF y ves la detección + explicación.

Es una cáscara fina sobre el motor de inferencia ya construido
(``neuropilot.inference.detector``): FastAPI recibe el archivo, corre el pipeline
con el modelo de despliegue y devuelve los eventos, el gráfico y la explicación.

Correr:
    .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Después abrir http://127.0.0.1:8000
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from neuropilot.explain.saliency import explain_event, plot_explanation
from neuropilot.inference.detector import (
    DEFAULT_MODEL, _build_model, _meta_from_ckpt, detect_from_edf, plot_detection,
)

app = FastAPI(title="NeuroPilot AI — MVP")
STATIC = Path(__file__).parent / "static"

# El modelo de despliegue se carga una sola vez.
_CKPT = torch.load(DEFAULT_MODEL, weights_only=False)
_META = _meta_from_ckpt(_CKPT, exp=Path("experiments/exp_005"))
_CAL = _META["calibration"]


def _png_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": str(DEFAULT_MODEL), "channels": len(_META["channels"])}


@app.get("/legal")
def legal():
    """Sirve el aviso legal completo (LEGAL.md) como texto."""
    return FileResponse("LEGAL.md", media_type="text/plain; charset=utf-8")


@app.post("/api/detect")
def detect(file: UploadFile = File(...)):
    """Corre el detector sobre el EDF subido y devuelve eventos + gráficos (base64)."""
    if not file.filename.lower().endswith(".edf"):
        return JSONResponse({"error": "El archivo debe ser un .edf"}, status_code=400)

    tmp_dir = Path(tempfile.mkdtemp(prefix="neuropilot_"))
    edf_path = tmp_dir / file.filename
    edf_path.write_bytes(file.file.read())

    try:
        det = detect_from_edf(
            edf_path, checkpoint_path=DEFAULT_MODEL, channels=_META["channels"],
            target_specificity=_CAL["target_specificity"], min_consecutive=_CAL["min_consecutive"],
            max_gap=_CAL["max_gap"], window_seconds=_META["window_seconds"], **_META["preprocess"],
        )
    except ValueError as e:
        return JSONResponse(
            {"error": f"Montaje incompatible: {e}. El modelo requiere los "
                      f"{len(_META['channels'])} canales canónicos de CHB-MIT."},
            status_code=422,
        )
    except Exception as e:  # noqa: BLE001 - superficie de error clara para el MVP
        return JSONResponse({"error": f"No pude procesar el archivo: {e}"}, status_code=500)

    timeline_png = tmp_dir / "timeline.png"
    plot_detection(det, timeline_png)

    result = {
        "filename": file.filename,
        "minutes": round(det.total_seconds / 60, 1),
        "threshold": round(det.threshold, 4),
        "n_events": len(det.events_sec),
        "events": [
            {"start_sec": round(a, 1), "end_sec": round(b, 1),
             "start_min": round(a / 60, 2), "dur_sec": round(b - a, 1)}
            for a, b in det.events_sec
        ],
        "timeline_png": _png_b64(timeline_png),
        "explanation_png": None,
        "top_channels": [],
    }

    # explicación del evento más fuerte
    if det.events_idx:
        strongest = max(det.events_idx, key=lambda e: det.scores[e[0]:e[1]].max())
        model = _build_model(_CKPT)
        exp = explain_event(model, det.windows, strongest, _META["channels"],
                            window_seconds=_META["window_seconds"])
        exp_png = tmp_dir / "explanation.png"
        plot_explanation(exp, exp_png)
        result["explanation_png"] = _png_b64(exp_png)
        result["top_channels"] = [
            {"channel": n, "weight": round(w, 4)} for n, w in exp.top_channels(6)
        ]

    return result
