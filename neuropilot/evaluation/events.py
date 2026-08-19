"""Detección y evaluación a nivel de **evento** (no de ventana).

Un neurólogo no evalúa ventanas de 4 s aisladas: evalúa si el sistema **detecta cada
crisis** y **cuántas falsas alarmas por hora** genera. Este módulo convierte las
predicciones por ventana en *eventos* (rangos temporales) y calcula las métricas que
determinan la viabilidad clínica de una herramienta de asistencia:

  - **Sensibilidad por evento**: fracción de crisis reales detectadas (con ≥1 ventana).
  - **Falsas alarmas por hora (FA/h)**: eventos detectados que no son crisis, por hora.

Dos perillas reducen falsas alarmas sin perder crisis:
  - ``min_consecutive``: exige N ventanas positivas seguidas para declarar un evento
    (mata FPs aislados; una crisis dura varias ventanas, así que apenas afecta la
    sensibilidad).
  - ``max_gap``: une eventos separados por huecos cortos (una crisis con un bache de
    score no se parte en dos).

Trabaja en espacio de índices de ventana (con ``overlap=0``, ventana k empieza en
``k * window_seconds``). numpy puro, testeable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    """Un evento como rango de índices de ventana ``[start, end)`` (end exclusivo)."""
    start: int
    end: int

    def overlaps(self, other: "Event") -> bool:
        return self.start < other.end and other.start < self.end

    def to_seconds(self, window_seconds: float) -> tuple[float, float]:
        return self.start * window_seconds, self.end * window_seconds


def _runs(mask: np.ndarray) -> list[Event]:
    """Rachas maximales de ``True`` en un array booleano -> lista de Event."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [Event(int(s), int(e)) for s, e in zip(starts, ends)]


def aggregate_events(
    scores: np.ndarray,
    threshold: float,
    *,
    min_consecutive: int = 1,
    max_gap: int = 0,
) -> list[Event]:
    """Agrupa ventanas positivas (``score >= threshold``) en eventos.

    1) une rachas separadas por <= ``max_gap`` ventanas; 2) descarta las de menos de
    ``min_consecutive`` ventanas.
    """
    mask = np.asarray(scores) >= threshold
    events = _runs(mask)
    if not events:
        return []
    # unir eventos separados por un hueco <= max_gap
    if max_gap > 0:
        merged = [events[0]]
        for ev in events[1:]:
            if ev.start - merged[-1].end <= max_gap:
                merged[-1] = Event(merged[-1].start, ev.end)
            else:
                merged.append(ev)
        events = merged
    # filtrar por duración mínima
    return [ev for ev in events if (ev.end - ev.start) >= min_consecutive]


@dataclass
class EventMetrics:
    n_true_events: int          # crisis reales
    n_detected_true: int        # crisis reales detectadas (>=1 evento predicho encima)
    n_pred_events: int          # eventos predichos totales
    n_false_alarms: int         # eventos predichos que no son crisis
    event_sensitivity: float    # n_detected_true / n_true_events
    false_alarms_per_hour: float
    hours: float

    def as_dict(self) -> dict:
        return {
            "n_true_events": self.n_true_events,
            "n_detected_true": self.n_detected_true,
            "n_pred_events": self.n_pred_events,
            "n_false_alarms": self.n_false_alarms,
            "event_sensitivity": round(self.event_sensitivity, 4),
            "false_alarms_per_hour": round(self.false_alarms_per_hour, 3),
            "hours": round(self.hours, 2),
        }


def event_metrics(
    true_labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    window_seconds: float = 4.0,
    min_consecutive: int = 1,
    max_gap: int = 0,
) -> EventMetrics:
    """Métricas por evento: sensibilidad de detección y falsas alarmas por hora.

    Parameters
    ----------
    true_labels : etiquetas por ventana (1 ictal / 0 normal), en orden temporal.
    scores      : probabilidad de crisis por ventana.
    threshold   : umbral de decisión (elegirlo con calibración por paciente).
    """
    true_labels = np.asarray(true_labels).astype(int)
    true_events = _runs(true_labels == 1)
    pred_events = aggregate_events(
        scores, threshold, min_consecutive=min_consecutive, max_gap=max_gap
    )

    # crisis reales detectadas (>=1 evento predicho que la solapa)
    detected = sum(any(t.overlaps(p) for p in pred_events) for t in true_events)
    # falsas alarmas: eventos predichos que no solapan ninguna crisis real
    false_alarms = sum(not any(p.overlaps(t) for t in true_events) for p in pred_events)

    hours = len(true_labels) * window_seconds / 3600.0
    n_true = len(true_events)
    return EventMetrics(
        n_true_events=n_true,
        n_detected_true=int(detected),
        n_pred_events=len(pred_events),
        n_false_alarms=int(false_alarms),
        event_sensitivity=(detected / n_true) if n_true else float("nan"),
        false_alarms_per_hour=(false_alarms / hours) if hours > 0 else float("nan"),
        hours=hours,
    )
