"""Explicabilidad para el CNN 1D — ¿por qué el modelo marcó esta crisis?

Principio #1 del proyecto: nada de caja negra. Para una herramienta de asistencia,
mostrar **qué canales** y **en qué momento** activaron la detección es parte del valor
(ayuda a localizar el foco y a que un humano valide).

Método: **saliencia por gradiente**. Se calcula la derivada del logit de crisis
respecto de la señal de entrada; su magnitud dice cuánto influye cada muestra
(canal × tiempo) en la decisión. Es barato (un backward por ventana), fiel al modelo
y estándar. Agregando en el tiempo se obtiene importancia por canal; agregando en los
canales, un mapa temporal.

No decide nada: solo explica una detección ya hecha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def window_saliency(model: torch.nn.Module, window: np.ndarray) -> np.ndarray:
    """|∂ logit / ∂ entrada| para una ventana normalizada ``(C, W)`` -> ``(C, W)``."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(window)[None]).float().requires_grad_(True)
    logit = model(x).sum()               # (1,) -> escalar
    model.zero_grad(set_to_none=True)
    logit.backward()
    return x.grad.detach().abs()[0].numpy()   # (C, W)


@dataclass
class Explanation:
    """Explicación de un evento detectado."""
    saliency: np.ndarray            # (C, T) saliencia a lo largo del evento
    channel_importance: np.ndarray  # (C,) importancia agregada por canal (normalizada)
    channels: list[str]
    start_sec: float
    end_sec: float

    def top_channels(self, k: int = 5) -> list[tuple[str, float]]:
        """Los ``k`` canales más influyentes en la detección (nombre, importancia)."""
        order = np.argsort(-self.channel_importance)[:k]
        return [(self.channels[i], float(self.channel_importance[i])) for i in order]


def explain_event(
    model: torch.nn.Module,
    windows: np.ndarray,
    event_idx: tuple[int, int],
    channels: list[str],
    *,
    window_seconds: float = 4.0,
) -> Explanation:
    """Calcula la saliencia agregada sobre las ventanas de un evento detectado.

    Parameters
    ----------
    windows   : ventanas normalizadas ``(N, C, W)`` (las que vio el modelo).
    event_idx : ``(start, end)`` del evento en índices de ventana.
    channels  : nombres de canal (montaje canónico), en el orden de ``windows``.
    """
    start, end = event_idx
    seg = windows[start:end]                      # (n, C, W)
    maps = [window_saliency(model, w) for w in seg]
    saliency = np.concatenate(maps, axis=1)       # (C, n*W) a lo largo del evento
    imp = saliency.sum(axis=1)                    # (C,)
    imp = imp / (imp.sum() + 1e-12)               # normalizada a fracción de contribución
    return Explanation(
        saliency=saliency, channel_importance=imp, channels=list(channels),
        start_sec=start * window_seconds, end_sec=end * window_seconds,
    )


def plot_explanation(exp: Explanation, out_path, *, top_k: int = 8):
    """Figura: ranking de canales + mapa de saliencia (canal × tiempo) del evento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = exp.top_channels(top_k)
    names = [n for n, _ in top][::-1]
    vals = [v for _, v in top][::-1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    axA.barh(names, vals, color="#4C78A8")
    axA.set_xlabel("contribución (fracción)")
    axA.set_title(f"Canales que activaron la detección\n(top {top_k})")

    # mapa de saliencia (canales por importancia, tiempo relativo al evento)
    order = np.argsort(-exp.channel_importance)
    sal = exp.saliency[order]
    t = np.linspace(0, exp.end_sec - exp.start_sec, sal.shape[1])
    im = axB.imshow(sal, aspect="auto", cmap="magma", origin="lower",
                    extent=[t[0], t[-1], 0, len(order)])
    axB.set_yticks(np.arange(len(order)) + 0.5)
    axB.set_yticklabels([exp.channels[i] for i in order], fontsize=6)
    axB.set_xlabel("tiempo dentro del evento (s)")
    axB.set_title(f"Mapa de saliencia — evento {exp.start_sec:.0f}–{exp.end_sec:.0f}s")
    fig.colorbar(im, ax=axB, label="|gradiente|")
    fig.suptitle("¿Por qué el modelo marcó crisis? (saliencia por gradiente)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
