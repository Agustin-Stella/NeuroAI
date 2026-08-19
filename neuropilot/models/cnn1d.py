"""CNN 1D para clasificación de ventanas EEG.

A diferencia del baseline (features de banda hechas a mano), esta red **aprende
sus propias features** directamente de la señal cruda ``(C, W)``. Los canales del
EEG entran como canales de la convolución 1D; la red desliza filtros a lo largo
del tiempo y va combinando/reduciendo hasta una probabilidad de crisis.

Arquitectura: bloques Conv1d → BatchNorm → ReLU → MaxPool que reducen la dimensión
temporal, un pooling promedio global, y una cabeza lineal que produce **un logit**
por ventana (se usa con ``BCEWithLogitsLoss``, que maneja el desbalance vía
``pos_weight``).

Diseñada para comparar de igual a igual contra el baseline con el mismo LOSO.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """CNN 1D para EEG multicanal.

    Parameters
    ----------
    n_channels : cantidad de canales de entrada (ej. 23 tras harmonizar el montaje).
    n_filters  : filtros por bloque convolucional.
    kernel_size: tamaño del kernel temporal (impar).
    pool       : factor de submuestreo temporal por bloque.
    dropout    : dropout antes de la cabeza lineal.

    Forward
    -------
    x : (B, C, W)  ->  logits (B,)   (sin sigmoide; aplicar en la pérdida/inferencia)
    """

    def __init__(
        self,
        n_channels: int = 23,
        *,
        n_filters: Sequence[int] = (32, 64, 128),
        kernel_size: int = 7,
        pool: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar (para padding simétrico)")

        blocks: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in n_filters:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # colapsa el tiempo restante
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Esperaba (B, C, W); recibí shape {tuple(x.shape)}")
        h = self.features(x)  # (B, F, W')
        h = self.global_pool(h).squeeze(-1)  # (B, F)
        return self.head(h).squeeze(-1)  # (B,)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidad de crisis (aplica sigmoide al logit)."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
