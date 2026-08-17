"""Entidad primordial E₀ — solo S, Δ_struct, H y posición (protocolo §48)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Entity:
    id: int
    S: float
    delta_struct: float
    H: float
    pos: np.ndarray
    alive: bool = True
    t_hist: float = 0.0
    lineage: int = 0

    def viability(self) -> float:
        return max(0.0, self.S)

    def record_history(self, net_delta: float, kappa_h: float) -> None:
        """Llamado UNA vez por paso con el cambio NETO de Δ_struct del paso.
        Solo la variación OBSERVABLE (≥ κ_H, §22) cuenta como historia persistente
        (§16.3). Así H y t_hist se desacoplan del mero conteo de interacciones (y de S):
        H = suma de cambios estructurales observables; t_hist = pasos observables."""
        if net_delta >= kappa_h:
            self.H += net_delta
            self.t_hist += 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "S": round(self.S, 6),
            "delta_struct": round(self.delta_struct, 6),
            "H": round(self.H, 6),
            "pos": [round(float(x), 3) for x in self.pos],
            "alive": self.alive,
            "t_hist": round(self.t_hist, 2),
            "lineage": self.lineage,
        }