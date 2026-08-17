"""Estado compartido del motor unificado 1→7."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EstadoMotor1a7:
    protocolo: str = "MOTOR_1A7_2026-07-23"
    modo: str = "smoke"
    # Etapa 1–2 (campo / r)
    campo_ok: Optional[bool] = None
    r0_lava: Optional[bool] = None
    r_cruz_ok: Optional[bool] = None
    P_r0: Optional[float] = None
    P_r_high: Optional[float] = None
    z_r_high: Optional[float] = None
    D_campo: Optional[float] = None
    N_campo: Optional[int] = None
    # Etapa 3–4
    stretch_ok: Optional[bool] = None
    rho_ok: Optional[bool] = None
    # Etapa 5–7
    mass_pre_e4_zero: Optional[bool] = None
    e3_ok: Optional[bool] = None
    rate_E3: Optional[float] = None
    e4_lineage_ok: Optional[bool] = None
    rate_e4_lineage: Optional[float] = None
    mass_nulls_clean: Optional[bool] = None
    # Cierre
    stages: dict = field(default_factory=dict)
    chain_pass: Optional[bool] = None
    notes: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EstadoMotor1a7":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
