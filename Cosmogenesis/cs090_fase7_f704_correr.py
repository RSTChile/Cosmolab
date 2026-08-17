"""
cs090_fase7_f704_correr.py — FASE VII, tarea F7-04: corre Phantom sobre las condiciones iniciales que
escribió `cs090_fase7_f704_brazos.py` (5 brazos × 12 grafos base, N=2000, masa total fija 18800).

NO REIMPLEMENTA NADA: importa `main` de `cs090_fase5b_correr.py` (el runner ya validado de Fase V-B) y
le pasa la lista explícita de carpetas. Ese runner es el que garantiza que TODOS los parámetros de
Phantom sean los mismos de toda la jerarquía CS073 — `icreate_sinks=1`, `rho_crit_cgs=1000`,
`r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001` — reescribiendo el bloque de sumideros de
`cosmog.in` tras el `phantomsetup`, con `assert` de que el bloque por defecto es el esperado (no se
edita a ciegas). Tampoco recomputa: si la carpeta ya tiene `cosmog_00500`, la salta.

USO
---
    ./venv/bin/python cs090_fase7_f704_correr.py                 # todas las carpetas
    ./venv/bin/python cs090_fase7_f704_correr.py --shard=0/4     # un cuarto (para paralelizar)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
from cs090_fase5b_correr import main as correr_phantom   # runner de Fase V-B, sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase7_f704_cortar_bien")


if __name__ == "__main__":
    carpetas = sorted(c for c in BASE.iterdir() if c.is_dir())
    for arg in sys.argv[1:]:
        if arg.startswith("--shard="):
            k, n = (int(x) for x in arg.split("=", 1)[1].split("/"))
            carpetas = [c for i, c in enumerate(carpetas) if i % n == k]
    print(f"[f704] {len(carpetas)} carpetas a correr", flush=True)
    correr_phantom(carpetas)
