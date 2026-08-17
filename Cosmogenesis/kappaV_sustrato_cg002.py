"""
kappaV_sustrato_cg002.py — ¿es κ_V medible en Cosmogénesis?
==========================================================

κ_V (canon): **A_sys-env ≥ κ_V > 0** — "el acoplamiento con el entorno no puede caer a cero
sin ruptura".

Para medirlo hace falta una PARTICIÓN sistema/entorno que el modelo dé por sí mismo. CG002
acopla nodo con nodo: la lista de N nodos es todo lo que existe. El único término que no es
un nodo es el decaimiento MU (s ← (1−MU)·s), que es un sumidero sin estado y sin
retroacción — no es un entorno, es una pérdida.

Este script convierte esa objeción en un NÚMERO, en vez de dejarla como opinión: si hay que
imponer la partición a mano, entonces A_sys-env es lo que uno haya cortado. Se calculan las
2^(N−1)−1 biparticiones posibles y se mira (a) cuánto vale A en cada una y (b) si el sistema
se rompe cuando A→0.

Salida: kappaV_sustrato_cg002.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from kappas_mortalidad_barrido import KAPPA_S_0, _matriz_g, motor

RAIZ = Path(__file__).resolve().parent
N = 8


def acoplamiento_de_corte(G, x, A_idx, B_idx):
    """A_sys-env de una bipartición = flujo de acoplamiento que cruza el corte,
    normalizado por el flujo total. Es la única lectura posible de 'acoplamiento
    con el entorno' en un modelo que sólo tiene relaciones nodo-nodo."""
    F = np.abs(G) * np.outer(x, x)
    np.fill_diagonal(F, 0.0)
    cruza = F[np.ix_(A_idx, B_idx)].sum() * 2.0
    total = F.sum()
    return float(cruza / total) if total > 0 else 0.0


def main() -> None:
    out = {"pregunta": "¿existe en CG002 una partición sistema/entorno que el modelo dé?",
           "respuesta_estructural": (
               "No. El estado es una lista cerrada de N nodos; el único término no-nodal es "
               "MU, un sumidero sin estado ni retroacción. Cualquier partición hay que "
               "imponerla. Lo que sigue mide cuánto depende A_sys-env de esa imposición."),
           "corridas": []}

    for seed in range(1, 11):
        r = motor(N=N, alpha=1.0, seed=seed)          # corrida original completa
        om = np.asarray(r["omega"])
        G = _matriz_g(om, 8, 0.0, np.float64)
        s = np.asarray(r["S_final"], float)
        x = np.sqrt(np.maximum(s, 0.0)) * (s > KAPPA_S_0)

        vals = []
        for k in range(1, N):
            for A_idx in combinations(range(N), k):
                B_idx = tuple(i for i in range(N) if i not in A_idx)
                vals.append(acoplamiento_de_corte(G, x, list(A_idx), list(B_idx)))
        vals = np.asarray(vals)
        out["corridas"].append({
            "semilla": seed,
            "n_vivos_fin": r["n_vivos"],
            "n_aristas": r["n_aristas"],
            "n_biparticiones": int(vals.size),
            "A_min": float(vals.min()), "A_max": float(vals.max()),
            "A_mediana": float(np.median(vals)),
            "frac_biparticiones_con_A_igual_0": float(np.mean(vals == 0.0)),
            "frac_biparticiones_con_A_menor_1e-3": float(np.mean(vals < 1e-3)),
            "sistema_roto": bool(r["n_vivos"] == 0),
        })

    A0 = [c["frac_biparticiones_con_A_igual_0"] for c in out["corridas"]]
    out["sintesis"] = {
        "frac_media_de_biparticiones_con_A_exactamente_0": float(np.mean(A0)),
        "corridas_con_alguna_biparticion_A_0": int(sum(1 for a in A0 if a > 0)),
        "corridas_rotas": int(sum(1 for c in out["corridas"] if c["sistema_roto"])),
        "lectura": (
            "Si hay biparticiones con A_sys-env = 0 exactamente y el sistema NO se rompe, "
            "entonces 'el acoplamiento con el entorno no puede caer a cero sin ruptura' no "
            "es falsable en CG002: el valor de A es una propiedad del corte que uno eligió, "
            "no del sistema."),
    }
    with (RAIZ / "kappaV_sustrato_cg002.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["sintesis"], indent=2, ensure_ascii=False))
    for c in out["corridas"][:5]:
        print(c)


if __name__ == "__main__":
    main()
